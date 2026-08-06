"""Analysis for large-scale checkpoint evaluations (see `rebuttal/tools/eval_ckpts.py`).

Reads the per-ckpt result JSONs written by the eval tool and reports how each
(task, group) setting did, where group is ``RL`` when ``run_id`` looks like
``seed<N>`` and ``BC`` otherwise.

Several results dirs can be loaded at once and tagged, so an old and a new
sweep sit side by side in every table::

    --results runs/eval_lot_results:v1 --results runs/eval_v2_results:v2

Subcommands
-----------
    summary   peak / abs-peak / distribution per (task, group)   [legacy table]
    delta     BC initializer vs. the RL runs it seeded           [headline]
    curve     success rate vs. RL iteration, mean +/- std over seeds
    seeds     one row per RL seed: its best iteration
    top       the K best ckpts of a (task, group), with paths
    select    pick the best BC + RL ckpt per task, emit JSON + a copy script
    report    everything above, plus a self-contained HTML file

Examples
--------
    python rebuttal/tools/analyze_eval.py summary --results runs/eval_lot_results
    python rebuttal/tools/analyze_eval.py delta   --results runs/eval_v2_results:v2 \\
        --baseline runs/weights/selection_v2.json
    python rebuttal/tools/analyze_eval.py curve   --results runs/eval_v2_results
    python rebuttal/tools/analyze_eval.py select  --results runs/eval_lot_results \\
        --out runs/weights/selection_v2.json --plan runs/weights/copy_selection_v2.sh
    python rebuttal/tools/analyze_eval.py report  --results runs/eval_lot_results:v1 \\
        --results runs/eval_v2_results:v2 --html runs/analysis/eval_v2.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # rebuttal/tools/ -> repo root

# Identifies one evaluated checkpoint within one results dir.
CKPT_KEY = ["variant", "task_name", "group", "run_id", "epoch_id"]
AGG_LEVELS = ("auto", "run", "topk", "ckpt")

# task -> (setting index used by policies/config/fs_bc.yaml, short config name).
TASK_INFO: Dict[str, Tuple[int, str]] = {
    "PushT-v2": (0, "pusht"),
    "RollBall-v1": (1, "rollball"),
    "PullCube-v2": (2, "pullcube"),
    "PullCubeTool-v1": (3, "pullcubetool"),
    "PegInsertionSide-v1": (4, "peginsertionside"),
    "PokeCube-v2": (5, "pokecube"),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def group_of(run_id: str) -> str:
    """RL runs are tagged ``seed<N>`` by `parse_run_id`; everything else is BC."""
    return "RL" if re.fullmatch(r"seed\d+", str(run_id)) else "BC"


def parse_results_arg(spec: str) -> Tuple[str, str]:
    """``dir[:label]`` -> (dir, label). Label defaults to the dir basename."""
    if ":" in spec:
        head, _, label = spec.rpartition(":")
        # A Windows-style drive or a bare path with no label keeps the whole spec.
        if head and label and not os.path.isdir(spec):
            return head, label
    return spec, Path(spec.rstrip("/")).name or spec


def resolve_ckpt_path(recorded: str, repo_root: Path = REPO_ROOT) -> str:
    """Re-anchor a recorded ckpt path onto this checkout.

    Results were written from several mounts (``/scratch/...``,
    ``/scache/scratch/...``, ``/common/users/...``), so the stored absolute path
    often does not exist here. Everything lives under a ``runs/`` directory, so
    we re-root at the last ``runs/`` component.
    """
    if os.path.exists(recorded):
        return os.path.abspath(recorded)
    parts = str(recorded).replace(os.sep, "/").split("/runs/")
    if len(parts) > 1:
        cand = repo_root / "runs" / parts[-1]
        if cand.exists():
            return str(cand)
    return str(recorded)


def load_results(specs: Sequence[str]) -> pd.DataFrame:
    """Flatten every result JSON into one row per (ckpt, set)."""
    rows: List[Dict[str, Any]] = []
    n_files = n_skipped = 0
    for spec in specs:
        results_dir, label = parse_results_arg(spec)
        if not os.path.isdir(results_dir):
            print(f"[analyze] no such results dir: {results_dir}", file=sys.stderr)
            continue
        n_here = 0
        for path in sorted(Path(results_dir).rglob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[analyze] skipping {path}: {e}", file=sys.stderr)
                n_skipped += 1
                continue
            if not isinstance(data, dict) or "sets" not in data:
                n_skipped += 1
                continue
            n_files += 1
            n_here += 1
            run_id = str(data.get("run_id", "default"))
            for s in data["sets"]:
                rows.append({
                    "variant": label,
                    "task_name": data.get("task_name", ""),
                    "group": group_of(run_id),
                    "run_id": run_id,
                    "epoch_id": int(data.get("epoch_id", -1)),
                    "set_id": int(s.get("set_id", 0)),
                    "success_rate": float(s.get("success_rate", float("nan"))),
                    "avg_reward": float(s.get("avg_reward", float("nan"))),
                    "n_episodes": int(s.get("n_episodes", 0)),
                    "ckpt_path": str(data.get("ckpt_path", "")),
                    "ckpt_format": str(data.get("ckpt_format", "")),
                    "timestamp": str(data.get("timestamp", "")),
                    "src_json": str(path),
                })
        print(f"[analyze] {results_dir} [{label}]: {n_here} result files", file=sys.stderr)
    df = pd.DataFrame(rows)
    print(
        f"[analyze] loaded {n_files} JSON files ({n_skipped} skipped) -> {len(df)} set rows",
        file=sys.stderr,
    )
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeat evaluations of the same ckpt, keeping the most complete one.

    The legacy tool wrote a timestamped file per run, so re-running a ckpt left
    two JSONs; loading both would double-count it in every distribution.
    """
    if df.empty:
        return df
    per_file = (
        df.groupby(CKPT_KEY + ["src_json"])
          .agg(n_sets=("set_id", "nunique"), timestamp=("timestamp", "max"))
          .reset_index()
          .sort_values(["n_sets", "timestamp"], ascending=[False, False])
    )
    keep = per_file.drop_duplicates(CKPT_KEY, keep="first")[CKPT_KEY + ["src_json"]]
    n_dropped = len(per_file) - len(keep)
    if n_dropped:
        print(f"[analyze] dropped {n_dropped} duplicate ckpt result file(s)", file=sys.stderr)
    return df.merge(keep, on=CKPT_KEY + ["src_json"])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def ckpt_means(df: pd.DataFrame, min_sets: int = 1) -> pd.DataFrame:
    """Mean success rate over sets for every ckpt with >= `min_sets` sets."""
    if df.empty:
        return df
    stats = (
        df.groupby(CKPT_KEY)
          .agg(success_rate=("success_rate", "mean"),
               sr_std=("success_rate", "std"),
               n_sets=("set_id", "nunique"),
               n_episodes=("n_episodes", "sum"),
               ckpt_path=("ckpt_path", "first"),
               ckpt_format=("ckpt_format", "first"),
               src_json=("src_json", "first"))
          .reset_index()
    )
    return stats.loc[stats["n_sets"] >= min_sets].reset_index(drop=True)


def rank_sorted(means: pd.DataFrame, within: Sequence[str] = ()) -> pd.DataFrame:
    """Sort ckpts best-first: highest mean, then tightest spread, then earliest.

    Success rates are averages of multiples of ``1/episodes_per_set``, so exact
    ties are common (two ckpts really can both score 0.741333...). Ranking on the
    raw float would let last-bit noise from ``groupby().mean()`` decide the
    winner, so the primary key is rounded first and the tie is broken on merit.
    """
    out = means.copy()
    out["_sr_rank"] = out["success_rate"].round(9)
    keys = list(within) + ["_sr_rank", "sr_std", "epoch_id", "ckpt_path"]
    asc = [True] * len(within) + [False, True, True, True]
    return out.sort_values(keys, ascending=asc).drop(columns="_sr_rank")


def aggregate_population(means: pd.DataFrame, agg: str = "auto", topk: int = 5) -> pd.DataFrame:
    """Collapse ckpts into the population whose mean / std we report.

    ``run`` keeps each run_id's best epoch, so an RL setting with 15 seeds gives
    15 points. ``topk`` keeps the K best ckpts of the setting, which is how a
    single-run BC pool gets a sample. ``ckpt`` keeps every epoch. ``auto`` picks
    ``run`` for multi-run settings and ``topk`` otherwise.
    """
    if agg not in AGG_LEVELS:
        raise ValueError(f"unknown agg: {agg!r} (expected one of {AGG_LEVELS})")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    setting = ["variant", "task_name", "group"]
    if means.empty:
        out = means.copy()
        for col in ("n_runs", "level", "level_label"):
            out[col] = pd.Series(dtype=object)
        return out

    n_runs = means.groupby(setting)["run_id"].nunique().rename("n_runs").reset_index()
    n_runs["level"] = np.where(n_runs["n_runs"] >= 2, "run", "topk") if agg == "auto" else agg
    n_runs["level_label"] = n_runs["level"].replace({"topk": f"top{topk}"})

    pop = means.merge(n_runs, on=setting)
    run_best = pop.loc[pop.groupby(setting + ["run_id"])["success_rate"].idxmax()]
    best_first = pop.sort_values(
        setting + ["success_rate", "epoch_id"], ascending=[True] * 3 + [False, True]
    )
    out = pd.concat([
        run_best[run_best["level"] == "run"],
        best_first[best_first["level"] == "topk"].groupby(setting, sort=False).head(topk),
        pop[pop["level"] == "ckpt"],
    ])
    return out.sort_values(setting + ["success_rate"], ascending=[True] * 3 + [False]) \
              .reset_index(drop=True)


def compute_summary(
    df: pd.DataFrame, min_sets: int = 1, agg: str = "auto", topk: int = 5
) -> pd.DataFrame:
    """Legacy-compatible peak table, one row per (variant, task, group)."""
    setting = ["variant", "task_name", "group"]
    means = ckpt_means(df, min_sets)
    if means.empty:
        return means

    # Same tie-break as `_pick_best`, so the peak row names the ckpt `select`
    # would actually pick when several ckpts share the top score.
    peak = (
        rank_sorted(means, setting).groupby(setting, sort=False).head(1)
             .rename(columns={"success_rate": "peak_sr", "n_sets": "peak_n_sets"})
             .reset_index(drop=True)
    )
    # Absolute peak: per set take the max over ckpts, then average. An optimistic
    # ceiling -- what you would get if you could pick the best ckpt per set.
    elig = df.merge(means[CKPT_KEY], on=CKPT_KEY)
    abs_peak = (
        elig.groupby(setting + ["set_id"])["success_rate"].max()
            .groupby(setting).mean().reset_index()
            .rename(columns={"success_rate": "abs_peak_sr"})
    )
    n_used = means.groupby(setting).size().rename("n_used").reset_index()
    pop = aggregate_population(means, agg, topk)
    stats = (
        pop.groupby(setting)
           .agg(dist_level=("level_label", "first"),
                dist_n=("success_rate", "size"),
                dist_mean=("success_rate", "mean"),
                dist_std=("success_rate", "std"))  # sample std (ddof=1)
           .reset_index()
    )

    summary = (
        peak.merge(abs_peak, on=setting)
            .merge(n_used, on=setting)
            .merge(stats, on=setting, how="left")
    )
    cols = setting + ["n_used", "run_id", "epoch_id", "peak_n_sets", "peak_sr",
                      "abs_peak_sr", "dist_level", "dist_n", "dist_mean", "dist_std"]
    return summary[cols].sort_values(setting).reset_index(drop=True)


def compute_delta(
    df: pd.DataFrame,
    min_sets: int = 1,
    baseline: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """BC starting point vs. what RL reached, per (variant, task).

    ``bc_sr`` is the BC initializer's success rate: taken from ``--baseline``
    (the actual ckpt RL was seeded from) when available, otherwise the best BC
    ckpt present in the results.
    """
    means = ckpt_means(df, min_sets)
    if means.empty:
        return means
    setting = ["variant", "task_name"]
    rows: List[Dict[str, Any]] = []
    for (variant, task), sub in means.groupby(setting):
        bc = sub[sub["group"] == "BC"]
        rl = sub[sub["group"] == "RL"]
        if rl.empty:
            continue
        run_best = rl.loc[rl.groupby("run_id")["success_rate"].idxmax()]
        rl_peak = _pick_best(rl)

        bc_sr = bc_src = bc_label = None
        if baseline and task in baseline:
            bc_sr, bc_src = float(baseline[task]), "baseline"
            bc_label = "(init)"
        elif not bc.empty:
            best_bc = bc.loc[bc["success_rate"].idxmax()]
            bc_sr, bc_src = float(best_bc["success_rate"]), "results"
            bc_label = f"{best_bc['run_id']}@{int(best_bc['epoch_id'])}"

        rows.append({
            "variant": variant,
            "task_name": task,
            "bc_sr": bc_sr,
            "bc_src": bc_src,
            "bc_ckpt": bc_label,
            "rl_n_seeds": int(run_best["run_id"].nunique()),
            "rl_mean": float(run_best["success_rate"].mean()),
            "rl_std": float(run_best["success_rate"].std()) if len(run_best) > 1 else float("nan"),
            "rl_median": float(run_best["success_rate"].median()),
            "rl_min": float(run_best["success_rate"].min()),
            "rl_peak": float(rl_peak["success_rate"]),
            "rl_peak_ckpt": f"{rl_peak['run_id']}@{int(rl_peak['epoch_id'])}",
            "d_mean": float(run_best["success_rate"].mean()) - bc_sr if bc_sr is not None else None,
            "d_peak": float(rl_peak["success_rate"]) - bc_sr if bc_sr is not None else None,
        })
    return pd.DataFrame(rows).sort_values(setting).reset_index(drop=True)


def compute_curve(df: pd.DataFrame, min_sets: int = 1) -> pd.DataFrame:
    """Mean / std across seeds at each RL iteration, per (variant, task)."""
    means = ckpt_means(df, min_sets)
    if means.empty:
        return means
    rl = means[means["group"] == "RL"]
    if rl.empty:
        return rl
    curve = (
        rl.groupby(["variant", "task_name", "epoch_id"])["success_rate"]
          .agg(n="size", mean="mean", std="std", min="min", max="max")
          .reset_index()
    )
    # `final.pt` is recorded as epoch -1; it is the last point, not the first.
    max_ep = curve.loc[curve["epoch_id"] >= 0, "epoch_id"].max()
    max_ep = 0 if pd.isna(max_ep) else int(max_ep)
    curve["order"] = np.where(curve["epoch_id"] < 0, max_ep + 25, curve["epoch_id"])
    curve["label"] = np.where(curve["epoch_id"] < 0, "final", curve["epoch_id"].astype(str))
    return curve.sort_values(["variant", "task_name", "order"]).reset_index(drop=True)


def compute_seed_table(df: pd.DataFrame, min_sets: int = 1) -> pd.DataFrame:
    """One row per RL seed: which iteration was its best, and by how much."""
    means = ckpt_means(df, min_sets)
    if means.empty:
        return means
    rl = means[means["group"] == "RL"]
    if rl.empty:
        return rl
    best = rl.loc[rl.groupby(["variant", "task_name", "run_id"])["success_rate"].idxmax()].copy()
    final = (
        rl[rl["epoch_id"] < 0][["variant", "task_name", "run_id", "success_rate"]]
        .rename(columns={"success_rate": "final_sr"})
    )
    best = best.merge(final, on=["variant", "task_name", "run_id"], how="left")
    best["seed"] = pd.to_numeric(
        best["run_id"].str.extract(r"seed(\d+)")[0], errors="coerce"
    ).astype("Int64")
    best["n_ckpts"] = best.merge(
        rl.groupby(["variant", "task_name", "run_id"]).size().rename("c").reset_index(),
        on=["variant", "task_name", "run_id"],
    )["c"].values
    best["ckpt_path"] = best["ckpt_path"].map(lambda p: resolve_ckpt_path(p))
    cols = ["variant", "task_name", "run_id", "seed", "n_ckpts", "epoch_id",
            "success_rate", "sr_std", "final_sr", "ckpt_path"]
    return best[cols].sort_values(["variant", "task_name", "seed"]).reset_index(drop=True)


def compute_top(
    df: pd.DataFrame, min_sets: int = 1, k: int = 10, group: Optional[str] = None
) -> pd.DataFrame:
    """The K best ckpts of every (variant, task, group), with resolved paths."""
    means = ckpt_means(df, min_sets)
    if means.empty:
        return means
    if group:
        means = means[means["group"] == group]
    setting = ["variant", "task_name", "group"]
    top = rank_sorted(means, setting).groupby(setting, sort=False).head(k).copy()
    top["rank"] = top.groupby(["variant", "task_name", "group"]).cumcount() + 1
    top["ckpt"] = top["ckpt_path"].map(lambda p: resolve_ckpt_path(p))
    cols = ["variant", "task_name", "group", "rank", "run_id", "epoch_id",
            "n_sets", "success_rate", "sr_std", "ckpt"]
    return top[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def _sets_of(df: pd.DataFrame, row: pd.Series) -> List[List[Any]]:
    sel = df[(df["variant"] == row["variant"]) & (df["task_name"] == row["task_name"])
             & (df["group"] == row["group"]) & (df["run_id"] == row["run_id"])
             & (df["epoch_id"] == row["epoch_id"])]
    return sorted([[int(r.set_id), round(float(r.success_rate), 6)] for r in sel.itertuples()])


def _pick_best(means: pd.DataFrame) -> pd.Series:
    """Highest mean; ties broken by lower spread across sets, then lower epoch."""
    return rank_sorted(means).iloc[0]


def build_selection(
    df: pd.DataFrame,
    min_sets: int = 1,
    bc_root: str = "runs/weights/wmrl_init_v2",
    rl_root: str = "runs/weights/rl_best_v2",
) -> Dict[str, Any]:
    """Best BC + best RL ckpt per task, with everything needed to copy them."""
    means = ckpt_means(df, min_sets)
    if means.empty:
        raise SystemExit("[analyze] no ckpts passed --min-sets; nothing to select")

    n_sets = int(means["n_sets"].max())
    eps = int(round(means["n_episodes"].max() / max(n_sets, 1)))
    out: Dict[str, Any] = {
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "num_sets": n_sets,
        "episodes_per_set": eps,
        "n_seeds": n_sets * eps,
        "weights": "model",  # eval_ckpts scores state_dicts['model'] for BC ckpts
        "bc": {},
        "rl": {},
    }

    for task, sub in means.groupby("task_name"):
        setting, short = TASK_INFO.get(task, (-1, re.sub(r"[^a-z0-9]", "", task.lower())))

        bc = sub[sub["group"] == "BC"]
        if not bc.empty:
            best = _pick_best(bc)
            src = Path(resolve_ckpt_path(str(best["ckpt_path"])))
            # <run_dir>/checkpoints/latest_epochN.ckpt -> <run_dir>
            src_run = src.parent.parent if src.parent.name == "checkpoints" else src.parent
            dst_dir = f"{bc_root}/{setting}_bc_{short}"
            runners = bc.sort_values("success_rate", ascending=False).iloc[1:4]
            out["bc"][task] = {
                "task": task, "setting": setting, "short": short,
                "run_id": str(best["run_id"]), "epoch_id": int(best["epoch_id"]),
                "success_rate": float(best["success_rate"]),
                "std_over_sets": float(best["sr_std"]),
                "n_sets": int(best["n_sets"]),
                "n_candidates": int(len(bc)),
                "runner_up": [
                    {"epoch_id": int(r.epoch_id), "success_rate": float(r.success_rate)}
                    for r in runners.itertuples()
                ],
                "src_ckpt": str(src), "src_run_dir": str(src_run),
                "src_hydra_cfg": str(src_run / ".hydra" / "config.yaml"),
                "dst_dir": dst_dir,
                "dst_ckpt": f"{dst_dir}/checkpoints/{src.name}",
                "sets": _sets_of(df, best),
            }

        rl = sub[sub["group"] == "RL"]
        if not rl.empty:
            best = _pick_best(rl)
            src = Path(resolve_ckpt_path(str(best["ckpt_path"])))
            dst_dir = f"{rl_root}/{short}_{best['run_id']}"
            runners = rl.sort_values("success_rate", ascending=False).iloc[1:4]
            out["rl"][task] = {
                "task": task, "short": short,
                "run_id": str(best["run_id"]), "epoch_id": int(best["epoch_id"]),
                "success_rate": float(best["success_rate"]),
                "std_over_sets": float(best["sr_std"]),
                "n_sets": int(best["n_sets"]),
                "n_candidates": int(len(rl)),
                "runner_up": [
                    {"run_id": str(r.run_id), "epoch_id": int(r.epoch_id),
                     "success_rate": float(r.success_rate)}
                    for r in runners.itertuples()
                ],
                "src_ckpt": str(src), "src_run_dir": str(src.parent),
                "src_config": str(src.parent / "config.yaml"),
                "dst_dir": dst_dir,
                "dst_ckpt": f"{dst_dir}/{src.name}",
                "sets": _sets_of(df, best),
            }
    return out


def render_copy_plan(sel: Dict[str, Any]) -> str:
    """A `cp -n` script: never clobbers, safe to re-run."""
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by rebuttal/tools/analyze_eval.py select -- copies the selected",
        "# checkpoints into runs/weights/. `cp -n` never overwrites; re-running is safe.",
        f"# generated: {sel['generated']}",
        f"# selection basis: {sel['n_seeds']} seeds "
        f"({sel['num_sets']} sets x {sel['episodes_per_set']} episodes), "
        f"BC weights='{sel['weights']}'",
        "set -euo pipefail",
        "# Destinations are repo-relative; override REPO to install elsewhere.",
        f'cd "${{REPO:-{REPO_ROOT}}}"',
        "",
    ]
    for task, e in sorted(sel.get("bc", {}).items()):
        lines += [
            f"# --- BC init | {task} | {e['run_id']} @ epoch {e['epoch_id']} "
            f"| SR {e['success_rate']:.4f} ---",
            f"mkdir -p {e['dst_dir']}/checkpoints {e['dst_dir']}/.hydra",
            f"cp -n {e['src_ckpt']!r} {e['dst_ckpt']!r}",
            f"cp -n {e['src_hydra_cfg']!r} {e['dst_dir']}/.hydra/config.yaml",
            "",
        ]
    for task, e in sorted(sel.get("rl", {}).items()):
        lines += [
            f"# --- RL best | {task} | {e['run_id']} @ epoch {e['epoch_id']} "
            f"| SR {e['success_rate']:.4f} ---",
            f"mkdir -p {e['dst_dir']}",
            f"cp -n {e['src_ckpt']!r} {e['dst_ckpt']!r}",
            f"cp -n {e['src_config']!r} {e['dst_dir']}/config.yaml",
            "",
        ]
    lines.append('echo "[copy_selection] done"')
    return "\n".join(lines) + "\n"


def render_selection_md(sel: Dict[str, Any]) -> str:
    out = [
        "# Selected checkpoints",
        "",
        f"Generated {sel['generated']} by `rebuttal/tools/analyze_eval.py select`.",
        "",
        f"Selection basis: **{sel['n_seeds']} seeds** "
        f"({sel['num_sets']} sets x {sel['episodes_per_set']} episodes, seeds "
        f"1..{sel['n_seeds']}), deterministic actions, BC weights "
        f"`state_dicts['{sel['weights']}']`.",
        "",
        "## BC initializers (for WMRL `pretrained_ckpt`)",
        "",
        "| Task | setting | run_id | epoch | SR | std over sets | candidates | destination |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task, e in sorted(sel.get("bc", {}).items()):
        out.append(
            f"| {task} | {e['setting']} | `{e['run_id']}` | {e['epoch_id']} | "
            f"**{e['success_rate']:.4f}** | {e['std_over_sets']:.3f} | {e['n_candidates']} | "
            f"`{e['dst_ckpt']}` |"
        )
    out += [
        "",
        "## Best RL checkpoints (for eval-tool testing)",
        "",
        "| Task | run_id | epoch | SR | std over sets | candidates | destination |",
        "|---|---|---|---|---|---|---|",
    ]
    for task, e in sorted(sel.get("rl", {}).items()):
        ep = "final" if e["epoch_id"] < 0 else str(e["epoch_id"])
        out.append(
            f"| {task} | `{e['run_id']}` | {ep} | **{e['success_rate']:.4f}** | "
            f"{e['std_over_sets']:.3f} | {e['n_candidates']} | `{e['dst_ckpt']}` |"
        )
    out += ["", "## Provenance", ""]
    for kind in ("bc", "rl"):
        for task, e in sorted(sel.get(kind, {}).items()):
            out.append(f"- **{kind.upper()} {task}** &larr; `{e['src_ckpt']}`")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #

def print_table(df: pd.DataFrame, title: str) -> None:
    print()
    print(f"### {title}")
    print()
    if df is None or df.empty:
        print("(no rows)")
        return
    print(df.to_markdown(index=False, floatfmt=".4f"))


def print_curves(curve: pd.DataFrame, width: int = 46) -> None:
    """Per (variant, task): success rate against RL iteration, as an ASCII bar."""
    print()
    print("### Success rate vs. RL iteration (mean +/- std over seeds)")
    if curve.empty:
        print("\n(no RL results)")
        return
    for (variant, task), sub in curve.groupby(["variant", "task_name"]):
        lo = float(max(0.0, (sub["mean"] - sub["std"].fillna(0)).min() - 0.02))
        hi = float(min(1.0, (sub["mean"] + sub["std"].fillna(0)).max() + 0.02))
        span = max(hi - lo, 1e-6)
        best = sub.loc[sub["mean"].idxmax()]
        print()
        print(f"{task}  [{variant}]  n_seeds={int(sub['n'].max())}  "
              f"best iter={best['label']} mean={best['mean']:.3f}  "
              f"axis=[{lo:.2f}, {hi:.2f}]")
        for r in sub.itertuples():
            pos = int(round(width * (r.mean - lo) / span))
            std = 0.0 if (r.std is None or math.isnan(r.std)) else r.std
            wl = int(round(width * max(r.mean - std, lo) / span - width * lo / span))
            wr = int(round(width * min(r.mean + std, hi) / span - width * lo / span))
            bar = [" "] * (width + 1)
            for i in range(max(wl, 0), min(wr, width) + 1):
                bar[i] = "-"
            bar[min(max(pos, 0), width)] = "*" if r.Index == best.name else "o"
            print(f"    iter {r.label:>5s} | n={int(r.n):>2d} | {''.join(bar)} | "
                  f"{r.mean:.3f} +/- {std:.3f}")


def print_delta(delta: pd.DataFrame) -> None:
    print()
    print("### BC initializer vs. RL")
    print("    rl_mean = mean over seeds of each seed's best iteration")
    print("    d_mean  = rl_mean - bc_sr    (bc_src='baseline' -> the actual init ckpt)")
    if delta.empty:
        print("\n(no RL results)")
        return
    cols = ["variant", "task_name", "bc_sr", "bc_src", "bc_ckpt", "rl_n_seeds",
            "rl_mean", "rl_std", "rl_median", "rl_min", "rl_peak", "rl_peak_ckpt",
            "d_mean", "d_peak"]
    print()
    print(delta[cols].to_markdown(index=False, floatfmt=".4f"))


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--fg:#111827;--muted:#6b7280;--card:#fff;--line:#e5e7eb;
--accent:#2563eb;--best:#f59e0b;--band:#c7d7fb;--pos:#059669;--neg:#dc2626;--hi:#fef3c7}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e5e7eb;--muted:#9ca3af;
--card:#161a21;--line:#272d38;--accent:#60a5fa;--best:#fbbf24;--band:#2b3d63;
--pos:#34d399;--neg:#f87171;--hi:#2f2a15}}
*{box-sizing:border-box}
body{margin:0;padding:20px;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto}
h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:26px 0 8px}
.meta{color:var(--muted);font-size:12px}
.meta code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 4px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:var(--card);
border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:10px 0}
input,select,button{font:inherit;color:inherit;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:5px 8px}
button{cursor:pointer}
label{font-size:12px;color:var(--muted);display:inline-flex;align-items:center;gap:5px}
#q{min-width:260px;flex:1}
.tablewrap{max-height:70vh;overflow:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);
font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}
th.l,td.l{text-align:left}
th{position:sticky;top:0;z-index:2;background:var(--card);cursor:pointer;user-select:none;
font-size:12px;font-weight:600;color:var(--muted)}
th:hover{color:var(--fg)}tbody tr:hover{background:var(--hi)}
tr.peak td{font-weight:600}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.badge{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:0 6px;font-size:11px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h3{font-size:14px;margin:0 0 2px}.card .sub{font-size:12px;color:var(--muted)}
svg{display:block;width:100%;height:170px;margin-top:6px}
.empty{color:var(--muted);padding:12px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1><div class="meta" id="meta"></div>
<h2>BC initializer vs. RL</h2>
<div class="meta">rl_mean averages, over seeds, each seed's best iteration.</div>
<div class="tablewrap"><table id="delta"></table></div>
<h2>Success rate vs. RL iteration</h2>
<div class="meta">Line = mean over seeds; band = &plusmn;1 std; dot = best iteration.</div>
<div class="cards" id="curves"></div>
<h2>Peak summary per (task, group)</h2>
<div class="tablewrap"><table id="summary"></table></div>
<h2>Checkpoint lookup <span class="meta" id="count"></span></h2>
<div class="controls">
  <input id="q" type="search" placeholder="search: task / run_id / group / variant (space = AND)">
  <label>group <select id="f-group"></select></label>
  <label>variant <select id="f-variant"></select></label>
  <label>SR &ge; <input id="f-sr" size="5" placeholder="0.0"></label>
  <button id="csv">download CSV</button>
</div>
<div class="tablewrap"><table id="ckpts"></table></div>
</div>
<script id="report-data" type="application/json">__DATA__</script>
<script>
(function(){
const D=JSON.parse(document.getElementById('report-data').textContent);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const f3=v=>(v===null||v===undefined||Number.isNaN(v))?'&mdash;':Number(v).toFixed(3);
const f4=v=>(v===null||v===undefined||Number.isNaN(v))?'&mdash;':Number(v).toFixed(4);
const sgn=v=>(v===null||v===undefined||Number.isNaN(v))?'&mdash;':
  `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${Number(v).toFixed(3)}</span>`;
const badge=g=>`<span class="badge">${esc(g)}</span>`;
document.getElementById('meta').innerHTML=D.meta_html;

function table(el,cols,rows,opts){
  opts=opts||{};let sk=opts.sortKey,sd=opts.sortDir||1;
  el.innerHTML='<thead><tr>'+cols.map((c,i)=>
    `<th data-i="${i}" class="${c.l?'l':''}" title="${esc(c.title||c.label)}">${esc(c.label)}</th>`
  ).join('')+'</tr></thead><tbody></tbody>';
  const tb=el.querySelector('tbody');
  el.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{
    const k=cols[+th.dataset.i].key;sd=(k===sk)?-sd:1;sk=k;draw();}));
  function draw(){
    let rs=(typeof rows==='function'?rows():rows).slice();
    if(sk)rs.sort((a,b)=>{const x=a[sk],y=b[sk];if(x===y)return 0;
      if(typeof x==='number'&&typeof y==='number')return (x-y)*sd;
      return String(x).localeCompare(String(y))*sd;});
    el.querySelectorAll('th').forEach((th,i)=>{const c=cols[i];
      th.textContent=c.label+(c.key===sk?(sd>0?' \\u25b2':' \\u25bc'):'');});
    if(!rs.length){tb.innerHTML=`<tr><td class="empty l" colspan="${cols.length}">no matching rows</td></tr>`;return;}
    tb.innerHTML=rs.map(r=>'<tr class="'+(r.rank===1?'peak':'')+'">'+cols.map(c=>
      `<td class="${c.l?'l':''}">${c.render?c.render(r):esc(r[c.key])}</td>`).join('')+'</tr>').join('');
  }
  draw();return {draw};
}

table(document.getElementById('delta'),[
 {key:'variant',label:'variant',l:true},{key:'task_name',label:'task',l:true},
 {key:'bc_sr',label:'bc_sr',render:r=>f4(r.bc_sr)},
 {key:'bc_ckpt',label:'bc ckpt',l:true},
 {key:'rl_n_seeds',label:'seeds'},
 {key:'rl_mean',label:'rl_mean',render:r=>f4(r.rl_mean)},
 {key:'rl_std',label:'rl_std',render:r=>f3(r.rl_std)},
 {key:'rl_peak',label:'rl_peak',render:r=>f4(r.rl_peak)},
 {key:'rl_peak_ckpt',label:'peak ckpt',l:true},
 {key:'d_mean',label:'\\u0394 mean',render:r=>sgn(r.d_mean)},
 {key:'d_peak',label:'\\u0394 peak',render:r=>sgn(r.d_peak)},
],D.delta,{sortKey:'task_name'});

table(document.getElementById('summary'),[
 {key:'variant',label:'variant',l:true},{key:'task_name',label:'task',l:true},
 {key:'group',label:'group',l:true,render:r=>badge(r.group)},
 {key:'n_used',label:'n_used'},{key:'run_id',label:'best run_id',l:true},
 {key:'epoch_id',label:'best epoch'},
 {key:'peak_sr',label:'peak_sr',render:r=>f4(r.peak_sr)},
 {key:'abs_peak_sr',label:'abs_peak_sr',title:'per set take max over ckpts, then average',
  render:r=>f4(r.abs_peak_sr)},
 {key:'dist_level',label:'unit',l:true},{key:'dist_n',label:'dist_n'},
 {key:'dist_mean',label:'dist_mean',render:r=>f4(r.dist_mean)},
 {key:'dist_std',label:'dist_std',render:r=>f3(r.dist_std)},
],D.summary,{sortKey:'task_name'});

/* --- curves as inline SVG --- */
const W=440,H=170,P={t:12,r:12,b:26,l:40};
document.getElementById('curves').innerHTML=D.curves.length?D.curves.map(c=>{
  const xs=c.points.map((p,i)=>i), n=c.points.length;
  const lo=Math.max(0,Math.min(...c.points.map(p=>p.mean-(p.std||0)))-0.02);
  const hi=Math.min(1,Math.max(...c.points.map(p=>p.mean+(p.std||0)))+0.02);
  const sx=i=>P.l+(n<2?0:(W-P.l-P.r)*i/(n-1));
  const sy=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/Math.max(hi-lo,1e-6));
  const up=c.points.map((p,i)=>`${sx(i)},${sy(Math.min(hi,p.mean+(p.std||0)))}`);
  const dn=c.points.map((p,i)=>`${sx(i)},${sy(Math.max(lo,p.mean-(p.std||0)))}`).reverse();
  const line=c.points.map((p,i)=>`${sx(i)},${sy(p.mean)}`).join(' ');
  const bi=c.points.reduce((b,p,i)=>p.mean>c.points[b].mean?i:b,0);
  const ticks=[lo,(lo+hi)/2,hi].map(v=>
    `<line x1="${P.l}" x2="${W-P.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--line)"/>
     <text x="${P.l-6}" y="${sy(v)+4}" text-anchor="end" font-size="10" fill="var(--muted)">${v.toFixed(2)}</text>`
  ).join('');
  const xlab=c.points.map((p,i)=>(n<=8||i%Math.ceil(n/7)===0||i===n-1)?
    `<text x="${sx(i)}" y="${H-8}" text-anchor="middle" font-size="10" fill="var(--muted)">${esc(p.label)}</text>`:'').join('');
  return `<div class="card"><h3>${esc(c.task)} ${badge(c.variant)}</h3>
    <div class="sub">${c.n_seeds} seeds &nbsp; best iter <b>${esc(c.points[bi].label)}</b>
      = ${f3(c.points[bi].mean)} &plusmn; ${f3(c.points[bi].std)}</div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${ticks}
      <polygon points="${up.concat(dn).join(' ')}" fill="var(--band)" opacity="0.55"/>
      <polyline points="${line}" fill="none" stroke="var(--accent)" stroke-width="2"/>
      <circle cx="${sx(bi)}" cy="${sy(c.points[bi].mean)}" r="4" fill="var(--best)"/>
      ${xlab}</svg></div>`;
}).join(''):'<div class="empty">no RL results</div>';

/* --- ckpt lookup --- */
const CK=D.ckpts;
const q=document.getElementById('q'),fg=document.getElementById('f-group'),
      fv=document.getElementById('f-variant'),fs=document.getElementById('f-sr');
const fill=(s,v)=>s.innerHTML='<option value="">all</option>'+v.map(x=>`<option>${esc(x)}</option>`).join('');
fill(fg,[...new Set(CK.map(r=>r.group))].sort());
fill(fv,[...new Set(CK.map(r=>r.variant))].sort());
const hay=r=>(r.variant+' '+r.task_name+' '+r.group+' '+r.run_id+' ep'+r.epoch_id).toLowerCase();
function filtered(){
  const tk=q.value.toLowerCase().split(/\\s+/).filter(Boolean);const mn=parseFloat(fs.value);
  return CK.filter(r=>{
    if(fg.value&&r.group!==fg.value)return false;
    if(fv.value&&r.variant!==fv.value)return false;
    if(!Number.isNaN(mn)&&r.success_rate<mn)return false;
    if(tk.length){const h=hay(r);if(!tk.every(t=>h.includes(t)))return false;}
    return true;});
}
const ct=table(document.getElementById('ckpts'),[
 {key:'variant',label:'variant',l:true},{key:'task_name',label:'task',l:true},
 {key:'group',label:'group',l:true,render:r=>badge(r.group)},
 {key:'run_id',label:'run_id',l:true},{key:'epoch_id',label:'epoch'},
 {key:'n_sets',label:'n_sets'},
 {key:'success_rate',label:'success_rate',render:r=>f4(r.success_rate)},
 {key:'sr_std',label:'std',render:r=>f3(r.sr_std)},
 {key:'rank',label:'rank',title:'rank within (variant, task, group)'},
],filtered,{sortKey:'task_name'});
function refresh(){ct.draw();
  document.getElementById('count').textContent=
    `\\u2014 ${filtered().length} of ${CK.length} ckpts`;}
[q,fg,fv,fs].forEach(e=>{e.addEventListener('input',refresh);e.addEventListener('change',refresh);});
document.getElementById('csv').addEventListener('click',()=>{
  const h=['variant','task_name','group','run_id','epoch_id','n_sets','success_rate','sr_std','rank'];
  const b=filtered().map(r=>h.map(k=>r[k]).join(',')).join('\\n');
  const u=URL.createObjectURL(new Blob([h.join(',')+'\\n'+b],{type:'text/csv'}));
  const a=document.createElement('a');a.href=u;a.download='eval_ckpts.csv';a.click();URL.revokeObjectURL(u);});
refresh();
})();
</script></body></html>
"""


def render_html(
    summary: pd.DataFrame, delta: pd.DataFrame, curve: pd.DataFrame,
    ckpts: pd.DataFrame, meta: Dict[str, Any], title: str = "Eval results",
) -> str:
    def recs(df: pd.DataFrame) -> List[Dict[str, Any]]:
        return json.loads(df.to_json(orient="records") or "[]") if not df.empty else []

    curves: List[Dict[str, Any]] = []
    if not curve.empty:
        for (variant, task), sub in curve.groupby(["variant", "task_name"]):
            curves.append({
                "variant": variant, "task": task, "n_seeds": int(sub["n"].max()),
                "points": [
                    {"label": str(r.label), "mean": float(r.mean),
                     "std": 0.0 if pd.isna(r.std) else float(r.std)}
                    for r in sub.itertuples()
                ],
            })
    payload = {
        "summary": recs(summary), "delta": recs(delta),
        "curves": curves, "ckpts": recs(ckpts),
        "meta_html": " &nbsp;|&nbsp; ".join(
            f"{html_lib.escape(k)}: <code>{html_lib.escape(str(v))}</code>"
            for k, v in meta.items()
        ),
    }
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return _HTML.replace("__TITLE__", html_lib.escape(title)).replace("__DATA__", data)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--results", action="append", required=True, metavar="DIR[:LABEL]",
        help="Results dir, optionally tagged. Repeatable, e.g. "
             "`--results runs/eval_lot_results:v1 --results runs/eval_v2_results:v2`.",
    )
    p.add_argument("--min-sets", type=int, default=1,
                   help="Require this many completed sets for a ckpt to count.")
    p.add_argument("--task", default=None, help="Substring filter on task_name.")
    p.add_argument("--group", default=None, choices=["BC", "RL"], help="Keep only this group.")
    p.add_argument("--csv", default=None, metavar="PATH", help="Also write the table as CSV.")
    return p


def load_filtered(args: argparse.Namespace) -> pd.DataFrame:
    df = dedupe(load_results(args.results))
    if df.empty:
        raise SystemExit("[analyze] no rows loaded")
    if getattr(args, "task", None):
        df = df[df["task_name"].str.contains(args.task, case=False, na=False)]
    if getattr(args, "group", None):
        df = df[df["group"] == args.group]
    if df.empty:
        raise SystemExit("[analyze] filters removed every row")
    return df


def load_baseline(path: Optional[str]) -> Optional[Dict[str, float]]:
    """Map task -> BC initializer success rate, from a selection JSON."""
    if not path:
        return None
    with open(path) as f:
        sel = json.load(f)
    return {t: float(e["success_rate"]) for t, e in sel.get("bc", {}).items()}


def maybe_csv(df: pd.DataFrame, path: Optional[str]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[analyze] wrote {out}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = _common(sub.add_parser("summary", help="Peak / distribution per (task, group)."))
    ps.add_argument("--agg", default="auto", choices=AGG_LEVELS,
                    help="Population for dist_mean / dist_std.")
    ps.add_argument("--topk", type=int, default=5, help="K for the 'topk' unit.")

    pd_ = _common(sub.add_parser("delta", help="BC initializer vs. the RL runs it seeded."))
    pd_.add_argument("--baseline", default=None, metavar="SELECTION_JSON",
                     help="selection JSON from `select`; its BC success rates become bc_sr.")

    _common(sub.add_parser("curve", help="Success rate vs. RL iteration."))
    _common(sub.add_parser("seeds", help="One row per RL seed."))

    pt = _common(sub.add_parser("top", help="The K best ckpts per (task, group)."))
    pt.add_argument("-k", "--k", type=int, default=10)

    psel = _common(sub.add_parser("select", help="Pick the best BC + RL ckpt per task."))
    psel.add_argument("--out", default="runs/weights/selection_v2.json", metavar="PATH")
    psel.add_argument("--plan", default=None, metavar="PATH",
                      help="Also write a `cp -n` shell script that materializes the picks.")
    psel.add_argument("--md", default=None, metavar="PATH", help="Also write a markdown summary.")
    psel.add_argument("--bc-root", default="runs/weights/wmrl_init_v2")
    psel.add_argument("--rl-root", default="runs/weights/rl_best_v2")

    pr = _common(sub.add_parser("report", help="All tables, plus an HTML file."))
    pr.add_argument("--html", default="runs/analysis/eval_report.html", metavar="PATH")
    pr.add_argument("--baseline", default=None, metavar="SELECTION_JSON")
    pr.add_argument("--agg", default="auto", choices=AGG_LEVELS)
    pr.add_argument("--topk", type=int, default=5)
    pr.add_argument("-k", "--k", type=int, default=10, help="Rows per group in the lookup table.")

    args = p.parse_args(argv)
    pd.set_option("display.width", 220)
    df = load_filtered(args)

    if args.cmd == "summary":
        out = compute_summary(df, args.min_sets, args.agg, args.topk)
        print_table(out, "Peak summary per (variant, task, group)")
        maybe_csv(out, args.csv)

    elif args.cmd == "delta":
        out = compute_delta(df, args.min_sets, load_baseline(args.baseline))
        print_delta(out)
        maybe_csv(out, args.csv)

    elif args.cmd == "curve":
        out = compute_curve(df, args.min_sets)
        print_curves(out)
        maybe_csv(out, args.csv)

    elif args.cmd == "seeds":
        out = compute_seed_table(df, args.min_sets)
        print_table(out, "Best iteration per RL seed")
        maybe_csv(out, args.csv)

    elif args.cmd == "top":
        out = compute_top(df, args.min_sets, args.k, args.group)
        print_table(out, f"Top {args.k} ckpts per (variant, task, group)")
        maybe_csv(out, args.csv)

    elif args.cmd == "select":
        sel = build_selection(df, args.min_sets, args.bc_root, args.rl_root)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
        print(render_selection_md(sel))
        print(f"[analyze] wrote {out_path}", file=sys.stderr)
        if args.md:
            Path(args.md).parent.mkdir(parents=True, exist_ok=True)
            Path(args.md).write_text(render_selection_md(sel), encoding="utf-8")
            print(f"[analyze] wrote {args.md}", file=sys.stderr)
        if args.plan:
            plan = Path(args.plan)
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text(render_copy_plan(sel), encoding="utf-8")
            plan.chmod(0o755)
            print(f"[analyze] wrote {plan}  (run it to copy the checkpoints)", file=sys.stderr)

    elif args.cmd == "report":
        summary = compute_summary(df, args.min_sets, args.agg, args.topk)
        delta = compute_delta(df, args.min_sets, load_baseline(args.baseline))
        curve = compute_curve(df, args.min_sets)
        top = compute_top(df, args.min_sets, args.k, args.group)
        print_table(summary, "Peak summary per (variant, task, group)")
        print_delta(delta)
        print_curves(curve)
        print_table(top, f"Top {args.k} ckpts per (variant, task, group)")
        maybe_csv(top, args.csv)
        meta = {
            "results": ", ".join(args.results),
            "min sets": args.min_sets,
            "task filter": args.task or "(none)",
            "dist unit": f"{args.agg} (k={args.topk})",
            "ckpts": len(top),
            "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        html_out = Path(args.html)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(
            render_html(summary, delta, curve, top, meta), encoding="utf-8"
        )
        print(f"\n[analyze] wrote HTML report -> {html_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
