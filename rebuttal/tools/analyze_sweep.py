#!/usr/bin/env python3
"""Tables and significance tests for the WMRL v2 evaluation sweep.

Reads the per-checkpoint JSONs written by ``rebuttal/tools/eval_ckpts.py`` and produces:

  A  headline    per task: BC initializer vs. mean/max over the 15 RL seed-bests
  A2 split-half  the same comparison with the max-over-checkpoints bias removed
  B  trajectory  length of *successful* trajectories, BC vs. post-WMRL
  C  tests       significance at two granularities, and combined across tasks

    python rebuttal/tools/analyze_sweep.py
    python rebuttal/tools/analyze_sweep.py --results runs/eval_v2/results --min-sets 3

Two things about the design are worth stating up front, because every number below
depends on them.

**Every model is evaluated on the same eval seeds (1..N).** ``env.reset(seed=s)`` gives
identical initial states, so BC and RL face the same problems and comparisons are
*paired*. That is why the episode-level test is McNemar on discordant pairs rather than a
two-sample proportion test, and why trajectory lengths can be compared on the subset both
policies solve.

**Max-over-checkpoints is optimistically biased, and BC has no such maximum.** Each RL
seed contributes the best of its 12 checkpoints; the BC initializer is a single
checkpoint (the rest of its epoch sweep no longer exists on disk). Comparing the two
directly flatters RL by construction, before any real effect. Table A reports it anyway
because it is the protocol asked for; Table A2 reports the cross-fitted version, which
selects each seed's checkpoint on one half of the episodes and scores it on the other.
The gap between them is the size of the bias.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]  # rebuttal/tools/ -> repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from rebuttal.tools import analyze_eval as A  # noqa: E402

TASK_ORDER = ["PushT-v2", "RollBall-v1", "PullCube-v2", "PullCubeTool-v1", "PokeCube-v2"]
ALPHA = 0.05
N_BOOT = 20000


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values. scipy has BH but not Holm."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> Tuple[float, float]:
    """Exact binomial CI. Used for the BC success rate, which is *not* a constant --
    it is a proportion over n episodes and carries its own uncertainty."""
    lo = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def mcnemar(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Exact McNemar for paired binary outcomes: does `a` beat `b`?

    Only discordant pairs carry information. scipy has no `mcnemar`, but the exact test
    *is* a binomial test on those pairs, so `binomtest` is not an approximation of it.
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    n01 = int((a & ~b).sum())   # a succeeds, b fails
    n10 = int((~a & b).sum())   # b succeeds, a fails
    n = n01 + n10
    p = float(stats.binomtest(n01, n, 0.5, alternative="greater").pvalue) if n else 1.0
    return {"a_only": n01, "b_only": n10, "n_discordant": n, "p": p,
            "delta": float(a.mean() - b.mean())}


def wilcoxon_safe(d: Sequence[float], alternative: str = "greater") -> Dict[str, Any]:
    """Signed-rank test that copes with ties.

    Episode lengths are integers in [1, 100], so paired differences tie constantly and
    the exact null distribution -- which assumes none -- would be invalid. Fall back to
    the normal approximation with continuity correction whenever ties are present, and
    report how many zero-differences were dropped, because a reviewer will ask.
    """
    d = np.asarray(d, float)
    nz = d[d != 0]
    n_zero = int(len(d) - len(nz))
    if nz.size == 0:
        return {"p": 1.0, "n": 0, "n_zero": n_zero, "method": "n/a"}
    has_ties = len(np.unique(np.abs(nz))) != len(nz)
    try:
        if has_ties:
            r = stats.wilcoxon(nz, alternative=alternative, method="approx",
                               correction=True)
            method = "approx (ties)"
        else:
            r = stats.wilcoxon(nz, alternative=alternative, method="exact")
            method = "exact"
        return {"p": float(r.pvalue), "n": int(nz.size), "n_zero": n_zero,
                "method": method}
    except ValueError:
        return {"p": float("nan"), "n": int(nz.size), "n_zero": n_zero, "method": "failed"}


def boot_ci(x: Sequence[float], statistic=np.mean) -> Tuple[float, float]:
    """95% bootstrap CI, BCa where it is defined and percentile where it is not.

    BCa's acceleration comes from a jackknife and is undefined when the statistic barely
    moves under leave-one-out -- degenerate data, or an order statistic like `max`. Never
    let that turn into a silent nan.
    """
    x = np.asarray(x, float)
    if x.size < 2 or np.allclose(x, x[0]):
        return float(x.mean()), float(x.mean())
    for method in ("BCa", "percentile"):
        try:
            r = stats.bootstrap((x,), statistic, confidence_level=1 - ALPHA,
                                n_resamples=N_BOOT, method=method, random_state=0)
            lo, hi = float(r.confidence_interval.low), float(r.confidence_interval.high)
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi
        except Exception:
            continue
    return float("nan"), float("nan")


def two_stage_boot(seed_best: np.ndarray, bc_episodes: np.ndarray) -> Tuple[float, float]:
    """CI for (mean of RL seed-bests) - (BC success rate), resampling both.

    A one-sample test against `bc_sr` treats BC as a known constant. It is not: it is a
    proportion over the evaluation set's Bernoulli episodes, SE around 0.012-0.018. Resampling the
    seeds *and* BC's episodes propagates both sources of uncertainty into the difference.
    """
    rng = np.random.default_rng(0)
    k, m = len(seed_best), len(bc_episodes)
    s = seed_best[rng.integers(0, k, size=(N_BOOT, k))].mean(axis=1)
    b = bc_episodes[rng.integers(0, m, size=(N_BOOT, m))].mean(axis=1)
    lo, hi = np.percentile(s - b, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

class Model:
    """One evaluated checkpoint, with its per-episode outcomes as dense arrays."""

    __slots__ = ("task", "group", "run_id", "seed", "epoch", "n_sets", "path",
                 "success", "length", "terminated", "set_of", "horizon", "sim_freq")

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def sr(self) -> float:
        return float(self.success.mean())

    def sr_on(self, mask: np.ndarray) -> float:
        return float(self.success[mask].mean())

    def __repr__(self) -> str:
        return f"<{self.task} {self.run_id}@{self.epoch} sr={self.sr:.3f}>"


def load_models(results_dir: str, min_sets: int) -> Tuple[List[Model], Dict[str, Any]]:
    """Read every result JSON that has per-episode data for at least `min_sets` sets."""
    notes: Dict[str, Any] = {"files": 0, "no_episodes": 0, "too_few_sets": 0,
                             "unreadable": 0, "duplicates": 0}
    by_key: Dict[Tuple[str, str, int], Model] = {}

    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            notes["unreadable"] += 1
            continue
        if not isinstance(d, dict) or "sets" not in d:
            notes["unreadable"] += 1
            continue
        notes["files"] += 1

        rows = {}
        for s in d["sets"]:
            ep = s.get("episodes")
            if isinstance(ep, dict) and len(ep.get("seed", [])) == s.get("n_episodes"):
                rows[int(s["set_id"])] = s
        if not rows:
            notes["no_episodes"] += 1
            continue
        # With --min-sets, take sets 1..min_sets from every checkpoint, so all models are
        # compared on identical episodes even when some were evaluated more deeply.
        keep = [k for k in sorted(rows) if k <= min_sets] if min_sets else sorted(rows)
        if len(keep) < min_sets:
            notes["too_few_sets"] += 1
            continue

        seeds, succ, leng, term, setid = [], [], [], [], []
        horizons, freqs = set(), set()
        for k in keep:
            s = rows[k]
            e = s["episodes"]
            seeds += list(e["seed"])
            succ += list(e["success"])
            leng += list(e["length"])
            term += list(e.get("terminated", [0] * len(e["seed"])))
            setid += [k] * len(e["seed"])
            horizons.add(int(s.get("max_episode_steps", -1)))
            freqs.add(int(s.get("sim_freq", -1)))

        order = np.argsort(np.asarray(seeds))
        run_id = str(d.get("run_id", "?"))
        m = Model(
            task=str(d.get("task_name", "?")), group=A.group_of(run_id), run_id=run_id,
            seed=int(run_id[4:]) if run_id.startswith("seed") and run_id[4:].isdigit() else -1,
            epoch=int(d.get("epoch_id", -1)), n_sets=len(keep), path=str(path),
            success=np.asarray(succ, np.int8)[order],
            length=np.asarray(leng, np.int16)[order],
            terminated=np.asarray(term, np.int8)[order],
            set_of=np.asarray(setid, np.int16)[order],
            horizon=sorted(horizons), sim_freq=sorted(freqs),
        )
        key = (m.task, m.run_id, m.epoch)
        prior = by_key.get(key)
        if prior is not None:
            notes["duplicates"] += 1
            if prior.n_sets >= m.n_sets:
                continue
        by_key[key] = m

    return list(by_key.values()), notes


def preflight(models: List[Model], args: argparse.Namespace) -> List[str]:
    """Assert the shape of the data before any table is built.

    Each check corresponds to a way the numbers could be quietly wrong rather than
    obviously broken: an unbalanced seed inflates its max-over-checkpoints, a mixed
    episode horizon makes lengths incomparable, and a mismatched seed grid breaks the
    pairing that every test here relies on.
    """
    problems = []
    by_task = defaultdict(list)
    for m in models:
        by_task[m.task].append(m)

    for task, ms in sorted(by_task.items()):
        bc = [m for m in ms if m.group == "BC"]
        rl = [m for m in ms if m.group == "RL"]
        if len(bc) != 1:
            problems.append(f"{task}: expected exactly 1 BC checkpoint, found {len(bc)}")
        per_seed = defaultdict(int)
        for m in rl:
            per_seed[m.run_id] += 1
        if per_seed and len(set(per_seed.values())) > 1:
            counts = dict(sorted(per_seed.items()))
            problems.append(
                f"{task}: seeds have unequal checkpoint counts {sorted(set(counts.values()))} "
                f"-- max-over-checkpoints grows with the count, so these are not comparable"
            )
        horizons = {tuple(m.horizon) for m in ms}
        freqs = {tuple(m.sim_freq) for m in ms}
        if len(horizons) > 1 or any(len(h) > 1 for h in horizons):
            problems.append(f"{task}: mixed max_episode_steps {horizons} -- "
                            f"episode lengths are censored differently and cannot be pooled")
        if len(freqs) > 1 or any(len(fq) > 1 for fq in freqs):
            problems.append(f"{task}: mixed sim_freq {freqs}")

        grids = {(len(m.success), int(m.success.size)) for m in ms}
        if len({g[0] for g in grids}) > 1:
            problems.append(f"{task}: models evaluated on different numbers of episodes "
                            f"{sorted({g[0] for g in grids})} -- pairing is invalid")
    return problems


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def seed_bests(rl: List[Model]) -> List[Model]:
    """Each seed's best checkpoint by overall success rate."""
    best: Dict[str, Model] = {}
    for m in rl:
        if m.run_id not in best or m.sr > best[m.run_id].sr:
            best[m.run_id] = m
    return [best[k] for k in sorted(best, key=lambda r: (len(r), r))]


def split_masks(model: Model, n_sets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two halves of the eval episodes, by set id."""
    half = n_sets // 2
    lo = model.set_of <= half
    hi = model.set_of > half
    return lo, hi


def table_headline(by_task: Dict[str, List[Model]], n_sets: int) -> pd.DataFrame:
    rows = []
    for task in sorted(by_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        ms = by_task[task]
        bc = next((m for m in ms if m.group == "BC"), None)
        rl = [m for m in ms if m.group == "RL"]
        if bc is None or not rl:
            continue
        best = seed_bests(rl)
        vals = np.array([m.sr for m in best])
        n_ep = int(bc.success.size)
        k = int(bc.success.sum())
        lo, hi = clopper_pearson(k, n_ep)
        blo, bhi = boot_ci(vals)
        top = max(best, key=lambda m: m.sr)
        rows.append({
            "task": task, "episodes": n_ep, "n_seeds": len(best),
            "bc_sr": bc.sr, "bc_ci_lo": lo, "bc_ci_hi": hi,
            "rl_mean_of_best": float(vals.mean()),
            "rl_std_of_best": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "rl_mean_ci_lo": blo, "rl_mean_ci_hi": bhi,
            "rl_max_of_best": float(vals.max()),
            "argmax": f"{top.run_id}@{'final' if top.epoch < 0 else top.epoch}",
            "d_mean": float(vals.mean()) - bc.sr,
            "d_max": float(vals.max()) - bc.sr,
            "per_seed": ", ".join(f"{v:.3f}" for v in vals),
        })
    return pd.DataFrame(rows)


def table_split_half(by_task: Dict[str, List[Model]], n_sets: int) -> pd.DataFrame:
    """Cross-fitted selection: choose on one half of the episodes, score on the other.

    Both directions are run and averaged, so every episode is used for selection once and
    for reporting once, and the estimate does not depend on which half was picked first.
    BC is scored on the same reporting half as RL, never on all the episodes -- otherwise
    the two sides would rest on different amounts of data.
    """
    rows = []
    for task in sorted(by_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        ms = by_task[task]
        bc = next((m for m in ms if m.group == "BC"), None)
        rl = [m for m in ms if m.group == "RL"]
        if bc is None or not rl:
            continue
        by_seed: Dict[str, List[Model]] = defaultdict(list)
        for m in rl:
            by_seed[m.run_id].append(m)

        dir_means, dir_bc, picks = [], [], []
        for sel_first in (True, False):
            lo, hi = split_masks(rl[0], n_sets)
            sel_mask, rep_mask = (lo, hi) if sel_first else (hi, lo)
            vals = []
            for run_id in sorted(by_seed):
                cand = by_seed[run_id]
                chosen = max(cand, key=lambda m: m.sr_on(sel_mask))
                vals.append(chosen.sr_on(rep_mask))
                picks.append(chosen.epoch)
            dir_means.append(np.array(vals))
            dir_bc.append(bc.sr_on(rep_mask))

        cross = (dir_means[0] + dir_means[1]) / 2.0
        bc_cross = float(np.mean(dir_bc))
        rows.append({
            "task": task, "n_seeds": len(cross),
            "bc_sr": bc_cross,
            "rl_mean_honest": float(cross.mean()),
            "rl_std_honest": float(cross.std(ddof=1)) if len(cross) > 1 else np.nan,
            "rl_max_honest": float(cross.max()),
            "d_mean": float(cross.mean()) - bc_cross,
            "dirA_mean": float(dir_means[0].mean()),
            "dirB_mean": float(dir_means[1].mean()),
            "per_seed": ", ".join(f"{v:.3f}" for v in cross),
        })
    return pd.DataFrame(rows)


def table_traj_len(by_task: Dict[str, List[Model]]) -> pd.DataFrame:
    """Success-trajectory length, BC vs. post-WMRL, two ways.

    `all successes` is each policy on whatever episodes it happens to solve, which is
    confounded: a policy that solves more episodes is also solving harder ones. `matched`
    restricts to episodes *both* solve and pairs them by seed, which is the comparison
    that actually answers "did it get faster".

    Length is control steps (`sim_freq=1`, so also env steps); failures are right-censored
    at `max_episode_steps` and are excluded throughout.
    """
    rows = []
    for task in sorted(by_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        ms = by_task[task]
        bc = next((m for m in ms if m.group == "BC"), None)
        rl = [m for m in ms if m.group == "RL"]
        if bc is None or not rl:
            continue
        best = seed_bests(rl)
        horizon = bc.horizon[0] if bc.horizon else -1

        bc_ok = bc.success.astype(bool)
        # terminated but not successful: an early *failure*, not a fast success. None of
        # these tasks emit a "fail" signal today, so this should be 0 -- report it so a
        # future task that does cannot silently shorten the mean.
        bc_failterm = int(((bc.terminated == 1) & (~bc_ok)).sum())

        per_seed_all, per_seed_matched, per_seed_bcmatched = [], [], []
        deltas, pvals, matched_n = [], [], []
        for m in best:
            ok = m.success.astype(bool)
            if ok.any():
                per_seed_all.append(float(m.length[ok].mean()))
            both = ok & bc_ok
            matched_n.append(int(both.sum()))
            if both.sum() >= 2:
                d = m.length[both].astype(float) - bc.length[both].astype(float)
                per_seed_matched.append(float(m.length[both].mean()))
                per_seed_bcmatched.append(float(bc.length[both].mean()))
                deltas.append(float(d.mean()))
                pvals.append(wilcoxon_safe(-d, alternative="greater")["p"])

        rl_failterm = float(np.mean([((m.terminated == 1) & (~m.success.astype(bool))).sum()
                                     for m in best]))
        rows.append({
            "task": task, "horizon": horizon, "n_seeds": len(best),
            "bc_sr": bc.sr, "bc_n_success": int(bc_ok.sum()),
            "bc_len_all": float(bc.length[bc_ok].mean()) if bc_ok.any() else np.nan,
            "bc_len_median": float(np.median(bc.length[bc_ok])) if bc_ok.any() else np.nan,
            "rl_sr": float(np.mean([m.sr for m in best])),
            "rl_len_all": float(np.mean(per_seed_all)) if per_seed_all else np.nan,
            "rl_len_all_std": float(np.std(per_seed_all, ddof=1)) if len(per_seed_all) > 1 else np.nan,
            "d_len_all": (float(np.mean(per_seed_all)) - float(bc.length[bc_ok].mean()))
                         if per_seed_all and bc_ok.any() else np.nan,
            "matched_n": float(np.mean(matched_n)) if matched_n else 0.0,
            "bc_len_matched": float(np.mean(per_seed_bcmatched)) if per_seed_bcmatched else np.nan,
            "rl_len_matched": float(np.mean(per_seed_matched)) if per_seed_matched else np.nan,
            "d_len_matched": float(np.mean(deltas)) if deltas else np.nan,
            "d_len_matched_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else np.nan,
            "n_seeds_faster": int(sum(1 for d in deltas if d < 0)),
            "p_seed_level": wilcoxon_safe([-d for d in deltas], "greater")["p"] if deltas else np.nan,
            "p_episode_median": float(np.median(pvals)) if pvals else np.nan,
            "bc_fail_terminations": bc_failterm,
            "rl_fail_terminations_mean": rl_failterm,
        })
    return pd.DataFrame(rows)


def table_tests(by_task: Dict[str, List[Model]], n_sets: int) -> pd.DataFrame:
    """Per task, at both granularities. Every row says what its unit of analysis is."""
    rows = []
    for task in sorted(by_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        ms = by_task[task]
        bc = next((m for m in ms if m.group == "BC"), None)
        rl = [m for m in ms if m.group == "RL"]
        if bc is None or not rl:
            continue
        best = seed_bests(rl)
        vals = np.array([m.sr for m in best])
        bc_sr = bc.sr
        d = vals - bc_sr

        # ---- seed level, n = number of training seeds --------------------------
        t = stats.ttest_1samp(vals, bc_sr, alternative="greater")
        w = wilcoxon_safe(d, "greater")
        n_win = int((vals > bc_sr).sum())
        sign_p = float(stats.binomtest(n_win, len(vals), 0.5, alternative="greater").pvalue)
        tsl, tsh = two_stage_boot(vals, bc.success.astype(float))

        # ---- episode level, paired on the eval seed ----------------------------
        top = max(best, key=lambda m: m.sr)
        mc_top = mcnemar(top.success, bc.success)

        lo, hi = split_masks(rl[0], n_sets)
        by_seed: Dict[str, List[Model]] = defaultdict(list)
        for m in rl:
            by_seed[m.run_id].append(m)
        honest_p, honest_d = [], []
        for run_id in sorted(by_seed):
            chosen = max(by_seed[run_id], key=lambda m: m.sr_on(lo))
            r = mcnemar(chosen.success[hi], bc.success[hi])
            honest_p.append(r["p"])
            honest_d.append(r["delta"])

        rows.append({
            "task": task, "n_seeds": len(vals), "n_episodes": int(bc.success.size),
            "bc_sr": bc_sr, "rl_mean": float(vals.mean()),
            "d_mean": float(d.mean()),
            "seed_t_p": float(t.pvalue),
            "seed_wilcoxon_p": w["p"], "seed_wilcoxon_method": w["method"],
            "seed_sign_wins": f"{n_win}/{len(vals)}", "seed_sign_p": sign_p,
            "cohens_d": float(d.mean() / vals.std(ddof=1)) if vals.std(ddof=1) > 0 else np.nan,
            "d_ci_lo": tsl, "d_ci_hi": tsh,
            "ep_best_ckpt": f"{top.run_id}@{'final' if top.epoch < 0 else top.epoch}",
            "ep_best_delta": mc_top["delta"],
            "ep_best_discordant": mc_top["n_discordant"],
            "ep_best_mcnemar_p": mc_top["p"],
            "ep_honest_median_p": float(np.median(honest_p)) if honest_p else np.nan,
            "ep_honest_mean_delta": float(np.mean(honest_d)) if honest_d else np.nan,
            "ep_honest_n_sig": int(sum(1 for p in honest_p if p < ALPHA)),
        })
    return pd.DataFrame(rows)


def combine_across_tasks(tests: pd.DataFrame) -> pd.DataFrame:
    """Family-wise and FDR correction over the per-task tests, plus omnibus combination."""
    out = []
    for label, col in [("seed-level sign test", "seed_sign_p"),
                       ("seed-level Wilcoxon", "seed_wilcoxon_p"),
                       ("seed-level t-test", "seed_t_p"),
                       ("episode-level McNemar (split-half, median)", "ep_honest_median_p")]:
        p = tests[col].to_numpy(float)
        p = np.clip(np.nan_to_num(p, nan=1.0), 1e-300, 1.0)
        row: Dict[str, Any] = {"test": label, "n_tasks": len(p),
                               "max_raw_p": float(p.max())}
        row["holm_max"] = float(holm(p).max())
        row["bh_max"] = float(np.max(stats.false_discovery_control(p, method="bh")))
        row["n_sig_holm"] = int((holm(p) < ALPHA).sum())
        for method in ("fisher", "stouffer"):
            row[f"{method}_p"] = float(stats.combine_pvalues(p, method=method).pvalue)
        out.append(row)
    return pd.DataFrame(out)


def pooled_seed_test(by_task: Dict[str, List[Model]]) -> Dict[str, Any]:
    """All (task, seed) pairs at once, as differences from that task's BC baseline."""
    diffs = []
    for task, ms in by_task.items():
        bc = next((m for m in ms if m.group == "BC"), None)
        rl = [m for m in ms if m.group == "RL"]
        if bc is None or not rl:
            continue
        diffs += [m.sr - bc.sr for m in seed_bests(rl)]
    d = np.array(diffs)
    if d.size == 0:
        return {}
    n_win = int((d > 0).sum())
    return {
        "n": int(d.size), "mean_delta": float(d.mean()),
        "wins": f"{n_win}/{d.size}",
        "sign_p": float(stats.binomtest(n_win, d.size, 0.5, alternative="greater").pvalue),
        "wilcoxon_p": wilcoxon_safe(d, "greater")["p"],
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def md_table(df: pd.DataFrame, cols: Sequence[str], floatfmt: str = ".4f") -> str:
    """Render, formatting p-value columns in scientific notation.

    A fixed-decimal format prints every p below 1e-4 as `0.0000`, which is exactly the
    range worth reporting -- an exact sign test on 15 seeds bottoms out at 3.05e-5.
    """
    if df.empty:
        return "_(no rows)_\n"
    sub = df[[c for c in cols if c in df.columns]].copy()
    p_idx = []
    for i, c in enumerate(sub.columns):
        if c.endswith("_p") or c.startswith("p_") or c == "p":
            sub[c] = sub[c].map(fmt_p)
            p_idx.append(i)
    # tabulate re-parses numeric-looking strings and re-applies floatfmt, which would
    # undo the formatting above; disable_numparse keeps these columns verbatim.
    return sub.to_markdown(index=False, floatfmt=floatfmt,
                           disable_numparse=p_idx) + "\n"


def build_report(head: pd.DataFrame, split: pd.DataFrame, traj: pd.DataFrame,
                 tests: pd.DataFrame, comb: pd.DataFrame, pooled: Dict[str, Any],
                 meta: Dict[str, Any], problems: Sequence[str]) -> str:
    n_ep = int(head["episodes"].max()) if not head.empty else 0
    L = [
        "# WMRL v2 — evaluation sweep",
        "",
        f"{meta['n_models']} checkpoints, {n_ep} evaluation episodes each "
        f"({meta['n_sets']} sets x {meta['episodes_per_set']}, seeds 1..{n_ep}), "
        "deterministic actions.",
        "",
    ]
    if problems:
        L += ["> **Pre-flight problems — read before trusting anything below:**", ""]
        L += [f"> - {p}" for p in problems] + [""]

    L += [
        "## A. Headline — max over each seed's checkpoints",
        "",
        "Per seed, the best of its checkpoints; then mean ± std and max over the seeds.",
        "`bc_sr` is the initializer those seeds were trained from, with an exact "
        "(Clopper–Pearson) interval — it is a proportion, not a constant.",
        "",
        md_table(head, ["task", "n_seeds", "bc_sr", "bc_ci_lo", "bc_ci_hi",
                        "rl_mean_of_best", "rl_std_of_best", "rl_mean_ci_lo",
                        "rl_mean_ci_hi", "rl_max_of_best", "argmax", "d_mean", "d_max"]),
        "",
        "Per-seed values:",
        "",
        md_table(head, ["task", "per_seed"]),
        "",
        "## A2. Same comparison, selection bias removed",
        "",
        "Each seed's checkpoint is chosen on one half of the episodes and scored on the "
        "other, averaged over both directions; BC is scored on the same reporting half. "
        "**The gap between `rl_mean_honest` here and `rl_mean_of_best` in Table A is the "
        "size of the max-over-checkpoints bias**, which Table A cannot avoid because the "
        "BC side has only one checkpoint to choose from.",
        "",
        md_table(split, ["task", "n_seeds", "bc_sr", "rl_mean_honest", "rl_std_honest",
                         "rl_max_honest", "d_mean", "dirA_mean", "dirB_mean"]),
        "",
        "## B. Trajectory length of successful episodes",
        "",
        "Control steps (`sim_freq=1`, so also environment steps; x0.05 s for simulated "
        "seconds). Failures run to the `horizon` and are excluded. RL columns average "
        "over the seed-best policies.",
        "",
        "`*_all` uses whatever episodes each policy solves — confounded, because a "
        "stronger policy also solves harder episodes. `*_matched` uses only episodes "
        "**both** solve, paired by seed: that is the comparison that answers whether the "
        "policy got faster. Negative `d_len_matched` means WMRL finishes sooner.",
        "",
        md_table(traj, ["task", "horizon", "bc_sr", "rl_sr", "bc_len_all", "rl_len_all",
                        "d_len_all", "matched_n", "bc_len_matched", "rl_len_matched",
                        "d_len_matched", "d_len_matched_std", "n_seeds_faster",
                        "p_seed_level"], ".2f"),
        "",
        "## C. Significance",
        "",
        "Every model saw the same eval seeds, so all comparisons are **paired on the "
        "environment's initial state**. Two units of analysis, answering different "
        "questions:",
        "",
        "- **Seed level (n = training seeds).** Does the *method* beat BC? This is the "
        "unit that generalises, and the sign test is the claim that rests on fewest "
        "assumptions.",
        "- **Episode level (n = episodes).** Does *this particular checkpoint* beat BC on "
        "these problems? `ep_best_*` picks the single best RL checkpoint out of all of "
        "them, so its p-value is inflated by the winner's curse and should be quoted as a "
        "description, not a test. `ep_honest_*` selects on one half and tests on the "
        "other, and is the defensible episode-level number.",
        "",
        md_table(tests, ["task", "n_seeds", "bc_sr", "rl_mean", "d_mean", "seed_t_p",
                         "seed_wilcoxon_p", "seed_wilcoxon_method", "seed_sign_wins", "seed_sign_p", "cohens_d",
                         "d_ci_lo", "d_ci_hi"]),
        "",
        "`d_ci_*` is a two-stage bootstrap resampling both the seeds and BC's own "
        "episodes, so BC's sampling error is not treated as zero.",
        "",
        md_table(tests, ["task", "ep_best_ckpt", "ep_best_delta", "ep_best_discordant",
                         "ep_best_mcnemar_p", "ep_honest_mean_delta",
                         "ep_honest_median_p", "ep_honest_n_sig"]),
        "",
        "### Across the five tasks",
        "",
        md_table(comb, ["test", "n_tasks", "max_raw_p", "holm_max", "n_sig_holm",
                        "bh_max", "fisher_p", "stouffer_p"]),
        "",
    ]
    if pooled:
        L += [
            f"Pooled over all {pooled['n']} (task, seed) pairs as differences from each "
            f"task's BC baseline: mean Δ = {pooled['mean_delta']:.4f}, "
            f"{pooled['wins']} seeds above BC, sign test p = {fmt_p(pooled['sign_p'])}, "
            f"Wilcoxon p = {fmt_p(pooled['wilcoxon_p'])}.",
            "",
        ]
    L += [
        "## Limitations",
        "",
        "- **Table A is optimistically biased.** Each RL seed takes a max over its "
        "checkpoints; the BC initializer is one checkpoint. Table A2 is the corrected "
        "version. A symmetric best-of-N for BC is not possible here — the rest of the BC "
        "epoch sweep is no longer on disk.",
        "- **`ep_best_mcnemar_p` is not a hypothesis test.** The checkpoint was chosen for "
        "being the best of ~180 on the same episodes it is then tested on.",
        "- The BC baseline is a proportion over the same episodes, with the interval "
        "shown; treating it as an exact constant would overstate every p-value's "
        "confidence.",
        "- Flow-matching noise is not seed-controlled (`docs/applications.md:51`), so "
        "re-running does not reproduce these numbers bit for bit.",
        "",
    ]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO / "runs/eval_v2/results"))
    p.add_argument("--out-dir", default=str(REPO / "runs/eval_v2/tables"))
    p.add_argument("--min-sets", type=int, default=0,
                   help="Use exactly this many sets per checkpoint, so every model is "
                        "compared on the same episodes. 0 = use whatever is complete.")
    p.add_argument("--episodes-per-set", type=int, default=50)
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any pre-flight check fails.")
    args = p.parse_args(argv)

    models, notes = load_models(args.results, args.min_sets)
    if not models:
        sys.exit(f"[analyze] no usable results with per-episode data in {args.results}")

    n_sets = min(m.n_sets for m in models)
    if len({m.success.size for m in models}) > 1:
        keep = max(sorted({m.success.size for m in models}))
        print(f"[analyze] models differ in episode count; "
              f"restricting to those with {keep}", file=sys.stderr)
        models = [m for m in models if m.success.size == keep]
        n_sets = min(m.n_sets for m in models)

    print(f"[analyze] {notes['files']} files -> {len(models)} checkpoints, "
          f"{n_sets} sets x {models[0].success.size // max(n_sets, 1)} episodes "
          f"= {models[0].success.size} episodes each", file=sys.stderr)
    for k in ("no_episodes", "too_few_sets", "unreadable", "duplicates"):
        if notes[k]:
            print(f"[analyze]   {notes[k]} {k}", file=sys.stderr)

    problems = preflight(models, args)
    for prob in problems:
        print(f"[analyze] PREFLIGHT: {prob}", file=sys.stderr)

    by_task: Dict[str, List[Model]] = defaultdict(list)
    for m in models:
        by_task[m.task].append(m)

    head = table_headline(by_task, n_sets)
    split = table_split_half(by_task, n_sets)
    traj = table_traj_len(by_task)
    tests = table_tests(by_task, n_sets)
    comb = combine_across_tasks(tests) if not tests.empty else pd.DataFrame()
    pooled = pooled_seed_test(by_task)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [("headline", head), ("split_half", split), ("traj_len", traj),
                     ("tests", tests), ("combined", comb)]:
        df.to_csv(out / f"{name}.csv", index=False)
    per_ckpt = pd.DataFrame([{
        "task": m.task, "group": m.group, "run_id": m.run_id, "epoch_id": m.epoch,
        "success_rate": m.sr, "n_episodes": int(m.success.size),
        "n_success": int(m.success.sum()),
        "avg_success_length": float(m.length[m.success.astype(bool)].mean())
        if m.success.any() else np.nan,
        "ckpt_json": m.path,
    } for m in models]).sort_values(["task", "group", "run_id", "epoch_id"])
    per_ckpt.to_csv(out / "per_ckpt.csv", index=False)

    meta = {"n_models": len(models), "n_sets": n_sets,
            "episodes_per_set": models[0].success.size // max(n_sets, 1)}
    report = build_report(head, split, traj, tests, comb, pooled, meta, problems)
    (out / "report.md").write_text(report)
    print(report)
    print(f"[analyze] wrote {out}/report.md and 6 CSVs", file=sys.stderr)
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
