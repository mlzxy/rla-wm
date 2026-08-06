"""Evaluate many checkpoints across many GPUs. Resumable, no job cache.

Replaces the older `run_eval_lot.py` + `run_eval_lot_multigpu.py` pair (not shipped:
they import a helper package that no longer exists),
whose three-step `gen-jobs -> run --piece i -> multigpu` flow needed a
jobs_cache.json on disk. Here the parent and every worker re-derive the same
job list from the same arguments, so a shard index is all that has to be passed.

Each job evaluates one checkpoint over `--num-sets` sets of `--episodes-per-set`
episodes (default 15 sets of 50 seeds), writing one JSON whose
schema matches the legacy tool -- `rebuttal/tools/analyze_eval.py` reads both.

Targets
-------
Any mix of: a checkpoint file (`.pt` / `.ckpt`), a run directory, a directory
containing run directories, or a `.txt` file listing any of those. Two run
layouts are recognised, matching how the two trainers save:

    bcrl_agent      <run>/config.yaml        + <run>/ckpt_*.pt, <run>/final.pt
    base_workspace  <run>/.hydra/config.yaml + <run>/checkpoints/latest_epoch*.ckpt

Examples
--------
    # smoke test: one ckpt, one set, in-process on GPU 0
    python rebuttal/tools/eval_ckpts.py runs/weights/rl_best_v2 \\
        --num-sets 1 --gpus 0 --out-dir runs/eval_smoke

    # the real sweep: every new RL ckpt, 7 GPUs, resumable
    python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 \\
        --out-dir runs/eval_v2_results --gpus 0-6

    # see the plan and the exact worker commands without running anything
    python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 --gpus 0-6 --dry-run

Resume is automatic: results are written to a deterministic filename after every
set, so re-running the same command skips finished checkpoints and continues
partially-finished ones from the first missing set.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import datetime as dt
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parents[2]  # rebuttal/tools/ -> repo root
CKPT_SUFFIXES = (".pt", ".ckpt")
SUMMARY_TOKEN = "EVAL_CKPTS_SUMMARY"
POLL_INTERVAL = 2.0
SHUTDOWN_GRACE = 30.0


def _bootstrap_sys_path() -> None:
    """This can be run as a plain script, so `import policies` needs help."""
    for p in (REPO_ROOT, REPO_ROOT / "third_party" / "diffusion_policy"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


_bootstrap_sys_path()

_LAZY: Optional[SimpleNamespace] = None


def lazy() -> SimpleNamespace:
    """Import torch/hydra/omegaconf once, after CUDA_VISIBLE_DEVICES is settled.

    Also registers the ``eval`` resolver that BC hydra configs need to resolve
    ``task: ${eval:'[...][${setting}]'}``.
    """
    global _LAZY
    if _LAZY is None:
        import dill
        import hydra
        import numpy as np
        import torch
        from omegaconf import DictConfig, OmegaConf
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval, replace=True)
        _LAZY = SimpleNamespace(
            dill=dill, hydra=hydra, np=np, torch=torch,
            DictConfig=DictConfig, OmegaConf=OmegaConf,
        )
    return _LAZY


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
# Everything recorded or hashed uses os.path.abspath, never realpath: `runs/` is
# a symlink onto another mount, and resolving it would change ckpt_hash8 for
# every result relative to the 1100 already on disk.

def parse_run_id(ckpt_path: str) -> str:
    """`seed<N>` from the path if present, else the run-dir basename."""
    s = str(ckpt_path)
    m = re.search(r"seed(\d+)", s)
    if m:
        return f"seed{m.group(1)}"
    p = Path(ckpt_path).resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent.name
    return p.parent.name


def parse_epoch_id(ckpt_path: str) -> int:
    """Training epoch / RL iteration; -1 for `final.pt`."""
    s = str(ckpt_path)
    for pat in (r"epoch[_=]?(\d+)", r"ckpt_(\d+)"):
        m = re.search(pat, s)
        if m:
            return int(m.group(1))
    return -1


def ckpt_hash8(ckpt_path: str) -> str:
    """Legacy abspath hash; goes in the JSON so old and new results agree."""
    return hashlib.sha1(os.path.abspath(ckpt_path).encode()).hexdigest()[:8]


def job_id(ckpt_path: str) -> str:
    return hashlib.sha1(os.path.abspath(ckpt_path).encode()).hexdigest()[:12]


def ckpt_key(ckpt_path: str) -> Tuple[str, ...]:
    """Mount-prefix-invariant key: the last 3 path components.

    Results were written from `/scratch/...`, `/scache/scratch/...` and
    `/common/users/...` views of the same files, so an abspath comparison
    misses them. This is what makes `--adopt-legacy` work.
    """
    return tuple(Path(ckpt_path).parts[-3:])


def stable_id8(ckpt_path: str) -> str:
    """Filename hash, stable across mount prefixes (unlike `ckpt_hash8`)."""
    return hashlib.sha1("/".join(ckpt_key(ckpt_path)).encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Checkpoint loading  (semantics preserved from run_eval_lot.py:91-146)
# --------------------------------------------------------------------------- #

_RELOCATE_CACHE: Dict[str, Optional[str]] = {}


def _maybe_relocate(value: str) -> Optional[str]:
    """A repo-relative artifact path that has moved -> where it lives now, else None.

    Only `runs/...` values are considered, so Hugging Face ids like
    `facebook/dinov3-...` are never touched, and only an unambiguous match counts: the
    artifacts are timestamped (`20260404_22-53-36`), so the basename identifies one
    directory. Anything with zero or several candidates is left alone to fail loudly.
    """
    if not value.startswith("runs/") or os.path.exists(os.path.join(REPO_ROOT, value)):
        return None
    if value in _RELOCATE_CACHE:
        return _RELOCATE_CACHE[value]
    base = os.path.basename(value.rstrip("/"))
    hits: List[str] = []
    if base:
        for root in (REPO_ROOT / "runs" / "weights", REPO_ROOT / "runs"):
            hits = sorted(str(p.relative_to(REPO_ROOT))
                          for p in root.glob(f"**/{base}") if p.is_dir())
            if hits:
                break
    out = hits[0] if len(hits) == 1 else None
    _RELOCATE_CACHE[value] = out
    return out


def relocate_stale_paths(cfg: Any) -> List[Tuple[str, str]]:
    """Repoint artifact paths in a checkpoint-embedded config that have since moved.

    The BC checkpoints were written when the tree was laid out differently, and their
    embedded config still says `runs/dec27/dino_to_image_v2/<ts>` for the image decoder
    that now lives under `runs/weights/dino-to-image_unet/maniskill/<ts>`. Hydra
    instantiation dies with a bare `FileNotFoundError` naming no path at all. The RL
    checkpoints are unaffected -- their configs were written after the move -- which is
    why this only ever bites the BC side.
    """
    L = lazy()
    fixed: List[Tuple[str, str]] = []

    def visit(node: Any) -> None:
        if L.OmegaConf.is_dict(node):
            keys = list(node.keys())
        elif L.OmegaConf.is_list(node):
            keys = list(range(len(node)))
        else:
            return
        for k in keys:
            try:
                v = node[k]
            except Exception:
                continue  # unresolvable interpolation: not ours to fix
            if isinstance(v, str):
                new = _maybe_relocate(v)
                if new:
                    node[k] = new
                    fixed.append((v, new))
            else:
                visit(v)

    visit(cfg)
    return fixed


def _load_format_b(payload: dict, device: Any, bc_weights: str) -> Tuple[Any, Any, str]:
    """BC workspace ckpt: policy comes from the cfg embedded in the payload."""
    L = lazy()
    cfg = payload["cfg"]
    for old, new in relocate_stale_paths(cfg.policy):
        log(f"[relocate] checkpoint config path moved: {old} -> {new}")
    policy = L.hydra.utils.instantiate(cfg.policy)
    sd = payload["state_dicts"]
    # 'model' is what the legacy sweep scored, and what wmrl loads when its
    # config says pretrained_weights: model. 'ema' is offered for comparison.
    key = "ema_model" if (bc_weights == "ema" and "ema_model" in sd) else "model"
    policy.load_state_dict(sd[key])
    policy.to(device)
    return policy, cfg, key


def _load_format_a(payload: Any, ckpt_path: str, device: Any) -> Tuple[Any, Any, str]:
    """WMRL agent ckpt: policy class + kwargs come from the sibling config.yaml."""
    L = lazy()
    sibling_cfg = Path(ckpt_path).parent / "config.yaml"
    if not sibling_cfg.exists():
        raise FileNotFoundError(
            f"bcrl_agent-format ckpt requires sibling config.yaml at {sibling_cfg}"
        )
    cfg = L.OmegaConf.load(sibling_cfg)
    L.OmegaConf.resolve(cfg)

    raw_kwargs = L.OmegaConf.to_container(cfg.policy_kwargs, resolve=True)
    if not isinstance(raw_kwargs, dict):
        raise TypeError("cfg.policy_kwargs must resolve to a dict")
    policy_kwargs = cast(Dict[str, Any], raw_kwargs)
    policy_kwargs["enable_rl_heads"] = True
    policy_kwargs["mlp_dropout"] = 0.0
    policy_kwargs.pop("critic_latent_dim", None)

    # Was `notes.wmrl_baseline.rl_utils` in run_eval_lot.py:116 -- that package
    # no longer exists, which is why the legacy tool fails on every RL ckpt.
    from wmrl.rl_utils import import_cls
    PolicyCls = import_cls(str(cfg.policy_cls))
    policy = PolicyCls(**policy_kwargs)

    state_dict = payload["policy"] if isinstance(payload, dict) and "policy" in payload else payload
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"state_dict mismatch loading {ckpt_path}: {e}") from e
    policy.to(device)
    return policy, cfg, "policy"


def load_policy_from_ckpt(
    ckpt_path: str, device: Any, bc_weights: str = "model",
) -> Tuple[Any, Any, str, str]:
    """Auto-detect the format. Returns (policy, cfg, format_name, weights_key)."""
    L = lazy()
    payload = L.torch.load(
        ckpt_path, map_location=device, pickle_module=L.dill, weights_only=False
    )
    if isinstance(payload, dict) and "cfg" in payload and "state_dicts" in payload:
        policy, cfg, weights = _load_format_b(payload, device, bc_weights)
        fmt = "base_workspace"
    else:
        policy, cfg, weights = _load_format_a(payload, ckpt_path, device)
        fmt = "bcrl_agent"

    policy.eval()
    # World-model rollouts feed DINO-decoded images, so training sets this True;
    # against the real simulator it must be False (mirrors wmrl/train.py:319).
    if hasattr(policy, "skip_dino_preprocess"):
        policy.skip_dino_preprocess = False
    return policy, cfg, fmt, weights


# --------------------------------------------------------------------------- #
# Eval-config resolution  (verbatim from run_eval_lot.py:153-264)
# --------------------------------------------------------------------------- #

def _get_nested(cfg: Any, *keys: str) -> Any:
    L = lazy()
    if cfg is None:
        return None
    if isinstance(cfg, L.DictConfig):
        return L.OmegaConf.select(
            cfg, ".".join(keys), default=None, throw_on_resolution_failure=False,
        )
    cur = cfg
    for k in keys:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else getattr(cur, k, None)
    return cur


def build_eval_namespace(ckpt_cfg: Any) -> Tuple[SimpleNamespace, Dict[str, str]]:
    """Resolve the eval env spec from the ckpt-embedded config, and say where
    each field came from (recorded as `field_sources` in the result JSON)."""
    sources: Dict[str, str] = {}

    def pick(name, ckpt_paths, default=None, required=False):
        for path in ckpt_paths:
            v = _get_nested(ckpt_cfg, *path)
            if v is not None:
                sources[name] = "ckpt:" + ".".join(path)
                return v
        if required:
            raise ValueError(f"Required eval-config field '{name}' not found in ckpt cfg")
        sources[name] = f"default:{default!r}"
        return default

    task = pick("task", [("task",)], required=True)
    robot = pick("robot", [("robot",), ("env_kwargs", "robot_uid")], required=True)
    control_mode = pick(
        "control_mode", [("control_mode",), ("env_kwargs", "control_mode")],
        default="pd_joint_pos",
    )
    cameras = pick(
        "cameras", [("env_kwargs", "cameras"), ("cameras",), ("env", "cameras")],
        default=["front_lower_camera"],
    )
    camera_h = pick(
        "camera_height", [("camera_height",), ("env", "camera_height"), ("img_size",)],
        required=True,
    )
    camera_w = pick(
        "camera_width", [("camera_width",), ("env", "camera_width"), ("img_size",)],
        required=True,
    )
    shader_dir = pick(
        "shader_dir", [("env_kwargs", "shader_dir"), ("env", "shader_dir")],
        default="rt-clean",
    )
    max_episode_steps = pick(
        "max_episode_steps", [("eval_max_episode_steps",), ("eval", "max_episode_steps")],
        default=100,
    )
    sim_freq = pick("sim_freq", [("eval_sim_freq",), ("eval", "sim_freq")], default=1)
    cpu = pick("cpu", [("eval_cpu",), ("eval", "cpu")], default=False)

    cameras_list = [cameras] if isinstance(cameras, str) else [str(c) for c in cameras]

    # `create_eval_env` calls .get() on these, so they must stay plain dicts.
    cfg_ns = SimpleNamespace(
        task=str(task),
        robot=str(robot),
        control_mode=str(control_mode),
        env={
            "num_envs": 1,
            "shader_dir": str(shader_dir),
            "camera_width": int(camera_w),
            "camera_height": int(camera_h),
            "cameras": cameras_list,
        },
        eval={
            "cpu": bool(cpu),
            "num_episodes": 1,
            "max_episode_steps": int(max_episode_steps),
            "sim_freq": int(sim_freq),
            "save_video": False,
            "n_vis": 0,
            "video_dir": "eval_videos",
        },
    )
    return cfg_ns, sources


def read_task_from_config_file(config_path: str) -> str:
    """Cheap read of `task` from a run config, without loading the checkpoint."""
    L = lazy()
    cfg = L.OmegaConf.load(config_path)
    v = _get_nested(cfg, "task")
    if v is None:
        raise ValueError(f"`task` not found in config {config_path}")
    return str(v)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def classify_dir(d: str) -> Optional[str]:
    if os.path.isfile(os.path.join(d, "config.yaml")):
        return "bcrl_agent"
    if os.path.isfile(os.path.join(d, ".hydra", "config.yaml")):
        return "base_workspace"
    return None


def iter_run_dirs(root: str, max_depth: int) -> Iterator[str]:
    """Yield run dirs under `root`, deepest-match-wins, stopping at a match."""
    if max_depth < 0:
        return
    if classify_dir(root):
        yield root
        return
    try:
        entries = sorted(e.path for e in os.scandir(root) if e.is_dir())
    except OSError:
        return
    for sub in entries:
        if os.path.basename(sub) in (".hydra", "checkpoints", "wandb", "__pycache__"):
            continue
        yield from iter_run_dirs(sub, max_depth - 1)


def expand_targets(
    targets: Sequence[str], max_depth: int, _depth: int = 0,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Resolve CLI targets to an ordered, de-duped list of (kind, abspath)."""
    out: List[Tuple[str, str]] = []
    unresolved: List[str] = []
    seen = set()

    def add(kind: str, path: str) -> None:
        key = os.path.realpath(path)  # de-dup only; the abspath is what we keep
        if key not in seen:
            seen.add(key)
            out.append((kind, os.path.abspath(path)))

    for t in targets:
        p = os.path.abspath(t)
        if os.path.isfile(p) and p.endswith(".txt"):
            if _depth >= 4:
                unresolved.append(f"{t} (list nesting too deep)")
                continue
            lines = [
                ln.strip() for ln in Path(p).read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            sub, sub_bad = expand_targets(lines, max_depth, _depth + 1)
            for kind, path in sub:
                add(kind, path)
            unresolved.extend(sub_bad)
        elif os.path.isfile(p) and p.endswith(CKPT_SUFFIXES):
            add("ckpt", p)
        elif os.path.isdir(p):
            if classify_dir(p):
                add("run", p)
            else:
                found = list(iter_run_dirs(p, max_depth))
                if not found:
                    unresolved.append(f"{t} (no run dirs within --max-depth)")
                for d in found:
                    add("run", d)
        else:
            unresolved.append(f"{t} (not a ckpt, dir, or .txt)")
    return out, unresolved


def discover_run(run_dir: str) -> Dict[str, Any]:
    """Inspect a run directory: format, checkpoint list, task."""
    fmt = classify_dir(run_dir)
    if fmt == "bcrl_agent":
        config_path = os.path.join(run_dir, "config.yaml")
        ckpts = sorted(Path(run_dir).glob("ckpt_*.pt")) + sorted(Path(run_dir).glob("final.pt"))
    elif fmt == "base_workspace":
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
        ckpts = sorted(Path(run_dir, "checkpoints").glob("latest_epoch*.ckpt"))
    else:
        raise FileNotFoundError(f"no config.yaml or .hydra/config.yaml under {run_dir}")
    if not ckpts:
        log(f"[discover] WARNING: no checkpoints in {run_dir}")
    return {
        "run_dir": run_dir, "ckpt_format": fmt, "config_path": config_path,
        "ckpt_paths": [str(c) for c in ckpts], "task": read_task_from_config_file(config_path),
    }


def discover_loose_ckpt(ckpt_path: str) -> Dict[str, Any]:
    """A single checkpoint given directly: infer its run dir and format."""
    parent = Path(ckpt_path).parent
    run_dir = str(parent.parent if parent.name == "checkpoints" else parent)
    fmt = classify_dir(run_dir)
    if fmt is None:
        # A bare .ckpt still works: format B carries its own cfg in the payload.
        if ckpt_path.endswith(".ckpt"):
            return {"run_dir": run_dir, "ckpt_format": "base_workspace",
                    "config_path": "", "ckpt_paths": [ckpt_path], "task": ""}
        raise FileNotFoundError(
            f"{ckpt_path}: no config.yaml beside it -- a bcrl_agent ckpt needs one"
        )
    config_path = (os.path.join(run_dir, "config.yaml") if fmt == "bcrl_agent"
                   else os.path.join(run_dir, ".hydra", "config.yaml"))
    return {"run_dir": run_dir, "ckpt_format": fmt, "config_path": config_path,
            "ckpt_paths": [ckpt_path], "task": read_task_from_config_file(config_path)}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Job:
    ckpt_path: str
    run_dir: str
    config_path: str
    ckpt_format: str
    task: str
    run_id: str
    epoch_id: int

    @property
    def jid(self) -> str:
        return job_id(self.ckpt_path)

    @property
    def sid(self) -> str:
        return stable_id8(self.ckpt_path)


def parse_epoch_filter(spec: Optional[str]) -> Optional[Callable[[int], bool]]:
    """`--epochs 100-300,final` -> a predicate on epoch_id (-1 means final)."""
    if not spec:
        return None
    exact: set = set()
    ranges: List[Tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("final", "last"):
            exact.add(-1)
        elif re.fullmatch(r"-?\d+", tok):
            exact.add(int(tok))
        elif re.fullmatch(r"\d*\s*-\s*\d*", tok):
            a, _, b = tok.partition("-")
            ranges.append((int(a) if a.strip() else -10**9, int(b) if b.strip() else 10**9))
        else:
            raise ValueError(f"--epochs: cannot parse {tok!r}")
    return lambda e: e in exact or any(lo <= e <= hi for lo, hi in ranges)


def build_jobs(args: argparse.Namespace) -> Tuple[List[Job], List[str]]:
    """Enumerate jobs from the CLI targets. Deterministic: reads only the
    targets and the run configs, never the results dir or the clock."""
    pairs, unresolved = expand_targets(args.targets, args.max_depth)
    jobs: List[Job] = []
    seen = set()
    now = time.time()
    for kind, path in pairs:
        try:
            info = discover_run(path) if kind == "run" else discover_loose_ckpt(path)
        except Exception as e:
            unresolved.append(f"{path} ({type(e).__name__}: {e})")
            continue
        for ckpt in info["ckpt_paths"]:
            if ckpt in seen:
                continue
            seen.add(ckpt)
            if args.min_age_sec > 0:
                try:
                    if now - os.path.getmtime(ckpt) < args.min_age_sec:
                        continue  # still being written by a live trainer
                except OSError:
                    continue
            jobs.append(Job(
                ckpt_path=ckpt, run_dir=info["run_dir"], config_path=info["config_path"],
                ckpt_format=info["ckpt_format"], task=info["task"],
                run_id=parse_run_id(ckpt), epoch_id=parse_epoch_id(ckpt),
            ))

    epoch_ok = parse_epoch_filter(args.epochs)
    inc = [re.compile(r) for r in args.include]
    exc = [re.compile(r) for r in args.exclude]
    fmt_alias = {"a": "bcrl_agent", "b": "base_workspace"}
    want_fmt = fmt_alias.get(args.formats, args.formats)

    def keep(j: Job) -> bool:
        if want_fmt != "both" and j.ckpt_format != want_fmt:
            return False
        if epoch_ok and not epoch_ok(j.epoch_id):
            return False
        if inc and not any(r.search(j.ckpt_path) for r in inc):
            return False
        if any(r.search(j.ckpt_path) for r in exc):
            return False
        return True

    jobs = [j for j in jobs if keep(j)]
    if args.order == "sorted":
        jobs.sort(key=lambda j: (j.task, j.run_id,
                                 j.epoch_id if j.epoch_id >= 0 else 10**9, j.ckpt_path))
    if args.max_ckpts:
        jobs = jobs[:args.max_ckpts]
    if args.shuffle_seed >= 0:
        random.Random(args.shuffle_seed).shuffle(jobs)
    return jobs, unresolved


def plan_hash(jobs: Sequence[Job]) -> str:
    """Fingerprint of the resolved job list, so a worker can prove it agrees."""
    h = hashlib.sha1()
    for j in jobs:
        h.update(f"{j.ckpt_path}\x00{j.ckpt_format}\x00{j.task}\x00"
                 f"{j.run_id}\x00{j.epoch_id}\n".encode())
    return h.hexdigest()[:16]


def print_plan(jobs: Sequence[Job], unresolved: Sequence[str], show: int = 0) -> None:
    log(f"[plan] {len(jobs)} ckpts  plan_hash={plan_hash(jobs)}")
    if unresolved:
        log(f"[plan] {len(unresolved)} unresolved target(s):")
        for u in unresolved[:10]:
            log(f"[plan]   {u}")
    by_fmt = Counter(j.ckpt_format for j in jobs)
    log(f"[plan] by format: {dict(by_fmt)}")
    by_rt = Counter((j.task, j.run_id) for j in jobs)
    log(f"[plan] ckpts per (task, run_id): {len(by_rt)} combos")
    for (task, rid), n in sorted(by_rt.items())[:show or 0]:
        log(f"[plan]   {task:>18s} | {rid:>26s} -> {n}")
    if show and len(by_rt) > show:
        log(f"[plan]   ... and {len(by_rt) - show} more")


# --------------------------------------------------------------------------- #
# Result IO
# --------------------------------------------------------------------------- #

def out_path_for(out_dir: str, job: Job, task: str, bc_weights: str) -> str:
    """Deterministic (no timestamp), so resume is a single stat."""
    safe_task = re.sub(r"[^A-Za-z0-9._-]+", "_", task or "unknown")
    suffix = "_ema" if (bc_weights == "ema" and job.ckpt_format == "base_workspace") else ""
    return os.path.join(
        out_dir, f"{safe_task}_{job.run_id}_epoch{job.epoch_id}{suffix}_{job.sid}.json"
    )


def build_payload(
    *, task: str, job: Job, ckpt_format: str, weights: str, set_results: List[Dict[str, Any]],
    deterministic: bool, field_sources: Dict[str, str], sets_expected: int, worker: str,
) -> Dict[str, Any]:
    L = lazy()
    srs = [r["success_rate"] for r in set_results]
    # Pooled over every episode of every set, so it is not an average of averages over
    # sets with different numbers of successes.
    succ_lens = [
        ln for r in set_results if isinstance(r.get("episodes"), dict)
        for s, ln in zip(r["episodes"]["success"], r["episodes"]["length"]) if s
    ]
    return {
        "task_name": task,
        "run_id": job.run_id,
        "epoch_id": int(job.epoch_id),
        "ckpt_path": os.path.abspath(job.ckpt_path),
        "ckpt_format": ckpt_format,
        "ckpt_hash8": ckpt_hash8(job.ckpt_path),
        "num_sets": len(set_results),
        "episodes_per_set": set_results[0]["n_episodes"] if set_results else 0,
        "deterministic": bool(deterministic),
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "summary": {
            "mean_success": float(L.np.mean(srs)) if srs else 0.0,
            "std_success": float(L.np.std(srs)) if srs else 0.0,
            "min_success": float(L.np.min(srs)) if srs else 0.0,
            "max_success": float(L.np.max(srs)) if srs else 0.0,
            "n_success": len(succ_lens),
            "avg_success_length": float(L.np.mean(succ_lens)) if succ_lens else None,
            "median_success_length": float(L.np.median(succ_lens)) if succ_lens else None,
            "elapsed_s": round(sum(float(r.get("elapsed_s") or 0.0) for r in set_results), 1),
        },
        "field_sources": field_sources,
        "extra": {
            "job_id": job.jid, "stable_id8": job.sid, "weights": weights,
            "sets_expected": int(sets_expected),
            "complete": len(set_results) >= sets_expected,
            "worker": worker, "host": socket.gethostname(),
        },
        "sets": set_results,
    }


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write via a temp file in the same directory, then rename.

    Same-directory rename is atomic on this NFS mount; a plain `open(w)` would
    leave a truncated file at the final path if interrupted -- and with a write
    after every set, interruptions are routine rather than rare. The temp suffix
    is `.tmp`, not `.json`, so a reader globbing `*.json` never sees it.
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ec-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) and "sets" in data else None
    except (json.JSONDecodeError, OSError):
        return None


def has_episodes(row: Dict[str, Any], episodes_per_set: int) -> bool:
    """True if `row` carries a full set of per-episode records."""
    ep = row.get("episodes")
    if not isinstance(ep, dict):
        return False
    return all(
        isinstance(ep.get(k), list) and len(ep[k]) == episodes_per_set
        for k in ("seed", "success", "length")
    )


def classify_existing(
    payload: Optional[Dict[str, Any]], *, num_sets: int, episodes_per_set: int,
    deterministic: bool, require_episodes: bool = False,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """-> ('complete' | 'partial' | 'stale' | 'bad', {set_id: row}).

    A stored set only counts if its seed range is the one we would run now, so a
    file written with different eval settings can never be silently reused. With
    `require_episodes`, a set written before per-episode recording existed also does
    not count -- that is how a directory holding a mix of old and new results gets
    healed without `--force` re-running the sets that are already fine.
    """
    if payload is None:
        return "bad", {}
    if int(payload.get("episodes_per_set", -1)) != episodes_per_set:
        return "stale", {}
    if bool(payload.get("deterministic", True)) != deterministic:
        return "stale", {}
    usable: Dict[int, Dict[str, Any]] = {}
    for row in payload["sets"]:
        try:
            sid = int(row["set_id"])
            if (int(row["n_episodes"]) == episodes_per_set
                    and int(row["seed_lo"]) == (sid - 1) * episodes_per_set + 1
                    and int(row["seed_hi"]) == sid * episodes_per_set
                    and (not require_episodes or has_episodes(row, episodes_per_set))):
                usable[sid] = row
        except (KeyError, TypeError, ValueError):
            continue
    if set(range(1, num_sets + 1)) <= set(usable):
        return "complete", usable
    return ("partial", usable) if usable else ("bad", {})


_LEGACY_INDEX: Dict[str, Dict[Tuple[str, ...], str]] = {}


def legacy_index(out_dir: str) -> Dict[Tuple[str, ...], str]:
    """Index results by mount-invariant ckpt key, so timestamped legacy files
    (and any file whose name we would not guess) still count for resume."""
    if out_dir in _LEGACY_INDEX:
        return _LEGACY_INDEX[out_dir]
    index: Dict[Tuple[str, ...], str] = {}
    best: Dict[Tuple[str, ...], int] = {}
    if os.path.isdir(out_dir):
        for path in sorted(Path(out_dir).glob("*.json")):
            data = load_json(str(path))
            if not data or not data.get("ckpt_path"):
                continue
            key = ckpt_key(str(data["ckpt_path"]))
            n = len(data.get("sets", []))
            if n > best.get(key, -1):
                best[key], index[key] = n, str(path)
    _LEGACY_INDEX[out_dir] = index
    return index


def find_existing(out_dir: str, job: Job, task: str, bc_weights: str,
                  adopt_legacy: bool) -> Optional[str]:
    p = out_path_for(out_dir, job, task, bc_weights)
    if os.path.exists(p):
        return p
    if adopt_legacy:
        return legacy_index(out_dir).get(ckpt_key(job.ckpt_path))
    return None


# --------------------------------------------------------------------------- #
# Running one checkpoint
# --------------------------------------------------------------------------- #

def _silence_inner_tqdm() -> None:
    """Mute the per-episode bar in eval_utils (18k lines per ckpt in a log)."""
    from policies.workspace import eval_utils
    if not getattr(eval_utils, "_ec_silenced", False):
        eval_utils.tqdm = functools.partial(eval_utils.tqdm, disable=True)
        eval_utils._ec_silenced = True  # type: ignore[attr-defined]


def summarize_episodes(per_ep: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar summaries of a set's per-episode arrays.

    `length` is control steps actually executed. The env terminates on success and the
    rollout loop breaks on it, so for a successful episode this is steps-to-success and
    for a failed one it is `max_episode_steps`. Averaging the two together would be
    meaningless, hence the split.
    """
    succ = per_ep["success"]
    lens = per_ep["length"]
    ok = [ln for s, ln in zip(succ, lens) if s]
    bad = [ln for s, ln in zip(succ, lens) if not s]
    L = lazy()
    return {
        "n_success": int(sum(succ)),
        "avg_success_length": float(L.np.mean(ok)) if ok else None,
        "avg_failure_length": float(L.np.mean(bad)) if bad else None,
    }


def iter_eval_sets(
    policy: Any, ns: SimpleNamespace, device: Any, *,
    num_sets: int, episodes_per_set: int, skip_sets: set,
) -> Iterator[Dict[str, Any]]:
    """Yield one result row per set. Set k covers seeds
    ((k-1)*eps+1 .. k*eps) -- the exact blocks the legacy tool used."""
    from policies.train_loop import create_eval_env
    from policies.workspace.eval_utils import evaluate_policy_in_sim_env

    reset_fn = getattr(policy, "reset", None)
    env = create_eval_env(ns)
    try:
        for k in range(1, num_sets + 1):
            if k in skip_sets:
                continue
            seeds = list(range((k - 1) * episodes_per_set + 1, k * episodes_per_set + 1))
            t0 = time.time()
            metrics, _ = evaluate_policy_in_sim_env(
                env, policy,
                eval_seeds=seeds,
                max_episode_steps=ns.eval["max_episode_steps"],
                sim_freq=ns.eval["sim_freq"],
                cameras=ns.env["cameras"],
                device=device,
                camera_height=ns.env["camera_height"],
                camera_width=ns.env["camera_width"],
                n_vis=0,
                progress_desc=f"set {k}/{num_sets}",
                reset_policy_fn=reset_fn if callable(reset_fn) else None,
                # Per-episode outcomes are what the trajectory-length study and the
                # seed-paired McNemar tests are computed from; a set-level success rate
                # cannot support either.
                return_per_episode=True,
            )
            per_ep = metrics["per_episode"]
            row = {
                "set_id": k, "seed_lo": seeds[0], "seed_hi": seeds[-1],
                "n_episodes": len(seeds),
                "success_rate": float(metrics["success_rate"]),
                "avg_reward": float(metrics["avg_reward"]),
                "elapsed_s": round(time.time() - t0, 2),
                # Episode lengths are right-censored at max_episode_steps and counted in
                # control steps, so a length table that mixed two horizons or two
                # sim_freqs would be nonsense. Recorded per set so the analysis can
                # assert they are constant rather than trusting that configs never drift.
                "max_episode_steps": int(ns.eval["max_episode_steps"]),
                "sim_freq": int(ns.eval["sim_freq"]),
                "episodes": per_ep,
            }
            row.update(summarize_episodes(per_ep))
            yield row
    finally:
        env.close()


def job_status(job: Job, args: argparse.Namespace) -> str:
    """Disk-only completeness check: 'complete' | 'partial' | 'stale' | 'bad'.

    Cheap enough (one stat plus at most one JSON read) for a worker to run over the whole
    job list on every queue pass, and it never touches CUDA.
    """
    existing = find_existing(args.out_dir, job, job.task, args.bc_weights, args.adopt_legacy)
    status, _ = classify_existing(
        load_json(existing) if existing else None,
        num_sets=args.num_sets, episodes_per_set=args.episodes_per_set,
        deterministic=not args.stochastic,
        require_episodes=args.require_episodes,
    )
    return status


def run_one_ckpt(
    job: Job, args: argparse.Namespace, worker: str,
    on_progress: Optional[Callable[[], None]] = None,
) -> str:
    """-> 'ok' | 'resumed' | 'skipped'. Raises on failure.

    `on_progress` is invoked after every set is written; the queue uses it to refresh its
    claim's lease so a healthy but slow worker is never mistaken for a dead one.
    """
    L = lazy()
    deterministic = not args.stochastic
    existing = find_existing(args.out_dir, job, job.task, args.bc_weights, args.adopt_legacy)
    status, done = classify_existing(
        load_json(existing) if existing else None,
        num_sets=args.num_sets, episodes_per_set=args.episodes_per_set,
        deterministic=deterministic, require_episodes=args.require_episodes,
    )
    if not args.force:
        if status == "complete":
            log(f"[skip] {job.task} {job.run_id}@{job.epoch_id} -> {existing}")
            return "skipped"
        if status == "stale":
            raise RuntimeError(
                f"{existing} was written with different eval settings; "
                f"pass --force to overwrite"
            )
    if args.force or args.redo_partial:
        done = {}

    device = L.torch.device(args.device)
    policy, cfg, fmt, weights = load_policy_from_ckpt(job.ckpt_path, device, args.bc_weights)
    ns, field_sources = build_eval_namespace(cfg)
    if job.task and ns.task != job.task:
        log(f"[warn] {job.ckpt_path}: config says task={job.task!r} but ckpt says "
            f"{ns.task!r}; using the ckpt's")
    task = ns.task
    out_path = out_path_for(args.out_dir, job, task, args.bc_weights)
    policy._deterministic_eval = deterministic

    n_todo = args.num_sets - len(set(done) & set(range(1, args.num_sets + 1)))
    log(f"[run] {task} {job.run_id}@{job.epoch_id} fmt={fmt} weights={weights} "
        f"sets={n_todo}/{args.num_sets}{' (resuming)' if done else ''} -> {out_path}")

    sets_done = dict(done)
    try:
        for row in iter_eval_sets(
            policy, ns, device,
            num_sets=args.num_sets, episodes_per_set=args.episodes_per_set,
            skip_sets=set(sets_done),
        ):
            sets_done[row["set_id"]] = row
            # Re-merge with disk before every write. `sets_done` started as a snapshot
            # taken before the first set, so without this a worker that was slow (or was
            # presumed dead and had its job taken over) would write its stale view over a
            # file another worker had since completed, silently deleting finished sets and
            # flipping `complete` back to false. Ours win on conflict -- they are the ones
            # just measured -- and `classify_existing` still vets the seed ranges, so a row
            # written under different settings can never be resurrected this way.
            disk = load_json(out_path)
            if disk is not None:
                _, on_disk = classify_existing(
                    disk, num_sets=args.num_sets,
                    episodes_per_set=args.episodes_per_set,
                    deterministic=deterministic,
                    require_episodes=args.require_episodes,
                )
                for k, prior in on_disk.items():
                    sets_done.setdefault(k, prior)
            ordered = [sets_done[k] for k in sorted(sets_done)]
            if not args.no_write:
                atomic_write_json(out_path, build_payload(
                    task=task, job=job, ckpt_format=fmt, weights=weights,
                    set_results=ordered, deterministic=deterministic,
                    field_sources=field_sources, sets_expected=args.num_sets,
                    worker=worker,
                ))
            if on_progress is not None:
                on_progress()
            mean = float(L.np.mean([r["success_rate"] for r in ordered]))
            log(f"[set {row['set_id']}/{args.num_sets}] "
                f"success={row['success_rate']:.3f} running_mean={mean:.3f} "
                f"({row['elapsed_s']:.0f}s)")
    finally:
        del policy
        if L.torch.cuda.is_available():
            L.torch.cuda.empty_cache()

    srs = [sets_done[k]["success_rate"] for k in sorted(sets_done)]
    log(f"[done] {task} {job.run_id}@{job.epoch_id} mean={L.np.mean(srs):.4f} "
        f"+/- {L.np.std(srs):.4f} over {len(srs)} sets")
    return "resumed" if done else "ok"


# --------------------------------------------------------------------------- #
# Claim queue
#
# Static `jobs[i::n]` sharding assumes the worker set is fixed for the whole sweep. It
# is not: SLURM allocations expire mid-run and machines are handed back and borrowed on
# the fly, and a departed worker's shard would simply never finish. With a claim dir on
# the shared filesystem, any worker on any host can take any unfinished job, so hosts
# join and leave without anyone re-planning anything.
# --------------------------------------------------------------------------- #

def _claim_stem(job: Job, args: argparse.Namespace) -> str:
    """Claim identity: the result file it will produce, plus the depth.

    Deriving it from `out_path_for` keeps claims 1:1 with results by construction, so
    `bc_weights` and the task name cannot drift apart between the two. The depth matters
    on its own: a checkpoint finished at 3 sets is *not* finished at 15, and the staged
    narrow-then-full passes must not see each other's claims.
    """
    base = os.path.basename(out_path_for(args.out_dir, job, job.task, args.bc_weights))
    stem = base[:-5] if base.endswith(".json") else base
    return f"{stem}.{args.num_sets}x{args.episodes_per_set}"


def _claim_path(claim_dir: str, job: Job, args: argparse.Namespace) -> str:
    return os.path.join(claim_dir, _claim_stem(job, args) + ".claim")


def _fail_path(claim_dir: str, job: Job, args: argparse.Namespace) -> str:
    return os.path.join(claim_dir, "failed", _claim_stem(job, args) + ".json")


def _done_path(claim_dir: str, job: Job, args: argparse.Namespace) -> str:
    return os.path.join(claim_dir, "done", _claim_stem(job, args) + ".json")


def fs_now(dir_: str) -> float:
    """"Now", as the shared filesystem stamps it.

    Staleness compares a claim's mtime -- written by whichever node holds it -- against
    "now". Reading "now" off the local clock means one node with a drifting clock
    declares every claim stale and steals the entire queue. Stamping a scratch file in
    the same directory puts both timestamps in the same clock domain.
    """
    try:
        fd, p = tempfile.mkstemp(dir=dir_, prefix=".now-")
        try:
            os.close(fd)
            return os.stat(p).st_mtime
        finally:
            try:
                os.unlink(p)
            except OSError:
                pass
    except OSError:
        return time.time()


def _read_json_any(path: str) -> Optional[Dict[str, Any]]:
    """`load_json` insists on a result payload; failure records are not one."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


@dataclasses.dataclass
class Claim:
    path: str

    def touch(self) -> None:
        """Refresh the lease."""
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def release(self) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass


# The claim currently held by this process, so a signal handler can hand it back
# instead of leaving it to time out. The parent SIGKILLs workers `SHUTDOWN_GRACE`
# seconds after SIGINT, which is shorter than one set, so `finally` alone is not enough.
_ACTIVE_CLAIM: Optional[Claim] = None


def _release_active_claim(signum: int, _frame: Any) -> None:
    if _ACTIVE_CLAIM is not None:
        log(f"[queue] signal {signum}: releasing {os.path.basename(_ACTIVE_CLAIM.path)}")
        _ACTIVE_CLAIM.release()
    raise SystemExit(128 + signum)


class Heartbeat(threading.Thread):
    """Refresh a claim on a timer for as long as the job runs.

    Refreshing only between sets would leave the lease frozen through checkpoint load
    plus a whole set -- minutes -- and any set slower than the lease would get the job
    stolen out from under a perfectly healthy worker.
    """

    def __init__(self, claim: Claim, interval: float = 60.0) -> None:
        super().__init__(daemon=True)
        # NOT `_stop`: threading.Thread._stop is a method, and Thread.join() calls it.
        self.claim, self.interval, self._finished = claim, interval, threading.Event()

    def run(self) -> None:
        while not self._finished.wait(self.interval):
            self.claim.touch()

    def __enter__(self) -> "Heartbeat":
        self.claim.touch()
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._finished.set()
        self.join(timeout=5.0)


def try_claim(
    claim_dir: str, job: Job, args: argparse.Namespace, worker: str
) -> Optional[Claim]:
    """Take ownership of `job`, or return None if someone else holds it.

    `O_CREAT|O_EXCL` is atomic on this GPFS mount, so exactly one worker wins a fresh
    claim. A claim whose mtime has not moved for `--lease-sec` belongs to a worker that
    died; it is stolen by renaming it aside -- also atomic, so exactly one racer's
    rename succeeds -- and then re-running the same `O_EXCL` create.
    """
    path = _claim_path(claim_dir, job, args)
    for retry in (False, True):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if retry:
                return None  # another worker re-created it first
            try:
                age = fs_now(claim_dir) - os.stat(path).st_mtime
            except OSError:
                continue  # released between our create and our stat
            if age < args.lease_sec:
                return None
            aside = f"{path}.stale-{socket.gethostname()}-{os.getpid()}"
            try:
                os.rename(path, aside)
                os.unlink(aside)
            except OSError:
                return None  # someone else stole it first
            log(f"[queue] stole claim {os.path.basename(path)} "
                f"({age / 60:.0f} min stale) for {job.run_id}@{job.epoch_id}")
            continue
        except OSError as e:
            log(f"[queue] cannot claim {path}: {e}")
            return None
        with os.fdopen(fd, "w") as f:
            json.dump({
                "ckpt": job.ckpt_path, "worker": worker, "host": socket.gethostname(),
                "pid": os.getpid(), "claimed": dt.datetime.now().isoformat(timespec="seconds"),
            }, f)
        return Claim(path)
    return None


def failure_count(claim_dir: str, job: Job, args: argparse.Namespace) -> int:
    data = _read_json_any(_fail_path(claim_dir, job, args))
    return len(data.get("attempts", [])) if data else 0


def mark_done(claim_dir: str, job: Job, args: argparse.Namespace, worker: str) -> None:
    """Drop a small completion marker, if the checkpoint really is finished.

    Progress reporting otherwise means opening all ~900 result files, which with
    per-episode records is ~100 MB a time. These markers make `dispatch.py status` a
    directory listing. They are strictly a cache: the result JSONs remain the truth, and
    `status --exact` ignores the markers entirely.
    """
    if job_status(job, args) != "complete":
        return
    payload = load_json(out_path_for(args.out_dir, job, job.task, args.bc_weights)) or {}
    summary = payload.get("summary", {})
    atomic_write_json(_done_path(claim_dir, job, args), {
        "task": job.task, "run_id": job.run_id, "epoch_id": job.epoch_id,
        "success_rate": summary.get("mean_success"),
        "n_success": summary.get("n_success"),
        "avg_success_length": summary.get("avg_success_length"),
        "elapsed_s": summary.get("elapsed_s"),
        "worker": worker, "host": socket.gethostname(),
        "t": dt.datetime.now().isoformat(timespec="seconds"),
    })


def record_failure(
    claim_dir: str, job: Job, args: argparse.Namespace, err: str
) -> None:
    path = _fail_path(claim_dir, job, args)
    data = _read_json_any(path) or {"ckpt": job.ckpt_path, "attempts": []}
    data["attempts"].append({
        "t": dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "err": err[:800],
    })
    atomic_write_json(path, data)


class StatusCache:
    """Memoized `job_status`, keyed on the result file's (mtime, size).

    A worker re-checks all ~900 jobs on every queue pass. Reading each result JSON every
    time would be tens of MB of shared-filesystem traffic per worker per pass once
    per-episode data is in them; a finished result never changes again, so after the
    first pass this is one `stat` per job.
    """

    def __init__(self) -> None:
        self._m: Dict[str, Tuple[Any, str]] = {}

    def status(self, job: Job, args: argparse.Namespace) -> str:
        path = out_path_for(args.out_dir, job, job.task, args.bc_weights)
        try:
            st = os.stat(path)
            key: Any = (st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        # With --adopt-legacy the answer can come from a file we did not stat, so the
        # key would not describe it. Only the miss path is affected: never cache that.
        if key is None and args.adopt_legacy:
            return job_status(job, args)
        hit = self._m.get(job.sid)
        if hit is not None and hit[0] == key:
            return hit[1]
        status = job_status(job, args)
        self._m[job.sid] = (key, status)
        return status


def plan_hash_stable(jobs: Sequence[Job]) -> str:
    """Like `plan_hash`, but invariant to which mount the targets were named through.

    `/scratch` and `/scache` are two views of one GPFS filesystem and only some nodes
    have both, so the same 905 checkpoints hash differently under `plan_hash`. `sid`
    depends only on the last three path components, so this agrees everywhere -- which is
    what a claim dir shared between hosts needs.
    """
    h = hashlib.sha1()
    for j in jobs:
        h.update(f"{j.sid}\x00{j.task}\x00{j.run_id}\x00{j.epoch_id}\n".encode())
    return h.hexdigest()[:16]


def check_queue_plan(claim_dir: str, jobs: Sequence[Job], args: argparse.Namespace) -> None:
    """Record what this queue is for, and refuse to join one built for something else.

    Two sweeps sharing a claim dir would hand each other jobs from the wrong plan. The
    first worker in writes the manifest; everyone after verifies it.
    """
    want = {
        "n_jobs": len(jobs), "plan": plan_hash_stable(jobs),
        "num_sets": args.num_sets, "episodes_per_set": args.episodes_per_set,
        "deterministic": not args.stochastic, "bc_weights": args.bc_weights,
        "out_dir": os.path.abspath(args.out_dir),
    }
    path = os.path.join(claim_dir, "PLAN.json")
    have = _read_json_any(path)
    if have is None:
        atomic_write_json(path, want)
        return
    diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    if diff:
        raise SystemExit(
            f"[queue] {path} describes a different sweep; refusing to join.\n"
            + "\n".join(f"    {k}: on disk {a!r} != requested {b!r}"
                        for k, (a, b) in diff.items())
            + "\n    Use a fresh --claim-dir, or delete that file if this is intended."
        )


def run_queue(args: argparse.Namespace, jobs: List[Job], worker: str) -> Dict[str, Any]:
    """Pull jobs from the shared claim dir until the whole sweep is done."""
    global _ACTIVE_CLAIM
    claim_dir = args.claim_dir
    os.makedirs(os.path.join(claim_dir, "failed"), exist_ok=True)
    os.makedirs(os.path.join(claim_dir, "done"), exist_ok=True)
    check_queue_plan(claim_dir, jobs, args)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _release_active_claim)
        except (ValueError, OSError):
            pass

    # Start each worker at a different point in the list so that N workers do not all
    # race for job 0, and so one host's workers spread over tasks instead of piling onto
    # the same one. Deterministic in the worker tag, so a restarted worker resumes where
    # it was rather than re-walking from the top.
    off = int(hashlib.sha1(worker.encode()).hexdigest(), 16) % max(len(jobs), 1)
    order = jobs[off:] + jobs[:off]

    cache = StatusCache()
    counts: Counter = Counter()
    failures: List[Dict[str, str]] = []
    stale_seen: set = set()
    gave_up: set = set()
    t0 = time.time()
    last_change = time.time()
    prev_complete = -1

    while True:
        claimed_any = False
        n_complete = n_remaining = 0
        for job in order:
            status = cache.status(job, args)
            if status == "complete":
                n_complete += 1
                continue
            if status == "stale":
                if job.sid not in stale_seen:
                    stale_seen.add(job.sid)
                    log(f"[queue] SKIP {job.ckpt_path}: existing result used different "
                        f"eval settings; pass --force to overwrite")
                continue
            if job.sid in gave_up:
                continue
            n_fail = failure_count(claim_dir, job, args)
            if n_fail >= args.max_attempts:
                gave_up.add(job.sid)
                log(f"[queue] GIVE UP {job.ckpt_path}: failed {n_fail}x")
                continue
            n_remaining += 1

            claim = try_claim(claim_dir, job, args, worker)
            if claim is None:
                continue
            claimed_any = True
            _ACTIVE_CLAIM = claim
            log(f"\n=== [{worker}] {job.task} {job.run_id}@{job.epoch_id} "
                f"(attempt {n_fail + 1}/{args.max_attempts}) ===")
            try:
                with Heartbeat(claim, args.heartbeat_sec):
                    counts[run_one_ckpt(job, args, worker,
                                        on_progress=claim.touch)] += 1
                mark_done(claim_dir, job, args, worker)
            except KeyboardInterrupt:
                claim.release()
                raise
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                counts["failed"] += 1
                failures.append({"ckpt": job.ckpt_path, "err": msg})
                record_failure(claim_dir, job, args, msg)
                log(f"[ERROR] {job.ckpt_path}: {msg}")
            finally:
                claim.release()
                _ACTIVE_CLAIM = None

        if n_remaining == 0:
            log(f"[queue] nothing left to do "
                f"({n_complete} complete, {len(gave_up)} abandoned, "
                f"{len(stale_seen)} stale)")
            break
        if n_complete != prev_complete:
            prev_complete, last_change = n_complete, time.time()
        if claimed_any:
            continue
        idle = time.time() - last_change
        if idle > args.max_idle_sec:
            log(f"[queue] {n_remaining} job(s) still held by other workers and no global "
                f"progress for {idle / 60:.0f} min -- exiting")
            break
        log(f"[queue] {n_complete} done, {n_remaining} in flight elsewhere, none "
            f"claimable -- sleeping {args.idle_sleep:.0f}s")
        time.sleep(args.idle_sleep)

    # In queue mode `jobs` is the whole sweep for every worker, so "total" would be a
    # meaningless multiple of it. Report what this worker actually did.
    summary = {
        "tag": worker, "mode": "queue", "total": counts["ok"] + counts["resumed"],
        "ok": counts["ok"], "resumed": counts["resumed"],
        "skipped": counts["skipped"], "failed": counts["failed"],
        "gave_up": len(gave_up), "sweep_size": len(jobs),
        "elapsed_s": round(time.time() - t0, 1), "failures": failures[:20],
    }
    log(f"{SUMMARY_TOKEN} {json.dumps(summary)}")
    return summary


# --------------------------------------------------------------------------- #
# Shard runner
# --------------------------------------------------------------------------- #

def run_shard(args: argparse.Namespace, jobs: List[Job], worker: str) -> Dict[str, Any]:
    if args.progress == "inner" or (args.progress == "auto" and sys.stdout.isatty()):
        pass  # keep the per-episode bar
    else:
        _silence_inner_tqdm()

    if args.claim_dir:
        return run_queue(args, jobs, worker)

    counts: Counter = Counter()
    failures: List[Dict[str, str]] = []
    t0 = time.time()
    for i, job in enumerate(jobs, start=1):
        log(f"\n=== [{worker}] {i}/{len(jobs)}  {job.task} {job.run_id}@{job.epoch_id} ===")
        try:
            counts[run_one_ckpt(job, args, worker)] += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            counts["failed"] += 1
            failures.append({"ckpt": job.ckpt_path, "err": f"{type(e).__name__}: {e}"})
            log(f"[ERROR] {job.ckpt_path}: {type(e).__name__}: {e}")
    summary = {
        "tag": worker, "total": len(jobs), "ok": counts["ok"], "resumed": counts["resumed"],
        "skipped": counts["skipped"], "failed": counts["failed"],
        "elapsed_s": round(time.time() - t0, 1), "failures": failures[:20],
    }
    log(f"{SUMMARY_TOKEN} {json.dumps(summary)}")
    return summary


# --------------------------------------------------------------------------- #
# Multi-GPU parent
# --------------------------------------------------------------------------- #

def detect_gpus() -> List[int]:
    """Ask nvidia-smi first so the parent never initializes a CUDA context."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return list(range(len([t for t in cvd.split(",") if t.strip()])))
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return []


def parse_gpus(spec: Optional[str]) -> List[int]:
    if spec is None:
        return detect_gpus()
    if spec.strip().lower() in ("all", ""):
        return detect_gpus()
    if spec.strip().lower() == "none":
        return []
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    seen: set = set()
    return [g for g in out if not (g in seen or seen.add(g))]


# Flags the parent must hand to every worker verbatim, so both sides enumerate
# the same jobs and evaluate them the same way.
FORWARDED = [
    ("include", "append"), ("exclude", "append"), ("epochs", "value"),
    ("formats", "value"), ("max_depth", "value"), ("min_age_sec", "value"),
    ("max_ckpts", "value"), ("order", "value"), ("shuffle_seed", "value"),
    ("num_sets", "value"), ("episodes_per_set", "value"), ("bc_weights", "value"),
    ("stochastic", "flag"), ("out_dir", "value"), ("force", "flag"),
    ("redo_partial", "flag"), ("no_write", "flag"), ("adopt_legacy", "bool"),
    ("require_episodes", "flag"), ("claim_dir", "value"), ("lease_sec", "value"),
    ("max_attempts", "value"), ("idle_sleep", "value"), ("max_idle_sec", "value"),
    ("heartbeat_sec", "value"),
]
PARENT_ONLY = {"gpus", "serial", "log_dir", "tee", "tee_only", "tail_on_fail",
               "retries", "threads", "dry_run", "show", "targets", "device",
               "workers_per_gpu"}
WORKER_ONLY = {"shard", "plan_hash", "worker_tag", "allow_plan_drift", "progress"}


def build_worker_argv(args: argparse.Namespace, shard: Tuple[int, int], tag: str,
                      phash: str) -> List[str]:
    argv = [sys.executable, "-u", str(Path(__file__).resolve())] + list(args.targets)
    for dest, kind in FORWARDED:
        v = getattr(args, dest)
        flag = "--" + dest.replace("_", "-")
        if kind == "append":
            for item in v or []:
                argv += [flag, str(item)]
        elif kind == "flag":
            if v:
                argv.append(flag)
        elif kind == "bool":
            argv.append(flag if v else "--no-" + dest.replace("_", "-"))
        elif v is not None:
            argv += [flag, str(v)]
    argv += ["--shard", f"{shard[0]}/{shard[1]}", "--plan-hash", phash,
             "--worker-tag", tag, "--progress", "none", "--device", "cuda"]
    return argv


def _forward(stream: IO[bytes], log_fp: IO[str], prefix: Optional[str]) -> None:
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode(errors="replace")
            try:
                log_fp.write(line)
                log_fp.flush()
            except Exception:
                pass
            if prefix is not None:
                for sub in line.rstrip("\n").split("\r"):
                    if sub:
                        sys.stdout.write(f"{prefix} {sub}\n")
                sys.stdout.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def launch_workers(args: argparse.Namespace, jobs: List[Job], gpus: List[int]) -> int:
    phash = plan_hash(jobs)
    host = socket.gethostname().split(".")[0]
    # One process per (gpu, replica). The simulator runs on CPU (`eval_cpu: true`), so a
    # single worker leaves the GPU mostly idle -- --workers-per-gpu is how that headroom
    # gets used, bounded by the node's cores.
    # Replica-major, so consecutive shard indices land on *different* GPUs -- with the
    # static `jobs[i::n]` split that keeps each GPU's slice spread over the job list
    # instead of two workers on one GPU getting adjacent (same-task) jobs.
    slots = [(g, k) for k in range(max(1, args.workers_per_gpu)) for g in gpus]
    slots = slots[:max(1, min(len(slots), len(jobs)))]
    n = len(slots)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    plan = []
    for i, (gpu, rep) in enumerate(slots):
        # In queue mode every worker can reach every job, so the shard is only an
        # identity; in static mode it is the actual slice.
        n_jobs = len(jobs) if args.claim_dir else len(jobs[i::n])
        if not n_jobs:
            continue
        tag = f"{host}-gpu{gpu}" + (f"-w{rep}" if args.workers_per_gpu > 1 else "")
        cmd = build_worker_argv(args, (i, n), tag, phash)
        lp = log_dir / f"{tag}_shard{i}of{n}_{stamp}.log"
        plan.append({"gpu": gpu, "tag": tag, "shard": (i, n), "n": n_jobs,
                     "cmd": cmd, "log": lp})

    mode = f"queue({args.claim_dir})" if args.claim_dir else "static shards"
    log(f"[parent] {len(jobs)} ckpts over {len(plan)} worker(s) on gpus={gpus} "
        f"x{max(1, args.workers_per_gpu)}/gpu  mode={mode}  plan_hash={phash}")
    for p in plan:
        log(f"[parent]   {p['tag']} shard={p['shard'][0]}/{p['shard'][1]} "
            f"jobs={p['n']} log={p['log']}")
        log(f"[parent]     CUDA_VISIBLE_DEVICES={p['gpu']} "
            f"{' '.join(shlex.quote(c) for c in p['cmd'])}")
    if args.dry_run:
        return 0

    tee_filter = ({int(x) for x in args.tee_only.split(",") if x.strip()}
                  if args.tee and args.tee_only else None)
    procs: List[Dict[str, Any]] = []
    files: List[IO[str]] = []
    threads: List[threading.Thread] = []

    def spawn(p: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(p["gpu"])
        env["PYTHONPATH"] = os.pathsep.join([
            str(REPO_ROOT), str(REPO_ROOT / "third_party" / "diffusion_policy"),
            env.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        env["EVAL_CKPTS_WORKER"] = "1"
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(args.threads)
        path = p["log"] if attempt == 0 else Path(f"{p['log']}.retry{attempt}")
        fp = open(path, "w", buffering=1)
        files.append(fp)
        fp.write(f"# CUDA_VISIBLE_DEVICES={p['gpu']}\n"
                 f"# CMD: {' '.join(shlex.quote(c) for c in p['cmd'])}\n\n")
        if args.tee:
            proc = subprocess.Popen(
                p["cmd"], env=env, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0, start_new_session=True,
            )
            show = tee_filter is None or p["gpu"] in tee_filter
            t = threading.Thread(target=_forward,
                                 args=(proc.stdout, fp, f"[{p['tag']}]" if show else None),
                                 daemon=True)
            t.start()
            threads.append(t)
        else:
            proc = subprocess.Popen(p["cmd"], env=env, cwd=str(REPO_ROOT), stdout=fp,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        log(f"[parent] started {p['tag']} pid={proc.pid} attempt={attempt} log={path}")
        return {**p, "proc": proc, "attempt": attempt, "active_log": path}

    running = [spawn(p, 0) for p in plan]
    results: List[Dict[str, Any]] = []
    try:
        while running:
            time.sleep(POLL_INTERVAL)
            still = []
            for r in running:
                rc = r["proc"].poll()
                if rc is None:
                    still.append(r)
                elif rc != 0 and r["attempt"] < args.retries:
                    log(f"[parent] {r['tag']} rc={rc} -> retrying "
                        f"(finished sets are already on disk)")
                    still.append(spawn(r, r["attempt"] + 1))
                else:
                    log(f"[parent] {r['tag']} -> "
                        f"{'ok' if rc == 0 else f'FAIL rc={rc}'}  log={r['active_log']}")
                    results.append({**r, "rc": rc})
            running = still
    except KeyboardInterrupt:
        log("\n[parent] interrupted -> stopping workers")
        for r in running:
            try:
                os.killpg(r["proc"].pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
        t0 = time.time()
        while running and time.time() - t0 < SHUTDOWN_GRACE:
            running = [r for r in running if r["proc"].poll() is None]
            time.sleep(1.0)
        for r in running:
            try:
                os.killpg(r["proc"].pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        return 130
    finally:
        for t in threads:
            t.join(timeout=5.0)
        for fp in files:
            try:
                fp.close()
            except Exception:
                pass

    return report(results, args.tail_on_fail)


def report(results: List[Dict[str, Any]], tail: int) -> int:
    """Aggregate the workers' summary lines out of their logs."""
    agg = Counter()
    failures: List[Tuple[str, Dict[str, str]]] = []
    for r in results:
        summary = None
        try:
            for line in reversed(Path(r["active_log"]).read_text().splitlines()[-400:]):
                if line.startswith(SUMMARY_TOKEN):
                    summary = json.loads(line[len(SUMMARY_TOKEN):])
                    break
        except OSError:
            pass
        if summary:
            for k in ("total", "ok", "resumed", "skipped", "failed"):
                agg[k] += summary.get(k, 0)
            for f in summary.get("failures", []):
                failures.append((summary["tag"], f))
        elif r["rc"] != 0:
            agg["worker_lost"] += 1

    n_bad = sum(1 for r in results if r["rc"] != 0)
    log(f"\n[parent] workers ok={len(results) - n_bad}/{len(results)}")
    log(f"[parent] ckpts: ok={agg['ok']} resumed={agg['resumed']} "
        f"skipped={agg['skipped']} failed={agg['failed']} (planned {agg['total']})")
    for tag, f in failures:
        log(f"[parent]   FAIL {tag}  {f['ckpt']}  {f['err']}")
    if tail > 0:
        for r in results:
            if r["rc"] == 0:
                continue
            log(f"\n[parent] --- tail of {r['tag']} (rc={r['rc']}) ---")
            try:
                for ln in Path(r["active_log"]).read_text().splitlines()[-tail:]:
                    log(ln)
            except OSError as e:
                log(f"[parent] could not read {r['active_log']}: {e}")
    return 0 if (n_bad == 0 and agg["failed"] == 0) else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_ckpts.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("targets", nargs="+",
                   help="ckpt file, run dir, dir of run dirs, or .txt list of any of these.")

    g = p.add_argument_group("selection")
    g.add_argument("--include", action="append", default=[], metavar="RE",
                   help="Keep only ckpts whose path matches ANY of these. Repeatable.")
    g.add_argument("--exclude", action="append", default=[], metavar="RE",
                   help="Drop ckpts matching ANY of these; wins over --include. Repeatable.")
    g.add_argument("--epochs", default=None, metavar="SPEC",
                   help="e.g. '100-300', '25,50,75', 'final', '100-300,final'.")
    g.add_argument("--formats", default="both",
                   choices=["both", "a", "b", "bcrl_agent", "base_workspace"])
    g.add_argument("--max-depth", type=int, default=3,
                   help="Recursion depth when a target dir is not itself a run dir.")
    g.add_argument("--min-age-sec", type=float, default=0.0,
                   help="Skip ckpts modified more recently than this (a live trainer "
                        "may still be writing them).")
    g.add_argument("--max-ckpts", type=int, default=0, metavar="N",
                   help="Take only the first N jobs, before sharding. 0 = all.")
    g.add_argument("--order", default="sorted", choices=["sorted", "given"])
    g.add_argument("--shuffle-seed", type=int, default=-1,
                   help="If >=0, deterministically shuffle after sorting.")

    g = p.add_argument_group("eval")
    g.add_argument("--num-sets", type=int, default=15)
    g.add_argument("--episodes-per-set", type=int, default=50,
                   help="Set k covers seeds ((k-1)*eps+1 .. k*eps). "
                        "Defaults match every previous sweep.")
    g.add_argument("--bc-weights", default="model", choices=["model", "ema"],
                   help="Which weight set to score in a BC workspace ckpt. 'model' is "
                        "what previous sweeps used and what wmrl loads when its config "
                        "says pretrained_weights: model.")
    g.add_argument("--stochastic", action="store_true",
                   help="Sample actions (default: deterministic).")
    g.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cuda:N' | 'cpu'.")

    g = p.add_argument_group("output and resume")
    g.add_argument("--out-dir", default="runs/eval_ckpts_results")
    g.add_argument("--force", action="store_true",
                   help="Re-run everything, ignoring complete and partial results.")
    g.add_argument("--redo-partial", action="store_true",
                   help="Restart partially-evaluated ckpts at set 1.")
    g.add_argument("--no-write", action="store_true", help="Run but persist nothing.")
    g.add_argument("--adopt-legacy", action=argparse.BooleanOptionalAction, default=True,
                   help="Also honour legacy timestamped JSONs in --out-dir when resuming.")
    g.add_argument("--require-episodes", action="store_true",
                   help="Treat a stored set without per-episode records as missing, so "
                        "results predating per-episode recording are recomputed. Needed "
                        "for the trajectory-length and paired-episode analyses.")

    g = p.add_argument_group("claim queue (multi-host)")
    g.add_argument("--claim-dir", default=None, metavar="DIR",
                   help="Shared directory of claim files. When set, workers pull from a "
                        "common queue instead of running a fixed --shard, so hosts can "
                        "join and leave mid-sweep. Put it on the shared filesystem.")
    g.add_argument("--lease-sec", type=float, default=900.0,
                   help="A claim not refreshed for this long is stolen. Safe well below "
                        "one set's runtime because a background thread refreshes it "
                        "every --heartbeat-sec; smaller means faster recovery from a "
                        "node that disappears.")
    g.add_argument("--heartbeat-sec", type=float, default=60.0,
                   help="How often the claim held by a running job is refreshed.")
    g.add_argument("--max-attempts", type=int, default=3,
                   help="Give up on a checkpoint after this many failures.")
    g.add_argument("--idle-sleep", type=float, default=60.0,
                   help="Wait this long before rescanning when nothing is claimable.")
    g.add_argument("--max-idle-sec", type=float, default=3600.0,
                   help="Exit if nothing is claimable and no worker anywhere completes a "
                        "checkpoint for this long.")

    g = p.add_argument_group("parallelism")
    g.add_argument("--gpus", default=None, metavar="SPEC",
                   help="'0-6' | '0,2,3' | 'all' | 'none'. Default: autodetect. "
                        "One GPU means in-process, no subprocess.")
    g.add_argument("--workers-per-gpu", type=int, default=1, metavar="N",
                   help="Processes per GPU. The sim runs on CPU, so N>1 usually helps "
                        "until the node's cores run out; lower --threads to match.")
    g.add_argument("--serial", action="store_true",
                   help="Force in-process even with several GPUs listed.")
    g.add_argument("--log-dir", default="runs/eval_ckpts_logs")
    g.add_argument("--tee", action="store_true", help="Mirror worker output here.")
    g.add_argument("--tee-only", default=None, metavar="GPUS",
                   help="With --tee, only mirror these GPUs.")
    g.add_argument("--tail-on-fail", type=int, default=60)
    g.add_argument("--retries", type=int, default=1,
                   help="Relaunch a worker that exits non-zero. Cheap: finished sets "
                        "are already on disk.")
    g.add_argument("--threads", type=int, default=4,
                   help="OMP/MKL threads per worker (the sim runs on CPU).")
    g.add_argument("--dry-run", action="store_true",
                   help="Print the plan and the exact worker commands, then exit.")
    g.add_argument("--show", type=int, default=0, metavar="N",
                   help="Print the first N (task, run_id) groups of the plan.")

    g = p.add_argument_group("worker (set by the parent)")
    g.add_argument("--shard", default=None, metavar="I/N")
    g.add_argument("--plan-hash", default=None, metavar="HEX")
    g.add_argument("--worker-tag", default=None)
    g.add_argument("--allow-plan-drift", action="store_true")
    g.add_argument("--progress", default="auto", choices=["auto", "inner", "none"])
    return p


def _check_flags_classified(p: argparse.ArgumentParser) -> None:
    """Every flag must be forwarded, parent-only, or worker-only -- otherwise a
    future flag could silently fail to reach the workers."""
    known = {d for d, _ in FORWARDED} | PARENT_ONLY | WORKER_ONLY | {"help"}
    dests = {a.dest for a in p._actions}
    missing = dests - known
    assert not missing, f"unclassified eval_ckpts flags: {sorted(missing)}"


def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    _check_flags_classified(parser)
    args = parser.parse_args(argv)
    args.targets = [os.path.abspath(t) for t in args.targets]

    jobs, unresolved = build_jobs(args)
    if not jobs:
        log("[eval_ckpts] no checkpoints matched the targets/filters")
        for u in unresolved[:10]:
            log(f"[eval_ckpts]   {u}")
        return 1

    is_worker = args.shard is not None
    if not is_worker:
        print_plan(jobs, unresolved, args.show)

    gpus = parse_gpus(args.gpus)
    n_slots = len(gpus) * max(1, args.workers_per_gpu)
    if not is_worker and not args.serial and n_slots > 1 and len(jobs) > 1:
        return launch_workers(args, jobs, gpus)

    # ---- in-process: the whole plan, or one shard of it ----
    if is_worker:
        i, n = (int(x) for x in args.shard.split("/"))
        mine = plan_hash(jobs)
        if args.plan_hash and args.plan_hash != mine:
            msg = (f"plan drift: parent={args.plan_hash} worker={mine} "
                   f"({len(jobs)} ckpts enumerated here)")
            if not args.allow_plan_drift:
                log(f"[worker] {msg}")
                return 3
            log(f"[worker] WARNING {msg}")
        # In queue mode the shard is only an identity -- every worker may claim any job,
        # which is the whole point: a host that leaves strands nothing.
        if not args.claim_dir:
            jobs = jobs[i::n]
        worker = args.worker_tag or f"shard{i}"
    else:
        worker = "local"
        if args.dry_run:
            return 0
        if gpus and args.device == "auto":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])

    args.device = resolve_device(args.device)
    log(f"[{worker}] {len(jobs)} ckpts  device={args.device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}  "
        f"sets={args.num_sets}x{args.episodes_per_set}  out={args.out_dir}"
        f"{'  queue=' + args.claim_dir if args.claim_dir else ''}")

    summary = run_shard(args, jobs, worker)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
