#!/usr/bin/env python3
"""
Evaluation Wrapper with Cached Deterministic Handles

Two-mode script:
  1. prepare-handles: Build validation dataset, capture deterministic eval handles
     for each requested horizon, and save them to a cache file.
  2. run-eval: Load cached handles, replay samples via DataLoader, run an external
     DynamicsPredictor module on each sample, and aggregate metrics + save images.

Usage:
    # Phase 1: Prepare handles (already shipped under data/eval_handles/, regeneration usually not needed)
    python eval/eval_wrapper.py prepare-handles \
        --config configs/rla_wm/panda.yaml \
        --cache-path data/eval_handles/maniskill/handles.panda.json \
        --horizons 5 15 30 \
        --handles-per-horizon 50 \
        --seed 2026

    # Phase 2: Run evaluation
    python eval/eval_wrapper.py run-eval \
        --config configs/rla_wm/panda.yaml \
        --cache-path data/eval_handles/maniskill/handles.panda.json \
        --module-path eval/predictors/rla_wm_predictor.py \
        --output-dir runs/eval_output/panda \
        --num-workers 4 \
        --seed 2026
"""

import argparse
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
import torch.multiprocessing as mp
from rich import print
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure project root is in path
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Heavy imports are deferred to avoid slow startup for unit tests.
# They are loaded on first use inside prepare_handles() / run_eval().
TrajectoryDataset = None  # type: Any
EvalSampleHandle = None  # type: Any
EVAL_HANDLE_VERSION_LOADED = None  # type: Any
_load_config = None  # type: Any
_make_worker_seed_init_fn = None  # type: Any


def _ensure_heavy_imports() -> None:
    """Lazy-load heavy project modules on first call."""
    global TrajectoryDataset, EvalSampleHandle, EVAL_HANDLE_VERSION_LOADED
    global _load_config, _make_worker_seed_init_fn
    if TrajectoryDataset is not None:
        return
    from src.datasets.trajectory_dataset import (
        EVAL_HANDLE_VERSION as _EHV,
        EvalSampleHandle as _ESH,
        TrajectoryDataset as _TD,
    )
    from utils.misc import load_config as _lc, make_worker_seed_init_fn as _mw

    TrajectoryDataset = _TD
    EvalSampleHandle = _ESH
    EVAL_HANDLE_VERSION_LOADED = _EHV
    _load_config = _lc
    _make_worker_seed_init_fn = _mw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_FORMAT_VERSION = 1
DEFAULT_HORIZONS = [5, 15, 30]
DEFAULT_HANDLES_PER_HORIZON = 1000
PREDICTOR_CLASS_NAME = "DynamicsPredictor"
RUN_STATE_VERSION = 1
RUN_META_FILENAME = "run_meta.json"
PROGRESS_DIRNAME = "progress"

# ---------------------------------------------------------------------------
# Handle Cache I/O
# ---------------------------------------------------------------------------


def _config_fingerprint(cfg: dict) -> str:
    serialised = json.dumps(
        {k: str(v) for k, v in sorted(cfg.items())}, sort_keys=True
    ).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]


def save_handle_cache(
    handles_by_horizon: dict[int, list],
    filepath: str,
    *,
    config_path: str,
    horizons: list[int],
    handles_per_horizon: int,
    seed: int,
    val_config: dict,
    eval_handle_version: int = 1,
) -> None:
    """Save horizon-grouped handles to a single JSON cache file."""
    payload = {
        "meta": {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "eval_handle_version": eval_handle_version,
            "config_path": config_path,
            "config_fingerprint": _config_fingerprint(val_config),
            "horizons": horizons,
            "handles_per_horizon": handles_per_horizon,
            "seed": seed,
            "created_at": time.time(),
            "total_handles": sum(len(v) for v in handles_by_horizon.values()),
        },
        "handles_by_horizon": {
            str(h): [dict(handle) for handle in handles]
            for h, handles in handles_by_horizon.items()
        },
    }
    os.makedirs(osp.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    total = payload["meta"]["total_handles"]
    print(f"[green]Saved {total} handles ({len(horizons)} horizons) to {filepath}[/green]")


def load_handle_cache(filepath: str) -> tuple[dict, dict[int, list[dict]]]:
    """Load cached handles. Returns (meta, handles_by_horizon)."""
    with open(filepath, "r") as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    fmt_version = meta.get("cache_format_version", 0)
    if fmt_version > CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Cache format version {fmt_version} is newer than supported "
            f"version {CACHE_FORMAT_VERSION}. Please upgrade."
        )

    raw = payload.get("handles_by_horizon", {})
    handles_by_horizon: dict[int, list[dict]] = {}
    for h_str, handle_list in raw.items():
        h = int(h_str)
        # Validate required keys
        required = {"dataset_idx", "traj_id", "frame_id", "sampled_horizon", "camera_keys", "rgb_variant"}
        for i, handle in enumerate(handle_list):
            missing = required - set(handle.keys())
            if missing:
                raise ValueError(
                    f"Handle at horizon={h}, index={i} is missing keys: {missing}"
                )
        handles_by_horizon[h] = handle_list

    return meta, handles_by_horizon


# ---------------------------------------------------------------------------
# Prepare Handles
# ---------------------------------------------------------------------------


def _build_val_dataset(val_cfg: dict, horizon: int):
    """Build a TrajectoryDataset from val config with a fixed horizon override."""
    _ensure_heavy_imports()
    args = deepcopy(val_cfg)
    args["horizon"] = horizon
    # Remove trajectory_info_cache_file to avoid stale cache issues with changed horizon
    args.pop("trajectory_info_cache_file", None)
    dataset_cls = cast(Any, TrajectoryDataset)
    return dataset_cls(None, **args)


def _allocate_balanced_trajectory_counts(
    capacities: list[int],
    n_samples: int,
    rng: np.random.RandomState,
) -> dict[int, int]:
    """Allocate samples across trajectories as evenly as possible."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")

    active = [traj_pos for traj_pos, cap in enumerate(capacities) if int(cap) > 0]
    counts = {traj_pos: 0 for traj_pos in active}
    remaining = int(n_samples)

    while remaining > 0 and active:
        shuffled_active = [int(x) for x in rng.permutation(active)]
        round_size = min(remaining, len(shuffled_active))
        for traj_pos in shuffled_active[:round_size]:
            counts[traj_pos] += 1
        remaining -= round_size
        active = [
            traj_pos
            for traj_pos in active
            if counts[traj_pos] < int(capacities[traj_pos])
        ]

    return {traj_pos: count for traj_pos, count in counts.items() if count > 0}


def _sample_balanced_dataset_indices(
    dataset: Any,
    n_samples: int,
    rng: np.random.RandomState,
) -> tuple[list[int], dict[int, int]]:
    """Sample global dataset indices with broad, even trajectory coverage."""
    capacities = [int(cap) for cap in getattr(dataset, "traj_valid_start_counts")]
    prefixes = [int(prefix) for prefix in getattr(dataset, "traj_valid_start_prefix")]

    if len(capacities) != len(prefixes):
        raise ValueError(
            "TrajectoryDataset has inconsistent traj_valid_start_counts/"
            "traj_valid_start_prefix lengths."
        )

    sampled_counts = _allocate_balanced_trajectory_counts(capacities, n_samples, rng)
    sampled_indices: list[int] = []

    for traj_pos in sorted(sampled_counts.keys()):
        count = sampled_counts[traj_pos]
        capacity = capacities[traj_pos]
        global_start = 0 if traj_pos == 0 else prefixes[traj_pos - 1]
        local_offsets = rng.choice(capacity, size=count, replace=False)
        sampled_indices.extend(global_start + int(offset) for offset in local_offsets)

    if sampled_indices:
        rng.shuffle(sampled_indices)

    return sampled_indices, sampled_counts


def prepare_handles(args: argparse.Namespace) -> None:
    """Mode 1: Build val dataset per horizon, capture N handles each, save cache."""
    _ensure_heavy_imports()
    load_cfg_fn = cast(Any, _load_config)
    cfg = load_cfg_fn(args.config)

    if "val_dataset" not in cfg or not cfg["val_dataset"]:
        raise ValueError("Config must have a 'val_dataset' section.")

    val_cfg = dict(cfg["val_dataset"]["args"])
    horizons = args.horizons
    n = args.handles_per_horizon
    seed = args.seed

    print(f"[bold]Preparing handles:[/bold] horizons={horizons}, N={n}, seed={seed}")

    handles_by_horizon: dict[int, list[EvalSampleHandle]] = {}

    for h in horizons:
        print(f"\n[cyan]Building val dataset with horizon={h}...[/cyan]")
        dataset = _build_val_dataset(val_cfg, horizon=h)
        dataset_len = dataset.total_valid_starts

        if dataset_len == 0:
            print(f"[yellow]Warning: dataset is empty for horizon={h}, skipping.[/yellow]")
            handles_by_horizon[h] = []
            continue

        effective_n = min(n, dataset_len)
        if effective_n < n:
            print(
                f"[yellow]Warning: only {dataset_len} samples available for "
                f"horizon={h}, capturing {effective_n} instead of {n}.[/yellow]"
            )

        capacities = [int(cap) for cap in dataset.traj_valid_start_counts]
        eligible_trajectories = sum(cap > 0 for cap in capacities)

        # Deterministic balanced sampling with seed-offset per horizon
        rng = np.random.RandomState(seed + h)
        sampled_indices, planned_counts = _sample_balanced_dataset_indices(
            dataset,
            effective_n,
            rng,
        )
        handles: list[EvalSampleHandle] = []
        skipped = 0
        sampled_traj_counts: dict[tuple[int, str], int] = defaultdict(int)
        for i in sampled_indices:
            handle = dataset.capture_eval_handle(int(i))
            # Force the exact requested horizon (if feasible)
            if handle["feasible_horizon"] >= h:
                handle["sampled_horizon"] = h
            else:
                skipped += 1
                continue
            handles.append(handle)
            traj_key = (int(handle["dataset_idx"]), str(handle["traj_id"]))
            sampled_traj_counts[traj_key] += 1

        if skipped > 0:
            print(
                f"[yellow]  Unexpectedly skipped {skipped}/{effective_n} handles "
                f"where feasible_horizon < {h}.[/yellow]"
            )

        handles_by_horizon[h] = handles
        if sampled_traj_counts:
            min_per_traj = min(sampled_traj_counts.values())
            max_per_traj = max(sampled_traj_counts.values())
        else:
            min_per_traj = 0
            max_per_traj = 0

        print(
            f"  Captured {len(handles)} handles for horizon={h} across "
            f"{len(sampled_traj_counts)}/{eligible_trajectories} trajectories "
            f"(min={min_per_traj}, max={max_per_traj})"
        )

        if effective_n < eligible_trajectories:
            print(
                f"[yellow]  Requested {effective_n} handles but {eligible_trajectories} "
                f"trajectories are eligible for horizon={h}; prioritizing "
                "trajectory breadth over repeat samples.[/yellow]"
            )
        elif sampled_traj_counts and max_per_traj - min_per_traj > 1:
            saturated_trajectories = sum(
                1
                for traj_pos, planned_count in planned_counts.items()
                if planned_count >= capacities[traj_pos]
            )
            print(
                f"[yellow]  Some trajectories saturated early for horizon={h}; "
                f"redistributed remaining samples across {saturated_trajectories} "
                "capacity-limited trajectories.[/yellow]"
            )

    save_handle_cache(
        handles_by_horizon,
        args.cache_path,
        config_path=args.config,
        horizons=horizons,
        handles_per_horizon=n,
        seed=seed,
        val_config=val_cfg,
        eval_handle_version=EVAL_HANDLE_VERSION_LOADED or 1,
    )


# ---------------------------------------------------------------------------
# External Module Loading
# ---------------------------------------------------------------------------


def load_predictor(
    module_path: str,
    init_kwargs: dict[str, Any] | None = None,
):
    """Dynamically import a module from filesystem path and return a DynamicsPredictor instance."""
    module_path = osp.abspath(module_path)
    if not osp.isfile(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")

    module_name = osp.splitext(osp.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, PREDICTOR_CLASS_NAME, None)
    if cls is None:
        raise AttributeError(
            f"Module {module_path} does not define class '{PREDICTOR_CLASS_NAME}'"
        )

    if not init_kwargs:
        return cls()

    # Keep backwards compatibility for predictors that only support zero-arg init.
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return cls(**init_kwargs)
        accepted = {k: v for k, v in init_kwargs.items() if k in params}
        return cls(**accepted)
    except TypeError:
        return cls()


def _make_handle_hash(handle: dict[str, Any]) -> str:
    text = json.dumps(handle, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _progress_dir(output_dir: str) -> str:
    return osp.join(output_dir, PROGRESS_DIRNAME)


def _worker_status_path(output_dir: str, worker_rank: int) -> str:
    return osp.join(_progress_dir(output_dir), f"worker_{worker_rank:02d}.jsonl")


def _append_jsonl_row(path: str, row: dict[str, Any]) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    line = json.dumps(row, default=str)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _iter_status_files(output_dir: str) -> list[str]:
    pdir = _progress_dir(output_dir)
    if not osp.isdir(pdir):
        return []
    out = []
    for name in sorted(os.listdir(pdir)):
        if name.startswith("worker_") and name.endswith(".jsonl"):
            out.append(osp.join(pdir, name))
    return out


def _load_latest_status_by_handle(output_dir: str) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for status_file in _iter_status_files(output_dir):
        with open(status_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                handle_idx = row.get("handle_idx")
                if not isinstance(handle_idx, int):
                    continue
                prev = latest.get(handle_idx)
                prev_ts = float(prev.get("timestamp", -1.0)) if prev else -1.0
                ts = float(row.get("timestamp", -1.0))
                if ts >= prev_ts:
                    latest[handle_idx] = row
    return latest


def _count_finished_for_indices(
    latest_status: dict[int, dict[str, Any]],
    indices: set[int],
) -> tuple[int, int, int]:
    """Return (done, success, failed) for a subset of handle indices."""
    done = 0
    success = 0
    failed = 0
    for idx in indices:
        row = latest_status.get(idx)
        if row is None:
            continue
        status = row.get("status")
        if status == "success":
            done += 1
            success += 1
        elif status == "failed":
            done += 1
            failed += 1
    return done, success, failed


def _write_run_meta(
    output_dir: str,
    *,
    args: argparse.Namespace,
    cache_meta: dict[str, Any],
    total_handles: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_path = osp.join(output_dir, RUN_META_FILENAME)
    current = {
        "run_state_version": RUN_STATE_VERSION,
        "config_path": osp.abspath(args.config),
        "cache_path": osp.abspath(args.cache_path),
        "module_path": osp.abspath(args.module_path),
        "cache_seed": cache_meta.get("seed", None),
        "seed": args.seed,
        "total_handles": total_handles,
        "created_at": time.time(),
    }
    if not osp.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump(current, f, indent=2)
        return

    with open(meta_path, "r") as f:
        old = json.load(f)

    mismatch_fields = []
    for key in ["config_path", "cache_path", "module_path"]:
        if old.get(key) != current.get(key):
            mismatch_fields.append(key)
    if mismatch_fields and not args.force_resume:
        raise ValueError(
            "Existing run metadata does not match current args for fields "
            f"{mismatch_fields}. Use --force-resume to ignore."
        )


def _partition_indices(indices: list[int], n_parts: int) -> list[list[int]]:
    parts: list[list[int]] = [[] for _ in range(max(n_parts, 1))]
    for i, idx in enumerate(indices):
        parts[i % len(parts)].append(idx)
    return parts


class WorkerHandleDataset:
    """Worker-local dataset view over a subset of global handle indices."""

    def __init__(
        self,
        dataset: Any,
        flat_handles: list[dict[str, Any]],
        flat_horizons: list[int],
        handle_indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.flat_handles = flat_handles
        self.flat_horizons = flat_horizons
        self.handle_indices = handle_indices

    def __len__(self) -> int:
        return len(self.handle_indices)

    def __getitem__(self, local_idx: int) -> dict[str, Any]:
        handle_idx = self.handle_indices[local_idx]
        handle = cast(EvalSampleHandle, self.flat_handles[handle_idx])
        raw = self.dataset.getitem_from_handle(handle, apply_augmentation=False)
        sample = {k: v for k, v in raw.items()}
        sample["_eval_handle_idx"] = int(handle_idx)
        sample["_eval_horizon"] = int(self.flat_horizons[handle_idx])
        return sample


def _worker_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    # batch_size is always 1 for predictor API compatibility.
    return batch[0]


def _seed_dataloader_worker(worker_id: int, base_seed: int) -> None:
    """Top-level worker seed hook; must be picklable under spawn."""
    seed = (int(base_seed) + int(worker_id)) % (2 ** 32)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Image Saving
# ---------------------------------------------------------------------------


def _save_images(images, sample_dir: str, sample_idx: int) -> list[dict]:
    """Save images returned by apply(). Returns manifest entries.

    Supports:
      - dict of str -> PIL.Image  (saved as individual PNGs)
      - list of PIL.Image          (saved as animated GIF)
    """
    from PIL import Image

    manifest = []
    os.makedirs(sample_dir, exist_ok=True)

    if isinstance(images, dict):
        for key, img in images.items():
            if not isinstance(img, Image.Image):
                continue
            fname = f"{sample_idx:06d}_{key}.png"
            fpath = osp.join(sample_dir, fname)
            img.save(fpath)
            manifest.append({"type": "png", "key": key, "path": fname})

    elif isinstance(images, (list, tuple)):
        pil_frames = [img for img in images if isinstance(img, Image.Image)]
        if pil_frames:
            fname = f"{sample_idx:06d}.gif"
            fpath = osp.join(sample_dir, fname)
            pil_frames[0].save(
                fpath,
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0,
            )
            manifest.append({"type": "gif", "num_frames": len(pil_frames), "path": fname})

    return manifest


def _save_videos(videos, sample_dir: str, sample_idx: int) -> list[dict]:
    """Save predictor rollout videos as MP4 and return manifest entries.

    Supported format:
      - dict[str, list[PIL.Image]] -> one MP4 per key
    """
    from PIL import Image

    manifest = []
    os.makedirs(sample_dir, exist_ok=True)

    if not isinstance(videos, dict):
        return manifest

    try:
        import imageio.v2 as imageio
    except Exception as e:
        print(f"[yellow]  Warning: failed to import imageio for MP4 export: {e}[/yellow]")
        return manifest

    for key, frames in videos.items():
        if not isinstance(frames, (list, tuple)):
            continue
        pil_frames = [img for img in frames if isinstance(img, Image.Image)]
        if not pil_frames:
            continue

        fname = f"{sample_idx:06d}_{key}.mp4"
        fpath = osp.join(sample_dir, fname)

        try:
            with imageio.get_writer(fpath, fps=10, codec="libx264") as writer:
                for frame in pil_frames:
                    writer.append_data(np.asarray(frame.convert("RGB")))
        except Exception as e:
            print(
                f"[yellow]  Warning: failed to save MP4 for key '{key}' "
                f"(sample {sample_idx}): {e}[/yellow]"
            )
            continue

        manifest.append(
            {
                "type": "mp4",
                "key": key,
                "num_frames": len(pil_frames),
                "path": fname,
            }
        )

    return manifest


# ---------------------------------------------------------------------------
# Metrics Aggregation
# ---------------------------------------------------------------------------


def _is_scalar(v) -> bool:
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, (np.generic, np.ndarray)):
        return np.ndim(v) == 0
    if isinstance(v, torch.Tensor):
        return v.dim() == 0
    return False


def _to_float(v) -> float:
    if isinstance(v, torch.Tensor):
        return v.item()
    return float(v)


def aggregate_metrics(
    raw_metrics: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute mean/std/min/max/count for each scalar metric key."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for m in raw_metrics:
        for k, v in m.items():
            buckets[k].append(v)

    summary = {}
    for k, vals in sorted(buckets.items()):
        arr = np.array(vals)
        summary[k] = {
            "count": len(vals),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return summary


def print_summary_table(summary: dict[str, dict], title: str = "Evaluation Summary"):
    console = Console()
    table = Table(title=title, show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="dim")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Min", style="magenta")
    table.add_column("Max", style="red")
    for k, stats in summary.items():
        table.add_row(
            k,
            str(stats["count"]),
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
        )
    console.print(table)


def build_markdown_summary(
    summary_payload: dict[str, Any],
    global_summary: dict[str, dict[str, float]],
    per_horizon_summary: dict[int, dict[str, dict[str, float]]],
) -> str:
    """Build a readable markdown report with global/per-horizon metric tables."""

    def _render_table(stats: dict[str, dict[str, float]]) -> str:
        if not stats:
            return "_No scalar metrics available._\n"

        lines = [
            "| Metric | Count | Mean | Std | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for metric_name in sorted(stats.keys()):
            metric = stats[metric_name]
            lines.append(
                "| "
                f"{metric_name} | "
                f"{int(metric['count'])} | "
                f"{metric['mean']:.4f} | "
                f"{metric['std']:.4f} | "
                f"{metric['min']:.4f} | "
                f"{metric['max']:.4f} |"
            )
        return "\n".join(lines) + "\n"

    meta = summary_payload.get("meta", {})
    lines: list[str] = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append("## Run Info")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Cache Path | {meta.get('cache_path', '')} |")
    lines.append(f"| Module Path | {meta.get('module_path', '')} |")
    lines.append(f"| Seed | {meta.get('seed', '')} |")
    lines.append(f"| Total Samples | {meta.get('total_samples', '')} |")
    lines.append(f"| Completed Samples | {meta.get('completed_samples', '')} |")
    lines.append(f"| Pending Samples | {meta.get('pending_samples', '')} |")
    lines.append(f"| Partial Progress | {meta.get('partial_progress', '')} |")
    lines.append(f"| Successful | {meta.get('successful', '')} |")
    lines.append(f"| Failures | {meta.get('failures', '')} |")
    elapsed = float(meta.get("elapsed_seconds", 0.0))
    lines.append(f"| Elapsed Seconds | {elapsed:.2f} |")
    lines.append("")

    lines.append("## Global Metrics")
    lines.append("")
    lines.append(_render_table(global_summary).rstrip())
    lines.append("")

    lines.append("## Per-Horizon Metrics")
    lines.append("")
    if not per_horizon_summary:
        lines.append("_No per-horizon metrics available._")
    else:
        for h in sorted(per_horizon_summary.keys()):
            lines.append(f"### Horizon {h}")
            lines.append("")
            lines.append(_render_table(per_horizon_summary[h]).rstrip())
            lines.append("")

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Run Eval
# ---------------------------------------------------------------------------


def _run_eval_worker(
    worker_rank: int,
    worker_ctx: dict[str, Any],
) -> None:
    """Worker entry for shard evaluation on one device."""
    _ensure_heavy_imports()

    module_path = worker_ctx["module_path"]
    config_path = worker_ctx["config_path"]
    output_dir = worker_ctx["output_dir"]
    worker_indices: list[int] = worker_ctx.get("worker_indices", [])
    shard_indices: list[list[int]] = worker_ctx.get("shard_indices", [])
    if shard_indices:
        worker_indices = shard_indices[worker_rank]
    flat_handles: list[dict[str, Any]] = worker_ctx["flat_handles"]
    flat_horizons: list[int] = worker_ctx["flat_horizons"]
    seed = int(worker_ctx["seed"])
    fail_fast = bool(worker_ctx["fail_fast"])
    worker_num_workers = int(worker_ctx.get("worker_num_workers", 0))
    gpu_id = worker_ctx.get("gpu_id", None)
    gpu_ids: list[int] = worker_ctx.get("gpu_ids", [])
    if gpu_id is None and gpu_ids:
        gpu_id = gpu_ids[worker_rank]
    world_size = int(worker_ctx["world_size"])

    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device_str = f"cuda:{int(gpu_id)}"
    else:
        device_str = "cpu"

    np.random.seed(seed + worker_rank)
    torch.manual_seed(seed + worker_rank)

    load_cfg_fn = cast(Any, _load_config)
    cfg = load_cfg_fn(config_path)
    if "val_dataset" not in cfg or not cfg["val_dataset"]:
        raise ValueError("Config must have a 'val_dataset' section.")
    val_cfg = deepcopy(dict(cfg["val_dataset"]["args"]))
    max_horizon = max(flat_horizons) if flat_horizons else 1
    val_cfg["horizon"] = max_horizon
    val_cfg.pop("trajectory_info_cache_file", None)

    dataset_cls = cast(Any, TrajectoryDataset)
    replay_dataset = dataset_cls(None, **val_cfg)

    worker_dataset = WorkerHandleDataset(
        dataset=replay_dataset,
        flat_handles=flat_handles,
        flat_horizons=flat_horizons,
        handle_indices=worker_indices,
    )
    worker_seed_fn = partial(_seed_dataloader_worker, base_seed=seed + worker_rank)
    loader = DataLoader(
        cast(Any, worker_dataset),
        batch_size=1,
        shuffle=False,
        num_workers=max(worker_num_workers, 0),
        pin_memory=False,
        collate_fn=_worker_collate,
        worker_init_fn=worker_seed_fn,
    )

    init_kwargs = {
        "rank": worker_rank,
        "world_size": world_size,
        "device": device_str,
        "gpu_id": gpu_id,
    }
    predictor = load_predictor(module_path, init_kwargs=init_kwargs)

    status_path = _worker_status_path(output_dir, worker_rank)

    for local_i, sample in enumerate(loader):
        handle_idx = int(sample.pop("_eval_handle_idx"))
        horizon = int(sample.pop("_eval_horizon"))

        status_row: dict[str, Any] = {
            "timestamp": time.time(),
            "worker_rank": worker_rank,
            "local_index": local_i,
            "handle_idx": int(handle_idx),
            "horizon": int(horizon),
            "handle_hash": _make_handle_hash(flat_handles[handle_idx]),
            "status": "started",
        }
        _append_jsonl_row(status_path, status_row)

        scalar_m: dict[str, float] = {}
        image_entries: list[dict[str, Any]] = []
        video_entries: list[dict[str, Any]] = []

        try:
            result = predictor.apply(sample)
            if not isinstance(result, dict):
                raise ValueError("apply() did not return dict")

            raw_m = result.get("metrics", {})
            if isinstance(raw_m, dict):
                for k, v in raw_m.items():
                    if _is_scalar(v):
                        scalar_m[k] = _to_float(v)

            images = result.get("images", None)
            if images is not None:
                sample_dir = osp.join(output_dir, f"horizon_{horizon}", "images")
                image_entries = _save_images(images, sample_dir, handle_idx)
                for entry in image_entries:
                    entry["handle_idx"] = int(handle_idx)
                    entry["horizon"] = int(horizon)

                trace_info = {
                    "handle_idx": int(handle_idx),
                    "horizon": int(horizon),
                    "handle": {k: v for k, v in flat_handles[handle_idx].items()},
                    "metrics": scalar_m,
                    "worker_rank": worker_rank,
                    "device": device_str,
                }
                trace_path = osp.join(sample_dir, f"{handle_idx:06d}_trace.json")
                with open(trace_path, "w") as f:
                    json.dump(trace_info, f, indent=2, default=str)

            videos = result.get("videos", None)
            if videos is not None:
                sample_dir = osp.join(output_dir, f"horizon_{horizon}", "videos")
                video_entries = _save_videos(videos, sample_dir, handle_idx)
                for entry in video_entries:
                    entry["handle_idx"] = int(handle_idx)
                    entry["horizon"] = int(horizon)

            _append_jsonl_row(
                status_path,
                {
                    "timestamp": time.time(),
                    "worker_rank": worker_rank,
                    "handle_idx": int(handle_idx),
                    "horizon": int(horizon),
                    "handle_hash": _make_handle_hash(flat_handles[handle_idx]),
                    "status": "success",
                    "metrics": scalar_m,
                    "image_entries": image_entries,
                    "video_entries": video_entries,
                    "device": device_str,
                },
            )
        except Exception as e:
            _append_jsonl_row(
                status_path,
                {
                    "timestamp": time.time(),
                    "worker_rank": worker_rank,
                    "handle_idx": int(handle_idx),
                    "horizon": int(horizon),
                    "handle_hash": _make_handle_hash(flat_handles[handle_idx]),
                    "status": "failed",
                    "error": str(e),
                    "device": device_str,
                },
            )
            if fail_fast:
                raise


def _run_eval_inline(
    *,
    module_path: str,
    config_path: str,
    output_dir: str,
    flat_handles: list[dict[str, Any]],
    flat_horizons: list[int],
    seed: int,
    fail_fast: bool,
    worker_num_workers: int,
    pending_indices: list[int],
    gpu_id: int | None,
) -> None:
    ctx = {
        "module_path": module_path,
        "config_path": config_path,
        "output_dir": output_dir,
        "flat_handles": flat_handles,
        "flat_horizons": flat_horizons,
        "seed": seed,
        "fail_fast": fail_fast,
        "worker_num_workers": worker_num_workers,
        "world_size": 1,
        "gpu_id": gpu_id,
        "worker_indices": pending_indices,
    }
    _run_eval_worker(0, ctx)


def run_eval(args: argparse.Namespace) -> None:
    """Mode 2: Load cached handles, run external predictor, save results.

    Resume behavior:
      - Successful handle_idx rows in progress/worker_*.jsonl are skipped.
      - Failed rows are retried by default unless --no-retry-failures is used.
    """
    # --- Load cache ---
    meta, handles_by_horizon = load_handle_cache(args.cache_path)
    total_handles = sum(len(v) for v in handles_by_horizon.values())
    print(
        f"[bold]Loaded {total_handles} handles from {args.cache_path}[/bold] "
        f"(horizons: {sorted(handles_by_horizon.keys())})"
    )

    if total_handles == 0:
        print("[red]No handles to evaluate. Exiting.[/red]")
        return

    # --- Flatten handles with horizon tags ---
    flat_handles: list[dict] = []
    flat_horizons: list[int] = []
    for h in sorted(handles_by_horizon.keys()):
        for handle in handles_by_horizon[h]:
            flat_handles.append(handle)
            flat_horizons.append(h)

    # --- Output dir ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if not args.just_stats:
        _write_run_meta(
            output_dir,
            args=args,
            cache_meta=meta,
            total_handles=len(flat_handles),
        )

    # Per-horizon output dirs
    horizon_dirs: dict[int, str] = {}
    for h in sorted(handles_by_horizon.keys()):
        hdir = osp.join(output_dir, f"horizon_{h}")
        os.makedirs(hdir, exist_ok=True)
        horizon_dirs[h] = hdir

    # --- Resume status ---
    latest_status = _load_latest_status_by_handle(output_dir)
    success_set = {
        idx
        for idx, row in latest_status.items()
        if row.get("status") == "success"
    }
    failed_set = {
        idx
        for idx, row in latest_status.items()
        if row.get("status") == "failed"
    }

    pending_indices = [
        idx
        for idx in range(len(flat_handles))
        if idx not in success_set and (args.retry_failures or idx not in failed_set)
    ]

    print(
        "[bold]Resume state:[/bold] "
        f"{len(success_set)} already successful, {len(failed_set)} failed, "
        f"{len(pending_indices)} pending."
    )

    if args.just_stats:
        print(
            "[cyan]--just-stats enabled:[/cyan] skipping inference/model loading and "
            "aggregating from stored progress only."
        )

    # --- Launch workers ---
    t_start = time.time()
    if pending_indices and not args.just_stats:
        _ensure_heavy_imports()
        if torch.cuda.is_available():
            if args.gpus == "all":
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
            if not gpu_ids:
                raise ValueError("No GPUs selected. Use --gpus all or comma-separated ids.")
        else:
            gpu_ids = []

        if len(gpu_ids) > 1:
            world_size = len(gpu_ids)
            shards = _partition_indices(pending_indices, world_size)
            print(
                f"[cyan]Launching {world_size} worker processes across GPUs: {gpu_ids}[/cyan]"
            )
            mp.set_start_method("spawn", force=True)
            pending_set = set(pending_indices)
            spawn_ctx = mp.spawn(
                _run_eval_worker,
                args=(
                    {
                        "module_path": args.module_path,
                        "config_path": args.config,
                        "output_dir": output_dir,
                        "flat_handles": flat_handles,
                        "flat_horizons": flat_horizons,
                        "seed": args.seed,
                        "fail_fast": args.fail_fast,
                        "worker_num_workers": args.num_workers,
                        "world_size": world_size,
                        "gpu_ids": gpu_ids,
                        "shard_indices": shards,
                    },
                ),
                nprocs=world_size,
                join=False,
            )
            with tqdm(
                total=len(pending_indices),
                desc="Global Eval",
                dynamic_ncols=True,
            ) as pbar:
                last_done = 0
                while True:
                    latest = _load_latest_status_by_handle(output_dir)
                    done, ok_count, fail_count = _count_finished_for_indices(
                        latest, pending_set
                    )
                    if done > last_done:
                        pbar.update(done - last_done)
                        last_done = done
                    pbar.set_postfix(ok=ok_count, fail=fail_count)

                    finished = spawn_ctx.join(timeout=1.0)
                    if finished:
                        latest = _load_latest_status_by_handle(output_dir)
                        done, ok_count, fail_count = _count_finished_for_indices(
                            latest, pending_set
                        )
                        if done > last_done:
                            pbar.update(done - last_done)
                        pbar.set_postfix(ok=ok_count, fail=fail_count)
                        break
        elif gpu_ids:
            gpu_id = gpu_ids[0]
            print(
                f"[cyan]Launching inline single GPU worker on GPU {gpu_id} without multiprocessing.[/cyan]"
            )
            _run_eval_inline(
                module_path=args.module_path,
                config_path=args.config,
                output_dir=output_dir,
                flat_handles=flat_handles,
                flat_horizons=flat_horizons,
                seed=args.seed,
                fail_fast=args.fail_fast,
                worker_num_workers=args.num_workers,
                pending_indices=pending_indices,
                gpu_id=gpu_id,
            )
        else:
            print("[yellow]CUDA not available; running single CPU worker.[/yellow]")
            _run_eval_inline(
                module_path=args.module_path,
                config_path=args.config,
                output_dir=output_dir,
                flat_handles=flat_handles,
                flat_horizons=flat_horizons,
                seed=args.seed,
                fail_fast=args.fail_fast,
                worker_num_workers=args.num_workers,
                pending_indices=pending_indices,
                gpu_id=None,
            )

    elapsed_total = time.time() - t_start

    # --- Consolidate persisted status ---
    latest_status = _load_latest_status_by_handle(output_dir)
    all_metrics: list[dict[str, float]] = []
    metrics_by_horizon: dict[int, list[dict[str, float]]] = defaultdict(list)
    image_manifest: list[dict[str, Any]] = []
    success_count = 0
    failures = 0

    for handle_idx, row in sorted(latest_status.items()):
        status = row.get("status")
        horizon = int(row.get("horizon", flat_horizons[handle_idx]))
        if status == "success":
            success_count += 1
            metrics = row.get("metrics", {})
            if isinstance(metrics, dict) and metrics:
                scalar_m = {k: float(v) for k, v in metrics.items()}
                all_metrics.append(scalar_m)
                metrics_by_horizon[horizon].append(scalar_m)

            for key in ["image_entries", "video_entries"]:
                entries = row.get(key, [])
                if isinstance(entries, list):
                    image_manifest.extend(entries)
        elif status == "failed":
            failures += 1

    # --- Save summary ---
    global_summary = aggregate_metrics(all_metrics)
    per_horizon_summary = {
        h: aggregate_metrics(mlist) for h, mlist in sorted(metrics_by_horizon.items())
    }
    completed_count = success_count + failures
    pending_count = max(len(flat_handles) - completed_count, 0)
    partial_progress = pending_count > 0

    if args.just_stats and partial_progress:
        print(
            "[yellow]Partial progress detected:[/yellow] "
            f"{completed_count}/{len(flat_handles)} completed, "
            f"{pending_count} pending. Aggregating available results only."
        )

    summary_payload = {
        "meta": {
            "cache_path": args.cache_path,
            "module_path": args.module_path or "",
            "seed": args.seed,
            "total_samples": len(flat_handles),
            "completed_samples": completed_count,
            "pending_samples": pending_count,
            "partial_progress": partial_progress,
            "successful": success_count,
            "failures": failures,
            "elapsed_seconds": elapsed_total,
        },
        "global": global_summary,
        "per_horizon": {str(h): s for h, s in per_horizon_summary.items()},
    }

    summary_path = osp.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\n[green]Summary saved to {summary_path}[/green]")

    markdown_summary = build_markdown_summary(
        summary_payload=summary_payload,
        global_summary=global_summary,
        per_horizon_summary=per_horizon_summary,
    )
    summary_md_path = osp.join(output_dir, "eval_summary.md")
    with open(summary_md_path, "w") as f:
        f.write(markdown_summary)
    print(f"[green]Markdown summary saved to {summary_md_path}[/green]")

    # Image manifest
    if image_manifest:
        manifest_path = osp.join(output_dir, "image_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(image_manifest, f, indent=2)
        print(f"[green]Image manifest saved to {manifest_path}[/green]")

    # --- Console summary ---
    print()
    print_summary_table(global_summary, title="Global Evaluation Summary")

    for h in sorted(per_horizon_summary.keys()):
        h_summary = per_horizon_summary[h]
        if h_summary:
            print_summary_table(h_summary, title=f"Horizon {h} Summary")

    print(
        f"\n[bold]Done.[/bold] {success_count} successful, {failures} failures, "
        f"{pending_count} pending, {elapsed_total:.1f}s total."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation Wrapper with Cached Deterministic Handles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- prepare-handles ---
    p_prep = subparsers.add_parser(
        "prepare-handles",
        help="Capture deterministic eval handles from val dataset and save to cache.",
    )
    p_prep.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (must have val_dataset section).",
    )
    p_prep.add_argument(
        "--cache-path", type=str, required=True,
        help="Output path for the handle cache JSON file.",
    )
    p_prep.add_argument(
        "--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
        help=f"List of fixed horizons to capture. Default: {DEFAULT_HORIZONS}",
    )
    p_prep.add_argument(
        "--handles-per-horizon", type=int, default=DEFAULT_HANDLES_PER_HORIZON,
        help=f"Number of handles to capture per horizon. Default: {DEFAULT_HANDLES_PER_HORIZON}",
    )
    p_prep.add_argument(
        "--seed", type=int, default=2026,
        help="RNG seed for deterministic capture. Default: 2026",
    )
    p_prep.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing cache file if present.",
    )

    # --- run-eval ---
    p_eval = subparsers.add_parser(
        "run-eval",
        help="Run evaluation using cached handles and an external predictor module.",
    )
    p_eval.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (for building replay dataset).",
    )
    p_eval.add_argument(
        "--cache-path", type=str, required=True,
        help="Path to the handle cache JSON file.",
    )
    p_eval.add_argument(
        "--module-path", type=str, default="",
        help="Filesystem path to the Python module defining DynamicsPredictor.",
    )
    p_eval.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for evaluation outputs (images, summary JSON).",
    )
    p_eval.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers per eval process. Default: 4",
    )
    p_eval.add_argument(
        "--seed", type=int, default=2026,
        help="RNG seed for reproducibility. Default: 2026",
    )
    p_eval.add_argument(
        "--fail-fast", action="store_true",
        help="Abort on first sample failure instead of continuing.",
    )
    p_eval.add_argument(
        "--gpus", type=str, default="all",
        help="GPU ids to use, e.g. '0,1,2'. Default: all visible GPUs.",
    )
    p_eval.add_argument(
        "--retry-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On resume, retry previously failed samples (default: true).",
    )
    p_eval.add_argument(
        "--force-resume", action="store_true",
        help="Allow resume even if run metadata (config/cache/module path) differs.",
    )
    p_eval.add_argument(
        "--just-stats", action="store_true",
        help=(
            "Skip inference and model loading; only aggregate existing progress "
            "files into summary outputs."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "prepare-handles":
        if osp.exists(args.cache_path) and not args.overwrite:
            print(
                f"[red]Cache file already exists: {args.cache_path}[/red]\n"
                "Use --overwrite to replace it."
            )
            sys.exit(1)
        prepare_handles(args)

    elif args.mode == "run-eval":
        if not osp.exists(args.cache_path):
            print(f"[red]Cache file not found: {args.cache_path}[/red]")
            sys.exit(1)
        if not args.just_stats and not args.module_path:
            print(
                "[red]--module-path is required unless --just-stats is set.[/red]"
            )
            sys.exit(1)
        run_eval(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
