"""
rla_targets.py

v4world: reads the precomputed RLA latent-action targets that `scripts/extract_rla_latents.py`
writes, and joins them onto the RLDS trajectory stream.

The join key is `sha1(observation["state"].astype(float32).tobytes())` for the whole episode.
`state` is a float32 Tensor feature in the TFRecord, so the bytes the extractor hashed and the bytes
`libero_dataset_transform` sees are identical -- that is what makes the join exact. The obvious
alternative, `episode_metadata/file_path`, is *not* usable: it names the source HDF5 *task* file,
which every demo of a task shares.

Lookup runs once per *episode* (inside the trajectory-level `standardize_fn`), not once per frame,
so a `tf.numpy_function` here costs nothing measurable. Arrays are mmapped.

See docs/rla-vla-adapter.md §5-6 for the design and §10 for what verifies it.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from prismatic.vla.constants import RLA_DIM, RLA_QUERIES, RLA_SIDECAR, RLA_STEPS


def rla_enabled() -> bool:
    return RLA_STEPS > 0 and bool(RLA_SIDECAR)


def episode_key(states: np.ndarray) -> str:
    """Must stay byte-for-byte identical to `scripts/extract_rla_latents.episode_key`."""
    return hashlib.sha1(np.ascontiguousarray(states, dtype=np.float32).tobytes()).hexdigest()


class RlaSidecar:
    """One sidecar directory: `keys.json` per suite, `stats.json`, one `.npy` per episode."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Generate the sidecar first:\n"
                "  .venv/bin/python scripts/extract_rla_latents.py --help"
            )
        self.manifest = json.loads(manifest_path.read_text())

        # Fail loudly on a shape mismatch rather than silently regressing against the wrong target.
        for name, want, got in (
            ("RLA_STEPS", RLA_STEPS, self.manifest["chunk"]),
            ("RLA_QUERIES", RLA_QUERIES, self.manifest["num_tokens"]),
            ("RLA_DIM", RLA_DIM, self.manifest["token_dim"]),
        ):
            if want != got:
                raise ValueError(
                    f"{name}={want} does not match the sidecar at {self.root} ({got}). "
                    "Re-extract, or fix the ARM= env in the launcher."
                )
        if self.manifest.get("random_encoder"):
            print(
                f"[rla_targets] WARNING: {self.root} was extracted with --random-encoder. "
                "These targets are noise -- fine for plumbing tests, meaningless for training."
            )

        stats = json.loads((self.root / "stats.json").read_text())
        self.mean = np.asarray(stats["mean"], dtype=np.float32)  # (Q, D)
        std = np.asarray(stats["std"], dtype=np.float32)
        # Floor the std so a dead latent slot (std ~ 0) cannot blow the normalised target up.
        self.std = np.maximum(std, float(stats.get("std_floor", 1e-3)))

        # key -> (suite, entry). Merged across suites: a training mix can interleave several.
        self.index: Dict[str, tuple] = {}
        for keys_path in sorted(self.root.glob("*/keys.json")):
            suite = keys_path.parent.name
            for key, entry in json.loads(keys_path.read_text()).items():
                self.index[key] = (suite, entry)
        if not self.index:
            raise RuntimeError(f"no */keys.json under {self.root}; the sidecar is empty")

        self._cache: Dict[str, np.ndarray] = {}
        self.num_misses = 0

    @property
    def target_dim(self) -> int:
        return RLA_QUERIES * RLA_DIM

    def lookup(self, states: np.ndarray) -> np.ndarray:
        """(N, 8) float32 states -> (N, RLA_STEPS, Q*D) float32, normalised.

        A miss returns zeros and bumps `num_misses` instead of raising: raising inside a
        `tf.numpy_function` surfaces as an opaque graph error, and the vanilla arms must not be able
        to crash on a stale sidecar. `scripts/verify_rla_targets.py` Gate 2 is what turns a miss
        into a hard failure.
        """
        states = np.asarray(states, dtype=np.float32)
        num_frames = states.shape[0]
        key = episode_key(states)

        entry = self.index.get(key)
        if entry is None:
            self.num_misses += 1
            if self.num_misses <= 5:
                print(
                    f"[rla_targets] MISS: no sidecar entry for episode key {key} "
                    f"(traj_len={num_frames}); returning zeros. Sidecar: {self.root}"
                )
            return np.zeros((num_frames, RLA_STEPS, self.target_dim), dtype=np.float32)

        suite, meta = entry
        cached = self._cache.get(key)
        if cached is None:
            cached = np.load(self.root / suite / meta["file"], mmap_mode="r")
            if len(self._cache) < 64:  # bounded: these are mmaps, but the dict is not
                self._cache[key] = cached

        z = np.asarray(cached, dtype=np.float32)  # (N, A, Q, D)
        if z.shape[0] != num_frames:
            self.num_misses += 1
            print(
                f"[rla_targets] MISS: {suite}/{meta['file']} has {z.shape[0]} frames but the "
                f"episode has {num_frames}; returning zeros."
            )
            return np.zeros((num_frames, RLA_STEPS, self.target_dim), dtype=np.float32)

        z = (z - self.mean) / self.std
        return np.ascontiguousarray(z.reshape(num_frames, RLA_STEPS, self.target_dim))

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """Undo `lookup`'s normalisation, for feeding a predicted z back through `f_dec`."""
        flat = z.reshape(*z.shape[:-1], RLA_QUERIES, RLA_DIM)
        return flat * self.std + self.mean


_SIDECAR: Optional[RlaSidecar] = None


def get_sidecar() -> RlaSidecar:
    """Process-wide sidecar handle. Built lazily so importing this module is free."""
    global _SIDECAR
    if _SIDECAR is None:
        _SIDECAR = RlaSidecar(os.environ.get("RLA_SIDECAR", RLA_SIDECAR))
    return _SIDECAR


def lookup_numpy(states) -> np.ndarray:
    """The `tf.numpy_function` entry point. Kept module-level so tf.data can pickle it."""
    return get_sidecar().lookup(np.asarray(states))
