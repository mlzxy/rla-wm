#!/usr/bin/env python
"""Read an RLA latent-action sidecar and join it onto LIBERO RLDS -- with **no RLA dependencies**.

This is the consumer-side counterpart of `extract_rla_sidecar.py`, meant to be **copied into a repo
that has none of this repository's machinery**: no torch, no DINOv3, no `src.trainers`, no `datalib`, no
`utils`, no config yaml, no checkpoints. It needs `numpy` to read the latents, and `tensorflow` +
`tensorflow_datasets` only if you also want to join them against `data/libero` (the `describe`,
`peek` and `selftest` commands never import TF).

Copy this one file, copy the sidecar directory, and the receiving repo has everything it needs to
know what the data is, prove the join works, and load the targets.

================================================================================================
1. WHAT IS IN A SIDECAR
================================================================================================

    <sidecar>/
        manifest.json              what produced this, and the shape contract -- read this first
        stats.json                 per-(q, d) mean/std over the whole corpus, + std_floor
        <suite>/keys.json          sha1(states) -> {file, episode_index, traj_len}   THE JOIN INDEX
        <suite>/ep_000000.npy      (N, A, Q, D) float16 -- one file per episode, mmap-able
        <suite>/_keys_rank*.json   per-worker shards already merged into keys.json -- ignore them
        viz/*.jpg                  filmstrips from the extraction run -- inspection only

`<suite>` is one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. A sidecar may
hold any subset of them; `manifest["suites"]` says which, and `keys.json` is per suite because the
episode numbering restarts in each.

Every episode array has the same four axes:

    z[t, i, q, d]        t = 0 .. N-1     frame index within the episode (N = traj_len)
                         i = 0 .. A-1     step within the action chunk that starts at frame t
                         q = 0 .. Q-1     latent query slot        (A, Q, D = manifest chunk /
                         d = 0 .. D-1     channel within a query    num_tokens / token_dim)

dtype is **float16 on disk** -- cast to float32 before you do arithmetic on it. Typical size for
A=8, Q=16, D=64 is 16 KB per frame, ~2 MB per episode, 4.3 GiB for all four suites.

================================================================================================
2. WHAT THE NUMBERS MEAN (the part you cannot recover from the files)
================================================================================================

`z[t, i]` is a **latent action**: the output of the RLA autoencoder's inverse-dynamics encoder
`f_enc` applied to the visual change between two frames of the same episode,

    z[t, i] = f_enc( s_{min(t + i + 1, N-1)} - s_t )          ("anchored" -- manifest["pairs"])

where `s` is the DINOv3 patch-token embedding of a frame across both cameras. Two consequences,
and getting either wrong silently trains against the wrong target:

  * **Everything is anchored on frame `t`**, the frame the policy conditions on -- not on `t+i`.
    So `z[t, i]` is the cumulative visual consequence of executing actions `a_t .. a_{t+i}`, and it
    lines up one-to-one with the `i`-th action token of the chunk predicted at frame `t`:

        action token   a_t      a_{t+1}   a_{t+2}   ...   a_{t+A-1}
        RLA target     z[t,0]   z[t,1]    z[t,2]    ...   z[t,A-1]

  * **The tail is clamped, exactly the way the action chunk is.** Near the end of an episode
    `t + i + 1` runs past the last frame, and the extractor clamps it to `N-1` -- mirroring
    VLA-Adapter's own `min(max(idx, 0), traj_len - 1)` on the action chunk. So the last few frames
    repeat a latent, and `z[N-1, :]` is `f_enc(0)`, the encoding of no visual change at all. That
    is not corruption; it is the same padding convention the actions use. `selftest` and `join`
    both check it.

`manifest["pairs"] == "consecutive"` instead means `z[t, i] = f_enc(s_{t+i+1} - s_{t+i})`, stored
in the same layout. It is an ablation arm: those latents are *not* anchored on `t`, so `f_dec(x_t,
z)` cannot decode them. Check the field before assuming.

Latents are **not normalised on disk**. `stats.json` carries the corpus statistics; the training
path applies

    z_norm = (z - mean) / maximum(std, std_floor)            mean, std are (Q, D)

and then usually flattens the last two axes to a single `Q*D` target vector. `RlaSidecar.lookup`
below does exactly this, and `denormalize` undoes it -- you need the inverse if you ever feed a
predicted latent back through the RLA decoder.

================================================================================================
3. HOW THE JOIN WORKS
================================================================================================

The sidecar carries no images and no episode ids that survive a shuffle. It is joined onto the
RLDS stream by hashing the episode's own proprioceptive states:

    key = sha1( np.ascontiguousarray(states, dtype=np.float32).tobytes() ).hexdigest()

`states` is the full `(N, 8)` float32 `observation/state` array of one episode. `state` is stored
as a float32 Tensor feature in the TFRecord, so the bytes the extractor hashed and the bytes any
consumer sees inside the tf.data graph are *identical* -- the join is exact, not approximate.

Do not be tempted by the two things that look like episode identifiers but are not:

  * `episode_metadata/file_path` names the source HDF5 **task** file, which every demo of that task
    shares -- it identifies a task, not an episode.
  * `episode_index` in `keys.json` is the episode's position **within the split slice the extractor
    read**, and TFDS split slicing is not position-stable. `as_dataset` interleaves shards, and a
    slice touches a different set of shards than a full scan does, so the same ordinal can name a
    different episode. Measured against this dataset (extraction used 4 ranks, i.e. four slices):

        libero_spatial  16 shards   only 108/432 episode_index values agree with a full scan
        libero_goal     16 shards   only 107/428 agree
        libero_object   32 shards       454/454 agree
        libero_10       32 shards       379/379 agree

    and `train[30:60]` read on its own agrees with positions 30..59 of a full scan on **0 of 30**
    episodes. Same data, same order on disk, different numbering -- and note that two of the four
    suites agree perfectly, which is exactly what makes this a trap: a spot check on
    `libero_object` "proves" indices are stable right up until you try `libero_spatial`.

The state hash is immune to all of this: it is computed from the episode's own contents, so it
does not care which shard the episode came from, which slice read it, or what order it arrived in.

Lookup belongs at the **trajectory** level (once per episode), not per frame: one sha1 over ~4 KB
plus a dict hit, then the whole `(N, A, Q, D)` block is available for that episode.

================================================================================================
4. USING IT
================================================================================================

**The recommended pattern: resolve the whole join up front, by hash, then never think about it
again.** Two checks at startup, one lookup per trajectory, and no index or file order anywhere:

    from read_rla_sidecar import RlaSidecar, preflight

    sc = RlaSidecar(SIDECAR_ROOT, chunk=NUM_ACTIONS_CHUNK,     # 1. shape contract, from the
                    num_tokens=RLA_QUERIES, token_dim=RLA_DIM) #    manifest -- raises on mismatch
    preflight(sc, "data/libero")                               # 2. data contract: pre-scan every
                                                               #    episode, raise unless the join
                                                               #    is a clean bijection
    ...
    targets = sc.lookup(states)                 # per trajectory: (N, A, Q*D) float32, normalised

Why pre-scan rather than discover misses lazily: a miss inside a tf.data graph is an opaque error
at best and a **silent block of zeros** at worst, and a target that is quietly zero for part of the
corpus looks exactly like a training run that is working. `preflight` costs one pass over the RLDS
states -- no image decoding, ~1.8 min for LIBERO's 1693 episodes -- and turns every failure mode
into one exception at startup with a message naming the cause. Run it in CI too; it is the same
code path as the `join` command, so the two cannot drift.

The rest of the API:

    sc.chunk, sc.num_tokens, sc.token_dim        # A, Q, D
    z = sc.lookup(states, normalize=False, flatten=False)   # (N, A, Q, D) float32, raw
    raw = sc.denormalize(z)                      # back to encoder units, for f_dec

    for ref in sc.episodes():                    # EpisodeRef(key, suite, episode_index, ...)
        block = sc.load(ref)                     # (N, A, Q, D) float16, mmapped

    episode_key(states)                          # the join key, if you index it yourself
    anchored_future_indices(N, A)                # (N, A): which frame each z[t, i] targets
    join_suite(sc, "data/libero", "libero_goal") # -> JoinReport, the per-suite join in detail

`lookup` returns zeros and counts a miss (`sc.num_misses`) when `on_miss="zeros"`, which is what a
tf.data pipeline wants -- raising inside `tf.numpy_function` surfaces as an opaque graph error.
The default here is `on_miss="raise"`, because in a script a miss means a stale sidecar. Neither is
a substitute for `preflight`: `on_miss="zeros"` hides the problem and `on_miss="raise"` finds it an
hour into training, on whichever episode the shuffle happened to reach first.

From the command line:

    python read_rla_sidecar.py describe data/libero_rla/16x64-a8-step037500
        the manifest, every array's shape/dtype, disk footprint, and the corpus statistics --
        including what the numbers actually look like once normalised. No TF needed.

    python read_rla_sidecar.py peek data/libero_rla/16x64-a8-step037500 --frame 40
        one frame's (A, Q, D) block printed as numbers: per-chunk-step norms, per-query norms and
        a raw corner of the array, so "what does the data look like" has a concrete answer.

    python read_rla_sidecar.py join data/libero_rla/16x64-a8-step037500 --rlds-root data/libero
        the real test: walk the RLDS episodes, hash each one's states, and confirm the join lands.
        Prints coverage per suite and an alignment table showing action `a_{t+i}` next to
        `z[t, i]`. Fails on a sidecar entry that no episode in this dataset hashes to, or on a
        frame-count disagreement; add `--require-full` to also fail when an RLDS episode has no
        latents (a partial extraction, which would train against zeros). Needs TF + tfds.

    python read_rla_sidecar.py selftest
        builds a synthetic sidecar in a temp dir and round-trips it, and pins down the join verdict
        logic. Proves this file works in the receiving repo before any real data is copied over --
        numpy only, ~1 second, 13 checks.

Exit status is non-zero if a check fails, so `join` and `selftest` work in CI.

================================================================================================
5. VERIFIED -- what a 100% clean join actually looked like
================================================================================================

Run on `data/libero_rla/16x64-a8-step040000-snapshot` (f_enc `encoder_step0040000.snapshot.pt`)
against `data/libero`, all four suites, 2026-07-26:

    python read_rla_sidecar.py join data/libero_rla/16x64-a8-step040000-snapshot \
        --suites libero_spatial libero_object libero_goal libero_10 --require-full

    libero_spatial   432/432 joined   0 uncovered   0 orphaned   BIJECTION
    libero_object    454/454 joined   0 uncovered   0 orphaned   BIJECTION
    libero_goal      428/428 joined   0 uncovered   0 orphaned   BIJECTION
    libero_10        379/379 joined   0 uncovered   0 orphaned   BIJECTION
    corpus-wide      1693 distinct state hashes over 1693 episodes  ->  0 collisions
                     1693/1693 (100.00%) have latents          exit 0, 1 min 47 s

That is the standard to hold a sidecar to, and every word of it is checked mechanically:

  * **1693 distinct hashes over 1693 episodes** -- the join key is unique corpus-wide, so no
    episode can be served another one's latents. Checked across suites, not within one, because
    `RlaSidecar` merges all suites into a single index.
  * **bijection per suite** -- `matched == indexed == scanned`, no uncovered, no orphans. Not just
    "high coverage": both directions, exactly once each.
  * **frame counts agree on all 1693** -- `keys.json.traj_len`, the array's own first axis, and the
    RLDS episode length are the same number. This is what catches a sidecar built from a differently
    preprocessed copy of LIBERO (e.g. with no-ops not stripped), which would otherwise join fine and
    then misalign every target by a few frames.
  * **`lookup` ran on all 1693** and returned finite, correctly shaped, normalised targets --
    the actual consumer call, not a proxy for it.

Corpus shape, for capacity planning: 273465 frames, 4.2 GiB at A=8, Q=16, D=64 fp16 (16 KiB per
frame); `libero_10` is 1.5 GiB of it because its episodes average 268 frames. Normalised targets
come out at mean +0.004, std 1.001, |max| 8.18; 0/1024 dead latent slots.

Three failure modes were exercised on purpose, to confirm the checks can actually fail:

  * a **partial** sidecar (`16x64-a8-step037500`, 120 of 432 spatial episodes) -> `--require-full`
    reports `312 RLDS episodes with no latents` and exits 1; without the flag it passes, because a
    subset is a legitimate thing to train on if you meant it.
  * `--limit N` -> the orphan check is suppressed and says so, since a prefix scan cannot prove an
    entry is absent from the whole dataset.
  * a manifest whose `chunk` disagrees with the consumer's -> `RlaSidecar(...)` raises at
    construction, before any data is read.

Practical lesson from building this: the *only* reliable identifier is the content hash. Episode
ordinals, file names, split positions and shard layout all vary with how the data was read, and
two of the four suites will happily agree with them anyway -- long enough to convince you they are
safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RlaSidecar",
    "EpisodeRef",
    "LiberoEpisode",
    "JoinReport",
    "episode_key",
    "anchored_future_indices",
    "iter_libero_episodes",
    "join_suite",
    "preflight",
    "SUITES",
]

# Sidecar suite name -> RLDS directory under `--rlds-root`. The same mapping the extractor used;
# the suffix is part of the dataset name, not a variant.
SUITES = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

# Which RLDS observation key each camera came from. Recorded here because `manifest["cameras"]`
# names the *autoencoder's* cameras, and their order is the order the encoder saw them in.
RLDS_IMAGE_KEY = {"agentview_camera": "image", "wrist_camera": "wrist_image"}


def episode_key(states: np.ndarray) -> str:
    """`(N, 8)` float32 states -> the sha1 hex key used by `keys.json`.

    Byte-for-byte identical to `extract_rla_sidecar.episode_key`. The `ascontiguousarray` and the
    explicit dtype are load-bearing: a sliced or float64 view would hash different bytes and every
    lookup would miss.
    """
    return hashlib.sha1(np.ascontiguousarray(states, dtype=np.float32).tobytes()).hexdigest()


def anchored_future_indices(num_frames: int, chunk: int) -> np.ndarray:
    """`(N, A)` int array: which frame `z[t, i]` describes the motion *to*.

    `future[t, i] = min(t + i + 1, N - 1)`. This is the whole alignment contract in one line --
    reuse it rather than re-deriving the clamp, and use it to line targets up with the action
    chunk when you want to check yourself against the data.
    """
    t = np.arange(num_frames)[:, None]
    i = np.arange(chunk)[None, :]
    return np.minimum(t + i + 1, num_frames - 1)


@dataclass(frozen=True)
class EpisodeRef:
    """One row of a suite's `keys.json`, plus the resolved path."""

    key: str
    suite: str
    file: str
    episode_index: int
    traj_len: int
    path: Path


class RlaSidecar:
    """A sidecar directory, opened read-only.

    Construction reads only the json: the manifest, the statistics and every suite's `keys.json`.
    Episode arrays are opened lazily and mmapped, so holding a `RlaSidecar` over a 4 GiB sidecar
    costs a few MB.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        chunk: Optional[int] = None,
        num_tokens: Optional[int] = None,
        token_dim: Optional[int] = None,
        cache_size: int = 64,
    ) -> None:
        """`chunk` / `num_tokens` / `token_dim`, when given, are asserted against the manifest.

        Pass your own constants (VLA-Adapter's `NUM_ACTIONS_CHUNK`, `RLA_QUERIES`, `RLA_DIM`) and a
        mismatched sidecar fails here, at startup, instead of training against a target whose chunk
        axis means something else. A sidecar built with `--chunk 16` is a different dataset from one
        built with `--chunk 8`, not a compatible superset.
        """
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found -- {self.root} is not an RLA sidecar. A sidecar is the "
                "directory holding manifest.json, stats.json and <suite>/ep_*.npy."
            )
        self.manifest: dict = json.loads(manifest_path.read_text())

        self.chunk = int(self.manifest["chunk"])            # A
        self.num_tokens = int(self.manifest["num_tokens"])  # Q
        self.token_dim = int(self.manifest["token_dim"])    # D
        self.pairs = str(self.manifest.get("pairs", "anchored"))
        self.format = str(self.manifest.get("format", "npy"))

        for name, want, got in (
            ("chunk", chunk, self.chunk),
            ("num_tokens", num_tokens, self.num_tokens),
            ("token_dim", token_dim, self.token_dim),
        ):
            if want is not None and int(want) != got:
                raise ValueError(
                    f"{self.root} was extracted with {name}={got}, but this consumer expects "
                    f"{want}. Re-extract with the matching setting, or point at the sidecar built "
                    f"for it -- the directory name encodes Q, D and the chunk size."
                )
        if self.manifest.get("random_encoder"):
            print(
                f"[rla] WARNING: {self.root} was extracted with a random encoder. These targets "
                "are noise -- fine for plumbing tests, meaningless for training.",
                file=sys.stderr,
            )

        stats = json.loads((self.root / "stats.json").read_text())
        self.stats_count = int(stats["count"])
        self.mean = np.asarray(stats["mean"], dtype=np.float32)            # (Q, D)
        self.std_raw = np.asarray(stats["std"], dtype=np.float32)          # (Q, D), unfloored
        self.std_floor = float(stats.get("std_floor", 1e-3))
        # Floor the std so a dead latent slot (std ~ 0) cannot blow up the normalised target.
        self.std = np.maximum(self.std_raw, self.std_floor)
        if self.mean.shape != (self.num_tokens, self.token_dim):
            raise ValueError(
                f"stats.json mean is {self.mean.shape}, but the manifest says "
                f"({self.num_tokens}, {self.token_dim}); the sidecar is inconsistent."
            )

        # key -> EpisodeRef, merged across suites: one training mix can interleave several, and the
        # key is a hash of the episode's own states, so it is unique corpus-wide.
        self.index: Dict[str, EpisodeRef] = {}
        self.suites: List[str] = []
        for keys_path in sorted(self.root.glob("*/keys.json")):
            suite = keys_path.parent.name
            self.suites.append(suite)
            for key, entry in json.loads(keys_path.read_text()).items():
                ref = EpisodeRef(
                    key=key,
                    suite=suite,
                    file=entry["file"],
                    episode_index=int(entry["episode_index"]),
                    traj_len=int(entry["traj_len"]),
                    path=keys_path.parent / entry["file"],
                )
                previous = self.index.get(key)
                if previous is not None and previous.path != ref.path:
                    raise RuntimeError(
                        f"state hash collision: {key} maps to both {previous.suite}/"
                        f"{previous.file} and {suite}/{ref.file}"
                    )
                self.index[key] = ref
        if not self.index:
            raise RuntimeError(
                f"no */keys.json under {self.root}; the sidecar has no join index. An interrupted "
                "extraction leaves ep_*.npy with no keys.json -- re-run it to rebuild the index."
            )

        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = int(cache_size)
        self.num_misses = 0

    # -- shape contract -----------------------------------------------------------------------

    @property
    def target_dim(self) -> int:
        """`Q * D` -- the flattened per-chunk-step target width."""
        return self.num_tokens * self.token_dim

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        """`(A, Q, D)` -- the per-frame block."""
        return (self.chunk, self.num_tokens, self.token_dim)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return (
            f"RlaSidecar({self.root}, episodes={len(self.index)}, suites={self.suites}, "
            f"A={self.chunk}, Q={self.num_tokens}, D={self.token_dim}, pairs={self.pairs!r})"
        )

    # -- access -------------------------------------------------------------------------------

    def episodes(self, suite: Optional[str] = None) -> List[EpisodeRef]:
        """Every indexed episode, ordered by (suite, episode_index)."""
        refs = [r for r in self.index.values() if suite is None or r.suite == suite]
        return sorted(refs, key=lambda r: (r.suite, r.episode_index))

    def load(self, ref: EpisodeRef | str) -> np.ndarray:
        """`(N, A, Q, D)` float16, mmapped for `.npy` (the default format).

        Left as float16 on purpose -- casting a whole episode to float32 doubles the memory for no
        gain if you only need a few frames. `lookup` casts; this does not.
        """
        if isinstance(ref, str):
            ref = self.index[ref]
        cached = self._cache.get(ref.key)
        if cached is None:
            if ref.path.suffix == ".npz":
                # `--compress` writes the same fp16 block DEFLATE-compressed under the name `z`.
                # No mmap is possible, so this decompresses the whole episode on every miss.
                with np.load(ref.path) as handle:
                    cached = handle["z"]
            else:
                cached = np.load(ref.path, mmap_mode="r")
            if len(self._cache) < self._cache_size:  # bounded: mmaps are cheap, the dict is not
                self._cache[ref.key] = cached
        return cached

    def lookup(
        self,
        states: np.ndarray,
        *,
        normalize: bool = True,
        flatten: bool = True,
        on_miss: str = "raise",
    ) -> np.ndarray:
        """`(N, 8)` float32 states -> `(N, A, Q*D)` float32 targets for that episode.

        This is the function a data pipeline calls, once per trajectory. `normalize=True` applies
        `(z - mean) / std` from `stats.json`; `flatten=True` folds `(Q, D)` into one `Q*D` vector.

        `on_miss="zeros"` returns a correctly shaped block of zeros and counts the miss instead of
        raising -- the behaviour tf.data needs, since an exception inside `tf.numpy_function`
        surfaces as an opaque graph error far from its cause. Check `num_misses` afterwards: a
        pipeline silently training on zero targets looks exactly like one that is working.
        """
        states = np.asarray(states, dtype=np.float32)
        num_frames = int(states.shape[0])
        key = episode_key(states)

        def _miss(reason: str) -> np.ndarray:
            self.num_misses += 1
            if on_miss == "raise":
                raise KeyError(f"{reason} Sidecar: {self.root}")
            if self.num_misses <= 5:
                print(f"[rla] MISS: {reason} returning zeros.", file=sys.stderr)
            shape = (num_frames, self.chunk, self.target_dim) if flatten else (
                num_frames, *self.latent_shape
            )
            return np.zeros(shape, dtype=np.float32)

        ref = self.index.get(key)
        if ref is None:
            return _miss(f"no sidecar entry for episode key {key} (traj_len={num_frames}).")

        z = np.asarray(self.load(ref), dtype=np.float32)  # (N, A, Q, D)
        if z.shape[0] != num_frames:
            return _miss(
                f"{ref.suite}/{ref.file} has {z.shape[0]} frames but the episode has "
                f"{num_frames}."
            )

        if normalize:
            z = (z - self.mean) / self.std
        if flatten:
            z = z.reshape(num_frames, self.chunk, self.target_dim)
        return np.ascontiguousarray(z)

    def normalize(self, z: np.ndarray) -> np.ndarray:
        """Raw `(..., Q, D)` latents -> normalised, the same way `lookup` does."""
        return (np.asarray(z, dtype=np.float32) - self.mean) / self.std

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """Undo the normalisation, accepting either `(..., Q, D)` or a flattened `(..., Q*D)`.

        Needed whenever a predicted latent goes back into the RLA decoder: `f_dec` was trained on
        encoder-scale latents, not on normalised ones.
        """
        z = np.asarray(z, dtype=np.float32)
        if z.shape[-1] == self.target_dim and z.shape[-2:] != (self.num_tokens, self.token_dim):
            z = z.reshape(*z.shape[:-1], self.num_tokens, self.token_dim)
        return z * self.std + self.mean


# ------------------------------------------------------------------------------------------------
# The LIBERO side of the join. Everything below here needs tensorflow + tensorflow_datasets.
# ------------------------------------------------------------------------------------------------


@dataclass
class LiberoEpisode:
    """One RLDS episode, decoded to numpy. `images` is populated only when asked for."""

    suite: str
    episode_index: int
    states: np.ndarray            # (N, 8) float32 -- what the join key is computed from
    actions: np.ndarray           # (N, 7) float32
    instruction: str
    file_path: str                # the source HDF5 *task* file -- shared by every demo of a task
    images: Optional[Dict[str, np.ndarray]] = None   # camera -> (N, H, W, 3) uint8

    @property
    def key(self) -> str:
        return episode_key(self.states)

    @property
    def traj_len(self) -> int:
        return int(self.states.shape[0])


def _import_tfds():
    """Import TF with the GPU hidden -- TF would otherwise grab all of it just to read TFRecords."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover - depends on the host env
        raise SystemExit(
            "reading data/libero needs tensorflow and tensorflow_datasets "
            f"({exc}). `describe`, `peek` and `selftest` work without them."
        )
    tf.config.set_visible_devices([], "GPU")
    return tf, tfds


def iter_libero_episodes(
    rlds_root: str | Path,
    suite: str,
    *,
    start: int = 0,
    limit: int = -1,
    with_images: bool = False,
) -> Iterator[LiberoEpisode]:
    """Yield `LiberoEpisode`s from `<rlds_root>/<SUITES[suite]>/1.0.0`, in split order.

    JPEG decoding dominates the cost of reading LIBERO and the join needs no pixels, so images are
    skipped at the decoder unless `with_images=True`. Steps are read one batched tensor per episode
    rather than one python iteration per frame -- an episode is 100-300 frames and the per-step
    loop is what makes naive RLDS reading slow.

    `episode_index` counts from 0 within the split, matching `keys.json`.
    """
    _, tfds = _import_tfds()
    rlds_dir = Path(rlds_root) / SUITES[suite] / "1.0.0"
    if not rlds_dir.is_dir():
        raise FileNotFoundError(f"{rlds_dir} does not exist")

    builder = tfds.builder_from_directory(str(rlds_dir))
    total = builder.info.splits["train"].num_examples
    stop = total if limit is None or limit < 0 else min(total, start + limit)
    if start >= stop:
        return

    decoders = None
    if not with_images:
        skip = tfds.decode.SkipDecoding()
        decoders = {"steps": {"observation": {k: skip for k in RLDS_IMAGE_KEY.values()}}}

    dataset = builder.as_dataset(
        split=f"train[{start}:{stop}]", shuffle_files=False, decoders=decoders
    )
    for offset, raw in enumerate(dataset):
        # One `batch` over the whole steps dataset: N frames come back as stacked tensors.
        steps = next(iter(raw["steps"].batch(1 << 30)))
        obs = steps["observation"]
        images = None
        if with_images:
            images = {
                cam: obs[key].numpy() for cam, key in RLDS_IMAGE_KEY.items() if key in obs
            }
        yield LiberoEpisode(
            suite=suite,
            episode_index=start + offset,
            states=obs["state"].numpy().astype(np.float32),
            actions=steps["action"].numpy().astype(np.float32),
            instruction=steps["language_instruction"].numpy()[0].decode("utf-8"),
            file_path=raw["episode_metadata"]["file_path"].numpy().decode("utf-8"),
            images=images,
        )


@dataclass
class JoinReport:
    """What one suite's join looked like. Every list is empty on a healthy sidecar."""

    suite: str
    scanned: int = 0
    matched: int = 0
    same_index: int = 0
    indexed: int = 0
    uncovered: List[Tuple[int, str]] = None          # RLDS episodes with no entry
    orphaned: List[str] = None                       # entries no episode hashed to
    wrong_len: List[Tuple[int, int, int]] = None     # (episode_index, rlds_N, sidecar_N)
    cross_suite: List[Tuple[int, str]] = None        # matched an entry filed under another suite
    collisions: List[Tuple[int, str, int]] = None    # (episode_index, other_suite, other_index)
    bad_targets: List[Tuple[int, str]] = None        # lookup returned a wrong shape / non-finite

    def __post_init__(self) -> None:
        for field in ("uncovered", "orphaned", "wrong_len", "cross_suite", "collisions",
                      "bad_targets"):
            if getattr(self, field) is None:
                setattr(self, field, [])

    def problems(self, *, require_full: bool, conclusive_orphans: bool) -> Dict[str, int]:
        """Failure counts by kind. See `cmd_join.__doc__` for what each one means."""
        found = {
            "state-hash collisions between episodes": len(self.collisions),
            "episodes matched to a different suite's entry": len(self.cross_suite),
            "frame-count disagreement": len(self.wrong_len),
            "lookup returned wrong-shaped or non-finite targets": len(self.bad_targets),
        }
        if conclusive_orphans:
            found["sidecar entries absent from this dataset (ORPHAN)"] = len(self.orphaned)
        if require_full:
            found["RLDS episodes with no latents"] = len(self.uncovered)
        return {kind: count for kind, count in found.items() if count}

    @property
    def is_bijection(self) -> bool:
        """1:1 with no gaps, no duplicates -- what a full extraction should always be."""
        return (
            self.matched == self.indexed == self.scanned
            and not (self.uncovered or self.orphaned or self.collisions or self.cross_suite)
        )


def join_suite(
    sidecar: RlaSidecar,
    rlds_root: str | Path,
    suite: str,
    *,
    limit: int = -1,
    corpus: Optional[Dict[str, Tuple[str, int]]] = None,
    deep: bool = True,
    on_episode=None,
) -> JoinReport:
    """Hash every RLDS episode of `suite` and check the sidecar answers for exactly that episode.

    `corpus` is an optional shared `key -> (suite, episode_index)` dict; pass the same one across
    suites to make the collision check corpus-wide, which is the only way it is meaningful -- the
    sidecar merges suites into one index, so a hash shared between a spatial and an object episode
    would be a real ambiguity even though neither suite alone contains a duplicate.

    `deep=True` also runs the actual `lookup` per episode and checks the result is finite and
    correctly shaped -- a few ms per episode, worth it once at startup.
    """
    report = JoinReport(suite=suite)
    indexed = {r.key: r for r in sidecar.episodes(suite)}
    report.indexed = len(indexed)
    corpus = {} if corpus is None else corpus
    seen_keys = set()

    for episode in iter_libero_episodes(rlds_root, suite, limit=limit):
        report.scanned += 1
        duplicate = corpus.get(episode.key)
        if duplicate is not None:
            report.collisions.append((episode.episode_index, duplicate[0], duplicate[1]))
        corpus[episode.key] = (suite, episode.episode_index)

        # The *global* index, not `indexed`: an entry filed under the wrong suite must surface as
        # a cross-suite error, not silently as "uncovered".
        ref = sidecar.index.get(episode.key)
        if ref is None:
            report.uncovered.append((episode.episode_index, episode.key))
            continue
        if ref.suite != suite:
            report.cross_suite.append((episode.episode_index, f"{ref.suite}/{ref.file}"))
        seen_keys.add(episode.key)
        report.matched += 1
        report.same_index += int(ref.episode_index == episode.episode_index)
        if ref.traj_len != episode.traj_len:
            report.wrong_len.append((episode.episode_index, episode.traj_len, ref.traj_len))

        if deep:
            targets = sidecar.lookup(episode.states)
            expected = (episode.traj_len, sidecar.chunk, sidecar.target_dim)
            if targets.shape != expected:
                report.bad_targets.append(
                    (episode.episode_index, f"shape {targets.shape}, expected {expected}")
                )
            elif not np.isfinite(targets).all():
                report.bad_targets.append((episode.episode_index, "non-finite values"))
        if on_episode is not None:
            on_episode(episode, ref)

    report.orphaned = sorted(set(indexed) - seen_keys)
    return report


def preflight(
    sidecar: RlaSidecar | str | Path,
    rlds_root: str | Path = "data/libero",
    suites: Optional[Sequence[str]] = None,
    *,
    require_full: bool = True,
    limit: int = -1,
    deep: bool = True,
    verbose: bool = True,
) -> Dict[str, JoinReport]:
    """**Call this once at startup.** Pre-scan the dataset, join by state hash, fail loudly if
    anything is off -- so a training run can never quietly regress against zeros or, worse, against
    another episode's latents.

    This is the programmatic form of the `join` command, and it is the recommended way to use a
    sidecar: resolve the whole join *up front*, by hash, and never let `episode_index` or file
    order enter the picture at any point. Costs one pass over the RLDS states (~2 min for LIBERO's
    1693 episodes, no image decoding) and raises `RuntimeError` unless the join is clean.

        from read_rla_sidecar import RlaSidecar, preflight

        sc = RlaSidecar(SIDECAR_ROOT, chunk=NUM_ACTIONS_CHUNK)   # shape contract
        preflight(sc, "data/libero")                             # data contract, fail fast
        ...                                                      # then, per trajectory:
        targets = sc.lookup(states)                              # hash lookup, never an index

    `require_full=False` allows a sidecar that covers only part of the corpus (those episodes will
    get zeros, or a `KeyError`, depending on `lookup`'s `on_miss`); everything else still has to be
    exactly right. `limit` shortens the scan for a smoke test, at the cost of the orphan check.
    """
    if not isinstance(sidecar, RlaSidecar):
        sidecar = RlaSidecar(sidecar)
    suites = list(suites) if suites else list(sidecar.suites)

    corpus: Dict[str, Tuple[str, int]] = {}
    reports: Dict[str, JoinReport] = {}
    problems: Dict[str, int] = {}
    for suite in suites:
        report = join_suite(sidecar, rlds_root, suite, limit=limit, corpus=corpus, deep=deep)
        reports[suite] = report
        for kind, count in report.problems(
            require_full=require_full, conclusive_orphans=limit < 0
        ).items():
            problems[kind] = problems.get(kind, 0) + count
        if verbose:
            print(f"[rla] preflight {suite}: {report.matched}/{report.scanned} episodes joined by "
                  f"state hash, {len(report.uncovered)} uncovered, {len(report.orphaned)} orphaned"
                  f"{'  (bijection)' if report.is_bijection else ''}")

    if problems:
        detail = "; ".join(f"{count} {kind}" for kind, count in problems.items())
        raise RuntimeError(
            f"RLA sidecar {sidecar.root} does not match {rlds_root}: {detail}. Run "
            f"`python read_rla_sidecar.py join {sidecar.root} --rlds-root {rlds_root} "
            "--require-full` for the per-episode breakdown."
        )
    if verbose:
        joined = sum(report.matched for report in reports.values())
        print(f"[rla] preflight OK: {joined}/{len(corpus)} episodes joined by state hash"
              + ("" if joined == len(corpus) else
                 "  (partial sidecar; the rest have no latents -- allowed by require_full=False)"))
    return reports


# ------------------------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------------------------


def _human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num_bytes < 1024 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return ""  # unreachable


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def cmd_describe(args: argparse.Namespace) -> int:
    """Everything about a sidecar that can be answered without touching LIBERO."""
    sidecar = RlaSidecar(args.sidecar)
    manifest = sidecar.manifest

    _rule(f"sidecar {sidecar.root}")
    print(f"latent shape per frame   (A, Q, D) = {sidecar.latent_shape}   "
          f"flattened target width Q*D = {sidecar.target_dim}")
    print(f"pairs                    {sidecar.pairs}"
          + ("" if sidecar.pairs == "anchored" else "   <- NOT anchored on t; f_dec cannot decode"))
    print(f"file format              {sidecar.format}")
    print(f"chunk size A             {sidecar.chunk}   must equal the consumer's action chunk size")
    for field in ("checkpoint", "checkpoint_sha256", "rla_config", "img_size", "cameras",
                  "inverse_input_mode", "dino_channels", "dino_batch", "dino_compile",
                  "random_encoder", "rlds_root"):
        if field in manifest:
            print(f"{field:24s} {manifest[field]}")

    _rule("episodes on disk")
    total_bytes = total_frames = 0
    shapes: Dict[Tuple[int, ...], int] = {}
    dtypes: Dict[str, int] = {}
    mismatched: List[str] = []
    for suite in sidecar.suites:
        refs = sidecar.episodes(suite)
        suite_frames = suite_bytes = 0
        missing = 0
        for ref in refs:
            if not ref.path.exists():
                missing += 1
                continue
            block = sidecar.load(ref)
            shapes[tuple(block.shape[1:])] = shapes.get(tuple(block.shape[1:]), 0) + 1
            dtypes[str(block.dtype)] = dtypes.get(str(block.dtype), 0) + 1
            if block.shape[0] != ref.traj_len:
                mismatched.append(
                    f"{suite}/{ref.file}: array has {block.shape[0]} frames, keys.json says "
                    f"{ref.traj_len}"
                )
            suite_frames += int(block.shape[0])
            suite_bytes += ref.path.stat().st_size
        total_frames += suite_frames
        total_bytes += suite_bytes
        note = f"  [{missing} FILES MISSING]" if missing else ""
        print(f"{suite:16s} {len(refs):5d} episodes  {suite_frames:7d} frames  "
              f"{_human(suite_bytes):>10s}  mean N {suite_frames / max(len(refs), 1):5.0f}{note}")
    print(f"{'TOTAL':16s} {len(sidecar):5d} episodes  {total_frames:7d} frames  "
          f"{_human(total_bytes):>10s}")
    print(f"per-frame block          {shapes}   dtype {dtypes}")
    print(f"bytes per frame          {_human(sidecar.chunk * sidecar.target_dim * 2)} "
          f"(A*Q*D*2, float16)")
    if mismatched:
        print("\nFRAME COUNT MISMATCH between keys.json and the arrays:")
        for line in mismatched[:10]:
            print(f"  {line}")

    _rule("corpus statistics (stats.json)")
    rows_expected = total_frames * sidecar.chunk
    print(f"rows counted             {sidecar.stats_count}  "
          f"(= frames x A; {rows_expected} on disk"
          f"{'' if rows_expected == sidecar.stats_count else '  <- DISAGREES'})")
    print(f"mean  (Q, D)             range [{sidecar.mean.min():+.2f}, {sidecar.mean.max():+.2f}]"
          f"   |mean| median {np.median(np.abs(sidecar.mean)):.2f}")
    print(f"std   (Q, D)             range [{sidecar.std_raw.min():.3f}, "
          f"{sidecar.std_raw.max():.3f}]   floor {sidecar.std_floor}")
    dead = int((sidecar.std_raw < sidecar.std_floor).sum())
    print(f"dead slots (std < floor) {dead} / {sidecar.mean.size}"
          + ("   the encoder uses its whole budget" if dead == 0 else "   <- unused latent slots"))
    if "global_rms" in json.loads((sidecar.root / "stats.json").read_text()):
        print(f"global rms               "
              f"{json.loads((sidecar.root / 'stats.json').read_text())['global_rms']:.3f}")

    _rule(f"what the values look like ({args.sample} episodes sampled)")
    refs = sidecar.episodes()
    picks = refs[:: max(len(refs) // max(args.sample, 1), 1)][: args.sample]
    raw = np.concatenate(
        [np.asarray(sidecar.load(r), dtype=np.float32).reshape(-1, sidecar.num_tokens,
                                                               sidecar.token_dim) for r in picks]
    )
    norm = (raw - sidecar.mean) / sidecar.std
    print(f"raw        mean {raw.mean():+.3f}  std {raw.std():.3f}  "
          f"|max| {np.abs(raw).max():.2f}  rows {len(raw)}")
    print(f"normalised mean {norm.mean():+.3f}  std {norm.std():.3f}  "
          f"|max| {np.abs(norm).max():.2f}"
          "     <- lookup() output; should be ~0.0 / ~1.0")

    # The end-of-episode clamp, verified rather than asserted. See anchored_future_indices.
    ref = picks[0]
    block = np.asarray(sidecar.load(ref), dtype=np.float32)
    num_frames = block.shape[0]
    tail_spread = float(np.abs(block[num_frames - 2] - block[num_frames - 2, 0]).max())
    noop = float(np.abs(block[num_frames - 1]).max())
    print(f"tail clamp ({ref.suite}/{ref.file}): every z[N-2, i] targets frame N-1, spread "
          f"{tail_spread:.3f}"
          + ("  OK" if tail_spread < 0.05 * max(np.abs(block).max(), 1e-6) else "  <- SUSPICIOUS"))
    print(f"z[N-1] is f_enc(0), the no-motion latent: |max| {noop:.2f} "
          f"vs corpus |max| {np.abs(raw).max():.2f}")

    _rule("first 3 index entries")
    for ref in refs[:3]:
        print(f"{ref.key}  {ref.suite}/{ref.file}  episode_index={ref.episode_index}  "
              f"traj_len={ref.traj_len}")
    return 1 if mismatched else 0


def cmd_peek(args: argparse.Namespace) -> int:
    """One frame's latents, printed as numbers."""
    sidecar = RlaSidecar(args.sidecar)
    refs = sidecar.episodes(args.suite)
    if not refs:
        raise SystemExit(f"no episodes for suite {args.suite!r}; have {sidecar.suites}")
    ref = next((r for r in refs if r.episode_index == args.episode), refs[0])

    block = np.asarray(sidecar.load(ref), dtype=np.float32)          # (N, A, Q, D)
    num_frames = block.shape[0]
    frame = args.frame if args.frame is not None else num_frames // 3
    frame = max(0, min(frame, num_frames - 1))
    z = block[frame]                                                 # (A, Q, D)
    future = anchored_future_indices(num_frames, sidecar.chunk)[frame]

    _rule(f"{ref.suite}/{ref.file}  episode_index={ref.episode_index}  N={num_frames}  "
          f"frame t={frame}")
    print(f"z[t] is {z.shape} = (A, Q, D); it is the target for the action chunk "
          f"a_{frame} .. a_{frame + sidecar.chunk - 1}\n")
    print("  i  targets frame  covers actions      |z| max   |z| rms   normalised rms")
    for i in range(sidecar.chunk):
        znorm = (z[i] - sidecar.mean) / sidecar.std
        clamped = frame + i + 1 > num_frames - 1
        covers = f"a_{frame}..a_{frame + i}"
        print(f"  {i}  {future[i]:>13d}  {covers:<16s} {np.abs(z[i]).max():8.2f}  "
              f"{np.sqrt((z[i] ** 2).mean()):8.2f}  {np.sqrt((znorm ** 2).mean()):14.2f}"
              f"{'  (clamped to N-1)' if clamped else ''}")

    _rule(f"raw z[t, 0] corner -- queries 0..3, channels 0..7 (of {sidecar.num_tokens} x "
          f"{sidecar.token_dim})")
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print(z[0, :4, :8])
    _rule("per-query rms of z[t, 0] (which latent slots carry the motion)")
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print(np.sqrt((z[0] ** 2).mean(axis=1)))
    _rule("how a chunk step differs from the previous one -- |z[t, i] - z[t, i-1]| mean")
    with np.printoptions(precision=3, suppress=True, linewidth=200):
        print(np.abs(np.diff(z, axis=0)).mean(axis=(1, 2)))
    print("\n(a run of zeros at the end means the chunk ran past the episode and was clamped)")
    return 0


def cmd_join(args: argparse.Namespace) -> int:
    """Walk the RLDS episodes, join them by state hash, and check what comes back.

    The question this answers is not "did most episodes find something" but "is the join a
    **bijection**": every RLDS episode gets exactly one episode's latents, and every sidecar entry
    is claimed by exactly one RLDS episode. Four distinct ways that can break, and they mean
    different things:

      * **collision** -- two RLDS episodes hash to the same key. The join would be ambiguous and
        one of them would silently get the other's latents. Always a hard failure.
      * **cross-suite** -- an episode matches an entry filed under a different suite. Either a
        collision or a mis-filed extraction; always a hard failure.
      * **orphan** -- an indexed episode that no RLDS episode hashes to. The sidecar was built from
        a different copy of the dataset, so the targets do not belong to this data. Hard failure,
        but only decidable on a full scan, so `--limit` suppresses it.
      * **uncovered** -- an RLDS episode with no entry at all. Just a partial extraction (`--limit`,
        a suite left out, an interrupted run). Reported always; a failure only under
        `--require-full`, which is what you want before a training run that expects every episode
        to carry a real target rather than a silent block of zeros.
    """
    sidecar = RlaSidecar(args.sidecar)
    suites = args.suites or sidecar.suites
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; known: {list(SUITES)}")

    problems: Dict[str, int] = {}
    # Shared across suites: key -> the one episode that produced it. See `join_suite`.
    corpus: Dict[str, Tuple[str, int]] = {}
    scanned_total = 0
    demo_shown = []

    def demo(episode: LiberoEpisode, ref: EpisodeRef) -> None:
        if not demo_shown and episode.traj_len > sidecar.chunk + 4:
            _print_alignment_demo(sidecar, episode, args.frame)
            demo_shown.append(True)

    for suite in suites:
        if suite not in sidecar.suites:
            print(f"\n[{suite}] not in this sidecar (has {sidecar.suites}) -- skipped")
            continue
        _rule(f"[{suite}] joining {Path(args.rlds_root) / SUITES[suite]} against the sidecar")
        report = join_suite(
            sidecar, args.rlds_root, suite, limit=args.limit, corpus=corpus, on_episode=demo
        )
        scanned_total += report.scanned

        for episode_index, other_suite, other_index in report.collisions[:5]:
            print(f"  COLLISION: ep{episode_index} has the same state hash as {other_suite} "
                  f"ep{other_index} -- the join is ambiguous")
        for episode_index, where in report.cross_suite[:5]:
            print(f"  CROSS-SUITE: ep{episode_index} matched an entry filed under {where}")
        for episode_index, rlds_n, side_n in report.wrong_len[:5]:
            print(f"  LEN MISMATCH ep{episode_index}: RLDS N={rlds_n}, sidecar traj_len={side_n}")
        for episode_index, why in report.bad_targets[:5]:
            print(f"  BAD TARGETS ep{episode_index}: {why}")

        print(f"  scanned {report.scanned} RLDS episodes"
              f"{f' (--limit {args.limit})' if args.limit >= 0 else ' (full split)'}, "
              f"sidecar indexes {report.indexed}")
        print(f"  {report.matched} joined by state hash, frame counts agree on all but "
              f"{len(report.wrong_len)}")
        print(f"  {len(report.uncovered)} RLDS episodes have no latents"
              + ("" if not report.uncovered else "  <- they would train against zeros"))
        print(f"  {len(report.orphaned)} sidecar entries not found in the RLDS stream"
              + ("" if args.limit < 0 else "  (not conclusive: --limit only scanned a prefix)"))
        print(f"  join is "
              f"{'a BIJECTION: 1:1, no gaps, no duplicates' if report.is_bijection else 'not 1:1'}"
              f" ({report.matched} episodes <-> {report.indexed} entries)")
        print(f"  episode_index happens to match the scan position for {report.same_index}/"
              f"{report.matched} -- informational only, TFDS slicing is not position-stable "
              "(see the docstring)")
        for episode_index, key in report.uncovered[:3]:
            print(f"    no latents: RLDS ep{episode_index} key {key}")
        if args.limit < 0:  # under --limit every unscanned entry looks orphaned; do not cry wolf
            for key in report.orphaned[:3]:
                print(f"    ORPHAN: sidecar {sidecar.index[key].suite}/{sidecar.index[key].file} "
                      f"key {key}")

        for kind, count in report.problems(
            require_full=args.require_full, conclusive_orphans=args.limit < 0
        ).items():
            problems[kind] = problems.get(kind, 0) + count

    _rule("corpus-wide")
    print(f"  {len(corpus)} distinct state hashes over {scanned_total} episodes scanned across "
          f"{len(suites)} suite(s)"
          + ("" if "state-hash collisions between episodes" not in problems else
             "  <- COLLISIONS, see above"))
    covered = sum(1 for key in corpus if key in sidecar.index)
    print(f"  {covered}/{len(corpus)} of them ({100 * covered / max(len(corpus), 1):.2f}%) have "
          f"latents in {sidecar.root}")
    print(f"  {len(sidecar.index)} entries in the sidecar index across {sidecar.suites}")

    if problems:
        print("\nFAILED:")
        for kind, count in problems.items():
            print(f"  {count:6d}  {kind}")
        if "sidecar entries absent from this dataset (ORPHAN)" in problems or (
            "frame-count disagreement" in problems
        ):
            print("\nAn orphan or a frame-count disagreement means this sidecar was built from a "
                  "different copy of the dataset -- re-extract against the data you train on.")
        if "RLDS episodes with no latents" in problems:
            print("\nUncovered episodes are a partial extraction, not a corrupt sidecar: they "
                  "would train against zeros. Extract the missing suites/episodes, or drop "
                  "--require-full if training on a subset is intended.")
        return 1
    print("\nOK: every sidecar entry belongs to an episode in this dataset, frame counts agree, "
          "and lookup() returns finite, correctly shaped targets.")
    return 0


def _print_alignment_demo(
    sidecar: RlaSidecar, episode: LiberoEpisode, frame: Optional[int]
) -> None:
    """Show, on one real episode, that `z[t, i]` belongs to action `a_{t+i}`."""
    num_frames = episode.traj_len
    t = frame if frame is not None else num_frames // 3
    t = max(0, min(t, num_frames - 1))
    raw = sidecar.lookup(episode.states, normalize=False, flatten=False)   # (N, A, Q, D)
    future = anchored_future_indices(num_frames, sidecar.chunk)[t]

    _rule(f"alignment demo -- ep{episode.episode_index} \"{episode.instruction}\"  N={num_frames}"
          f"  t={t}")
    print("the policy sees frame t, predicts A actions, and is supervised by A latents:\n")
    print("  i   action a_{t+i} (dx dy dz  droll dpitch dyaw  grip)          z[t,i] -> frame "
          "| |z| rms")
    for i in range(sidecar.chunk):
        act = episode.actions[min(t + i, num_frames - 1)]
        act_text = " ".join(f"{v:+6.2f}" for v in act)
        print(f"  {i}   {act_text}   {future[i]:>14d} | {np.sqrt((raw[t, i] ** 2).mean()):7.2f}")
    print("\n'z[t,i] -> frame' is min(t+i+1, N-1): the frame the latent says the arm will have "
          "reached\nafter executing a_t..a_{t+i}. Both are anchored on t, which is what makes "
          "them comparable.")


def cmd_selftest(args: argparse.Namespace) -> int:
    """Build a synthetic sidecar, read it back, and check every documented property.

    Runs on numpy alone, so it is the first thing to run in the receiving repo: it proves this
    reader works there before any real data is copied over, and it doubles as an executable
    specification of the layout -- `_write_synthetic` below writes exactly what the extractor does.
    """
    root = Path(tempfile.mkdtemp(prefix="rla_selftest_"))
    try:
        chunk, queries, dim = 8, 4, 6
        lengths = {"libero_spatial": [11, 17], "libero_goal": [13]}
        truth = _write_synthetic(root, chunk, queries, dim, lengths)

        sidecar = RlaSidecar(root, chunk=chunk, num_tokens=queries, token_dim=dim)
        checks: List[Tuple[str, bool, str]] = []

        checks.append(("manifest shape contract", sidecar.latent_shape == (chunk, queries, dim),
                       str(sidecar.latent_shape)))
        checks.append(("index covers both suites", len(sidecar) == 3 and
                       sidecar.suites == ["libero_goal", "libero_spatial"], str(sidecar.suites)))

        states, expected = truth["libero_spatial"][1]
        got = sidecar.lookup(states, normalize=False, flatten=False)
        checks.append(("lookup finds the episode by state hash and returns it intact",
                       got.shape == expected.shape and np.array_equal(got, expected.astype(
                           np.float32)), f"{got.shape} vs {expected.shape}"))

        flat = sidecar.lookup(states)
        checks.append(("normalised + flattened shape",
                       flat.shape == (len(states), chunk, queries * dim), str(flat.shape)))
        round_trip = sidecar.denormalize(flat)
        checks.append(("denormalize(lookup(x)) == raw",
                       np.allclose(round_trip, expected.astype(np.float32), atol=1e-3),
                       f"max |diff| {np.abs(round_trip - expected.astype(np.float32)).max():.2e}"))

        future = anchored_future_indices(len(states), chunk)
        checks.append(("anchored_future_indices clamps at N-1",
                       future[-1].tolist() == [len(states) - 1] * chunk and
                       future[0].tolist() == list(range(1, chunk + 1)), str(future[-1].tolist())))

        sidecar.num_misses = 0
        zeros = sidecar.lookup(states + 1.0, on_miss="zeros")
        checks.append(("a miss returns zeros and counts, when asked to",
                       zeros.shape == flat.shape and not zeros.any() and sidecar.num_misses == 1,
                       f"misses={sidecar.num_misses}"))
        try:
            sidecar.lookup(states + 1.0)
            raised = False
        except KeyError:
            raised = True
        checks.append(("a miss raises by default", raised, ""))

        try:
            RlaSidecar(root, chunk=chunk + 8)
            rejected = False
        except ValueError:
            rejected = True
        checks.append(("a chunk-size mismatch is rejected at construction", rejected, ""))

        # The join verdict logic, without TF: JoinReport is what both `preflight` and the `join`
        # command decide on, so its rules are worth pinning down here where they cost nothing.
        clean = JoinReport(suite="libero_spatial", scanned=2, matched=2, indexed=2)
        checks.append(("a clean report is a bijection with no problems",
                       clean.is_bijection and not clean.problems(
                           require_full=True, conclusive_orphans=True), ""))

        partial = JoinReport(suite="libero_spatial", scanned=3, matched=2, indexed=2,
                             uncovered=[(2, "deadbeef")])
        checks.append(("uncovered episodes fail only under require_full",
                       not partial.problems(require_full=False, conclusive_orphans=True) and
                       partial.problems(require_full=True, conclusive_orphans=True) ==
                       {"RLDS episodes with no latents": 1} and not partial.is_bijection, ""))

        orphan = JoinReport(suite="libero_spatial", scanned=1, matched=1, indexed=2,
                            orphaned=["deadbeef"])
        checks.append(("an orphan fails, but only on a full scan",
                       orphan.problems(require_full=False, conclusive_orphans=True) ==
                       {"sidecar entries absent from this dataset (ORPHAN)": 1} and
                       not orphan.problems(require_full=False, conclusive_orphans=False), ""))

        broken = JoinReport(suite="libero_spatial", scanned=2, matched=2, indexed=2,
                            collisions=[(1, "libero_goal", 7)], cross_suite=[(1, "libero_goal/x")],
                            wrong_len=[(0, 10, 11)], bad_targets=[(0, "non-finite values")])
        checks.append(("collisions, cross-suite, frame-count and bad targets always fail",
                       len(broken.problems(require_full=False, conclusive_orphans=False)) == 4 and
                       not broken.is_bijection, ""))

        for name, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
        failed = [c for c in checks if not c[1]]
        print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed "
              f"(synthetic sidecar at {root}, removed on exit)")
        return 1 if failed else 0
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"kept {root}")


def _write_synthetic(
    root: Path, chunk: int, queries: int, dim: int, lengths: Dict[str, List[int]]
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Write a miniature sidecar with the exact layout the extractor produces.

    Deliberately spelled out rather than factored: this is the format, in code, for anyone who
    wants to fake a sidecar for a unit test in the receiving repo.
    """
    rng = np.random.default_rng(0)
    truth: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    all_z: List[np.ndarray] = []

    for suite, episode_lengths in lengths.items():
        suite_dir = root / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        keys: Dict[str, dict] = {}
        truth[suite] = []
        for episode_index, num_frames in enumerate(episode_lengths):
            states = rng.normal(size=(num_frames, 8)).astype(np.float32)
            z = (rng.normal(size=(num_frames, chunk, queries, dim)) * 3.0).astype(np.float16)
            # The tail clamp: z[t, i] for t + i + 1 >= N - 1 all describe the same pair, so make
            # the synthetic data obey it too -- a test that ignores this would not catch a reader
            # that mis-handles the end of an episode.
            future = anchored_future_indices(num_frames, chunk)
            for t in range(num_frames):
                for i in range(1, chunk):
                    if future[t, i] == future[t, i - 1]:
                        z[t, i] = z[t, i - 1]
            name = f"ep_{episode_index:06d}.npy"
            np.save(suite_dir / name, z)
            keys[episode_key(states)] = {
                "file": name,
                "episode_index": episode_index,
                "traj_len": num_frames,
            }
            truth[suite].append((states, z))
            all_z.append(z.astype(np.float32).reshape(-1, queries, dim))
        (suite_dir / "keys.json").write_text(json.dumps(keys, indent=1, sort_keys=True))

    rows = np.concatenate(all_z)
    (root / "stats.json").write_text(json.dumps({
        "count": int(rows.shape[0]),
        "mean": rows.mean(axis=0).tolist(),
        "std": rows.std(axis=0).tolist(),
        "std_floor": 1e-3,
        "global_rms": float(np.sqrt((rows ** 2).mean())),
    }, indent=1, sort_keys=True))
    (root / "manifest.json").write_text(json.dumps({
        "chunk": chunk,
        "num_tokens": queries,
        "token_dim": dim,
        "pairs": "anchored",
        "format": "npy",
        "cameras": ["agentview_camera", "wrist_camera"],
        "img_size": 224,
        "random_encoder": True,
        "suites": {s: {"num_episodes": len(v)} for s, v in lengths.items()},
    }, indent=1, sort_keys=True))
    return truth


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser(
        "describe", help="manifest, shapes, disk footprint and corpus statistics (no TF)"
    )
    describe.add_argument("sidecar")
    describe.add_argument("--sample", type=int, default=8,
                          help="episodes to read in full for the value statistics (default 8)")
    describe.set_defaults(func=cmd_describe)

    peek = subparsers.add_parser("peek", help="print one frame's latents as numbers (no TF)")
    peek.add_argument("sidecar")
    peek.add_argument("--suite", default=None, help="default: the first suite in the sidecar")
    peek.add_argument("--episode", type=int, default=0, help="episode_index within the suite")
    peek.add_argument("--frame", type=int, default=None, help="default: N // 3")
    peek.set_defaults(func=cmd_peek)

    join = subparsers.add_parser(
        "join", help="join the sidecar onto data/libero and verify coverage (needs TF + tfds)"
    )
    join.add_argument("sidecar")
    join.add_argument("--rlds-root", default="data/libero")
    join.add_argument("--suites", nargs="+", default=None,
                      help="default: every suite present in the sidecar")
    join.add_argument("--limit", type=int, default=-1,
                      help="only the first N episodes per suite; disables the orphan check, which "
                           "needs a full scan to mean anything")
    join.add_argument("--require-full", action="store_true",
                      help="also fail if any RLDS episode has no latents (i.e. would train "
                           "against zeros), not just if the sidecar has entries this data lacks")
    join.add_argument("--frame", type=int, default=None,
                      help="frame to use for the alignment demo (default: N // 3)")
    join.set_defaults(func=cmd_join)

    selftest = subparsers.add_parser(
        "selftest", help="round-trip a synthetic sidecar; numpy only, no data needed"
    )
    selftest.add_argument("--keep", action="store_true", help="do not delete the temp sidecar")
    selftest.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
