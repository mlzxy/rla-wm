"""
rla/sidecar.py

Process-wide handle on the RLA sidecar, plus the small integer index that lets the join survive the
tf.data graph without carrying any payload through it.

The join itself is `read_rla_sidecar`'s: `sha1(observation/state)` per episode. What is added here is
a **slot number** -- the position of an episode in a deterministic corpus-wide ordering -- so the
graph only has to carry one `int32` per frame instead of a `(A, Q, D)` block. The block is fetched in
numpy at batch-transform time, straight out of the mmap.

That matters for two reasons beyond tidiness:

  * at `shuffle_buffer_size=100_000` the payload would be 3.2 GB of resident buffer per rank
    (8 x 1024 float32 = 32 KB per frame); the slot is 400 KB.
  * the `.npy` files are mmapped, so every DDP rank on a node shares one page-cache copy of the
    4.2 GB corpus rather than each materialising its own.

Nothing here is imported at module scope by the training entry point until it is needed, so `import
rla.sidecar` stays free; `get_store()` builds the handle on first use, once per process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from rla.config import CFG
from rla.read_rla_sidecar import RlaSidecar, episode_key

MISS = -1


class RlaStore:
    """A sidecar plus a stable episode<->slot mapping.

    Slots come from `RlaSidecar.episodes()`, which sorts by `(suite, episode_index)` -- a pure
    function of the sidecar's own `keys.json`, so every rank and every process derives the same
    numbering from the same directory with no coordination.
    """

    def __init__(
        self,
        root: str,
        *,
        chunk: int,
        num_tokens: int,
        token_dim: int,
        norm: str = "center",
        shuffle_targets: bool = False,
    ) -> None:
        # `cache_size` large enough for the whole corpus, but **only for the mmap-able format**.
        # For `.npy` an entry is a memmap, so caching all 1693 costs a dict of handles plus
        # file-backed page cache -- shared between the DDP ranks on a node and reclaimable under
        # memory pressure. The bounded default of 64 would instead re-open a mmap on nearly every
        # frame, since the shuffle buffer interleaves ~100k frames from every episode at once.
        #
        # For `.npz` the same cache would be a leak: there is no mmap, so each entry is the
        # *decompressed* fp16 block in anonymous memory, and an unbounded cache would grow to the
        # whole corpus (4.2 GB on this sidecar) in **every rank**, non-reclaimable. That is a host-RAM
        # OOM at 4 ranks -- see tools/OOM_DEBUGGING.md for what that failure looks like. So a
        # compressed sidecar keeps a small cache and pays the decompression instead.
        # Read straight from the manifest rather than constructing an `RlaSidecar` twice: the format
        # has to be known before the one that is kept is built.
        fmt = json.loads((Path(root) / "manifest.json").read_text()).get("format", "npy")
        self.sc = RlaSidecar(
            root,
            chunk=chunk,
            num_tokens=num_tokens,
            token_dim=token_dim,
            cache_size=(1 << 16) if fmt == "npy" else 64,
        )
        self.refs = self.sc.episodes()
        self.slot_of = {ref.key: i for i, ref in enumerate(self.refs)}
        self.num_misses = 0
        self.norm = norm
        self.offset, self.scale = self._scaling(norm)

        # Control arm: serve every episode another episode's latents. Same target distribution, same
        # per-frame shapes, alignment destroyed. Seeded from a constant so all ranks agree.
        self.perm = np.arange(len(self.refs))
        if shuffle_targets:
            self.perm = np.random.default_rng(0).permutation(len(self.refs))
            derangement = int((self.perm != np.arange(len(self.refs))).sum())
            print(
                f"[rla] RLA_SHUFFLE_TARGETS: {derangement}/{len(self.refs)} episodes serve a "
                "different episode's latents. This is the control arm.",
                file=sys.stderr,
            )

    def _scaling(self, norm: str):
        """`(offset, scale)` for the requested mode -- scalars, `(Q, D)` arrays, or a mix.

        The two scalars are derived from the per-`(q, d)` statistics already in `stats.json`, since
        every channel was accumulated over the same number of rows: `E[z] = mean(mean)` and
        `E[z^2] = mean(mean^2 + std^2)`. No re-extraction, and `global_std` reproduces the stored
        `global_rms` to 4 decimals on this sidecar, which is the cross-check that the algebra is right.
        """
        mean, std = self.sc.mean, self.sc.std_raw                   # (Q, D) each
        global_mean = float(mean.mean())
        global_std = float(np.sqrt((mean ** 2 + std ** 2).mean() - global_mean ** 2))
        return {
            "none": (np.float32(0.0), np.float32(1.0)),
            "global": (np.float32(global_mean), np.float32(global_std)),
            "center": (mean, np.float32(global_std)),
            "perdim": (mean, self.sc.std),                          # floored std
        }[norm]

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        """Undo `frame`'s scaling, for feeding a predicted latent back through `f_dec`.

        `read_rla_sidecar.denormalize` assumes the per-dim mode, so it is not a substitute -- this is
        the inverse of whatever mode is actually in force.
        """
        z = np.asarray(z, dtype=np.float32)
        if z.shape[-1] == self.sc.target_dim:
            z = z.reshape(*z.shape[:-1], self.sc.num_tokens, self.sc.token_dim)
        return z * self.scale + self.offset

    def __len__(self) -> int:
        return len(self.refs)

    # -- the tf.data side ------------------------------------------------------------------------

    def slot(self, states: np.ndarray) -> int:
        """`(N, 8)` float32 episode states -> slot number, or `MISS`.

        Called once per trajectory from inside a `tf.numpy_function`, which is the one place in this
        integration that must not raise: an exception there surfaces as an opaque graph error far
        from its cause. So a miss is recorded and returned as `MISS`, and `frame()` -- which runs in
        plain python, outside the graph -- is what turns it into a legible failure.
        """
        slot = self.slot_of.get(episode_key(states), MISS)
        if slot == MISS:
            self.num_misses += 1
            if self.num_misses <= 5:
                print(
                    f"[rla] MISS: no sidecar entry for an episode with {len(states)} frames "
                    f"(key {episode_key(states)}). Sidecar: {self.sc.root}",
                    file=sys.stderr,
                )
        return slot

    # -- the numpy side --------------------------------------------------------------------------

    def frame(self, slot: int, t: int) -> np.ndarray:
        """`(A, Q*D)` float32, normalised -- the target for the action chunk anchored at frame `t`.

        `t` is the frame the policy conditions on, and `z[t, i]` is the cumulative visual consequence
        of executing `a_t .. a_{t+i}`. The pipeline never asks for a `t` past `N - A` (chunk_act_obs
        drops the tail), but the clamp is kept so a shuffled-target run -- where the substituted
        episode can be shorter -- cannot index out of bounds.

        **A miss raises.** This runs inside `RLDSBatchTransform.__call__`, which is plain python in
        the training process (`RLDSDataset.__iter__` iterates `as_numpy_iterator()` and then calls
        the transform), so an exception here is a clean traceback rather than a graph error. The
        alternative -- returning zeros -- is the failure mode that matters most: a target that is
        quietly zero for part of the corpus looks exactly like a training run that is working.
        """
        if slot == MISS:
            self.num_misses += 1
            raise RuntimeError(
                f"RLA join miss: an episode in the RLDS stream has no entry in {self.sc.root} "
                f"({self.num_misses} so far). Training would regress against zeros for it. Run\n"
                f"    .venv/bin/python -m rla.tests --gates join\n"
                "for the per-suite breakdown; a miss means the sidecar was built from a different "
                "copy of the dataset, or covers only some suites."
            )

        ref = self.refs[self.perm[slot]]
        block = self.sc.load(ref)                                   # (N, A, Q, D) fp16, mmapped
        z = np.asarray(block[min(int(t), block.shape[0] - 1)], dtype=np.float32)   # (A, Q, D)
        z = (z - self.offset) / self.scale
        return np.ascontiguousarray(z.reshape(self.sc.chunk, self.sc.target_dim))


_STORE: Optional[RlaStore] = None


def get_store() -> RlaStore:
    """The process's sidecar handle, built on first use."""
    global _STORE
    if _STORE is None:
        if not CFG.sidecar:
            raise RuntimeError(
                "RLA_SIDECAR is unset but a target lookup was requested. Training with RLA_STEPS>0 "
                "needs a sidecar; evaluation does not (z is an output, never an input) and must not "
                "reach this code path."
            )
        _STORE = RlaStore(
            CFG.sidecar,
            chunk=CFG.steps,
            num_tokens=CFG.queries,
            token_dim=CFG.dim,
            norm=CFG.norm,
            shuffle_targets=CFG.shuffle_targets,
        )
        print(
            f"[rla] sidecar ready: {len(_STORE)} episodes from {CFG.sidecar}"
            f"  (scaling {CFG.norm})",
            file=sys.stderr,
        )
    return _STORE


def set_store(store: Optional[RlaStore]) -> None:
    """Override the process handle. Used by `rla/tests/` to swap in a synthetic sidecar."""
    global _STORE
    _STORE = store
