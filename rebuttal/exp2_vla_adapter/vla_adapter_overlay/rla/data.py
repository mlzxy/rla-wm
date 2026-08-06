"""
rla/data.py

Getting the RLA target from the sidecar into `batch["rla"]`, without editing a single upstream file.

The whole join is one wrapped module-global function. `prismatic/vla/datasets/rlds/dataset.py`
imports `normalize_action_and_proprio` at `:25` and calls it as a module global at `:241`, in a
`traj_map` that runs immediately after `restructure`. Two properties of that exact point make it the
right hook, and neither is true anywhere else in the pipeline:

  1. `traj["observation"]["proprio"]` is still **raw**, and for LIBERO it is bit-identical to
     `observation/state` -- `libero_dataset_transform` sets `EEF_state = state[:, :6]` and
     `gripper_state = state[:, -2:]`, and `restructure` concatenates exactly those two in that order
     (`dataset.py:157-168`). That `(N, 8)` float32 array is what `read_rla_sidecar.episode_key`
     hashes, so the join key is available here and nowhere downstream.
  2. Anything written into `observation` from here survives. `restructure` -- which rebuilds the
     trajectory dict from scratch and silently drops every key it does not know -- has already run.
     `chunk_act_obs` then maps over every observation key (`traj_transforms.py:46`), so with
     `window_size=1` the new key is windowed to `(T', 1)` for free, and
     `obs_transforms.decode_and_resize` / `augment` only touch keys prefixed `image_`/`depth_`
     (`obs_transforms.py:51-52, :19`), so an int32 key passes through untouched.

Consequence worth stating explicitly: the per-dataset `standardize_fn` is **not** modified, and
`get_dataset_statistics` hashes `inspect.getsource(standardize_fn)` (`dataset.py:213-221`). So the
cached `dataset_statistics_<sha>.json` is reused byte-identically and an RLA run is
normalisation-comparable to the vanilla baselines by construction -- no recompute pass, no drift.
`rla/tests/` gate 4 checks this rather than trusting it.

Only a 4-byte slot id crosses the graph; the `(A, Q*D)` payload is fetched from the mmap in
`RlaBatchTransform`, in numpy, one frame at a time.
"""

from __future__ import annotations

import numpy as np
import torch

# Import order is load-bearing: this module hides the GPU from TensorFlow at import scope, and TF
# must not have grabbed it before torch does.
from prismatic.vla.datasets.rlds import dataset as _rlds_dataset
import tensorflow as tf  # noqa: E402

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets import RLDSBatchTransform
from rla.config import CFG
from rla.sidecar import MISS, get_store

_ORIG_NORMALIZE = _rlds_dataset.normalize_action_and_proprio
_INSTALLED = False


def _slot_lookup(states: np.ndarray) -> np.ndarray:
    """`(N, 8)` raw states -> `(N,)` int32, the episode's slot repeated per frame.

    One sha1 over ~4 KB and a dict hit, once per trajectory. Broadcast per frame because the
    trajectory is about to be flattened into frames and each one has to carry its own copy.
    """
    states = np.asarray(states)
    slot = get_store().slot(states)
    return np.full((states.shape[0],), slot, dtype=np.int32)


def _normalize_with_rla(traj, metadata, normalization_type):
    """`normalize_action_and_proprio`, plus `observation["rla_slot"]`."""
    obs = traj.get("observation", {})
    if "proprio" not in obs:
        raise KeyError(
            "observation/proprio is missing, so the RLA join key cannot be recovered. The join "
            "needs `state_obs_keys` populated (RLDSDataset passes load_proprio=True); a dataset "
            "configured without proprio cannot carry RLA targets."
        )
    raw_states = obs["proprio"]                    # (N, 8) float32, still un-normalised
    traj = _ORIG_NORMALIZE(traj, metadata, normalization_type)

    slot = tf.numpy_function(_slot_lookup, [tf.cast(raw_states, tf.float32)], tf.int32)
    slot.set_shape([None])
    traj["observation"]["rla_slot"] = slot
    return traj


def install_dataset_hook() -> None:
    """Swap `normalize_action_and_proprio` for the version that attaches the slot id.

    Must run before `RLDSDataset(...)` is constructed. Idempotent, because the entry point and the
    verifier both call it.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _rlds_dataset.normalize_action_and_proprio = _normalize_with_rla
    _INSTALLED = True


class RlaBatchTransform(RLDSBatchTransform):
    """`RLDSBatchTransform` plus the per-frame RLA target.

    Everything upstream does is untouched; this only reads two integers off the frame and turns them
    into the `(A, Q*D)` block for that frame's action chunk.
    """

    def __call__(self, rlds_batch):
        out = super().__call__(rlds_batch)
        obs = rlds_batch["observation"]
        if "rla_slot" not in obs:
            raise KeyError(
                "observation/rla_slot is missing -- rla.data.install_dataset_hook() did not run "
                "before the dataset was built. Use rla/train.py as the entry point."
            )
        # Both are `(window_size,) == (1,)` after chunk_act_obs; [0] is the anchor frame t, the
        # frame the policy conditions on and the frame z is anchored on.
        slot = int(np.asarray(obs["rla_slot"]).reshape(-1)[0])
        t = int(np.asarray(obs["timestep"]).reshape(-1)[0])
        out["rla"] = get_store().frame(slot, t)                # (A, Q*D) float32, normalised
        return out


class RlaCollator(PaddedCollatorForActionPrediction):
    """The upstream collator plus one stacked field. Stacked exactly the way `actions` is."""

    def __call__(self, instances):
        output = super().__call__(instances)
        if "rla" in instances[0]:
            output["rla"] = torch.stack(
                [torch.from_numpy(np.copy(instance["rla"])) for instance in instances]
            )                                                   # (B, A, Q*D) float32
        return output


__all__ = ["install_dataset_hook", "RlaBatchTransform", "RlaCollator", "MISS", "CFG"]
