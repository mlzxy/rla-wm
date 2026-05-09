"""Deterministic RNG helpers for wmrl baseline training and env resets."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


RESET_SAMPLE_STREAM_ID = 1
SYNC_RESET_STREAM_ID = 2
PPO_SHUFFLE_STREAM_ID = 3
BC_DATALOADER_STREAM_ID = 4
FLOW_NOISE_STREAM_ID = 5

_MASK64 = (1 << 64) - 1
_POSITIVE_SEED_MASK = (1 << 63) - 1


def _splitmix64(value: int) -> int:
    """Mix an integer into a well-distributed 64-bit value."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def fold_in_seed(base_seed: int, *components: int) -> int:
    """Derive a stable positive seed from a base seed and components."""
    mixed = _splitmix64(int(base_seed) & _MASK64)
    for component in components:
        mixed = _splitmix64(mixed ^ _splitmix64(int(component) & _MASK64))
    return int(mixed & _POSITIVE_SEED_MASK)


def seeded_randperm(
    length: int,
    base_seed: int,
    *components: int,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Return a deterministic permutation while keeping shuffle order random."""
    generator = torch.Generator()
    generator.manual_seed(fold_in_seed(base_seed, *components))
    indices = torch.randperm(int(length), generator=generator)
    if device is None:
        return indices
    return indices.to(device=device)


def sample_episode_start(
    valid_ep_indices: np.ndarray,
    episode_lengths: np.ndarray,
    chunk_size: int,
    p_initial_frame: float,
    base_seed: int,
    *components: int,
) -> tuple[int, int]:
    """Sample an episode index and start timestep from an explicit seed."""
    if len(valid_ep_indices) == 0:
        raise ValueError("valid_ep_indices must not be empty")

    rng = np.random.default_rng(fold_in_seed(base_seed, *components))
    initial = bool(rng.random() < float(p_initial_frame))
    ep_pos = int(rng.integers(0, len(valid_ep_indices)))
    ep_idx = int(valid_ep_indices[ep_pos])
    episode_length = int(episode_lengths[ep_idx])
    max_t = episode_length - 1 - int(chunk_size)
    if initial or max_t <= 0:
        return ep_idx, 0
    return ep_idx, int(rng.integers(0, max_t + 1))