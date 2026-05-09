"""
Few-shot mixed dataset: proxy over two ManiSkillSequenceDatasets.

Combines a small robot-data dataset (with full state/action/images) and a
larger pixel-only dataset (images only, dummy state/action).  Controls
the approximate per-batch ratio of robot vs pixel-only samples via
deterministic index routing in __getitem__.

Usage:
    dataset = FewShotMixedDataset(
        robot_dataset_cfg={
            "dataset_dir": "data/.../success",
            "cameras": ["front_lower_camera"],
            "start_traj_id": 0, "end_traj_id": 9,
            "horizon": 16, "img_size": 512,
        },
        pixel_dataset_cfg={
            "dataset_dir": "data/.../success",
            "cameras": ["front_lower_camera"],
            "start_traj_id": 10, "end_traj_id": 999,
            "horizon": 16, "img_size": 512,
        },
        robot_ratio=0.5,
    )
"""

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from policies.dataset.maniskill_sequence_dataset import ManiSkillSequenceDataset


class FewShotMixedDataset(Dataset):
    """Proxy dataset mixing robot-data and pixel-only sub-datasets.

    Deterministically routes each index to one of the two sub-datasets
    so that approximately ``robot_ratio`` fraction of indices map to
    robot data.  Each sample includes a ``has_robot_data`` boolean.

    The normalizer is fit only on robot-data episodes.

    Args:
        robot_dataset_cfg: Kwargs for the robot-data ManiSkillSequenceDataset.
        pixel_dataset_cfg: Kwargs for the pixel-only ManiSkillSequenceDataset.
            Will automatically set ``pixel_only=True`` and propagate
            ``pixel_only_state_dim`` / ``pixel_only_action_dim`` from the
            robot dataset.
        robot_ratio: Target fraction of robot-data samples per epoch
            (0.0–1.0).  Default 0.5.
    """

    def __init__(
        self,
        robot_dataset_cfg: Dict[str, Any],
        pixel_dataset_cfg: Optional[Dict[str, Any]] = None,
        robot_ratio: float = 0.5,
    ):
        super().__init__()
        self.robot_ratio = robot_ratio

        # --- Build robot sub-dataset ---
        robot_dataset_cfg = dict(robot_dataset_cfg)
        robot_dataset_cfg.pop("pixel_only", None)
        self.robot_ds = ManiSkillSequenceDataset(**robot_dataset_cfg)

        # Infer dims from robot dataset
        self._state_dim = self.robot_ds.qpos.shape[1]
        self._action_dim = self.robot_ds.target_qpos.shape[1]

        # --- Build pixel-only sub-dataset (optional) ---
        if pixel_dataset_cfg is not None:
            pixel_dataset_cfg = dict(pixel_dataset_cfg)
            pixel_dataset_cfg["pixel_only"] = True
            pixel_dataset_cfg["pixel_only_state_dim"] = self._state_dim
            pixel_dataset_cfg["pixel_only_action_dim"] = self._action_dim
            self.pixel_ds: Optional[ManiSkillSequenceDataset] = ManiSkillSequenceDataset(**pixel_dataset_cfg)
        else:
            self.pixel_ds = None

        # --- Compute virtual length and routing ---
        self._robot_len = len(self.robot_ds)
        self._pixel_len = len(self.pixel_ds) if self.pixel_ds is not None else 0

        if self._pixel_len == 0:
            # No pixel data — all robot
            self._total_len = self._robot_len
            self._robot_count = self._total_len
        else:
            # Scale virtual length so both pools are roughly exhausted once,
            # respecting the ratio.
            # robot_count / total = robot_ratio
            # pixel_count / total = 1 - robot_ratio
            # We want robot_count >= robot_len and pixel_count >= pixel_len.
            if self.robot_ratio > 0:
                total_from_robot = int(self._robot_len / self.robot_ratio)
            else:
                total_from_robot = 0
            if self.robot_ratio < 1:
                total_from_pixel = int(self._pixel_len / (1.0 - self.robot_ratio))
            else:
                total_from_pixel = 0
            self._total_len = max(total_from_robot, total_from_pixel)
            self._robot_count = max(1, int(round(self._total_len * self.robot_ratio)))

        print(
            f"FewShotMixedDataset: robot={self._robot_len} samples, "
            f"pixel={self._pixel_len} samples, virtual_len={self._total_len}, "
            f"robot_ratio={self.robot_ratio:.2f} "
            f"(~{self._robot_count} robot per epoch)"
        )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> dict:
        if self._pixel_len == 0:
            # No pixel data — all robot
            inner_idx = idx % self._robot_len
            sample = self.robot_ds[inner_idx]
            sample["has_robot_data"] = torch.tensor(True)
            return sample

        # Deterministic routing: interleave robot and pixel indices
        # Every _total_len indices, the first _robot_count go to robot.
        # Use modular arithmetic to spread evenly.
        cycle_pos = idx % self._total_len
        if cycle_pos < self._robot_count:
            # Robot sample
            inner_idx = cycle_pos % self._robot_len
            sample = self.robot_ds[inner_idx]
            sample["has_robot_data"] = torch.tensor(True)
        else:
            # Pixel-only sample
            pixel_pos = cycle_pos - self._robot_count
            inner_idx = pixel_pos % self._pixel_len
            sample = self.pixel_ds[inner_idx]  # type: ignore[union-attr]
            sample["has_robot_data"] = torch.tensor(False)
        return sample

    # ------------------------------------------------------------------
    # Normalizer & validation
    # ------------------------------------------------------------------

    def get_normalizer(self, mode="limits", **kwargs):
        """Fit normalizer on robot sub-dataset only."""
        return self.robot_ds.get_normalizer(mode=mode, **kwargs)

    def get_validation_dataset(self) -> "FewShotMixedDataset":
        """Return a copy with validation splits from both sub-datasets."""
        val = copy.copy(self)
        val.robot_ds = self.robot_ds.get_validation_dataset()
        if self.pixel_ds is not None:
            val.pixel_ds = self.pixel_ds.get_validation_dataset()

        val._robot_len = len(val.robot_ds)
        val._pixel_len = len(val.pixel_ds) if val.pixel_ds is not None else 0

        if val._pixel_len == 0:
            val._total_len = val._robot_len
            val._robot_count = val._total_len
        else:
            if self.robot_ratio > 0:
                total_from_robot = int(val._robot_len / self.robot_ratio)
            else:
                total_from_robot = 0
            if self.robot_ratio < 1:
                total_from_pixel = int(val._pixel_len / (1.0 - self.robot_ratio))
            else:
                total_from_pixel = 0
            val._total_len = max(total_from_robot, total_from_pixel)
            val._robot_count = max(1, int(round(val._total_len * self.robot_ratio)))
        return val
