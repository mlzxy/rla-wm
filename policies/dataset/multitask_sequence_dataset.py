"""
Multi-task sequence dataset that wraps multiple ManiSkillSequenceDatasets.

Balances sampling across tasks/robots, zero-pads state/action to max dims
for collation, and provides per-robot normalizers and dimension metadata.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from policies.dataset.maniskill_sequence_dataset import ManiSkillSequenceDataset


class MultiTaskSequenceDataset(Dataset):
    """Wraps multiple single-task datasets with balanced sampling.

    Each __getitem__ uniformly selects a sub-dataset, then samples from it.
    States and actions are zero-padded to the max dims across all robots
    for easy batching, along with ``robot_ind`` so the policy can slice
    out real dims per sample.

    Args:
        task_configs: List of dicts, each containing kwargs for
            ManiSkillSequenceDataset plus ``robot_ind`` (int).
        robot_dim_map: Mapping from robot_ind to
            ``{"state_dim": int, "action_dim": int}``.
    """

    def __init__(
        self,
        task_configs: List[Dict[str, Any]],
        robot_dim_map: Dict[int, Dict[str, int]],
    ):
        super().__init__()
        self.robot_dim_map = {int(k): v for k, v in robot_dim_map.items()}
        self.max_state_dim = max(v["state_dim"] for v in self.robot_dim_map.values())
        self.max_action_dim = max(v["action_dim"] for v in self.robot_dim_map.values())

        # Build sub-datasets
        self.sub_datasets: List[ManiSkillSequenceDataset] = []
        self.sub_robot_inds: List[int] = []
        for tc in task_configs:
            tc = dict(tc)  # copy so we don't mutate
            robot_ind = int(tc.pop("robot_ind"))
            self.sub_datasets.append(ManiSkillSequenceDataset(**tc))
            self.sub_robot_inds.append(robot_ind)

        self.n_sub = len(self.sub_datasets)
        # Compute cumulative lengths for __len__
        self.sub_lengths = [len(d) for d in self.sub_datasets]
        # Virtual length: max sub-dataset length * n_sub (so each is sampled equally)
        self._max_sub_len = max(self.sub_lengths)

    def __len__(self) -> int:
        return self._max_sub_len * self.n_sub

    def __getitem__(self, idx: int) -> dict:
        # Balanced: pick sub-dataset from idx, wrap index within that dataset
        sub_idx = idx % self.n_sub
        inner_idx = (idx // self.n_sub) % self.sub_lengths[sub_idx]

        ds = self.sub_datasets[sub_idx]
        robot_ind = self.sub_robot_inds[sub_idx]
        sample = ds[inner_idx]

        real_state_dim = self.robot_dim_map[robot_ind]["state_dim"]
        real_action_dim = self.robot_dim_map[robot_ind]["action_dim"]

        # Pad state: (H, real_dim) -> (H, max_state_dim)
        state = sample["obs"]["state"]  # (H, real_dim)
        if state.shape[-1] < self.max_state_dim:
            pad = torch.zeros(*state.shape[:-1], self.max_state_dim - state.shape[-1])
            state = torch.cat([state, pad], dim=-1)

        # Pad action: (H, real_dim) -> (H, max_action_dim)
        action = sample["action"]  # (H, real_dim)
        if action.shape[-1] < self.max_action_dim:
            pad = torch.zeros(*action.shape[:-1], self.max_action_dim - action.shape[-1])
            action = torch.cat([action, pad], dim=-1)

        return {
            "obs": {
                "image": sample["obs"]["image"],
                "state": state,
            },
            "action": action,
            "robot_ind": robot_ind,
        }

    def get_normalizer(self, mode="limits", **kwargs) -> Dict[int, Any]:
        """Fit a LinearNormalizer per robot (on real dims only).

        Returns:
            dict mapping robot_ind -> LinearNormalizer
        """
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer

        # Collect data per robot
        robot_data: Dict[int, Dict[str, list]] = {}
        for ds, robot_ind in zip(self.sub_datasets, self.sub_robot_inds):
            if robot_ind not in robot_data:
                robot_data[robot_ind] = {"action": [], "state": []}
            robot_data[robot_ind]["action"].append(ds.target_qpos)
            robot_data[robot_ind]["state"].append(ds.qpos)

        normalizers: Dict[int, LinearNormalizer] = {}
        for robot_ind, data_lists in robot_data.items():
            action_data = np.concatenate(data_lists["action"], axis=0)
            state_data = np.concatenate(data_lists["state"], axis=0)
            normalizer = LinearNormalizer()
            normalizer.fit(
                data={"action": action_data, "state": state_data},
                last_n_dims=1, mode=mode, **kwargs,
            )
            normalizer["image"] = get_image_range_normalizer()
            normalizers[robot_ind] = normalizer

        return normalizers

    def get_validation_dataset(self) -> "MultiTaskSequenceDataset":
        """Return a shallow copy with validation indices for each sub-dataset."""
        val = copy.copy(self)
        val.sub_datasets = [ds.get_validation_dataset() for ds in self.sub_datasets]
        val.sub_lengths = [len(d) for d in val.sub_datasets]
        val._max_sub_len = max(val.sub_lengths) if val.sub_lengths else 0
        return val
