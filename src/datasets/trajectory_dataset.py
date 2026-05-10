"""
Trajectory Dataset Module

This module provides a PyTorch Dataset implementation for loading and sampling from
multiple trajectory datasets, particularly designed for robotics and manipulation tasks.
It supports loading multi-camera video streams, depth maps, segmentation masks, and
robot state information from ManiSkill trajectory datasets.

Main Components:
    - TrajectoryDataset: Main dataset class that aggregates multiple trajectory datasets
      and provides unified sampling interface
    - TypedDict classes: Type definitions for configuration and data structures
      (DataConfig, DatasetConfig, TrajectoryInfo, FrameIndex, TrajectorySample)

Key Features:
    - Multi-dataset support: Combine multiple trajectory datasets with weighted sampling
    - Wildcard path expansion: Support for glob patterns in dataset paths
    - Camera sampling: Randomly sample a subset of cameras per trajectory
    - Horizon-based frame sampling: Sample evenly-spaced frames across a horizon window
    - Frame windowing: Load sequences of consecutive frames
    - Caching: Cache trajectory metadata for faster initialization
    - Deterministic and random sampling: Support both indexed and random frame sampling

Data Format:
    The dataset returns TrajectorySample dictionaries with the following structure:
        - rgbs: np.ndarray [num_frames, Cam, 3, H, W] - RGB images (uint8), sampled evenly across horizon
        - depths: np.ndarray [num_frames, Cam, H, W] - Depth maps (float32), sampled evenly across horizon
        - intrinsics: np.ndarray [num_frames, Cam, 3, 3] - Camera intrinsic matrices, sampled evenly across horizon
        - extrinsics: np.ndarray [num_frames, Cam, 4, 4] - Camera extrinsic matrices (4x4), sampled evenly across horizon
        - foreground_masks: np.ndarray [num_frames, Cam, H, W] - Foreground segmentation masks (uint8), sampled evenly across horizon
        - static_masks: np.ndarray [num_frames, Cam, H, W] - Static object masks (uint8), sampled evenly across horizon
        - robot_masks: np.ndarray [num_frames, Cam, H, W] - Robot segmentation masks (uint8), sampled evenly across horizon
        - target_qpos: np.ndarray [horizon, J] - Target joint positions (full horizon, all frames)
        - qpos: np.ndarray [horizon, J] - Current joint positions (full horizon, all frames)
        - success: np.ndarray [1] - Task success flag (bool)
        - task_description: str - Text description of the task

    Where:
        num_frames = number of frames to sample (evenly spaced across horizon)
        horizon = total horizon length (for target_qpos and qpos, all frames are included)
        Cam = number of cameras
        H, W = image height and width
        J = number of robot joints

Usage Example:
    ```python
    from src.datasets.trajectory_dataset import TrajectoryDataset, DatasetConfig

    config: DatasetConfig = {
        'root': 'data/dec13',
        'max_num_cameras': 4,
        'num_frames': 2,
        'trajectory_info_cache_file': 'runs/cache/trajectory_info.json',
        'configs': {
            'pushT': {
                'paths': ['PushT-v1'],  # Supports wildcards like ['PushT-v1/*']
                'w': 1.0  # Sampling weight
            }
        }
    }

    dataset = TrajectoryDataset(config)
    sample = dataset[0]  # Get first sample
    ```

Notes:
    - Extrinsics are automatically converted from 3x4 to 4x4 format
    - RGB images are converted from [T, H, W, 3] to [T, 3, H, W] (CHW format)
    - Camera keys are automatically detected from available video streams
    - The dataset handles missing data gracefully by initializing with zeros
"""

import os
import glob
import time
import re
import bisect
from PIL import Image
import json
import numpy as np
import pandas as pd
import torch
import utils3d.torch
from rich import print
from tqdm import tqdm
from datalib.dataset import ManiSkillTrajectoryDataset, TrajectoryData, RobotInfo
from typing import TypedDict, List, Dict, Optional, Union, Literal, cast
from torch.utils.data import Dataset
from utils.cam import convert_extrinsics_3x4_to_4x4
from jaxtyping import Bool, Float32, Int32, UInt8
from torch import Tensor
import numpy as np
import albumentations as A
from datalib.augmentation import build_augmentation_pipeline
# from datalib.remote_dataset import RemoteQueueDataset
from datalib.object_interaction_detect import detect_interactions
import multiprocessing
import concurrent.futures


TASKS = [
    "PushT-v2",
    "RollBall-v1",
    "PegInsertionSide-v1",
    "PokeCube-v2",
    "PullCube-v2",
    "PullCubeTool-v1",
    "TableOnly-v2",
]


def _build_single_trajectory_info(args):
    (
        dataset,
        dataset_idx,
        group_name,
        traj_id,
        min_start_frame,
        min_num_frames,
        interaction_hparams,
    ) = args
    # Try to get frame count
    num_frames = None
    keys = [k for k in dataset.list_keys(traj_id) if "cam" in k]
    for key in keys:
        try:
            num_frames = dataset._get_video_frame_count(traj_id, key)
        except Exception as e:
            print(
                f"Warning: Could not get frame count for {traj_id}/{key} using decord: {e}"
            )
            num_frames = None
        if num_frames is not None:
            break

    if num_frames is None:
        print(
            f"[red]Warning: Could not get frame count for {traj_id}, skipping...[/red]"
        )
        return None

    if num_frames < min_num_frames:
        return None

    camera_keys = []
    video_suffixes = [
        "_rgb.wo_robot",
        "_rgb",
        "_depth",
        "_robot_mask",
        "_foreground_mask",
        "_static_mask",
    ]
    camera_names = set()
    for key in keys:
        for suffix in video_suffixes:
            if key.endswith(suffix):
                cam_name = key[: -len(suffix)]
                camera_names.add(cam_name)
                break
    camera_keys = sorted(list(camera_names))
    interacting_frame_indices = []

    if interaction_hparams is not None:
        traj_path = os.path.join(dataset.root, f"traj_{traj_id}")
        try:
            mask, _, _ = detect_interactions(traj_path, **interaction_hparams)
            interacting_frame_indices = np.where(mask)[0].tolist()
        except Exception as e:
            print(
                f"[red]Warning: Could not run interaction detection for {traj_id}: {e}[/red]"
            )
            interacting_frame_indices = []

    return {
        "dataset_idx": dataset_idx,
        "group_name": group_name,
        "traj_id": traj_id,
        "num_frames": num_frames,
        "camera_keys": camera_keys,
        "min_start_frame": min_start_frame,
        "interacting_frame_indices": interacting_frame_indices,
    }


class DataConfig(TypedDict):
    paths: list[str]  # support wildcard, relative paths
    w: float  # sampling weight
    min_start_frame: int  # Optional: skip first N frames
    horizon_stride: int  # Optional: stride for the sampled horizon
    rgb_variant_mode: Literal["base", "wo_robot", "both", "random"]


class InteractionDetectHParams(TypedDict):
    dist_threshold: float
    movement_threshold: float
    buffer_window: int


class DatasetConfig(TypedDict):
    configs: dict[str, DataConfig]
    root: str
    max_num_cameras: int
    cameras: Optional[List[str]]

    num_frames: int  # Number of frames to load (starting from frame_id)
    min_num_frames: int
    horizon: Union[int, list[int], tuple[int]]

    trajectory_info_cache_file: str
    manual_limit: int
    img_size: int
    augmentations: (
        dict  # Optional: {"brightness": 0.2, "contrast": 0.2, "p_blur": 0.0, ...}
    )
    interaction_detect: Optional[InteractionDetectHParams]
    interaction_prob: float
    start_traj_id: Optional[str]
    end_traj_id: Optional[str]

    mode: str  # Deprecated: only default linspace behavior is supported.
    history_n: int  # Deprecated
    n_parts: Optional[int]  # Deprecated

    use_mp_for_indexing: bool
    rgb_variant_mode: Literal["base", "wo_robot", "both", "random"]
    rgb_random_base_prob: float  # Used when rgb_variant_mode=random (default: 0.75)
    # Optional streams to read/decode. Supported values: ["depth", "robot_mask", "static_mask"]
    # Backward-compatible: bool is also accepted at runtime.
    read_optional_streams: Union[bool, list[str], tuple[str, ...], set[str]]
    read_eef_pose: bool  # False by default
    read_object_poses: bool  # False by default

    traj_id_starts_with_0: bool  # Optional: defaults to True
    nonlinear_sampling: bool
    pad_horizon_to_max: bool  # Optional: defaults to False
    bg_mask_top: float


class TrajectoryInfo(TypedDict):
    dataset_idx: int  # Index into self.datasets
    group_name: str  # Name of the data group this trajectory belongs to
    traj_id: str  # Trajectory ID
    num_frames: int  # Number of frames in the trajectory
    camera_keys: list[str]
    min_start_frame: int
    interacting_frame_indices: list[int]
    valid_start_count: int


class FrameIndex(TypedDict):
    dataset_idx: int  # Index into self.datasets and self.weights
    traj_id: str  # Trajectory ID
    frame_id: int  # Frame index within the trajectory
    camera_keys: List[str]  # List of camera IDs for this frame
    sampled_horizon: int
    feasible_horizon: int
    interaction_frame_indices: Optional[
        List[int]
    ]  # Interaction frame indices for horizon (when interaction_prob >= 1.0)


class EvalSampleHandle(TypedDict):
    """Persistable handle that uniquely identifies a dataset sample for deterministic replay.

    Captures all stochastic decisions (camera subset, rgb variant, horizon, interaction
    frames) at creation time so that replaying the handle later produces identical data
    reads regardless of global RNG state.
    """
    version: int  # Schema version (currently 1)
    dataset_idx: int  # Index into TrajectoryDataset.datasets
    group_name: str  # Data-group name from config
    traj_id: str  # Trajectory ID string
    frame_id: int  # Start frame index within the trajectory
    sampled_horizon: int  # Number of horizon steps for this sample
    feasible_horizon: int  # Maximum feasible horizon at frame_id
    camera_keys: List[str]  # Ordered camera list (post-subsampling)
    rgb_variant: Literal["base", "wo_robot"]  # Resolved RGB stream choice
    interaction_frame_indices: Optional[List[int]]  # Interaction frame list slice (if applicable)
    dataset_root: str  # Root path of the underlying ManiSkillTrajectoryDataset


EVAL_HANDLE_VERSION = 1


class TrajectorySample(TypedDict):
    rgbs: UInt8[Tensor, "num_frames cameras 3 height width"]  # type: ignore  # RGB images (uint8), sampled evenly across horizon
    depths: Float32[Tensor, "num_frames cameras height width"]  # type: ignore  # Depth maps (float32), sampled evenly across horizon
    intrinsics: Float32[Tensor, "num_frames cameras 3 3"]  # type: ignore  # Camera intrinsic matrices, sampled evenly across horizon
    w2c: Float32[Tensor, "num_frames cameras 4 4"]  # type: ignore  # World-to-Camera matrices (4x4), sampled evenly across horizon
    foreground_masks: UInt8[Tensor, "num_frames cameras height width"]  # type: ignore  # Foreground segmentation masks (uint8), sampled evenly across horizon
    static_masks: UInt8[Tensor, "num_frames cameras height width"]  # type: ignore  # Static object masks (uint8), sampled evenly across horizon
    robot_masks: UInt8[Tensor, "num_frames cameras height width"]  # type: ignore  # Robot segmentation masks (uint8), sampled evenly across horizon
    target_qpos: Float32[Tensor, "horizon joints"]  # type: ignore  # Target joint positions (full horizon, all frames)
    qpos: Float32[Tensor, "horizon joints"]  # type: ignore  # Current joint positions (full horizon, all frames)
    root_poses: Float32[Tensor, "horizon rx7"]
    success: Bool[Tensor, "1"]  # type: ignore  # Task success flag (bool)
    task_description: str  # Text description of the task
    traj_id: Int32[Tensor, "1"]  # type: ignore  # Integer trajectory ID
    robot_infos: list[RobotInfo]

    frame_id: Int32[Tensor, " num_frames "]
    max_frames: Int32[Tensor, " num_frames "]
    horizon: Int32[Tensor, " 1"]
    task_ind: Int32[Tensor, " 1 "]
    robot_id: Int32[Tensor, " 1 "]

    eef_pose: Float32[Tensor, "horizon 7"]
    object_poses: Optional[Float32[Tensor, "horizon num_objects 7"]]  # type: ignore
    horizon_is_pad: Tensor  # bool tensor, shape [horizon]


class TrajectoryBatch(TypedDict):
    rgbs: UInt8[Tensor, "batch num_frames cameras 3 height width"]  # type: ignore  # RGB images (uint8), sampled evenly across horizon
    depths: Float32[Tensor, "batch num_frames cameras height width"]  # type: ignore  # Depth maps (float32), sampled evenly across horizon
    intrinsics: Float32[Tensor, "batch num_frames cameras 3 3"]  # type: ignore  # Camera intrinsic matrices, sampled evenly across horizon
    w2c: Float32[Tensor, "batch num_frames cameras 4 4"]  # type: ignore  # World-to-Camera matrices (4x4), sampled evenly across horizon
    foreground_masks: UInt8[Tensor, "batch num_frames cameras height width"]  # type: ignore  # Foreground segmentation masks (uint8), sampled evenly across horizon
    static_masks: UInt8[Tensor, "batch num_frames cameras height width"]  # type: ignore  # Static object masks (uint8), sampled evenly across horizon
    robot_masks: UInt8[Tensor, "batch num_frames cameras height width"]  # type: ignore  # Robot segmentation masks (uint8), sampled evenly across horizon
    target_qpos: list[Float32[Tensor, "horizon joints"]]  # type: ignore  # Target joint positions (full horizon, all frames)
    qpos: list[Float32[Tensor, "horizon joints"]]  # type: ignore  # Current joint positions (full horizon, all frames)
    root_poses: list[Float32[Tensor, "horizon rx7"]]
    success: Bool[Tensor, "batch 1"]  # type: ignore  # Task success flag (bool)
    task_description: list[str]  # Text description of the task
    traj_id: Int32[Tensor, "batch 1"]  # type: ignore  # Integer trajectory IDs
    robot_infos: list[list[RobotInfo]]  # Robot URDF paths

    frame_id: Int32[Tensor, " batch num_frames "]
    max_frames: Int32[Tensor, " batch num_frames "]
    horizon: Int32[Tensor, " batch "]
    task_ind: Int32[Tensor, " batch "]
    robot_id: Int32[Tensor, " batch "]

    eef_pose: Float32[Tensor, "batch horizon 7"]
    object_poses: Optional[list[Float32[Tensor, "horizon num_objects 7"]]]  # type: ignore
    horizon_is_pad: Tensor  # bool tensor, shape [batch, horizon]


class TrajectoryDataset(Dataset):
    HORIZON_INDEX_VERSION = 4

    @staticmethod
    def _normalize_rgb_variant_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"base", "wo_robot", "both", "random"}:
            raise ValueError(
                f"Invalid rgb_variant_mode={mode}. Expected one of: base, wo_robot, both, random"
            )
        return normalized

    @staticmethod
    def _reader_rgb_variant_mode(mode: str) -> str:
        # Underlying trajectory reader supports {base, wo_robot, both}.
        # "random" needs access to both streams at read time.
        return "both" if mode == "random" else mode

    @staticmethod
    def _resolve_sample_rgb_variant(mode: str, base_prob: float = 0.75) -> str:
        if mode == "random":
            return "base" if np.random.rand() < base_prob else "wo_robot"
        if mode == "wo_robot":
            return "wo_robot"
        return "base"

    @staticmethod
    def _fallback_rgb_variant(mode: str, variant: str) -> Optional[str]:
        if mode in {"both", "random"}:
            return "wo_robot" if variant == "base" else "base"
        return None

    @staticmethod
    def _normalize_optional_streams(
        read_optional_streams: Union[bool, list[str], tuple[str, ...], set[str], None],
    ) -> set[str]:
        """
        Normalize config into a set of optional video suffixes.

        Always-available mandatory streams are RGB and foreground mask.
        Optional streams are: depth, robot_mask, static_mask.
        """
        all_optional = {"_depth", "_robot_mask", "_static_mask"}
        if read_optional_streams is None or read_optional_streams is True:
            return set(all_optional)
        if read_optional_streams is False:
            return set()

        if not isinstance(read_optional_streams, (list, tuple, set)):
            raise ValueError(
                "read_optional_streams must be bool, list[str], tuple[str, ...], or set[str]"
            )

        normalized = set()
        for item in read_optional_streams:
            if not isinstance(item, str):
                raise ValueError("read_optional_streams items must be strings")
            key = item.strip().lower()
            key_map = {
                "depth": "_depth",
                "_depth": "_depth",
                "robot_mask": "_robot_mask",
                "_robot_mask": "_robot_mask",
                "static_mask": "_static_mask",
                "_static_mask": "_static_mask",
            }
            if key not in key_map:
                raise ValueError(
                    f"Invalid optional stream '{item}'. Supported: depth, robot_mask, static_mask"
                )
            normalized.add(key_map[key])
        return normalized

    @staticmethod
    def _parse_traj_id_as_int(traj_id: str) -> int:
        """
        Parse trajectory id string into an integer.

        Accepts plain integer strings ("12") and common prefixed formats
        like "traj_000012" by extracting the trailing digit group.
        """
        try:
            return int(traj_id)
        except (TypeError, ValueError):
            pass

        if not isinstance(traj_id, str):
            raise ValueError(f"Trajectory ID must be a string, got: {type(traj_id)}")

        match = re.search(r"(\d+)$", traj_id)
        if match is None:
            raise ValueError(
                f"Cannot parse integer trajectory ID from '{traj_id}'. Expected numeric id or suffix digits."
            )
        return int(match.group(1))

    def _traj_id_to_tensor(self, traj_id: str) -> torch.Tensor:
        traj_id_int = self._parse_traj_id_as_int(traj_id)
        if self.traj_id_starts_with_0:
            base_traj_id = self.start_traj_id_int or 0
            traj_id_int -= base_traj_id
        if traj_id_int < 0:
            raise ValueError(
                f"Computed negative trajectory ID {traj_id_int} from '{traj_id}'. "
                f"Check traj_id_starts_with_0/start_traj_id configuration (start={self.start_traj_id!r})."
            )
        return torch.tensor([traj_id_int], dtype=torch.int32)

    def _get_rgb_mode_for_dataset(self, dataset_idx: int) -> str:
        return self.dataset_rgb_variant_modes[dataset_idx]

    def _horizon_bounds(self) -> tuple[int, int]:
        if self.horizon_range is not None:
            return int(self.horizon_range[0]), int(self.horizon_range[1])
        return int(self.horizon), int(self.horizon)

    def _effective_min_start_frame(self, traj_info: TrajectoryInfo) -> int:
        return int(traj_info["min_start_frame"])

    def _valid_start_count(
        self, traj_info: TrajectoryInfo, min_horizon: Optional[int] = None
    ) -> int:
        min_h, _ = self._horizon_bounds()
        h = min_h if min_horizon is None else int(min_horizon)

        dataset_idx = traj_info["dataset_idx"]
        stride = self.dataset_horizon_strides[dataset_idx]
        span = (h - 1) * stride + 1

        if self.interaction_only:
            n_interact = len(traj_info.get("interacting_frame_indices", []))
            return max(0, n_interact - span)

        num_frames = int(traj_info["num_frames"])
        offset = self._effective_min_start_frame(traj_info)

        return max(0, num_frames - (offset + span))

    def _feasible_horizon_for_start(
        self,
        traj_info: TrajectoryInfo,
        frame_id: int,
        interaction_start_idx: Optional[int] = None,
    ) -> int:
        dataset_idx = traj_info["dataset_idx"]
        stride = self.dataset_horizon_strides[dataset_idx]

        # Keep one-frame tail reservation to preserve legacy indexing semantics.
        if self.interaction_only:
            if interaction_start_idx is None:
                interact_frames = traj_info.get("interacting_frame_indices", [])
                try:
                    interaction_start_idx = interact_frames.index(int(frame_id))
                except ValueError:
                    return 0
            n_interact = len(traj_info.get("interacting_frame_indices", []))
            max_span = max(0, n_interact - int(interaction_start_idx) - 1)
            if max_span < 1:
                return 0
            return (max_span - 1) // stride + 1

        max_span = max(0, int(traj_info["num_frames"]) - int(frame_id) - 1)
        if max_span < 1:
            return 0
        return (max_span - 1) // stride + 1

    def _sample_horizon(
        self, feasible_horizon: int, deterministic_token: Optional[int] = None
    ) -> int:
        min_h, max_h = self._horizon_bounds()
        upper = min(max_h, int(feasible_horizon))
        if upper < min_h:
            raise ValueError(
                f"No feasible horizon in [{min_h}, {max_h}] with feasible_horizon={feasible_horizon}."
            )

        if deterministic_token is not None:
            span = upper - min_h + 1
            return min_h + (int(deterministic_token) % span)

        if upper == min_h:
            return min_h

        if not self.nonlinear_sampling:
            return np.random.randint(min_h, upper + 1)
        else:
            # Oversample longer horizons
            p = np.random.beta(a=5.0, b=2.0)
            h = min_h + int(p * (upper - min_h + 1))
            return min(upper, h)

    def _rebuild_valid_start_index(self) -> None:
        self.traj_valid_start_counts: list[int] = []
        self.traj_valid_start_prefix: list[int] = []
        running = 0

        min_h, _ = self._horizon_bounds()
        for i, traj_info in enumerate(self.trajectory_info):
            count = self._valid_start_count(traj_info, min_horizon=min_h)
            self.trajectory_info[i]["valid_start_count"] = count
            self.traj_valid_start_counts.append(count)
            running += count
            self.traj_valid_start_prefix.append(running)

        self.total_valid_starts = running

    def _locate_global_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_valid_starts:
            raise IndexError(
                f"Index {index} is out of range (max: {self.total_valid_starts - 1})"
            )
        traj_pos = bisect.bisect_right(self.traj_valid_start_prefix, index)
        prev_prefix = 0 if traj_pos == 0 else self.traj_valid_start_prefix[traj_pos - 1]
        local_index = index - prev_prefix
        return traj_pos, local_index

    def __init__(self, *args, **config: DatasetConfig):
        super().__init__()
        use_mp_for_indexing = config.get("use_mp_for_indexing", True)
        self.cfg = config
        self.nonlinear_sampling = config.get("nonlinear_sampling", True)
        self.pad_horizon_to_max = bool(config.get("pad_horizon_to_max", False))
        self.bg_mask_top = cast(float, config.get("bg_mask_top", 0.0) or 0.0)
        if not (0.0 <= self.bg_mask_top <= 1.0):
            raise ValueError(f"bg_mask_top must be in [0, 1], got {self.bg_mask_top}")
        self.rgb_random_base_prob = cast(
            float, config.get("rgb_random_base_prob", 0.75)
        )
        if not (0.0 <= self.rgb_random_base_prob <= 1.0):
            raise ValueError(
                f"rgb_random_base_prob must be in [0, 1], got {self.rgb_random_base_prob}"
            )
        self.traj_id_starts_with_0 = config.get("traj_id_starts_with_0", True)
        self.start_traj_id_int: Optional[int] = None
        self.max_num_cameras = config["max_num_cameras"]
        self.cameras = config.get("cameras")
        self.optional_video_suffixes = self._normalize_optional_streams(
            config.get("read_optional_streams", True)
        )
        # Metadata stream toggles. Default is disabled to reduce metadata I/O.
        self.read_eef_pose = bool(config.get("read_eef_pose", False))
        self.read_object_poses = bool(config.get("read_object_poses", False))
        self.value_range = (0.0, 1.0)
        root = config["root"]

        # Create all ManiSkillTrajectoryDataset instances and record trajectory info
        self.datasets: List[ManiSkillTrajectoryDataset] = []
        self.dataset_robot_infos: List[List[RobotInfo]] = []
        self.dataset_min_start_frames: List[int] = []
        self.dataset_task_inds: List[int] = []
        self.dataset_robot_ids: List[int] = []
        self.dataset_rgb_variant_modes: List[str] = []
        self.dataset_horizon_strides: List[int] = []

        # Data group hierarchy: group_name -> { "weight": float, "dataset_indices": [int, ...] }
        self.data_groups: Dict[str, Dict] = {}
        # Reverse mapping: dataset_idx -> group_name
        self.dataset_to_group: Dict[int, str] = {}
        self.trajectory_info: List[TrajectoryInfo] = []

        self.start_traj_id = config.get("start_traj_id")
        self.end_traj_id = config.get("end_traj_id")
        if self.start_traj_id is not None:
            self.start_traj_id_int = self._parse_traj_id_as_int(self.start_traj_id)

        # Check if cache file exists and try to load trajectory_info from it
        cache_file = config.get("trajectory_info_cache_file")
        cached_info = None
        cached_header = None
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache_content = json.loads(f.read())
                    if isinstance(cache_content, list):
                        cached_info = cache_content
                        cached_header = {}
                    elif isinstance(cache_content, dict):
                        cached_info = cache_content.get("trajectory_info", [])
                        cached_header = cache_content.get("header", {})
                    print(f"Found trajectory_info cache: {cache_file}")
            except Exception as e:
                print(
                    f"Warning: Could not load trajectory_info from cache file {cache_file}: {e}"
                )

        self.interaction_hparams = config.get(
            "interaction_detect",
            {
                "dist_threshold": 0.25,
                "movement_threshold": 0.002,
                "buffer_window": 5,
            },
        )
        self.interaction_prob = config.get("interaction_prob", 0.0)

        # First, create all datasets to get the correct dataset indices
        for config_name, data_config in config["configs"].items():
            weight = data_config["w"]
            paths = data_config["paths"]
            group_dataset_indices = []

            # Expand wildcard paths
            expanded_paths = []
            for path in paths:
                if "*" in path or "?" in path:
                    # Wildcard pattern - expand it
                    if os.path.isabs(path):
                        expanded = sorted(glob.glob(path))
                    else:
                        expanded = sorted(glob.glob(os.path.join(root, path)))
                    expanded_paths.extend(expanded)
                else:
                    # Regular path
                    if os.path.isabs(path):
                        expanded_paths.append(path)
                    else:
                        expanded_paths.append(os.path.join(root, path))

            # Create dataset for each expanded path
            for dataset_path in expanded_paths:
                if not os.path.exists(dataset_path):
                    print(f"Warning: Dataset path does not exist: {dataset_path}")
                    continue

                group_rgb_mode = self._normalize_rgb_variant_mode(
                    data_config.get(
                        "rgb_variant_mode", config.get("rgb_variant_mode", "base")
                    )
                )

                dataset = ManiSkillTrajectoryDataset(
                    dataset_path,
                    start_traj_id=self.start_traj_id,
                    end_traj_id=self.end_traj_id,
                    rgb_variant_mode=self._reader_rgb_variant_mode(group_rgb_mode),
                )
                if len(dataset.list_trajectories()) == 0:
                    print(
                        f"[red]Warning: Dataset path {dataset_path} has no trajectories. Rebuilding index.[/red]",
                        end="",
                    )
                    dataset = ManiSkillTrajectoryDataset(
                        dataset_path,
                        start_traj_id=self.start_traj_id,
                        end_traj_id=self.end_traj_id,
                        force_reindex=True,
                        rgb_variant_mode=self._reader_rgb_variant_mode(group_rgb_mode),
                    )
                    print(
                        f"[green] Found {len(dataset.list_trajectories())} trajectories.[/green]"
                    )

                dataset_idx = len(self.datasets)
                self.datasets.append(dataset)
                robot_infos = dataset.get_robot_infos()
                if robot_infos is None:
                    # Fallback or empty if not available
                    robot_infos = []
                self.dataset_robot_infos.append(robot_infos)
                self.dataset_min_start_frames.append(
                    data_config.get("min_start_frame", 0)
                )

                # Extract task index
                task_ind = 0
                if "TableOnly-V2" in dataset_path:
                    task_ind = -1
                else:
                    for i, task_name in enumerate(TASKS):
                        if task_name in dataset_path:
                            task_ind = i
                            break

                self.dataset_task_inds.append(task_ind)

                # Extract robot ID
                robot_id = 0  # default to panda

                # Using robot_infos if available
                if robot_infos and len(robot_infos) > 0:
                    uid = robot_infos[0].uid.lower()
                    if "xarm" in uid:
                        robot_id = 1
                    elif "ur10" in uid:
                        robot_id = 2
                else:
                    path_lower = dataset_path.lower()
                    if "xarm" in path_lower:
                        robot_id = 1
                    elif "ur10" in path_lower:
                        robot_id = 2

                self.dataset_robot_ids.append(robot_id)
                self.dataset_rgb_variant_modes.append(group_rgb_mode)
                self.dataset_horizon_strides.append(
                    data_config.get("horizon_stride", 1)
                )

                group_dataset_indices.append(dataset_idx)
                self.dataset_to_group[dataset_idx] = config_name

            # Register this data group (even if empty, for config validation)
            self.data_groups[config_name] = {
                "weight": weight,
                "dataset_indices": group_dataset_indices,
            }

        # Store num_frames and horizon settings
        requested_mode = str(config.get("mode", "linspace")).strip().lower()
        if requested_mode != "linspace":
            print(
                f"[yellow]Warning: mode='{requested_mode}' is deprecated and ignored. Using default 'linspace'.[/yellow]"
            )
        if int(config.get("history_n", 0) or 0) != 0:
            print(
                "[yellow]Warning: history_n is deprecated and ignored in TrajectoryDataset.[/yellow]"
            )
        if config.get("n_parts", None) is not None:
            print(
                "[yellow]Warning: n_parts is deprecated and ignored in TrajectoryDataset.[/yellow]"
            )

        self.mode = "linspace"
        self.num_frames = config["num_frames"]
        horizon_cfg = config.get("horizon", self.num_frames)
        if isinstance(horizon_cfg, (list, tuple)):
            self.horizon_range = (int(horizon_cfg[0]), int(horizon_cfg[1]))
            self.horizon = self.horizon_range[1]
        else:
            self.horizon = int(horizon_cfg)
            self.horizon_range = None

        self.min_num_frames = config.get("min_num_frames", 5)
        self.history_n = 0

        # Now build trajectory_info (either from cache if valid, or from scratch)
        if cached_info is not None:
            # Verify cache is valid by checking dataset_idx range, camera_keys, and group_names
            max_dataset_idx = max(
                (info.get("dataset_idx", -1) for info in cached_info), default=-1
            )
            has_camera_keys = all("camera_keys" in info for info in cached_info)
            has_group_names = all("group_name" in info for info in cached_info)
            if has_group_names:
                cached_groups = set(info["group_name"] for info in cached_info)
                config_groups = set(config["configs"].keys())
                groups_match = cached_groups == config_groups
            else:
                groups_match = False

            hparams_match = True
            if (
                cached_header is None
                or cached_header.get("interaction_detect") != self.interaction_hparams
            ):
                hparams_match = False

            if cached_header is not None:
                if (
                    cached_header.get(
                        "horizon_index_version", self.HORIZON_INDEX_VERSION
                    )
                    != self.HORIZON_INDEX_VERSION
                ):
                    hparams_match = False
                if cached_header.get("min_num_frames") != self.min_num_frames:
                    hparams_match = False
                if cached_header.get("horizon") != horizon_cfg:
                    hparams_match = False
                if (
                    cached_header.get("dataset_min_start_frames")
                    != self.dataset_min_start_frames
                ):
                    hparams_match = False
                if (
                    cached_header.get("dataset_horizon_strides")
                    != self.dataset_horizon_strides
                ):
                    hparams_match = False

            if (
                max_dataset_idx < len(self.datasets)
                and has_camera_keys
                and has_group_names
                and groups_match
                and hparams_match
            ):
                # Cache seems valid, filter it if needed
                if self.start_traj_id is not None or self.end_traj_id is not None:
                    # Use a temporary dataset instance to access sorting/filtering logic
                    # or replicate it here if appropriate. ManiSkillTrajectoryDataset
                    # already has _traj_id_in_range.
                    # We'll use the logic from ManiSkillTrajectoryDataset
                    temp_ds = self.datasets[0] if self.datasets else None
                    if temp_ds:
                        self.trajectory_info = [
                            {**info, "min_start_frame": info.get("min_start_frame", 0)}
                            for info in cached_info
                            if temp_ds._traj_id_in_range(info["traj_id"])
                            and info["num_frames"] >= self.min_num_frames
                        ]
                    else:
                        self.trajectory_info = [
                            {**info, "min_start_frame": info.get("min_start_frame", 0)}
                            for info in cached_info
                            if info["num_frames"] >= self.min_num_frames
                        ]
                else:
                    self.trajectory_info = [
                        {**info, "min_start_frame": info.get("min_start_frame", 0)}
                        for info in cached_info
                        if info["num_frames"] >= self.min_num_frames
                    ]
                print(
                    f"Using trajectory_info from cache ({len(self.trajectory_info)} trajectories)"
                )
            else:
                # Cache is invalid, rebuild
                if max_dataset_idx >= len(self.datasets):
                    print(
                        f"Cache invalid (max dataset_idx {max_dataset_idx} >= {len(self.datasets)} datasets), rebuilding..."
                    )
                if not has_camera_keys:
                    print("Cache invalid (missing camera_keys field), rebuilding...")
                if not has_group_names:
                    print("Cache invalid (missing group_name field), rebuilding...")
                elif not groups_match:
                    print(
                        f"Cache invalid (group names mismatch: cached={cached_groups}, config={config_groups}), rebuilding..."
                    )
                elif not hparams_match:
                    print(f"Cache invalid (interaction_detect mismatch), rebuilding...")
                cached_info = None

        # Build trajectory_info if not using cache
        # Build trajectory_info if not using cache
        if cached_info is None:
            build_workers = min(os.cpu_count() or 1, 32)

            if build_workers > 1 and use_mp_for_indexing:
                ctx = multiprocessing.get_context("fork")
                tasks = []
                for dataset_idx, dataset in enumerate(self.datasets):
                    trajectories = dataset.list_trajectories()
                    min_start_frame = self.dataset_min_start_frames[dataset_idx]
                    group_name = self.dataset_to_group[dataset_idx]
                    for traj_id in trajectories:
                        tasks.append(
                            (
                                dataset,
                                dataset_idx,
                                group_name,
                                traj_id,
                                min_start_frame,
                                self.min_num_frames,
                                self.interaction_hparams,
                            )
                        )

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=build_workers, mp_context=ctx
                ) as executor:
                    futures = [
                        executor.submit(_build_single_trajectory_info, t) for t in tasks
                    ]
                    results = []
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"Building trajectory info (parallel x{build_workers})",
                    ):
                        results.append(future.result())

                for res in results:
                    if res is not None:
                        self.trajectory_info.append(res)

                # Sort by dataset_idx then natural sort pattern for traj_id to match sequential order
                import re

                def natural_keys(text):
                    return [
                        int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)
                    ]

                self.trajectory_info.sort(
                    key=lambda x: (x["dataset_idx"], natural_keys(x["traj_id"]))
                )

            else:
                for dataset_idx, dataset in enumerate(self.datasets):
                    # Retrieve the min_start_frame for this dataset_idx
                    trajectories = dataset.list_trajectories()
                    min_start_frame = self.dataset_min_start_frames[dataset_idx]

                    # Get frame count and camera keys for each trajectory
                    for traj_id in tqdm(
                        trajectories,
                        desc=f"Building trajectory info for dataset {dataset_idx}",
                    ):
                        res = _build_single_trajectory_info(
                            (
                                dataset,
                                dataset_idx,
                                self.dataset_to_group[dataset_idx],
                                traj_id,
                                min_start_frame,
                                self.min_num_frames,
                                self.interaction_hparams,
                            )
                        )
                        if res is not None:
                            self.trajectory_info.append(res)

        # Store num_frames and horizon settings
        self.num_frames = config["num_frames"]
        horizon_cfg = config.get("horizon", self.num_frames)
        if isinstance(horizon_cfg, (list, tuple)):
            self.horizon_range = (int(horizon_cfg[0]), int(horizon_cfg[1]))
            # For indexing and __len__, we use the maximum possible horizon
            # to ensure we don't sample a starting frame that would go out of bounds
            # for any horizon in the range.
            self.horizon = self.horizon_range[1]
        else:
            self.horizon = int(horizon_cfg)
            self.horizon_range = None

        # Save trajectory_info to cache file if specified
        if cache_file and cached_info is None:
            try:
                # Create directory if it doesn't exist
                cache_dir = os.path.dirname(cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                header_info = {
                    "horizon_index_version": self.HORIZON_INDEX_VERSION,
                    "interaction_detect": self.interaction_hparams,
                    "min_num_frames": self.min_num_frames,
                    "horizon": horizon_cfg,
                    "dataset_min_start_frames": self.dataset_min_start_frames,
                    "dataset_horizon_strides": self.dataset_horizon_strides,
                }

                cache_obj = {
                    "header": header_info,
                    "trajectory_info": self.trajectory_info,
                }
                with open(cache_file, "w") as f:
                    json.dump(cache_obj, f, indent=2)
                print(f"Saved trajectory_info to cache: {cache_file}")
            except Exception as e:
                print(
                    f"Warning: Could not save trajectory_info to cache file {cache_file}: {e}"
                )

        # Filter trajectories that don't have enough frames for the current settings
        # We do this after cache saving so the cache remains valid for other horizons
        self.interaction_only = self.interaction_prob >= 1.0
        valid_trajectory_info = []
        min_h, max_h = self._horizon_bounds()
        for traj_info in self.trajectory_info:
            num_frames_valid = self._valid_start_count(traj_info, min_horizon=min_h)

            if num_frames_valid > 0:
                traj_info["valid_start_count"] = num_frames_valid
                valid_trajectory_info.append(traj_info)

        filtered_out = len(self.trajectory_info) - len(valid_trajectory_info)
        if filtered_out > 0:
            horizon_desc = (
                f"{min_h}-{max_h}"
                if self.horizon_range is not None
                else str(self.horizon)
            )
            print(
                f"Filtered out {filtered_out} trajectories that are too short for horizon {horizon_desc}."
            )

        self.trajectory_info = valid_trajectory_info
        self._rebuild_valid_start_index()

        # Update data_groups to remove empty dataset indices
        valid_dataset_indices = set(
            info["dataset_idx"] for info in self.trajectory_info
        )
        for group_name, group_data in self.data_groups.items():
            group_data["dataset_indices"] = [
                idx
                for idx in group_data["dataset_indices"]
                if idx in valid_dataset_indices
            ]

        # Print dataset summary (only on master process)
        is_master = int(os.environ.get("LOCAL_RANK", 0)) == 0
        if is_master:
            show_interaction = self.interaction_prob > 0
            interaction_only = self.interaction_prob >= 1.0
            frame_label = "interaction frames" if interaction_only else "frames"
            print(
                f"\n[bold]Dataset Summary{'  (interaction frames only)' if interaction_only else ''}:[/bold]"
            )
            # Collect per-dataset and per-group stats
            dataset_stats: Dict[int, Dict[str, int]] = {}
            group_stats: Dict[str, Dict[str, int]] = {}
            for traj_info in self.trajectory_info:
                gname = traj_info["group_name"]
                didx = traj_info["dataset_idx"]
                n_interact = len(traj_info.get("interacting_frame_indices", []))
                n_frames = n_interact if interaction_only else traj_info["num_frames"]
                if gname not in group_stats:
                    group_stats[gname] = {
                        "trajectories": 0,
                        "frames": 0,
                        "interaction_frames": 0,
                    }
                group_stats[gname]["trajectories"] += 1
                group_stats[gname]["frames"] += n_frames
                group_stats[gname]["interaction_frames"] += n_interact
                if didx not in dataset_stats:
                    dataset_stats[didx] = {
                        "trajectories": 0,
                        "frames": 0,
                        "interaction_frames": 0,
                        "group_name": gname,
                    }
                dataset_stats[didx]["trajectories"] += 1
                dataset_stats[didx]["frames"] += n_frames
                dataset_stats[didx]["interaction_frames"] += n_interact
            total_trajs = 0
            total_frames = 0
            for gname, gdata in self.data_groups.items():
                gstats = group_stats.get(
                    gname, {"trajectories": 0, "frames": 0, "interaction_frames": 0}
                )
                line = f"  [cyan]{gname}[/cyan]: {gstats['trajectories']} trajectories, {gstats['frames']} {frame_label}"
                if show_interaction and not interaction_only:
                    line += f" ({gstats['interaction_frames']} interaction)"
                print(line)
                total_trajs += gstats["trajectories"]
                total_frames += gstats["frames"]
                for didx in gdata["dataset_indices"]:
                    dstats = dataset_stats.get(
                        didx, {"trajectories": 0, "frames": 0, "interaction_frames": 0}
                    )
                    dpath = self.datasets[didx].root
                    line = f"    [dim]└ {dpath}[/dim]: {dstats['trajectories']} trajectories, {dstats['frames']} {frame_label}"
                    if show_interaction and not interaction_only:
                        line += f" ({dstats['interaction_frames']} interaction)"
                    print(line)
            print(
                f"  [bold green]Total[/bold green]: {total_trajs} trajectories, {total_frames} {frame_label}\n"
            )

        # Setup augmentations
        self.transform = None
        aug_config = config.get("augmentations", None)
        if aug_config:
            self.transform = build_augmentation_pipeline(aug_config)

    def __len__(self):
        """Return total number of frame indices across all datasets, trajectories, and frames."""
        if self.cfg.get("manual_limit", -1) > 0:
            return self.cfg["manual_limit"]
        return self.total_valid_starts

    def sample_frame_index(self, index: int = -1) -> FrameIndex:
        """Sample a frame index from all available trajectories.

        Args:
            index: If >= 0, deterministically select the frame at this index.
                   If < 0, randomly sample a frame.
        """
        if not self.trajectory_info:
            raise ValueError("No trajectories available")

        min_h, _ = self._horizon_bounds()

        def _pick_interaction_start(
            info: TrajectoryInfo, local_index: Optional[int] = None
        ) -> tuple[int, int, list[int], int]:
            interact_frames = info.get("interacting_frame_indices", [])
            valid_count = max(0, len(interact_frames) - min_h)
            if valid_count <= 0:
                raise ValueError(
                    f"Trajectory {info['traj_id']} has no valid interaction starts"
                )
            start_idx = (
                int(local_index)
                if local_index is not None
                else int(np.random.randint(0, valid_count))
            )
            frame = int(interact_frames[start_idx])
            feasible = self._feasible_horizon_for_start(
                info, frame_id=frame, interaction_start_idx=start_idx
            )
            return frame, feasible, interact_frames, start_idx

        def _pick_regular_start(
            info: TrajectoryInfo, local_index: Optional[int] = None
        ) -> tuple[int, int]:
            offset = self._effective_min_start_frame(info)

            valid_count = self._valid_start_count(info, min_horizon=min_h)
            if valid_count <= 0:
                raise ValueError(f"Trajectory {info['traj_id']} has no valid frames")

            frame_offset = (
                int(local_index)
                if local_index is not None
                else int(np.random.randint(0, valid_count))
            )
            frame = offset + frame_offset
            feasible = self._feasible_horizon_for_start(info, frame_id=frame)
            return frame, feasible

        if self.interaction_only:
            # Interaction-only mode: horizon advances through interaction frames
            interact_indices_list = None
            if index >= 0:
                traj_pos, local_index = self._locate_global_index(index)
                traj_info = self.trajectory_info[traj_pos]
                frame_id, feasible_horizon, interact_frames, start_interact_idx = (
                    _pick_interaction_start(traj_info, local_index=local_index)
                )
                sampled_horizon = self._sample_horizon(
                    feasible_horizon, deterministic_token=index
                )
                stride = self.dataset_horizon_strides[traj_info["dataset_idx"]]
                span = (sampled_horizon - 1) * stride + 1
                interact_indices_list = interact_frames[
                    start_interact_idx : start_interact_idx + span : stride
                ]
            else:
                # Random sampling with group hierarchy
                active_groups = [
                    (name, grp)
                    for name, grp in self.data_groups.items()
                    if grp["dataset_indices"]
                ]
                if not active_groups:
                    raise ValueError("No data groups with datasets available")

                group_weights = np.array([grp["weight"] for _, grp in active_groups])
                group_probs = group_weights / group_weights.sum()
                group_idx = np.random.choice(len(active_groups), p=group_probs)
                selected_group = active_groups[group_idx][1]

                dataset_indices = selected_group["dataset_indices"]
                dataset_idx = dataset_indices[np.random.randint(len(dataset_indices))]

                dataset_trajectories = [
                    ti
                    for ti in self.trajectory_info
                    if ti["dataset_idx"] == dataset_idx
                ]
                if not dataset_trajectories:
                    raise ValueError(
                        f"No trajectories available for dataset {dataset_idx}"
                    )

                traj_info = dataset_trajectories[
                    np.random.randint(len(dataset_trajectories))
                ]
                frame_id, feasible_horizon, interact_frames, start_interact_idx = (
                    _pick_interaction_start(traj_info)
                )
                sampled_horizon = self._sample_horizon(feasible_horizon)
                stride = self.dataset_horizon_strides[traj_info["dataset_idx"]]
                span = (sampled_horizon - 1) * stride + 1
                interact_indices_list = interact_frames[
                    start_interact_idx : start_interact_idx + span : stride
                ]

            if len(interact_indices_list) != sampled_horizon:
                raise RuntimeError(
                    f"Interaction horizon mismatch: expected {sampled_horizon}, got {len(interact_indices_list)}"
                )

        elif index >= 0:
            traj_pos, local_index = self._locate_global_index(index)
            traj_info = self.trajectory_info[traj_pos]
            frame_id, feasible_horizon = _pick_regular_start(
                traj_info, local_index=local_index
            )
            sampled_horizon = self._sample_horizon(
                feasible_horizon, deterministic_token=index
            )
            interact_indices_list = None
        else:
            # Random sampling: hierarchical group-first sampling
            # Step 1: Sample a data group based on group weights
            active_groups = [
                (name, grp)
                for name, grp in self.data_groups.items()
                if grp["dataset_indices"]  # skip empty groups
            ]
            if not active_groups:
                raise ValueError("No data groups with datasets available")

            group_names = [name for name, _ in active_groups]
            group_weights = np.array([grp["weight"] for _, grp in active_groups])
            total_weight = group_weights.sum()
            if total_weight == 0:
                raise ValueError("All data group weights are zero")

            group_probs = group_weights / total_weight
            group_idx = np.random.choice(len(group_names), p=group_probs)
            selected_group = active_groups[group_idx][1]

            # Step 2: Sample a dataset uniformly within the selected group
            dataset_indices = selected_group["dataset_indices"]
            dataset_idx = dataset_indices[np.random.randint(len(dataset_indices))]

            # Step 3: Get all trajectories for the selected dataset
            dataset_trajectories = [
                traj_info
                for traj_info in self.trajectory_info
                if traj_info["dataset_idx"] == dataset_idx
            ]

            if not dataset_trajectories:
                raise ValueError(f"No trajectories available for dataset {dataset_idx}")

            # Step 4: Randomly sample a trajectory from the selected dataset
            traj_idx = np.random.randint(0, len(dataset_trajectories))
            traj_info = dataset_trajectories[traj_idx]

            # Step 5: Sample a random frame from the selected trajectory
            frame_id, feasible_horizon = _pick_regular_start(traj_info)

            if self.interaction_prob > 0.0 and np.random.rand() < self.interaction_prob:
                offset = self._effective_min_start_frame(traj_info)
                num_frames_valid = self._valid_start_count(traj_info, min_horizon=min_h)
                valid_interacting_frames = [
                    f
                    for f in traj_info.get("interacting_frame_indices", [])
                    if offset <= f < offset + num_frames_valid
                ]
                if len(valid_interacting_frames) > 0:
                    frame_id = int(np.random.choice(valid_interacting_frames))
                    feasible_horizon = self._feasible_horizon_for_start(
                        traj_info, frame_id=frame_id
                    )

            sampled_horizon = self._sample_horizon(feasible_horizon)
            interact_indices_list = None

        # Get camera keys for this trajectory
        camera_keys = traj_info.get("camera_keys", [])
        if not camera_keys:
            # Fallback: try to get camera keys from the dataset
            dataset = self.datasets[traj_info["dataset_idx"]]
            keys = dataset.list_keys(traj_info["traj_id"])
            video_suffixes = [
                "_rgb.wo_robot",
                "_rgb",
                "_depth",
                "_robot_mask",
                "_foreground_mask",
                "_static_mask",
            ]
            camera_names = set()
            for key in keys:
                for suffix in video_suffixes:
                    if key.endswith(suffix):
                        cam_name = key[: -len(suffix)]
                        camera_names.add(cam_name)
                        break
            camera_keys = sorted(list(camera_names))

        # Filter and order camera keys according to self.cameras if provided
        if self.cameras:
            filtered_cameras = []
            for cam in self.cameras:
                if cam in camera_keys:
                    filtered_cameras.append(cam)
            if filtered_cameras:
                camera_keys = filtered_cameras
            else:
                # If none of the requested cameras are found, use the first available one as fallback
                # or keep the original list? Usually it's better to keep the original or log a warning.
                # The user request implies they want THESE cameras.
                print(
                    f"[yellow]Warning: Requested cameras {self.cameras} not found in trajectory {traj_info['traj_id']}. Available: {camera_keys}[/yellow]"
                )

        # Randomly sample camera keys if max_num_cameras is specified
        if self.max_num_cameras > 0 and len(camera_keys) > self.max_num_cameras:
            sampled_indices = np.random.choice(
                len(camera_keys), size=self.max_num_cameras, replace=False
            )
            camera_keys = [camera_keys[i] for i in sampled_indices]

        # Sample horizon if a range is provided
        if sampled_horizon > feasible_horizon:
            raise RuntimeError(
                f"Sampled horizon {sampled_horizon} exceeds feasible horizon {feasible_horizon}"
            )

        return {
            "dataset_idx": traj_info["dataset_idx"],
            "traj_id": traj_info["traj_id"],
            "frame_id": frame_id,
            "camera_keys": camera_keys,
            "sampled_horizon": sampled_horizon,
            "feasible_horizon": feasible_horizon,
            "interaction_frame_indices": interact_indices_list
            if self.interaction_only
            else None,
        }

    def _compute_sampled_frame_indices(
        self, num_frames: int, horizon: int
    ) -> np.ndarray:
        """
        Compute which frame indices to sample based on horizon and num_frames.

        Args:
            num_frames: Number of frames to sample
            horizon: Total horizon length

        Returns:
            Array of frame indices to sample (0-indexed within the horizon)

        Examples:
            horizon=5, num_frames=2 -> [0, 4]
            horizon=10, num_frames=3 -> [0, 5, 9]
        """
        if num_frames == 1:
            indices = np.array([0])
        elif num_frames >= horizon:
            # If we need more frames than horizon, just return all frames
            indices = np.arange(horizon)
        else:
            # Evenly space frames across the horizon
            indices = np.linspace(0, horizon - 1, num_frames, dtype=int)

        return indices

    def frame_index_to_trajectory_data(
        self, frame_index: FrameIndex, horizon: Optional[int] = None
    ) -> tuple[TrajectoryData, List[int]]:
        """
        Convert a FrameIndex to TrajectoryData by reading the necessary frames.

        This method efficiently reads only the frames needed:
        - Video streams (rgbs, depths, masks): Only reads sampled frames (num_frames frames)
        - Metadata (target_qpos, qpos): Reads full horizon (horizon frames)
        - Camera metadata (intrinsics, extrinsics): Reads full horizon then slices to sampled frames

        Args:
            frame_index: FrameIndex containing dataset_idx, traj_id, frame_id, and camera_keys
            horizon: Optional specific horizon to use. If None, uses frame_index["sampled_horizon"]
                     or self.horizon.

        Returns:
            Tuple of (TrajectoryData, video_frame_indices)
        """
        dataset_idx = frame_index["dataset_idx"]
        traj_id = frame_index["traj_id"]
        frame_id = frame_index["frame_id"]
        camera_keys = frame_index["camera_keys"]

        if horizon is None:
            horizon = frame_index.get("sampled_horizon", self.horizon)

        feasible_horizon = frame_index.get("feasible_horizon", horizon)
        if int(horizon) > int(feasible_horizon):
            raise ValueError(
                f"sampled_horizon ({horizon}) exceeds feasible_horizon ({feasible_horizon})"
            )

        # Get the dataset
        dataset = self.datasets[dataset_idx]

        interaction_frame_indices = frame_index.get("interaction_frame_indices")

        if interaction_frame_indices is not None:
            # Interaction-only mode: horizon advances through interaction frames
            # interaction_frame_indices contains the interaction frames for this horizon window
            interact_arr = np.array(interaction_frame_indices)
            # Use _compute_sampled_frame_indices to pick which interaction frames to sample
            sampled_indices_relative = self._compute_sampled_frame_indices(
                self.num_frames, len(interact_arr)
            )
            # Map relative indices to actual frame indices via the interaction frame list
            video_frame_indices = interact_arr[sampled_indices_relative].tolist()
            # Full horizon = all interaction frames in the window (for metadata)
            full_horizon_indices = interact_arr.tolist()
            sampled_indices_relative = np.arange(len(video_frame_indices))
        else:
            stride = self.dataset_horizon_strides[dataset_idx]
            # Compute full horizon indices for target_qpos and qpos
            full_horizon_indices = list(
                range(frame_id, frame_id + (horizon - 1) * stride + 1, stride)
            )

            # Compute sampled frame indices within the horizon (relative to full horizon)
            sampled_indices_relative = self._compute_sampled_frame_indices(
                self.num_frames, horizon
            )

            # Convert to absolute frame indices for video streams (camera data)
            video_frame_indices = [
                full_horizon_indices[i] for i in sampled_indices_relative
            ]

            if len(full_horizon_indices) != int(horizon):
                raise RuntimeError(
                    f"Horizon mismatch: expected {horizon}, got {len(full_horizon_indices)}"
                )

        # Build video_keys list for all camera-related data.
        # Only request wo_robot RGB streams when mode needs them.
        rgb_mode = self._get_rgb_mode_for_dataset(dataset_idx)
        if rgb_mode == "wo_robot":
            rgb_suffixes = ["_rgb.wo_robot"]
        elif rgb_mode in {"both", "random"}:
            # Prefer base first; wo_robot is optional in read_trajectory.
            rgb_suffixes = ["_rgb", "_rgb.wo_robot"]
        else:
            rgb_suffixes = ["_rgb"]
        video_suffixes = rgb_suffixes + ["_foreground_mask"]
        video_suffixes.extend(
            [
                suffix
                for suffix in ["_depth", "_robot_mask", "_static_mask"]
                if suffix in self.optional_video_suffixes
            ]
        )
        video_keys = []
        for cam_key in camera_keys:
            for suffix in video_suffixes:
                video_keys.append(f"{cam_key}{suffix}")

        # Build metadata_keys list
        metadata_keys = [
            "target_qpos",
            "qpos",
            "root_poses",
            "success",
            "task_description",
        ]
        if self.read_eef_pose:
            metadata_keys.append("eef_pose")
        if self.read_object_poses:
            metadata_keys.append("object_poses")
        for cam_key in camera_keys:
            metadata_keys.append(f"{cam_key}_intrinsics")
            metadata_keys.append(f"{cam_key}_extrinsics")

        # Read trajectory data efficiently
        img_size = self.cfg.get("img_size", None)
        if img_size == 0:
            img_size = None
        traj_data = dataset.read_trajectory(
            traj_id=traj_id,
            video_keys=video_keys,
            metadata_keys=metadata_keys,
            frame_indices=video_frame_indices,
            metadata_frame_indices=full_horizon_indices,
            img_size=img_size,
        )

        for key in list(traj_data.video_streams.keys()):
            if "_depth" in self.optional_video_suffixes and "_depth" in key:
                traj_data.video_streams[key] = (
                    traj_data.video_streams[key].astype(np.float32) / 1000.0
                )
            if self.bg_mask_top > 0.0 and key.endswith("_foreground_mask"):
                traj_data.video_streams[key] = self._apply_top_bg_to_mask(
                    traj_data.video_streams[key], self.bg_mask_top
                )

        # Slice camera intrinsics/extrinsics from full horizon to sampled frames
        for key in list(traj_data.metadata.keys()):
            if key.endswith("_intrinsics") or key.endswith("_extrinsics"):
                metadata_value = traj_data.metadata[key]
                if (
                    isinstance(metadata_value, np.ndarray)
                    and len(metadata_value.shape) > 0
                ):
                    metadata_indices = sampled_indices_relative
                    if metadata_value.shape[0] >= len(metadata_indices):
                        traj_data.metadata[key] = metadata_value[metadata_indices]
                    else:
                        traj_data.metadata[key] = metadata_value
                elif (
                    isinstance(metadata_value, np.ndarray)
                    and metadata_value.shape[0] > 0
                ):
                    traj_data.metadata[key] = metadata_value
            elif key in [
                "target_qpos",
                "qpos",
                "root_poses",
                "eef_pose",
                "object_poses",
            ]:
                metadata_value = traj_data.metadata.get(key)
                if metadata_value is not None:
                    traj_data.metadata[key] = metadata_value

        return traj_data, video_frame_indices

    @staticmethod
    def _pad_array_to_horizon(array: np.ndarray, target_horizon: int) -> np.ndarray:
        current_horizon = int(array.shape[0])
        if current_horizon == target_horizon:
            return array
        if current_horizon > target_horizon:
            raise ValueError(
                f"Cannot pad array of horizon {current_horizon} to smaller target {target_horizon}"
            )
        if current_horizon <= 0:
            raise ValueError("Cannot border-pad empty array with horizon <= 0")
        pad_len = target_horizon - current_horizon
        border = array[-1:]
        pad = np.repeat(border, repeats=pad_len, axis=0)
        return np.concatenate([array, pad], axis=0)

    @staticmethod
    def _pad_tensor_to_horizon(tensor: Tensor, target_horizon: int) -> Tensor:
        current_horizon = int(tensor.shape[0])
        if current_horizon == target_horizon:
            return tensor
        if current_horizon > target_horizon:
            raise ValueError(
                f"Cannot pad tensor of horizon {current_horizon} to smaller target {target_horizon}"
            )
        if current_horizon <= 0:
            raise ValueError("Cannot border-pad empty tensor with horizon <= 0")
        pad_len = target_horizon - current_horizon
        border = tensor[-1:].expand(pad_len, *tensor.shape[1:])
        return torch.cat([tensor, border], dim=0)

    @staticmethod
    def _get_eef_pose_or_default(
        metadata: Dict[str, object], horizon: int
    ) -> np.ndarray:
        value = metadata.get("eef_pose", None)
        if isinstance(value, np.ndarray):
            return value
        return np.zeros((int(horizon), 7), dtype=np.float32)

    @staticmethod
    def _apply_top_bg_to_mask(mask: np.ndarray, top_ratio: float) -> np.ndarray:
        """Set the top ratio of a foreground mask to background (0)."""
        if top_ratio <= 0.0:
            return mask
        if mask.ndim < 2:
            return mask

        height = int(mask.shape[-2])
        top_height = int(height * float(top_ratio))
        if top_height <= 0:
            return mask

        masked = mask.copy()
        masked[..., :top_height, :] = 0
        return masked

    def _maybe_pad_sample_horizon(
        self, sample: Dict[str, object], sampled_horizon: int
    ) -> None:
        if not self.pad_horizon_to_max:
            return

        target_horizon = int(self.horizon)
        if sampled_horizon > target_horizon:
            raise ValueError(
                f"sampled_horizon ({sampled_horizon}) exceeds target_horizon ({target_horizon})"
            )

        horizon_is_pad = np.zeros((target_horizon,), dtype=np.bool_)
        horizon_is_pad[sampled_horizon:] = True
        sample["horizon_is_pad"] = torch.from_numpy(horizon_is_pad)

        for key in ["target_qpos", "qpos", "root_poses", "eef_pose", "object_poses"]:
            value = sample.get(key)
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                sample[key] = self._pad_array_to_horizon(value, target_horizon)
            elif torch.is_tensor(value):
                sample[key] = self._pad_tensor_to_horizon(value, target_horizon)
            else:
                raise TypeError(
                    f"Unsupported type for horizon padding key '{key}': {type(value)}"
                )

    @staticmethod
    def collate_fn(batch: List[TrajectorySample]) -> TrajectoryBatch:
        """
        Custom collate function to handle variable-sized keys by keeping them as lists.
        """
        # In padded mode, horizon-dependent tensors can be stacked.
        pad_enabled = "horizon_is_pad" in batch[0]
        list_keys = ["robot_infos", "task_description", "object_poses"]
        if not pad_enabled:
            list_keys.extend(["eef_pose", "qpos", "root_poses", "target_qpos"])
        else:
            list_keys.extend(["target_qpos", "qpos"])

        # Separate the batch into parts
        collated = {}
        batch_dicts = {key: [] for key in list_keys}
        remaining_batch = []

        for sample in batch:
            # Extract list keys
            for key in list_keys:
                if key in sample and sample[key] is not None:
                    batch_dicts[key].append(sample[key])

            # Create a copy of the sample without list keys for default collation
            remaining_sample = {k: v for k, v in sample.items() if k not in list_keys}
            remaining_batch.append(remaining_sample)

        # Use default collation for the remaining parts (images, tensors, etc.)
        from torch.utils.data import default_collate

        collated = default_collate(remaining_batch)

        # Add the list keys back to the collated dictionary
        collated.update(batch_dicts)

        return collated

    def _select_rgb_key(
        self,
        video_streams: Dict[str, np.ndarray],
        cam_key: str,
        mode: str,
        selected_variant: Optional[str] = None,
    ) -> str:
        base_key = f"{cam_key}_rgb"
        wo_key = f"{cam_key}_rgb.wo_robot"
        if selected_variant is not None:
            return self._select_rgb_variant_key(
                video_streams,
                cam_key,
                variant=selected_variant,
                fallback_variant=self._fallback_rgb_variant(mode, selected_variant),
            )

        if mode == "random":
            selected_variant = self._resolve_sample_rgb_variant(
                mode, base_prob=self.rgb_random_base_prob
            )
            return self._select_rgb_variant_key(
                video_streams,
                cam_key,
                variant=selected_variant,
                fallback_variant=self._fallback_rgb_variant(mode, selected_variant),
            )

        if mode == "wo_robot":
            candidates = [wo_key]
        elif mode == "both":
            # Keep backward-compatible default for training inputs.
            candidates = [base_key, wo_key]
        else:
            candidates = [base_key]
        for k in candidates:
            if k in video_streams:
                return k
        raise KeyError(f"Missing RGB stream for camera={cam_key}, mode={mode}")

    @staticmethod
    def _select_rgb_variant_key(
        video_streams: Dict[str, np.ndarray],
        cam_key: str,
        variant: str,
        fallback_variant: str | None = None,
    ) -> str:
        """
        Select an explicit RGB variant key.

        Args:
            variant: "base" or "wo_robot"
            fallback_variant: optional fallback variant when primary is missing.
        """
        if variant not in {"base", "wo_robot"}:
            raise ValueError(f"Invalid RGB variant: {variant}")
        if fallback_variant is not None and fallback_variant not in {
            "base",
            "wo_robot",
        }:
            raise ValueError(f"Invalid fallback RGB variant: {fallback_variant}")

        base_key = f"{cam_key}_rgb"
        wo_key = f"{cam_key}_rgb.wo_robot"
        key_map = {"base": base_key, "wo_robot": wo_key}
        primary = key_map[variant]
        if primary in video_streams:
            return primary
        if fallback_variant is not None:
            fallback = key_map[fallback_variant]
            if fallback in video_streams:
                return fallback
        raise KeyError(
            f"Missing RGB stream for camera={cam_key}, variant={variant}, fallback={fallback_variant}"
        )

    def iter_trajectory_frames(self, dataset_idx: int, traj_id: str):
        """Iterate over all frames of a specific trajectory, yielding one TrajectorySample per frame.

        This is a generator that reads one frame at a time (horizon=1, num_frames=1),
        suitable for batch encoding inference.

        Args:
            dataset_idx: Index into self.datasets
            traj_id: Trajectory ID string

        Yields:
            TrajectorySample for each frame in the trajectory (sequentially)
        """
        # Find the TrajectoryInfo for this trajectory
        traj_info = None
        for info in self.trajectory_info:
            if info["dataset_idx"] == dataset_idx and info["traj_id"] == traj_id:
                traj_info = info
                break
        if traj_info is None:
            raise ValueError(
                f"Trajectory {traj_id} not found in dataset_idx={dataset_idx}"
            )

        camera_keys = traj_info.get("camera_keys", [])
        if not camera_keys:
            dataset = self.datasets[dataset_idx]
            keys = dataset.list_keys(traj_id)
            video_suffixes = [
                "_rgb.wo_robot",
                "_rgb",
                "_depth",
                "_robot_mask",
                "_foreground_mask",
                "_static_mask",
            ]
            camera_names = set()
            for key in keys:
                for suffix in video_suffixes:
                    if key.endswith(suffix):
                        camera_names.add(key[: -len(suffix)])
                        break
            camera_keys = sorted(list(camera_names))

        min_f = traj_info.get("min_start_frame", 0)
        max_f = traj_info["num_frames"]

        for frame_id in range(min_f, max_f):
            fidx: FrameIndex = {
                "dataset_idx": dataset_idx,
                "traj_id": traj_id,
                "frame_id": frame_id,
                "camera_keys": camera_keys,
                "sampled_horizon": 1,
                "feasible_horizon": 1,
                "interaction_frame_indices": None,
            }

            traj_data, video_frame_indices = self.frame_index_to_trajectory_data(
                fidx, horizon=1
            )

            num_cameras = len(camera_keys)

            if not traj_data.video_streams:
                continue

            first_rgb_key = None
            mode = self._get_rgb_mode_for_dataset(dataset_idx)
            selected_variant = self._resolve_sample_rgb_variant(
                mode, base_prob=self.rgb_random_base_prob
            )
            for cam_key in camera_keys:
                try:
                    first_rgb_key = self._select_rgb_key(
                        traj_data.video_streams,
                        cam_key,
                        mode=mode,
                        selected_variant=selected_variant,
                    )
                    break
                except KeyError:
                    continue
            if first_rgb_key is None:
                continue

            rgb_data = traj_data.video_streams[first_rgb_key]
            if len(rgb_data.shape) == 4 and rgb_data.shape[-1] == 3:
                H, W = rgb_data.shape[1], rgb_data.shape[2]
            else:
                continue

            n_frames = 1
            rgbs = np.zeros((n_frames, num_cameras, 3, H, W), dtype=np.uint8)
            rgbs_wo_robot = (
                np.zeros((n_frames, num_cameras, 3, H, W), dtype=np.uint8)
                if mode == "both"
                else None
            )
            depths = np.zeros((n_frames, num_cameras, H, W), dtype=np.float32)
            intr = np.zeros((n_frames, num_cameras, 3, 3), dtype=np.float32)
            w2c_arr = np.zeros((n_frames, num_cameras, 4, 4), dtype=np.float32)
            foreground_masks = np.zeros((n_frames, num_cameras, H, W), dtype=np.uint8)
            static_masks = np.zeros((n_frames, num_cameras, H, W), dtype=np.uint8)
            robot_masks = np.zeros((n_frames, num_cameras, H, W), dtype=np.uint8)

            for cam_idx, cam_key in enumerate(camera_keys):
                intr_data = traj_data.metadata[f"{cam_key}_intrinsics"]
                intr[:, [cam_idx]] = intr_data

                ext_data = traj_data.metadata[f"{cam_key}_extrinsics"]
                c2w_4x4 = convert_extrinsics_3x4_to_4x4(ext_data)
                w2c_arr[:, [cam_idx]] = c2w_4x4

                rgb = traj_data.video_streams[
                    self._select_rgb_key(
                        traj_data.video_streams,
                        cam_key,
                        mode=mode,
                        selected_variant=selected_variant,
                    )
                ]
                rgbs[:, cam_idx] = np.transpose(rgb, (0, 3, 1, 2))
                if rgbs_wo_robot is not None:
                    rgb_wo = traj_data.video_streams[
                        self._select_rgb_variant_key(
                            traj_data.video_streams,
                            cam_key,
                            variant="wo_robot",
                            fallback_variant="base",
                        )
                    ]
                    rgbs_wo_robot[:, cam_idx] = np.transpose(rgb_wo, (0, 3, 1, 2))

                mask = traj_data.video_streams[f"{cam_key}_foreground_mask"]
                foreground_masks[:, cam_idx] = mask.astype(np.uint8)

                if "_depth" in self.optional_video_suffixes:
                    depth = traj_data.video_streams[f"{cam_key}_depth"]
                    depths[:, cam_idx] = depth.astype(np.float32)

                if "_static_mask" in self.optional_video_suffixes:
                    mask = traj_data.video_streams[f"{cam_key}_static_mask"]
                    static_masks[:, cam_idx] = mask.astype(np.uint8)

                if "_robot_mask" in self.optional_video_suffixes:
                    mask = traj_data.video_streams[f"{cam_key}_robot_mask"]
                    robot_masks[:, cam_idx] = mask.astype(np.uint8)

            robot_id_value = (
                -1
                if selected_variant == "wo_robot"
                else self.dataset_robot_ids[dataset_idx]
            )

            sample = {
                "rgbs": rgbs,
                "depths": depths,
                "intrinsics": intr,
                "w2c": w2c_arr,
                "foreground_masks": foreground_masks,
                "static_masks": static_masks,
                "robot_masks": robot_masks,
                "target_qpos": traj_data.metadata["target_qpos"],
                "qpos": traj_data.metadata["qpos"],
                "root_poses": traj_data.metadata["root_poses"],
                "eef_pose": self._get_eef_pose_or_default(traj_data.metadata, 1),
                "success": np.array([traj_data.success], dtype=np.bool_),
                "task_description": traj_data.metadata["task_description"],
                "traj_id": self._traj_id_to_tensor(traj_id),
                "robot_infos": self.dataset_robot_infos[dataset_idx],
                "frame_id": torch.tensor(video_frame_indices, dtype=torch.int32)
                - min_f,
                "max_frames": torch.tensor(
                    [max_f] * len(video_frame_indices), dtype=torch.int32
                )
                - min_f,
                "horizon": torch.tensor([1], dtype=torch.int32),
                "task_ind": torch.tensor(
                    [self.dataset_task_inds[dataset_idx]], dtype=torch.int32
                )[0],
                "robot_id": torch.tensor([robot_id_value], dtype=torch.int32)[0],
            }
            if rgbs_wo_robot is not None:
                sample["rgbs_wo_robot"] = rgbs_wo_robot
            self._maybe_pad_sample_horizon(sample, sampled_horizon=1)
            yield sample

            self._maybe_pad_sample_horizon(sample, sampled_horizon=1)
            yield sample

    # ── Reproducible Eval Index API ─────────────────────────────────────────

    def capture_eval_handle(self, index: int) -> EvalSampleHandle:
        """Deterministically resolve a global dataset index into a fully-specified
        :class:`EvalSampleHandle` that can be persisted and replayed later.

        All stochastic decisions (camera subset, RGB variant, horizon length,
        interaction frame window) are resolved at capture time using a
        deterministic RNG seeded from *index*, so the same index always produces
        the same handle regardless of the global numpy RNG state.

        Args:
            index: Global dataset index (must be >= 0).

        Returns:
            An :class:`EvalSampleHandle` dict that can be serialised to JSON.
        """
        if index < 0:
            raise ValueError(
                "capture_eval_handle requires index >= 0 for determinism"
            )

        # 1. Deterministic frame index (reuses existing deterministic path)
        fidx = self.sample_frame_index(index)

        # 2. Deterministic RNG for remaining stochastic choices
        traj_info = next(
            info
            for info in self.trajectory_info
            if info["traj_id"] == fidx["traj_id"]
            and info["dataset_idx"] == fidx["dataset_idx"]
        )
        seed = (index ^ hash(fidx["traj_id"])) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)

        # 3. Camera subset (mirrors __getitem__ logic but deterministic)
        camera_keys = list(fidx["camera_keys"])  # copy
        if self.cameras:
            filtered = [c for c in self.cameras if c in camera_keys]
            if filtered:
                camera_keys = filtered
        if self.max_num_cameras > 0 and len(camera_keys) > self.max_num_cameras:
            sampled_indices = rng.choice(
                len(camera_keys), size=self.max_num_cameras, replace=False
            )
            camera_keys = [camera_keys[int(i)] for i in sorted(sampled_indices)]

        # 4. RGB variant
        dataset_idx = fidx["dataset_idx"]
        mode = self._get_rgb_mode_for_dataset(dataset_idx)
        if mode == "random":
            rgb_variant = "base" if rng.rand() < self.rgb_random_base_prob else "wo_robot"
        elif mode == "wo_robot":
            rgb_variant = "wo_robot"
        else:
            rgb_variant = "base"

        return EvalSampleHandle(
            version=EVAL_HANDLE_VERSION,
            dataset_idx=dataset_idx,
            group_name=traj_info["group_name"],
            traj_id=fidx["traj_id"],
            frame_id=fidx["frame_id"],
            sampled_horizon=fidx["sampled_horizon"],
            feasible_horizon=fidx["feasible_horizon"],
            camera_keys=camera_keys,
            rgb_variant=rgb_variant,
            interaction_frame_indices=fidx.get("interaction_frame_indices"),
            dataset_root=self.datasets[dataset_idx].root,
        )

    def save_eval_index(
        self, handles: List[EvalSampleHandle], filepath: str
    ) -> None:
        """Serialise a list of :class:`EvalSampleHandle` to a JSON file.

        The output includes a top-level ``meta`` block with schema version and
        dataset config fingerprint so that stale index files can be detected on
        reload.

        Args:
            handles: List of handles (from :meth:`capture_eval_handle`).
            filepath: Destination JSON path. Parent directories are created
                automatically.
        """
        import hashlib

        config_fingerprint = hashlib.sha256(
            json.dumps(
                {k: str(v) for k, v in sorted(self.cfg.items())}, sort_keys=True
            ).encode()
        ).hexdigest()[:16]

        payload = {
            "meta": {
                "version": EVAL_HANDLE_VERSION,
                "config_fingerprint": config_fingerprint,
                "num_samples": len(handles),
                "created_at": time.time(),
            },
            "samples": [dict(h) for h in handles],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {len(handles)} eval handles to {filepath}")

    @staticmethod
    def load_eval_index(filepath: str) -> List[EvalSampleHandle]:
        """Load a list of :class:`EvalSampleHandle` from a JSON file previously
        written by :meth:`save_eval_index`.

        Basic schema validation is performed; a ``ValueError`` is raised if the
        file version is incompatible or required fields are missing.

        Args:
            filepath: Path to the JSON eval-index file.

        Returns:
            List of :class:`EvalSampleHandle` dicts.
        """
        with open(filepath, "r") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            # Legacy flat list format
            samples = payload
        elif isinstance(payload, dict):
            meta = payload.get("meta", {})
            file_version = meta.get("version", 1)
            if file_version > EVAL_HANDLE_VERSION:
                raise ValueError(
                    f"Eval-index file version {file_version} is newer than "
                    f"supported version {EVAL_HANDLE_VERSION}. Please upgrade."
                )
            samples = payload.get("samples", [])
        else:
            raise ValueError("Unexpected eval-index file format")

        required_keys = {
            "dataset_idx", "traj_id", "frame_id", "sampled_horizon",
            "camera_keys", "rgb_variant",
        }
        for i, s in enumerate(samples):
            missing = required_keys - set(s.keys())
            if missing:
                raise ValueError(
                    f"Eval handle at index {i} is missing required keys: {missing}"
                )

        return samples

    def getitem_from_handle(
        self, handle: EvalSampleHandle, apply_augmentation: bool = False
    ) -> TrajectorySample:
        """Replay a persisted :class:`EvalSampleHandle` to produce the
        corresponding :class:`TrajectorySample` deterministically.

        This bypasses all random sampling in the normal ``__getitem__`` path.
        The handle's ``camera_keys``, ``rgb_variant``, ``sampled_horizon``, and
        ``interaction_frame_indices`` are used directly.

        Args:
            handle: A previously captured or loaded eval handle.
            apply_augmentation: Whether to apply the dataset's augmentation
                pipeline. Defaults to ``False`` for deterministic evaluation.

        Returns:
            A :class:`TrajectorySample` dictionary identical in structure to
            what ``__getitem__`` returns.
        """
        dataset_idx = handle["dataset_idx"]
        if dataset_idx >= len(self.datasets):
            raise ValueError(
                f"dataset_idx={dataset_idx} out of range "
                f"(have {len(self.datasets)} datasets). Index file may be stale."
            )

        # Validate dataset root hasn't changed
        expected_root = handle.get("dataset_root")
        actual_root = self.datasets[dataset_idx].root
        if expected_root and os.path.abspath(expected_root) != os.path.abspath(actual_root):
            raise ValueError(
                f"Dataset root mismatch for dataset_idx={dataset_idx}: "
                f"expected {expected_root}, got {actual_root}"
            )

        # Build FrameIndex from handle
        fidx: FrameIndex = {
            "dataset_idx": dataset_idx,
            "traj_id": handle["traj_id"],
            "frame_id": handle["frame_id"],
            "camera_keys": list(handle["camera_keys"]),
            "sampled_horizon": handle["sampled_horizon"],
            "feasible_horizon": handle.get("feasible_horizon", handle["sampled_horizon"]),
            "interaction_frame_indices": handle.get("interaction_frame_indices"),
        }

        traj_data, video_frame_indices = self.frame_index_to_trajectory_data(fidx)

        camera_keys = fidx["camera_keys"]
        num_cameras = len(camera_keys)

        if not traj_data.video_streams:
            raise ValueError("No video streams found in trajectory data")

        # Use the persisted rgb_variant directly
        selected_variant = handle["rgb_variant"]
        mode = self._get_rgb_mode_for_dataset(dataset_idx)

        first_video = list(traj_data.video_streams.values())[0]
        num_frames = first_video.shape[0]

        first_rgb_key = None
        for cam_key in camera_keys:
            try:
                first_rgb_key = self._select_rgb_key(
                    traj_data.video_streams,
                    cam_key,
                    mode=mode,
                    selected_variant=selected_variant,
                )
                break
            except KeyError:
                continue
        if first_rgb_key is None:
            raise ValueError(f"No RGB video stream found for cameras: {camera_keys}")

        rgb_data = traj_data.video_streams[first_rgb_key]
        if len(rgb_data.shape) == 4 and rgb_data.shape[-1] == 3:
            H, W = rgb_data.shape[1], rgb_data.shape[2]
        else:
            raise ValueError(f"Unexpected RGB shape: {rgb_data.shape}")

        # Optional augmentation
        aug_replay = None
        if apply_augmentation and self.transform is not None:
            ref_img = rgb_data[0]
            aug_data = self.transform(image=ref_img)
            aug_replay = aug_data["replay"]

        # Allocate output arrays
        rgbs = np.zeros((num_frames, num_cameras, 3, H, W), dtype=np.uint8)
        rgbs_wo_robot = (
            np.zeros((num_frames, num_cameras, 3, H, W), dtype=np.uint8)
            if mode == "both"
            else None
        )
        depths = np.zeros((num_frames, num_cameras, H, W), dtype=np.float32)
        intrinsics = np.zeros((num_frames, num_cameras, 3, 3), dtype=np.float32)
        w2c = np.zeros((num_frames, num_cameras, 4, 4), dtype=np.float32)
        foreground_masks = np.zeros((num_frames, num_cameras, H, W), dtype=np.uint8)
        static_masks = np.zeros((num_frames, num_cameras, H, W), dtype=np.uint8)
        robot_masks = np.zeros((num_frames, num_cameras, H, W), dtype=np.uint8)

        for cam_idx, cam_key in enumerate(camera_keys):
            intrinsics_data = traj_data.metadata[f"{cam_key}_intrinsics"]
            intrinsics[:, [cam_idx]] = intrinsics_data

            extrinsics_data = traj_data.metadata[f"{cam_key}_extrinsics"]
            c2w_4x4 = convert_extrinsics_3x4_to_4x4(extrinsics_data)
            w2c[:, [cam_idx]] = c2w_4x4

            rgb = traj_data.video_streams[
                self._select_rgb_key(
                    traj_data.video_streams, cam_key, mode=mode,
                    selected_variant=selected_variant,
                )
            ]
            rgb_wo = None
            if rgbs_wo_robot is not None:
                rgb_wo = traj_data.video_streams[
                    self._select_rgb_variant_key(
                        traj_data.video_streams, cam_key,
                        variant="wo_robot", fallback_variant="base",
                    )
                ]

            if aug_replay is not None:
                augmented_rgb = np.zeros_like(rgb)
                for t in range(rgb.shape[0]):
                    res = A.ReplayCompose.replay(aug_replay, image=rgb[t])
                    augmented_rgb[t] = res["image"]
                rgb = augmented_rgb
                if rgb_wo is not None:
                    augmented_rgb_wo = np.zeros_like(rgb_wo)
                    for t in range(rgb_wo.shape[0]):
                        res = A.ReplayCompose.replay(aug_replay, image=rgb_wo[t])
                        augmented_rgb_wo[t] = res["image"]
                    rgb_wo = augmented_rgb_wo

            rgbs[:, cam_idx] = np.transpose(rgb, (0, 3, 1, 2))
            if rgbs_wo_robot is not None and rgb_wo is not None:
                rgbs_wo_robot[:, cam_idx] = np.transpose(rgb_wo, (0, 3, 1, 2))

            mask = traj_data.video_streams[f"{cam_key}_foreground_mask"]
            foreground_masks[:, cam_idx] = mask.astype(np.uint8)

            if "_depth" in self.optional_video_suffixes:
                depth = traj_data.video_streams[f"{cam_key}_depth"]
                depths[:, cam_idx] = depth.astype(np.float32)

            if "_static_mask" in self.optional_video_suffixes:
                mask = traj_data.video_streams[f"{cam_key}_static_mask"]
                static_masks[:, cam_idx] = mask.astype(np.uint8)

            if "_robot_mask" in self.optional_video_suffixes:
                mask = traj_data.video_streams[f"{cam_key}_robot_mask"]
                robot_masks[:, cam_idx] = mask.astype(np.uint8)

        # Metadata
        traj_info = next(
            info
            for info in self.trajectory_info
            if info["traj_id"] == fidx["traj_id"]
            and info["dataset_idx"] == fidx["dataset_idx"]
        )
        min_f = traj_info["min_start_frame"]
        max_f = traj_info["num_frames"]

        object_poses = (
            traj_data.metadata.get("object_poses", None)
            if self.read_object_poses
            else None
        )
        robot_id_value = (
            -1 if selected_variant == "wo_robot"
            else self.dataset_robot_ids[dataset_idx]
        )

        sample = {
            "rgbs": rgbs,
            "depths": depths,
            "intrinsics": intrinsics,
            "w2c": w2c,
            "foreground_masks": foreground_masks,
            "static_masks": static_masks,
            "robot_masks": robot_masks,
            "target_qpos": traj_data.metadata["target_qpos"],
            "qpos": traj_data.metadata["qpos"],
            "root_poses": traj_data.metadata["root_poses"],
            "eef_pose": self._get_eef_pose_or_default(
                traj_data.metadata, int(fidx["sampled_horizon"])
            ),
            "object_poses": object_poses,
            "success": np.array([traj_data.success], dtype=np.bool_),
            "task_description": traj_data.metadata["task_description"],
            "traj_id": self._traj_id_to_tensor(fidx["traj_id"]),
            "robot_infos": self.dataset_robot_infos[dataset_idx],
            "frame_id": torch.tensor(video_frame_indices, dtype=torch.int32) - min_f,
            "max_frames": torch.tensor(
                [max_f] * len(video_frame_indices), dtype=torch.int32
            ) - min_f,
            "horizon": torch.tensor([fidx["sampled_horizon"]], dtype=torch.int32),
            "task_ind": torch.tensor(
                [self.dataset_task_inds[dataset_idx]], dtype=torch.int32
            ),
            "robot_id": torch.tensor([robot_id_value], dtype=torch.int32),
        }
        if rgbs_wo_robot is not None:
            sample["rgbs_wo_robot"] = rgbs_wo_robot
        self._maybe_pad_sample_horizon(
            sample, sampled_horizon=int(fidx["sampled_horizon"])
        )
        return sample

    def __getitem__(self, index: int) -> TrajectorySample:
        """Get a trajectory data by index. What shall be in the batch?

        - rgbs [num_frames, Cam, 3, H, W] - Sampled frames evenly spaced across horizon
        - depths [num_frames, Cam, H, W] - Sampled frames evenly spaced across horizon
        - intrinsics [num_frames, Cam, 3, 3] - Sampled frames evenly spaced across horizon
        - w2c [num_frames, Cam, 4, 4] - Sampled frames evenly spaced across horizon
        - foreground_masks [num_frames, Cam, H, W] - Sampled frames evenly spaced across horizon
        - static_masks [num_frames, Cam, H, W] - Sampled frames evenly spaced across horizon
        - robot_masks [num_frames, Cam, H, W] - Sampled frames evenly spaced across horizon
        - target_qpos [horizon, J] - Full horizon data (all frames)
        - qpos [horizon, J] - Full horizon data (all frames)
        - success: [1]
        - task_description: str
        """
        fidx = (
            self.sample_frame_index()
            if self.cfg.get("manual_limit", -1) > 0
            else self.sample_frame_index(index)
        )
        traj_data, video_frame_indices = self.frame_index_to_trajectory_data(fidx)

        camera_keys = fidx["camera_keys"]
        num_cameras = len(camera_keys)

        # Get number of frames from first video stream
        if not traj_data.video_streams:
            raise ValueError("No video streams found in trajectory data")

        first_video = list(traj_data.video_streams.values())[0]
        num_frames = first_video.shape[0]

        # Get spatial dimensions from RGB images
        first_rgb_key = None
        mode = self._get_rgb_mode_for_dataset(fidx["dataset_idx"])
        selected_variant = self._resolve_sample_rgb_variant(
            mode, base_prob=self.rgb_random_base_prob
        )
        for cam_key in camera_keys:
            try:
                first_rgb_key = self._select_rgb_key(
                    traj_data.video_streams,
                    cam_key,
                    mode=mode,
                    selected_variant=selected_variant,
                )
                break
            except KeyError:
                continue

        if first_rgb_key is None:
            raise ValueError(f"No RGB video stream found for cameras: {camera_keys}")

        rgb_data = traj_data.video_streams[first_rgb_key]
        # RGB data is [T, H, W, 3] from decord (already resized if img_size was specified)
        if len(rgb_data.shape) == 4 and rgb_data.shape[-1] == 3:
            H, W = rgb_data.shape[1], rgb_data.shape[2]
        else:
            raise ValueError(f"Unexpected RGB shape: {rgb_data.shape}")

        # Initialize augmentation parameters if enabled
        aug_replay = None
        if self.transform is not None:
            # Use the first frame to generate augmentation parameters
            # We need to make sure we have at least one frame
            ref_img = rgb_data[0]  # [H, W, 3]
            # Generating parameters
            aug_data = self.transform(image=ref_img)
            aug_replay = aug_data["replay"]

        # Initialize arrays for all cameras
        rgbs = np.zeros((num_frames, num_cameras, 3, H, W), dtype=np.uint8)
        rgbs_wo_robot = (
            np.zeros((num_frames, num_cameras, 3, H, W), dtype=np.uint8)
            if mode == "both"
            else None
        )
        depths = np.zeros((num_frames, num_cameras, H, W), dtype=np.float32)
        intrinsics = np.zeros((num_frames, num_cameras, 3, 3), dtype=np.float32)
        w2c = np.zeros((num_frames, num_cameras, 4, 4), dtype=np.float32)
        foreground_masks = np.zeros((num_frames, num_cameras, H, W), dtype=np.uint8)
        static_masks = np.zeros((num_frames, num_cameras, H, W), dtype=np.uint8)
        robot_masks = np.zeros((num_frames, num_cameras, H, W), dtype=np.uint8)

        # Extract data for each camera (resizing and intrinsics adjustment already done by read_trajectory)
        for cam_idx, cam_key in enumerate(camera_keys):
            intrinsics_data = traj_data.metadata[f"{cam_key}_intrinsics"]
            intrinsics[:, [cam_idx]] = intrinsics_data

            extrinsics_data = traj_data.metadata[f"{cam_key}_extrinsics"]
            c2w_4x4 = convert_extrinsics_3x4_to_4x4(extrinsics_data)
            # Invert C2W to get W2C
            # w2c[:, [cam_idx]] = np.linalg.inv(c2w_4x4)
            w2c[:, [cam_idx]] = c2w_4x4

            rgb = traj_data.video_streams[
                self._select_rgb_key(
                    traj_data.video_streams,
                    cam_key,
                    mode=mode,
                    selected_variant=selected_variant,
                )
            ]  # [T, H, W, 3]
            rgb_wo = None
            if rgbs_wo_robot is not None:
                rgb_wo = traj_data.video_streams[
                    self._select_rgb_variant_key(
                        traj_data.video_streams,
                        cam_key,
                        variant="wo_robot",
                        fallback_variant="base",
                    )
                ]

            if aug_replay is not None:
                # Apply consistent augmentation to all frames
                # Note: We iterate because A.ReplayCompose.replay expects a single image
                augmented_rgb = np.zeros_like(rgb)
                for t in range(rgb.shape[0]):
                    res = A.ReplayCompose.replay(aug_replay, image=rgb[t])
                    augmented_rgb[t] = res["image"]
                rgb = augmented_rgb
                if rgb_wo is not None:
                    augmented_rgb_wo = np.zeros_like(rgb_wo)
                    for t in range(rgb_wo.shape[0]):
                        res = A.ReplayCompose.replay(aug_replay, image=rgb_wo[t])
                        augmented_rgb_wo[t] = res["image"]
                    rgb_wo = augmented_rgb_wo

            rgbs[:, cam_idx] = np.transpose(rgb, (0, 3, 1, 2))
            if rgbs_wo_robot is not None and rgb_wo is not None:
                rgbs_wo_robot[:, cam_idx] = np.transpose(rgb_wo, (0, 3, 1, 2))

            mask = traj_data.video_streams[f"{cam_key}_foreground_mask"]  # [T, H, W]
            foreground_masks[:, cam_idx] = mask.astype(np.uint8)

            if "_depth" in self.optional_video_suffixes:
                depth = traj_data.video_streams[f"{cam_key}_depth"]  # [T, H, W]
                depths[:, cam_idx] = depth.astype(np.float32)

            if "_static_mask" in self.optional_video_suffixes:
                mask = traj_data.video_streams[f"{cam_key}_static_mask"]  # [T, H, W]
                static_masks[:, cam_idx] = mask.astype(np.uint8)

            if "_robot_mask" in self.optional_video_suffixes:
                mask = traj_data.video_streams[f"{cam_key}_robot_mask"]  # [T, H, W]
                robot_masks[:, cam_idx] = mask.astype(np.uint8)

        # Extract frame_id and max_frames
        traj_info = next(
            info
            for info in self.trajectory_info
            if info["traj_id"] == fidx["traj_id"]
            and info["dataset_idx"] == fidx["dataset_idx"]
        )
        min_f = traj_info["min_start_frame"]
        max_f = traj_info["num_frames"]

        object_poses = (
            traj_data.metadata.get("object_poses", None)
            if self.read_object_poses
            else None
        )
        robot_id_value = (
            -1
            if selected_variant == "wo_robot"
            else self.dataset_robot_ids[fidx["dataset_idx"]]
        )

        sample = {
            "rgbs": rgbs,
            "depths": depths,
            "intrinsics": intrinsics,
            "w2c": w2c,
            "foreground_masks": foreground_masks,
            "static_masks": static_masks,
            "robot_masks": robot_masks,
            "target_qpos": traj_data.metadata["target_qpos"],
            "qpos": traj_data.metadata["qpos"],
            "root_poses": traj_data.metadata["root_poses"],
            "eef_pose": self._get_eef_pose_or_default(
                traj_data.metadata, int(fidx["sampled_horizon"])
            ),
            "object_poses": object_poses,
            "success": np.array([traj_data.success], dtype=np.bool_),
            "task_description": traj_data.metadata["task_description"],
            "traj_id": self._traj_id_to_tensor(fidx["traj_id"]),
            "robot_infos": self.dataset_robot_infos[fidx["dataset_idx"]],
            "frame_id": torch.tensor(video_frame_indices, dtype=torch.int32) - min_f,
            "max_frames": torch.tensor(
                [max_f] * len(video_frame_indices), dtype=torch.int32
            )
            - min_f,
            "horizon": torch.tensor([fidx["sampled_horizon"]], dtype=torch.int32),
            "task_ind": torch.tensor(
                [self.dataset_task_inds[fidx["dataset_idx"]]], dtype=torch.int32
            ),
            "robot_id": torch.tensor([robot_id_value], dtype=torch.int32),
        }
        if rgbs_wo_robot is not None:
            sample["rgbs_wo_robot"] = rgbs_wo_robot
        self._maybe_pad_sample_horizon(
            sample, sampled_horizon=int(fidx["sampled_horizon"])
        )
        return sample


if __name__ == "__main__":
    """
    CLI utility to load TrajectoryDataset from a YAML config file.
    Useful for initializing/warming up the trajectory info cache.
    """
    import argparse
    import time
    import os
    import sys

    # Add project root to python path for relative imports if run directly
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    from utils.misc import load_config
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(
        description="Initialize TrajectoryDataset to build/warmup cache."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the config file (YAML/JSON)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Loading config from {args.config}")
    print("=" * 60)

    try:
        config_dict = load_config(args.config, debug=args.debug)
        cfg = edict(config_dict)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        sys.exit(1)

    # Initialize the train dataset
    if hasattr(cfg, "dataset") and cfg.dataset:
        print("\nInitializing TrajectoryDataset (train)...")
        start_time = time.perf_counter()

        train_args = dict(cfg.dataset.args)
        if "max_num_cameras" not in train_args:
            train_args = dict(cfg.dataset.full_args.args)
        train_dataset = TrajectoryDataset(None, **train_args)
        elapsed_time = time.perf_counter() - start_time

        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Total frames: {len(train_dataset)}")
        print(f"Total trajectories: {len(train_dataset.trajectory_info)}")
    else:
        print("\nNo 'dataset' configuration found.")

    # Initialize the validation dataset
    if hasattr(cfg, "val_dataset") and cfg.val_dataset:
        print("\nInitializing TrajectoryDataset (val)...")
        start_time = time.perf_counter()

        val_args = dict(cfg.val_dataset.args)
        val_dataset = TrajectoryDataset(None, **val_args)
        elapsed_time = time.perf_counter() - start_time

        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Total frames: {len(val_dataset)}")
        print(f"Total trajectories: {len(val_dataset.trajectory_info)}")
    else:
        print("\nNo 'val_dataset' configuration found.")

    print("\n" + "=" * 60)
    print("Initialization Complete")
    print("=" * 60)

    # ── Reproducible Eval Index Example ─────────────────────────────────
    #
    # The following shows how to capture, save, load, and replay
    # deterministic eval handles. This workflow is useful for building
    # fixed evaluation sets that produce identical data across runs.
    #
    #   1. capture_eval_handle(index) → EvalSampleHandle
    #   2. save_eval_index(handles, path) → JSON file
    #   3. load_eval_index(path) → list[EvalSampleHandle]
    #   4. getitem_from_handle(handle) → TrajectorySample
    #
    # Example (requires a dataset to be loaded above):
    #
    #   dataset = train_dataset  # or val_dataset
    #
    #   # Step 1: Capture handles for the first 8 dataset indices
    #   handles = [dataset.capture_eval_handle(i) for i in range(8)]
    #
    #   # Step 2: Save to disk
    #   dataset.save_eval_index(handles, "runs/eval_index/demo.json")
    #
    #   # Step 3: Load from disk (can be a different process / later run)
    #   loaded_handles = TrajectoryDataset.load_eval_index(
    #       "runs/eval_index/demo.json"
    #   )
    #
    #   # Step 4: Replay each handle deterministically (no augmentation)
    #   for h in loaded_handles:
    #       sample = dataset.getitem_from_handle(h)
    #       print(
    #           f"traj_id={h['traj_id']}, frame={h['frame_id']}, "
    #           f"horizon={h['sampled_horizon']}, cams={h['camera_keys']}, "
    #           f"rgb={h['rgb_variant']}, rgbs_shape={sample['rgbs'].shape}"
    #       )
    #
    # To run this example with a real dataset, uncomment the block below
    # and pass a config that has a 'dataset' section, e.g.:
    #
    #   python -m src.datasets.trajectory_dataset -c configs/generation/my_config.yaml
    #
    # ────────────────────────────────────────────────────────────────────

    if "train_dataset" in dir():
        n_eval = min(8, len(train_dataset))
        if n_eval > 0:
            print("\n" + "=" * 60)
            print(f"Eval-Index Demo: capturing {n_eval} handles")
            print("=" * 60)

            handles = [train_dataset.capture_eval_handle(i) for i in range(n_eval)]

            eval_index_path = os.path.join(
                os.path.dirname(cache_file) if cache_file else "runs",
                "eval_index_demo.json",
            )
            train_dataset.save_eval_index(handles, eval_index_path)

            loaded = TrajectoryDataset.load_eval_index(eval_index_path)
            print(f"Loaded {len(loaded)} handles from {eval_index_path}")

            for i, h in enumerate(loaded):
                sample = train_dataset.getitem_from_handle(h)
                print(
                    f"  [{i}] traj={h['traj_id']}, frame={h['frame_id']}, "
                    f"horizon={h['sampled_horizon']}, cams={len(h['camera_keys'])}, "
                    f"rgb={h['rgb_variant']}, rgbs={sample['rgbs'].shape}"
                )
