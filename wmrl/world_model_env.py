"""World-model VecEnv that runs rollouts inside LatentActionFlowModelV2.

This env never touches the real simulator at training time.  It uses a recorded
ManiSkill dataset for initial conditions and ground-truth future frames for
dense rewards; all stepping happens through the frozen flow world-model →
latent decoder → image decoder stack.

Operates on a **single camera** (default ``front_lower_camera``).

Pipeline for one decision step
-------------------------------
1. Start with cached DINO tokens ``x_t`` (extracted once at reset).
2. Policy target_qpos chunk ``(N, K, A)`` → full_qpos via :class:`ActionNormalizer`.
3. Flow model sampled via Euler ODE → normalized latent tokens.
4. Latent decoder: ``(x_t, pred_latent)`` → predicted x_{t+K} DINO tokens.
5. Reward = ``-distance(pred_tokens, recorded_tokens)``.
6. Image decoder: predicted tokens → RGB observation for the policy.
7. ``t`` advances by ``K``; terminates when episode exhausted or ``max_chunk_steps``.

Observations are ``(n_obs_steps, 3, H, W)`` image histories (no camera dim).
Full-qpos state history is provided via ``StepResult.info["state_history"]``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich import print

from policies.action_normalizer import ActionNormalizer
from policies.dataset.maniskill_sequence_dataset import ManiSkillSequenceDataset
from wmrl.rng_utils import (
    FLOW_NOISE_STREAM_ID,
    RESET_SAMPLE_STREAM_ID,
    SYNC_RESET_STREAM_ID,
    fold_in_seed,
    sample_episode_start,
)
from wmrl.rl_types import StepResult
from src import models as _model_module
from src.datasets.trajectory_dataset import TASKS
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels
from utils.misc import fetch_state_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ROBOT_UID_MAP: Dict[int, str] = {
    0: "panda",
    1: "xarm6_robotiq",
    2: "ur10e_stick",
}


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _build_from_cfg(model_cfg: Any) -> nn.Module:
    """Instantiate a model from its ``{name, args}`` config block."""
    name = str(model_cfg.name)
    args = cast(
        Mapping[str, Any],
        OmegaConf.to_container(model_cfg.args, resolve=True) or {},
    )
    cls = getattr(_model_module, name)
    return cls(**args)


@torch.no_grad()
def _euler_sample(
    flow_model: nn.Module,
    noise: torch.Tensor,
    xt_tokens: torch.Tensor,
    task_inds: torch.Tensor,
    horizons: torch.Tensor,
    robot_ids: torch.Tensor,
    full_qpos_list: List[Optional[torch.Tensor]],
    steps: int = 10,
) -> torch.Tensor:
    """Euler ODE sampling: compute conditioning once, then iterate flow steps."""
    model = cast(Any, flow_model)
    cond = model.forward_cond(
        xt_tokens=xt_tokens,
        task_inds=task_inds,
        horizons=horizons,
        robot_ids=robot_ids,
        full_qpos_list=full_qpos_list,
    )
    dt = 1.0 / steps
    x = noise.clone()
    for i in range(steps):
        t_val = 1.0 - i / steps
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
        v = model.forward_flow(cond, x, t)
        x = x - v * dt
    return x


@torch.no_grad()
def _extract_dino_tokens(
    extractor: nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Extract DINO patch tokens from single-camera images.

    Args:
        extractor: A :class:`DINOv3FeatureExtractor`.
        images: ``(B, 3, H, W)`` float in [0, 1].

    Returns:
        tokens: ``(B, Lp, C)`` where ``Lp = pH * pW``.
        patch_hw: ``(pH, pW)``.
    """
    flat = images.float()
    # Guard feature extraction against rare non-finite pixels from IO/masks.
    flat = torch.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)
    if flat.max() > 1.5:
        flat = flat / 255.0
    flat = flat.clamp(0.0, 1.0)
    _, patch_grid = extractor(flat, return_spatial_grid=True)
    _, ch, ph, pw = patch_grid.shape
    tokens = patch_grid.flatten(2).transpose(1, 2).float()  # (B, Lp, C)
    tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
    return tokens, (int(ph), int(pw))


@torch.no_grad()
def _decode_tokens_to_image(
    decoder: nn.Module,
    tokens: torch.Tensor,
    patch_hw: Tuple[int, int],
    dino_channels: int,
    target_size: int,
) -> torch.Tensor:
    """Decode DINO tokens back to single-camera RGB.

    Args:
        decoder: Image decoder (expects ``(N, 1, Lp, C)`` → ``(N, 1, 3, H, W)``).
        tokens: ``(N, Lp, C)`` predicted DINO tokens.
        patch_hw: ``(pH, pW)`` spatial grid size.
        dino_channels: DINO feature dimensionality.
        target_size: Desired output resolution (square).

    Returns:
        ``(N, 3, target_size, target_size)`` float in [0, 1].
    """
    pH, pW = patch_hw
    N = tokens.shape[0]
    # Image decoder interface expects a camera dimension.
    x = tokens.reshape(N, 1, pH * pW, dino_channels)
    out = decoder(x, patch_hw=patch_hw).clamp(0.0, 1.0)  # (N, 1, 3, H', W')
    out = out.squeeze(1)  # (N, 3, H', W')
    if out.shape[-1] != target_size or out.shape[-2] != target_size:
        out = F.interpolate(
            out, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
    return out


# ---------------------------------------------------------------------------
# Running reward normalizer
# ---------------------------------------------------------------------------


class RunningRewardNormalizer:
    """EMA-based reward normalizer that maps raw feature distance to [0, 1].

    Tracks running ``d_min`` and ``d_max`` via exponential moving average and
    returns ``(d_max - d) / (d_max - d_min + epsilon)`` clamped to [0, 1].
    Higher reward = lower distance (closer to goal).
    """

    def __init__(self, ema_decay: float = 0.99, epsilon: float = 1e-8) -> None:
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self._d_min: Optional[float] = None
        self._d_max: Optional[float] = None

    def normalize(self, raw_err: torch.Tensor) -> torch.Tensor:
        """Map raw distance tensor to [0, 1] reward (higher = better).

        Args:
            raw_err: ``(N,)`` non-negative distance values.

        Returns:
            ``(N,)`` normalized reward in [0, 1].
        """
        batch_min = raw_err.min().item()
        batch_max = raw_err.max().item()

        if self._d_min is None:
            self._d_min = batch_min
            self._d_max = batch_max
        else:
            assert self._d_max is not None
            a = self.ema_decay
            self._d_min = a * self._d_min + (1.0 - a) * batch_min
            self._d_max = a * self._d_max + (1.0 - a) * batch_max

        span = self._d_max - self._d_min + self.epsilon
        reward = (self._d_max - raw_err.float()) / span
        return reward.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# FlowWorldModelVecEnv
# ---------------------------------------------------------------------------


class FlowWorldModelVecEnv:
    """VecEnv powered by a frozen latent-action flow world model.

    Operates on a single camera (no camera dimension in tensors).
    Observations: ``(n_obs_steps, 3, H, W)`` image histories.
    State history (full_qpos via ActionNormalizer): ``(n_obs_steps, state_dim)``.

    Only :meth:`step_chunked` is implemented.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        # --- dataset ---
        dataset_dir: str,
        camera: str = "front_lower_camera",
        img_size: int = 256,
        camera_width: Optional[int] = None,
        camera_height: Optional[int] = None,
        use_foreground_mask: bool = True,
        start_traj_id: int = 0,
        end_traj_id: int = 999,
        min_episode_length: int = 5,
        max_train_episodes: Optional[int] = None,
        # --- world model ---
        flow_model_work_dir: str = "",
        encoder_work_dir: str = "",
        image_decoder_work_dir: str = "",
        euler_steps: int = 10,
        # --- policy-facing shapes ---
        n_obs_steps: int = 2,
        chunk_size: int = 8,
        # --- reset / termination ---
        p_initial_frame: float = 0.5,
        max_chunk_steps: int = 6,
        seed: int = 0,
        episode_partition_rank: int = 0,
        episode_partition_size: int = 1,
        # --- robot / task ---
        robot_uid: str = "ur10e_stick",
        control_mode: str = "pd_joint_pos",
        task_ind: int = 0,
        # --- reward ---
        reward_token_metric: str = "l2",
        reward_mode: str = "corresponding",
        reward_scale: float = 1.0,
        success_token_threshold: float = 0.05,
        terminal_success_bonus: float = 0.0,
        # --- misc ---
        latent_scalar_normalization: float = 10.0,
        env_batch_num: int = 1,
        flow_seed: Optional[int] = None,
        global_env_offset: int = 0,
        worker_id: int = 0,
        **_unused,
    ):
        if not image_decoder_work_dir:
            raise ValueError("image_decoder_work_dir is required.")

        # Backward compat: accept `cameras=[...]` from old configs.
        if "cameras" in _unused and camera == "front_lower_camera":
            cams = _unused.pop("cameras")
            if isinstance(cams, (list, tuple)) and len(cams) > 0:
                camera = str(cams[0])

        # --- Store config ---
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.num_envs = int(num_envs)
        self.camera = str(camera)
        self.img_size = int(img_size)
        self.camera_width = int(camera_width) if camera_width else self.img_size
        self.camera_height = int(camera_height) if camera_height else self.img_size
        self.use_foreground_mask = bool(use_foreground_mask)
        self.n_obs_steps = int(n_obs_steps)
        self.chunk_size = int(chunk_size)
        self.p_initial_frame = float(p_initial_frame)
        self.max_chunk_steps = int(max_chunk_steps)
        self.euler_steps = int(euler_steps)
        self.reward_token_metric = str(reward_token_metric)
        self.reward_mode = self._normalize_reward_mode(reward_mode)
        self.reward_scale = float(reward_scale)
        self.success_token_threshold = float(success_token_threshold)
        self.terminal_success_bonus = float(terminal_success_bonus)
        self.latent_scalar_normalization = float(latent_scalar_normalization)
        self.robot_uid = str(robot_uid)
        self.control_mode = str(control_mode)
        self.task_ind = int(task_ind)
        self._base_seed = int(seed)
        self._flow_seed = int(flow_seed if flow_seed is not None else self._base_seed)
        self._global_env_offset = int(global_env_offset)
        self._worker_id = int(worker_id)
        print(f"[red]flow_seed = {self._flow_seed}[/red]")
        self._env_batch_num = int(env_batch_num)
        if self._env_batch_num < 1:
            raise ValueError("env_batch_num must be >= 1")
        if self.num_envs % self._env_batch_num != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by "
                f"env_batch_num ({self._env_batch_num})"
            )
        self._comp_batch_size = self.num_envs // self._env_batch_num

        self.episode_partition_rank = int(episode_partition_rank)
        self.episode_partition_size = int(episode_partition_size)
        if self.episode_partition_size < 1:
            raise ValueError("episode_partition_size must be >= 1")
        if not (0 <= self.episode_partition_rank < self.episode_partition_size):
            raise ValueError(
                "episode_partition_rank must satisfy "
                "0 <= rank < episode_partition_size"
            )

        # ------------------------------------------------------------------
        # 1. Dataset (metadata: episode boundaries, qpos, traj ids).
        # ------------------------------------------------------------------
        self.dataset = ManiSkillSequenceDataset(
            dataset_dir=dataset_dir,
            cameras=[self.camera],
            horizon=max(self.chunk_size + self.n_obs_steps + 1, 2),
            pad_before=0,
            pad_after=0,
            img_size=self.img_size,
            seed=self._base_seed,
            val_ratio=0.0,
            max_train_episodes=max_train_episodes,
            max_val_episodes=None,
            start_traj_id=start_traj_id,
            end_traj_id=end_traj_id,
            use_foreground_mask=self.use_foreground_mask,
            min_episode_length=max(min_episode_length, self.chunk_size + self.n_obs_steps + 1),
        )
        self._episode_lengths = self.dataset.episode_ends - self.dataset.episode_starts
        self._valid_ep_indices = np.where(
            self._episode_lengths > self.chunk_size + self.n_obs_steps
        )[0]
        if len(self._valid_ep_indices) == 0:
            raise RuntimeError(
                f"No episodes long enough (need > {self.chunk_size + self.n_obs_steps})"
            )
        # Reset sampling is keyed by base seed and global env id, so every
        # worker uses the same episode pool regardless of worker topology.
        self._sample_ep_indices = self._valid_ep_indices

        # ------------------------------------------------------------------
        # 2. Frozen world-model components.
        # ------------------------------------------------------------------
        self._flow_model_work_dir = flow_model_work_dir
        self._encoder_work_dir = encoder_work_dir
        self._image_decoder_work_dir = image_decoder_work_dir
        self._init_world_model()

        # ------------------------------------------------------------------
        # 3. ActionNormalizer: target_qpos → full_qpos conversion.
        # ------------------------------------------------------------------
        norm_uid = self.robot_uid
        task_name = TASKS[self.task_ind] if 0 <= self.task_ind < len(TASKS) else ""
        if ("PushT" in task_name or "RollBall" in task_name) and "ur10" not in norm_uid:
            norm_uid += "_closed"
        self.action_normalizer = ActionNormalizer(
            robot_uid=norm_uid,
            control_mode=cast(Any, self.control_mode),
            state_source="target_qpos",
            device=self.device,
        )

        # Robot ID for the flow model's qpos encoder.
        self._robot_id = next(
            (k for k, v in _DEFAULT_ROBOT_UID_MAP.items() if v == self.robot_uid), -1,
        )
        if self._robot_id < 0:
            raise ValueError(f"Unknown robot_uid={self.robot_uid}")

        # ------------------------------------------------------------------
        # 4. VecEnv protocol attributes.
        # ------------------------------------------------------------------
        action_dim = self.action_normalizer._action_dim
        if action_dim is None:
            raise RuntimeError("ActionNormalizer did not initialize action_dim")
        self.action_dim = int(action_dim)

        # State dim = full_qpos_dim from ActionNormalizer.
        self.state_dim = int(self.action_normalizer._spec.full_qpos_dim)

        # Obs shape: single-camera image history (no Cam dimension).
        self.obs_shape = (self.n_obs_steps, 3, self.camera_height, self.camera_width)
        self.state_obs_shape = (self.n_obs_steps, self.state_dim)

        # ------------------------------------------------------------------
        # 5. Per-env mutable state.
        # ------------------------------------------------------------------
        self._ep_idx = np.zeros(self.num_envs, dtype=np.int64)
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self._step_count = np.zeros(self.num_envs, dtype=np.int64)
        self._reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._sync_reset_count = 0
        self._global_env_ids = np.arange(
            self._global_env_offset,
            self._global_env_offset + self.num_envs,
            dtype=np.int64,
        )

        # Image history: (N, n_obs_steps, 3, H_obs, W_obs)
        self._obs_history = torch.zeros(
            (self.num_envs,) + self.obs_shape,
            dtype=torch.float32, device=self.device,
        )
        # Current WM-resolution frame: (N, 3, img_size, img_size)
        self._current_wm_frames = torch.zeros(
            (self.num_envs, 3, self.img_size, self.img_size),
            dtype=torch.float32, device=self.device,
        )
        # State history: (N, n_obs_steps, state_dim)
        self._state_history = torch.zeros(
            (self.num_envs,) + self.state_obs_shape,
            dtype=torch.float32, device=self.device,
        )
        # Cached DINO tokens of current frame: (N, Lp, dino_channels)
        self._cached_x_t_tokens: Optional[torch.Tensor] = None
        self._patch_hw: Optional[Tuple[int, int]] = None
        # Cached GT WM-resolution frames for render (set after reset/step).
        self._gt_wm_frames: Optional[torch.Tensor] = None
        # Goal (final) WM-resolution frames per env for success evaluation.
        self._goal_wm_frames = torch.zeros(
            (self.num_envs, 3, self.img_size, self.img_size),
            dtype=torch.float32, device=self.device,
        )
        # Cached DINO tokens of goal frames: (N, Lp, dino_channels)
        self._goal_tokens: Optional[torch.Tensor] = None

        # RNG for flow-model noise — derived per-env from
        # ``(flow_seed, global_env_id, step)`` in ``_sample_flow_noise``;
        # see that method for determinism guarantees.
        self._flow_step_counter = 0

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_world_model(self) -> None:
        """Load frozen world-model components. Subclasses can override."""
        flow_model_work_dir = self._flow_model_work_dir
        encoder_work_dir = self._encoder_work_dir
        image_decoder_work_dir = self._image_decoder_work_dir

        if not flow_model_work_dir or not encoder_work_dir:
            raise ValueError(
                "flow_model_work_dir and encoder_work_dir are required for FlowWorldModelVecEnv."
            )

        flow_cfg = OmegaConf.load(os.path.join(flow_model_work_dir, "config.yaml"))
        self.dino_channels = int(
            OmegaConf.select(flow_cfg, "vars.dino_channels", default=1024)
        )

        # Flow model, latent decoder, image decoder — all frozen.
        self.flow_model = self._load_frozen(flow_cfg.models, "flow_model", flow_model_work_dir)
        self.latent_decoder = self._load_frozen(flow_cfg.models, "decoder", encoder_work_dir)
        img_cfg = OmegaConf.load(os.path.join(image_decoder_work_dir, "config.yaml"))
        self.image_decoder = self._load_frozen(img_cfg.models, "decoder", image_decoder_work_dir)

        # DINO extractor (frozen).
        dino_name = get_dinov3_model_for_channels(self.dino_channels)
        self.dino_extractor = DINOv3FeatureExtractor(
            model_name=dino_name, use_compile=False,
        ).to(self.device)
        self._freeze(self.dino_extractor)

    def _load_frozen(self, models_cfg: Any, key: str, work_dir: str) -> nn.Module:
        """Build model from config, load checkpoint, freeze, move to device."""
        if key not in models_cfg:
            raise RuntimeError(f"Config at {work_dir} has no models.{key}")
        model = _build_from_cfg(models_cfg[key]).to(self.device)
        model.load_state_dict(
            fetch_state_dict(key, work_dir, self.device), strict=True,
        )
        self._freeze(model)
        return model

    @torch.no_grad()
    def _ensure_patch_hw(self, images: torch.Tensor) -> None:
        if self._patch_hw is not None:
            return
        _, self._patch_hw = self._extract_tokens(images[:1])

    @staticmethod
    def _freeze(model: nn.Module) -> None:
        """Set model to eval mode and disable gradients."""
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # DINO / image helpers (single camera, no Cam dimension)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_tokens(
        self, images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract DINO tokens on the policy device.

        Args:
            images: ``(B, 3, H, W)`` float in [0, 1].

        Returns:
            tokens ``(B, Lp, C)``, patch_hw ``(pH, pW)``.
        """
        B = images.shape[0]
        if self._env_batch_num <= 1 or B <= self._comp_batch_size:
            return _extract_dino_tokens(self.dino_extractor, images)
        token_parts: List[torch.Tensor] = []
        patch_hw: Optional[Tuple[int, int]] = None
        for chunk in images.split(self._comp_batch_size, dim=0):
            t, phw = _extract_dino_tokens(self.dino_extractor, chunk)
            token_parts.append(t)
            patch_hw = phw
        assert patch_hw is not None
        return torch.cat(token_parts, dim=0), patch_hw

    @torch.no_grad()
    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode DINO tokens to RGB on the policy device.

        Args:
            tokens: ``(N, Lp, C)``.

        Returns:
            ``(N, 3, img_size, img_size)`` float in [0, 1].
        """
        assert self._patch_hw is not None
        N = tokens.shape[0]
        if self._env_batch_num <= 1 or N <= self._comp_batch_size:
            return _decode_tokens_to_image(
                self.image_decoder, tokens, self._patch_hw,
                self.dino_channels, self.img_size,
            )
        parts: List[torch.Tensor] = []
        for chunk in tokens.split(self._comp_batch_size, dim=0):
            parts.append(_decode_tokens_to_image(
                self.image_decoder, chunk, self._patch_hw,
                self.dino_channels, self.img_size,
            ))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _resize_for_policy(self, images: torch.Tensor) -> torch.Tensor:
        """Resize ``(N, 3, H, W)`` to policy observation resolution."""
        if images.shape[-2] == self.camera_height and images.shape[-1] == self.camera_width:
            return images
        return F.interpolate(
            images, size=(self.camera_height, self.camera_width),
            mode="bilinear", align_corners=False,
        )

    # ------------------------------------------------------------------
    # Action normalizer: target_qpos → full_qpos
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _target_qpos_to_full_qpos(
        self, target_qpos: torch.Tensor,
    ) -> torch.Tensor:
        """Convert target_qpos ``(K, A)`` to full_qpos ``(K, full_qpos_dim)``.

        Uses :class:`ActionNormalizer` normalize → denormalize round-trip,
        following the pattern from ``DinoLatentActionFlowTrainer``.
        """
        batch = {"target_qpos": [target_qpos]}
        action = self.action_normalizer.normalize(cast(Any, batch))
        full_qpos = self.action_normalizer.denormalize(action, return_full_qpos=True)
        return full_qpos.squeeze(0).detach().float()  # (K, full_qpos_dim)

    # ------------------------------------------------------------------
    # Frame I/O (single camera)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _read_frame(self, traj_id: str, t: int) -> torch.Tensor:
        """Read one masked RGB frame for the single camera.

        Returns:
            ``(3, img_size, img_size)`` float in [0, 1] on ``self.device``.
        """
        rgb_key = f"{self.camera}_rgb"
        traj_rgb = self.dataset.traj_ds.read_trajectory(
            traj_id,
            video_keys=[rgb_key],
            metadata_keys=[],
            frame_indices=[t],
            img_size=self.img_size,
        )
        rgb = np.asarray(traj_rgb.video_streams[rgb_key]).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (0, 3, 1, 2))  # (1, 3, H, W)
        rgb_t = torch.from_numpy(rgb).to(self.device)

        if self.use_foreground_mask:
            mask_key = f"{self.camera}_foreground_mask"
            traj_mask = self.dataset.traj_ds.read_trajectory(
                traj_id,
                video_keys=[mask_key],
                metadata_keys=[],
                frame_indices=[t],
                img_size=None,
            )
            fg = np.asarray(traj_mask.video_streams[mask_key]).astype(np.float32)
            fg_t = torch.from_numpy(fg).to(self.device)
            if fg_t.shape[1] != rgb_t.shape[2] or fg_t.shape[2] != rgb_t.shape[3]:
                fg_t = F.interpolate(
                    fg_t.unsqueeze(1),
                    size=(rgb_t.shape[2], rgb_t.shape[3]),
                    mode="nearest",
                ).squeeze(1)
            rgb_t = rgb_t * fg_t.unsqueeze(1)

        rgb_t = torch.nan_to_num(rgb_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        return rgb_t.squeeze(0)  # (3, H, W)

    # ------------------------------------------------------------------
    # Episode sampling & reset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_episode_start(self, env_i: int) -> Tuple[int, int]:
        """Sample ``(episode_index, t_start)`` for one global env stream."""
        return sample_episode_start(
            self._sample_ep_indices,
            self._episode_lengths,
            self.chunk_size,
            self.p_initial_frame,
            self._base_seed,
            RESET_SAMPLE_STREAM_ID,
            int(self._global_env_ids[env_i]),
            int(self._reset_counts[env_i]),
        )

    @torch.no_grad()
    def _sample_sync_episode_start(self) -> Tuple[int, int]:
        """Sample a shared reset used when all envs must align."""
        return sample_episode_start(
            self._sample_ep_indices,
            self._episode_lengths,
            self.chunk_size,
            self.p_initial_frame,
            self._base_seed,
            SYNC_RESET_STREAM_ID,
            int(self._sync_reset_count),
        )

    @torch.no_grad()
    def _reset_one(self, env_i: int, force_ep_t: Optional[Tuple[int, int]] = None) -> None:
        """Reset a single env slot in-place."""
        if force_ep_t is None:
            ep_idx, t0 = self._sample_episode_start(env_i)
        else:
            ep_idx, t0 = force_ep_t
        self._ep_idx[env_i] = ep_idx
        self._t[env_i] = t0
        self._step_count[env_i] = 0
        self._reset_counts[env_i] += 1

        ep_start_abs = int(self.dataset.episode_starts[ep_idx])
        traj_id = self.dataset.traj_id_list[ep_idx]

        # Fill obs & state history: frames [t0-(n_obs_steps-1) .. t0].
        for step_i in range(self.n_obs_steps):
            target_t = max(0, t0 - (self.n_obs_steps - 1 - step_i))
            wm_frame = self._read_frame(traj_id, target_t)  # (3, H, W)
            obs_frame = self._resize_for_policy(wm_frame.unsqueeze(0)).squeeze(0)
            self._obs_history[env_i, step_i] = obs_frame

            # Dataset qpos is already full_qpos from the simulator.
            qpos_row = self.dataset.qpos[ep_start_abs + target_t]
            n = min(len(qpos_row), self.state_dim)
            self._state_history[env_i, step_i] = 0.0
            self._state_history[env_i, step_i, :n] = torch.from_numpy(
                qpos_row[:n]
            ).to(self.device)

        self._current_wm_frames[env_i] = self._read_frame(traj_id, t0)

        # Cache the final frame of this episode as the goal reference.
        final_t = int(self._episode_lengths[ep_idx]) - 1
        self._goal_wm_frames[env_i] = self._read_frame(traj_id, final_t)

    # ------------------------------------------------------------------
    # Cached token management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _refresh_cached_tokens(self) -> None:
        """Recompute DINO tokens from current WM frames on ``self.device``."""
        self._ensure_patch_hw(self._current_wm_frames)
        tokens, _ = self._extract_tokens(self._current_wm_frames)
        self._cached_x_t_tokens = tokens


    @torch.no_grad()
    def _refresh_cached_tokens_at(self, indices: np.ndarray) -> None:
        """Recompute DINO tokens only for specific env indices (after reset)."""
        assert self._cached_x_t_tokens is not None
        if len(indices) == 0:
            return
        self._ensure_patch_hw(self._current_wm_frames[indices])
        tokens, _ = self._extract_tokens(self._current_wm_frames[indices])
        idx_t = torch.from_numpy(indices).to(self.device)
        self._cached_x_t_tokens[idx_t] = tokens


    @torch.no_grad()
    def _refresh_goal_tokens(self) -> None:
        """Recompute DINO tokens of goal (final) frames for all envs."""
        self._ensure_patch_hw(self._goal_wm_frames)
        self._goal_tokens, _ = self._extract_tokens(self._goal_wm_frames)

    @torch.no_grad()
    def _refresh_goal_tokens_at(self, indices: np.ndarray) -> None:
        """Recompute goal DINO tokens only for specific env indices."""
        assert self._goal_tokens is not None
        self._ensure_patch_hw(self._goal_wm_frames[indices])
        tokens, _ = self._extract_tokens(self._goal_wm_frames[indices])
        self._goal_tokens[
            torch.from_numpy(indices).to(self.device)
        ] = tokens

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self, sync_all_envs: bool = False) -> torch.Tensor:
        """Reset all envs and return initial observations."""
        shared_ep_t: Optional[Tuple[int, int]] = None
        if sync_all_envs:
            shared_ep_t = self._sample_sync_episode_start()
            self._sync_reset_count += 1

        for i in range(self.num_envs):
            self._reset_one(i, force_ep_t=shared_ep_t)
        self._refresh_cached_tokens()
        self._refresh_goal_tokens()
        # Cache GT frames at reset position for render.
        self._gt_wm_frames = self._current_wm_frames.clone()
        return self._obs_history.clone()

    @torch.no_grad()
    def get_state_history(self) -> torch.Tensor:
        """Return state history ``(num_envs, n_obs_steps, state_dim)``."""
        return self._state_history.clone()

    @staticmethod
    def _normalize_reward_mode(reward_mode: str) -> str:
        reward_mode = str(reward_mode)
        if reward_mode not in ("goal", "corresponding"):
            raise ValueError(
                "reward_mode must be 'goal' or 'corresponding', "
                f"got '{reward_mode}'"
            )
        return reward_mode

    def _reward_mode_requires_gt(self) -> bool:
        return self.reward_mode == "corresponding"

    def _compute_reward_for_mode(
        self,
        pred_tokens: torch.Tensor,
        gt_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reward, diagnostic token_err, and goal_err for the active mode."""
        assert self._goal_tokens is not None

        goal_reward, goal_err = self._compute_reward(pred_tokens, self._goal_tokens)
        if not self._reward_mode_requires_gt():
            return goal_reward, goal_err, goal_err

        if gt_tokens is None:
            raise ValueError(f"reward_mode={self.reward_mode!r} requires gt_tokens")

        corresponding_reward, corresponding_err = self._compute_reward(pred_tokens, gt_tokens)
        return corresponding_reward, corresponding_err, goal_err

    def step(self, actions: torch.Tensor) -> StepResult:
        raise NotImplementedError("Use step_chunked()")

    def step_fast(self, actions: torch.Tensor) -> StepResult:
        raise NotImplementedError("Use step_chunked()")

    # ------------------------------------------------------------------
    # Flow noise sampling (per-env, seed + worker-count invariant)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_flow_noise(
        self, N: int, num_latent_tokens: int, token_dim: int,
    ) -> torch.Tensor:
        """Draw per-env Gaussian noise keyed by ``(flow_seed, global_env_id, step)``.

        Independent across envs, deterministic given ``flow_seed``, and
        invariant to how envs are sharded across workers/GPUs. Uses NumPy
        on CPU so results match bit-for-bit across devices.

        Returns:
            ``(N, num_latent_tokens, token_dim)`` float32 tensor on ``self.device``.
        """
        step = self._flow_step_counter
        noises = np.empty((N, num_latent_tokens, token_dim), dtype=np.float32)
        for i in range(N):
            seed = fold_in_seed(
                self._flow_seed,
                FLOW_NOISE_STREAM_ID,
                int(self._global_env_ids[i]),
                step,
            )
            rng = np.random.default_rng(seed)
            noises[i] = rng.standard_normal(
                (num_latent_tokens, token_dim), dtype=np.float32,
            )
        self._flow_step_counter += 1
        return torch.from_numpy(noises).to(self.device)

    # ------------------------------------------------------------------
    # step_chunked — main world-model rollout method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_chunked(self, action_chunks: torch.Tensor) -> StepResult:
        """Run one chunked decision step through the world model.

        Args:
            action_chunks: ``(N, K, A)`` raw target_qpos from the policy.

        Returns:
            :class:`StepResult` with obs, reward, done, truncated, success, info.
        """
        N, K, A = action_chunks.shape
        assert N == self.num_envs and K == self.chunk_size and A == self.action_dim
        device = self.device
        action_chunks = action_chunks.to(device).float()

        # 1. Convert target_qpos → full_qpos via ActionNormalizer.
        full_qpos_list: List[Optional[torch.Tensor]] = []
        full_qpos_tensors: List[torch.Tensor] = []
        for i in range(N):
            fq = self._target_qpos_to_full_qpos(action_chunks[i])
            full_qpos_list.append(fq)
            full_qpos_tensors.append(fq)

        # 2. Prepare flow-model inputs.
        assert self._cached_x_t_tokens is not None
        x_t = self._cached_x_t_tokens  # (N, Lp, C)
        num_latent_tokens = int(getattr(self.latent_decoder, "num_tokens"))
        token_dim = int(getattr(self.flow_model, "token_dim"))

        noise = self._sample_flow_noise(N, num_latent_tokens, token_dim)
        # noise = torch.randn(N, num_latent_tokens, token_dim, device=device)

        task_inds = torch.full((N,), self.task_ind, device=device, dtype=torch.long)
        horizons = torch.full((N,), float(K), device=device, dtype=torch.float32)
        robot_ids = torch.full((N,), self._robot_id, device=device, dtype=torch.long)

        # 3. Advance timestep.
        t_new = np.clip(
            self._t + K, 0,
            (self._episode_lengths[self._ep_idx] - 1).astype(np.int64),
        )

        # 4. World-model forward + reward.
        #    "corresponding" mode: load GT frame at t+K, reward = -dist(pred, gt).
        #    "goal" mode: reward = -dist(pred, goal_tokens), skip GT frame I/O.
        assert self._goal_tokens is not None

        if self._reward_mode_requires_gt():
            gt_frames = torch.stack([
                self._read_frame(
                    self.dataset.traj_id_list[int(self._ep_idx[i])], int(t_new[i]),
                )
                for i in range(N)
            ])  # (N, 3, H, W)
            pred_tokens, gt_tokens = self._world_model_forward(
                x_t, noise, task_inds, horizons, robot_ids, full_qpos_list, gt_frames,
            )
            if pred_tokens.isnan().any():
                raise ValueError("NaN detected in predicted tokens")
            self._gt_wm_frames = gt_frames
        else:
            # Goal-only reward: no GT frame loading, no GT DINO extraction.
            pred_tokens = self._world_model_forward_no_gt(
                x_t, noise, task_inds, horizons, robot_ids, full_qpos_list,
            )
            if pred_tokens.isnan().any():
                raise ValueError("NaN detected in predicted tokens")

        # 5. Decode predicted tokens → next observation image.
        pred_img_wm = self._decode_tokens(pred_tokens)             # (N, 3, H_wm, W_wm)
        pred_img_obs = self._resize_for_policy(pred_img_wm)        # (N, 3, H_obs, W_obs)

        if pred_img_obs.isnan().any():
            raise ValueError("NaN detected in predicted observation image")

        reward, per_env_err, goal_err = self._compute_reward_for_mode(
            pred_tokens,
            gt_tokens=gt_tokens if self._reward_mode_requires_gt() else None,
        )

        # 6. Done / success / truncation.
        ep_lens = self._episode_lengths[self._ep_idx]
        natural_done = (self._t + K) >= (ep_lens - 1)
        new_step_count = self._step_count + 1
        truncated_np = new_step_count >= self.max_chunk_steps
        done_np = natural_done | truncated_np
        success_np = done_np & (
            goal_err.detach().cpu().numpy() <= self.success_token_threshold
        )

        # Terminal success bonus.
        if self.terminal_success_bonus != 0.0:
            bonus_mask = torch.from_numpy(success_np).to(device)
            reward = reward + torch.where(
                bonus_mask,
                torch.full_like(reward, self.terminal_success_bonus),
                torch.zeros_like(reward),
            )

        # Also include goal_err in info for diagnostics.
        _goal_err_info = goal_err.detach()

        # 7. Advance observation and state history.
        #    State = full_qpos of the last action in chunk (via ActionNormalizer).
        state_next = torch.stack(
            [fq[-1] for fq in full_qpos_tensors]
        )  # (N, full_qpos_dim)

        if self.n_obs_steps > 1:
            self._obs_history[:, :-1] = self._obs_history[:, 1:].clone()
            self._state_history[:, :-1] = self._state_history[:, 1:].clone()
        self._obs_history[:, -1] = pred_img_obs
        self._state_history[:, -1] = state_next
        self._current_wm_frames = pred_img_wm

        self._t = t_new
        self._step_count = new_step_count

        # Reuse predicted tokens as next x_t cache (no extra DINO pass).
        self._cached_x_t_tokens = pred_tokens

        # 7b. Capture pre-reset obs/state for truncation bootstrapping.
        bootstrap_mask = torch.from_numpy(truncated_np & ~natural_done).to(device)
        chunk_final_obs: Optional[torch.Tensor] = None
        chunk_final_state: Optional[torch.Tensor] = None
        if bootstrap_mask.any():
            chunk_final_obs = self._obs_history[bootstrap_mask].clone()
            chunk_final_state = self._state_history[bootstrap_mask].clone()

        # 7c. Capture pre-reset obs for *all* done envs (for visualization).
        done_mask = torch.from_numpy(done_np).to(device)
        pre_reset_obs: Optional[torch.Tensor] = None
        if done_mask.any():
            pre_reset_obs = self._obs_history[done_mask].clone()

        # 8. Reset done envs.
        if done_np.any():
            done_idx = np.where(done_np)[0]
            for i in done_idx:
                self._reset_one(int(i))
            self._refresh_cached_tokens_at(done_idx)
            self._refresh_goal_tokens_at(done_idx)
        
        if self._obs_history.isnan().any():
            raise ValueError("NaN detected in observation history")

        # 9. Assemble result.
        info: Dict[str, Any] = {
            "state_history": self._state_history.clone(),
            "chunk_return_sum": reward.detach().clone(),
            "token_err": per_env_err.detach(),
            "goal_err": _goal_err_info,
        }
        if chunk_final_obs is not None:
            info["chunk_bootstrap_mask"] = bootstrap_mask
            info["chunk_final_obs_tensor"] = chunk_final_obs
            info["chunk_final_state_obs"] = chunk_final_state
        if pre_reset_obs is not None:
            info["pre_reset_obs"] = pre_reset_obs
            info["pre_reset_done_mask"] = done_mask

        return StepResult(
            obs=self._obs_history.clone(),
            reward=reward.detach(),
            done=torch.from_numpy(done_np).to(device),
            truncated=torch.from_numpy(truncated_np).to(device),
            success=torch.from_numpy(success_np).to(device),
            info=info,
        )

    # ------------------------------------------------------------------
    # World-model forward (single device)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _world_model_forward(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        task_inds: torch.Tensor,
        horizons: torch.Tensor,
        robot_ids: torch.Tensor,
        full_qpos_list: List[Optional[torch.Tensor]],
        gt_frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run flow sampling + latent decode with GT token extraction.

        Used by ``reward_mode in {'corresponding', 'both'}`` to compute
        reward against the ground-truth frame at t+K.

        Returns:
            pred_tokens: ``(N, Lp, C)`` predicted x_{t+K} DINO tokens.
            gt_tokens: ``(N, Lp, C)`` GT DINO tokens extracted from *gt_frames*.
        """
        gt_tokens, _ = self._extract_tokens(gt_frames)
        N = x_t.shape[0]
        bs = self._comp_batch_size
        if self._env_batch_num <= 1 or N <= bs:
            pred_latent_flow = _euler_sample(
                self.flow_model, noise, x_t, task_inds, horizons,
                robot_ids, full_qpos_list, steps=self.euler_steps,
            )
            pred_latent = pred_latent_flow * self.latent_scalar_normalization
            _, pred_x_T = self.latent_decoder(x_t, tokens=pred_latent)
            return pred_x_T.float(), gt_tokens

        pred_parts: List[torch.Tensor] = []
        from tqdm.auto import tqdm
        for i in tqdm(
            range(0, N, bs),
            desc=f"flow fwd (env_batch_num={self._env_batch_num})",
            leave=False,
            total=(N + bs - 1) // bs,
            disable=(self._worker_id != 0),
        ):
            j = min(i + bs, N)
            chunk_flow = _euler_sample(
                self.flow_model,
                noise[i:j], x_t[i:j], task_inds[i:j], horizons[i:j],
                robot_ids[i:j], full_qpos_list[i:j],
                steps=self.euler_steps,
            )
            chunk_latent = chunk_flow * self.latent_scalar_normalization
            _, chunk_pred = self.latent_decoder(x_t[i:j], tokens=chunk_latent)
            pred_parts.append(chunk_pred.float())
        return torch.cat(pred_parts, dim=0), gt_tokens

    @torch.no_grad()
    def _world_model_forward_no_gt(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        task_inds: torch.Tensor,
        horizons: torch.Tensor,
        robot_ids: torch.Tensor,
        full_qpos_list: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Run flow sampling + latent decode without GT token extraction.

        Used by ``reward_mode='goal'`` to skip GT frame I/O and DINO pass.

        Returns:
            pred_tokens: ``(N, Lp, C)`` predicted x_{t+K} DINO tokens.
        """
        N = x_t.shape[0]
        bs = self._comp_batch_size
        if self._env_batch_num <= 1 or N <= bs:
            pred_latent_flow = _euler_sample(
                self.flow_model, noise, x_t, task_inds, horizons,
                robot_ids, full_qpos_list, steps=self.euler_steps,
            )
            pred_latent = pred_latent_flow * self.latent_scalar_normalization
            _, pred_x_T = self.latent_decoder(x_t, tokens=pred_latent)
            return pred_x_T.float()

        pred_parts: List[torch.Tensor] = []
        from tqdm.auto import tqdm
        for i in tqdm(
            range(0, N, bs),
            desc=f"flow fwd (env_batch_num={self._env_batch_num})",
            leave=False,
            total=(N + bs - 1) // bs,
            disable=(self._worker_id != 0),
        ):
            j = min(i + bs, N)
            chunk_flow = _euler_sample(
                self.flow_model,
                noise[i:j], x_t[i:j], task_inds[i:j], horizons[i:j],
                robot_ids[i:j], full_qpos_list[i:j],
                steps=self.euler_steps,
            )
            chunk_latent = chunk_flow * self.latent_scalar_normalization
            _, chunk_pred = self.latent_decoder(x_t[i:j], tokens=chunk_latent)
            pred_parts.append(chunk_pred.float())
        return torch.cat(pred_parts, dim=0)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        pred_tokens: torch.Tensor,
        gt_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-env reward and error from predicted vs GT tokens.

        Returns:
            reward ``(N,)``, per_env_err ``(N,)``.
        """
        diff = pred_tokens - gt_tokens
        if self.reward_token_metric == "l2":
            per_env_err = diff.pow(2).mean(dim=(-1, -2))
        elif self.reward_token_metric == "l1":
            per_env_err = diff.abs().mean(dim=(-1, -2))
        elif self.reward_token_metric == "cosine":
            cos = F.cosine_similarity(
                pred_tokens.flatten(1), gt_tokens.flatten(1), dim=-1,
            )
            per_env_err = 1.0 - cos
        else:
            raise ValueError(f"Unknown reward_token_metric={self.reward_token_metric}")

        if self.reward_token_metric == "cosine":
            reward = 1.0 - per_env_err  # = cos
        else:
            reward = -per_env_err
        return reward * self.reward_scale, per_env_err

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def sample_actions_from_dataset(self) -> torch.Tensor:
        """Return ground-truth action chunks for all envs at current timestep.

        Reads ``target_qpos`` from the dataset for each env's current episode
        and timestep, yielding a chunk of length ``chunk_size``.

        Returns:
            ``(num_envs, chunk_size, action_dim)`` tensor on ``self.device``.
        """
        K = self.chunk_size
        chunks = []
        for i in range(self.num_envs):
            ep_idx = int(self._ep_idx[i])
            ep_start = int(self.dataset.episode_starts[ep_idx])
            ep_end = int(self.dataset.episode_ends[ep_idx])
            t = int(self._t[i])
            t_abs = ep_start + t
            t_end = min(t_abs + K, ep_end)
            chunk = self.dataset.target_qpos[t_abs:t_end]  # (<=K, A)
            chunk_t = torch.from_numpy(chunk).to(self.device, dtype=torch.float32)
            # Pad if near episode end.
            if chunk_t.shape[0] < K:
                pad = chunk_t[-1:].expand(K - chunk_t.shape[0], -1)
                chunk_t = torch.cat([chunk_t, pad], dim=0)
            chunks.append(chunk_t)
        return torch.stack(chunks)  # (N, K, A)

    def render(
        self,
        env_index: Optional[int] = None,
        mode: str = "side_by_side",
        grid_cols: Optional[int] = None,
        label: bool = True,
    ) -> np.ndarray:
        """Render observations as a uint8 HWC numpy array.

        Args:
            env_index: Which env slot to visualize. ``None`` (default) renders
                all envs in a grid.
            mode: ``"pred"`` for predicted frame only,
                  ``"gt"`` for ground-truth frame only,
                  ``"side_by_side"`` for pred | gt concatenated horizontally.
            grid_cols: Number of columns when rendering all envs. Defaults to
                ``ceil(sqrt(num_envs))``.
            label: If True, overlay env index and mode labels on each cell.

        Returns:
            ``(H, W, 3)`` uint8 numpy array.
        """
        import math

        def _to_hwc(img_chw: torch.Tensor) -> np.ndarray:
            arr = img_chw.detach().float().cpu().clamp(0.0, 1.0).numpy()
            arr = np.transpose(arr, (1, 2, 0))
            return (arr * 255.0).round().astype(np.uint8)

        def _render_one(idx: int) -> np.ndarray:
            pred_obs = self._obs_history[idx, -1]  # (3, H, W)
            pred_hwc = _to_hwc(pred_obs)

            if mode == "pred":
                return pred_hwc

            gt_hwc: Optional[np.ndarray] = None
            if self._gt_wm_frames is not None:
                gt_obs = self._resize_for_policy(
                    self._gt_wm_frames[idx].unsqueeze(0)
                ).squeeze(0)
                gt_hwc = _to_hwc(gt_obs)

            if mode == "gt":
                if gt_hwc is None:
                    raise RuntimeError("No GT frames available (call step_chunked first)")
                return gt_hwc

            # side_by_side
            if gt_hwc is None:
                return pred_hwc
            sep = np.zeros((pred_hwc.shape[0], 4, 3), dtype=np.uint8)
            return np.concatenate([pred_hwc, sep, gt_hwc], axis=1)

        def _add_label(img: np.ndarray, text: str) -> np.ndarray:
            """Burn a simple text label into the top-left corner."""
            img = img.copy()
            h, w = img.shape[:2]
            # Draw a dark background strip for readability.
            bar_h = min(16, h // 6)
            img[:bar_h, :] = (img[:bar_h, :].astype(np.int32) * 4 // 10).astype(np.uint8)
            # Render text with simple bitmap (no PIL/cv2 dependency).
            _GLYPHS = {
                '0': [0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E],
                '1': [0x0C,0x1C,0x0C,0x0C,0x0C,0x0C,0x3F],
                '2': [0x3E,0x63,0x03,0x1E,0x30,0x60,0x7F],
                '3': [0x3E,0x63,0x03,0x1E,0x03,0x63,0x3E],
                '4': [0x06,0x0E,0x1E,0x36,0x7F,0x06,0x06],
                '5': [0x7F,0x60,0x7E,0x03,0x03,0x63,0x3E],
                '6': [0x1E,0x30,0x60,0x7E,0x63,0x63,0x3E],
                '7': [0x7F,0x03,0x06,0x0C,0x18,0x18,0x18],
                '8': [0x3E,0x63,0x63,0x3E,0x63,0x63,0x3E],
                '9': [0x3E,0x63,0x63,0x3F,0x03,0x06,0x3C],
                'E': [0x7F,0x60,0x60,0x7E,0x60,0x60,0x7F],
                'P': [0x7E,0x63,0x63,0x7E,0x60,0x60,0x60],
                'G': [0x3E,0x63,0x60,0x6F,0x63,0x63,0x3E],
                'T': [0x7F,0x08,0x08,0x08,0x08,0x08,0x08],
                '#': [0x24,0x7E,0x24,0x24,0x7E,0x24,0x00],
                ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
                '|': [0x08,0x08,0x08,0x08,0x08,0x08,0x08],
            }
            cx = 2
            for ch in text.upper():
                glyph = _GLYPHS.get(ch)
                if glyph is None:
                    cx += 6
                    continue
                for row_i, bits in enumerate(glyph):
                    py = 2 + row_i
                    if py >= h:
                        break
                    for col_i in range(7):
                        px = cx + col_i
                        if px >= w:
                            break
                        if bits & (0x40 >> col_i):
                            img[py, px] = [255, 255, 255]
                cx += 8
            return img

        # Single env.
        if env_index is not None:
            cell = _render_one(env_index)
            if label:
                cell = _add_label(cell, f"E{env_index}")
            return cell

        # All envs → grid.
        cells = []
        for i in range(self.num_envs):
            cell = _render_one(i)
            if label:
                lbl = f"E{i}"
                if mode == "side_by_side":
                    lbl += " P|GT"
                elif mode == "pred":
                    lbl += " P"
                else:
                    lbl += " GT"
                cell = _add_label(cell, lbl)
            cells.append(cell)

        cell_h, cell_w = cells[0].shape[:2]
        cols = grid_cols or int(math.ceil(math.sqrt(self.num_envs)))
        rows = int(math.ceil(self.num_envs / cols))
        pad = 2  # pixel gap between cells

        canvas_h = rows * cell_h + (rows - 1) * pad
        canvas_w = cols * cell_w + (cols - 1) * pad
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for idx, cell in enumerate(cells):
            r, c = divmod(idx, cols)
            y0 = r * (cell_h + pad)
            x0 = c * (cell_w + pad)
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell

        return canvas

    def close(self) -> None:
        """No-op cleanup (no simulator to close)."""
        pass

