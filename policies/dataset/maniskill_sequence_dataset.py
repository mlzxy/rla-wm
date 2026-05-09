"""
Diffusion-policy-style sequence dataset for ManiSkill trajectories.

Each ManiSkill trajectory = one episode.
Metadata (qpos, target_qpos) is preloaded into flat arrays.
Images are lazy-loaded per __getitem__ call.
State/action outputs always have shape (horizon, ...).
Image output length depends on the image sampling mode.

Usage:
    dataset = ManiSkillSequenceDataset(
        dataset_dir="data/better/ppo/ur10e_stick/PushT-v2/success",
        cameras=["front_lower_camera"],
        horizon=16,
        img_size=512,
    )
    sample = dataset[0]
    # sample['obs']['image'].shape == (1, 1, 3, 512, 512)
    # sample['obs']['state'].shape == (16, state_dim)
    # sample['action'].shape       == (16, action_dim)

    dataset_full = ManiSkillSequenceDataset(
        dataset_dir="data/better/ppo/ur10e_stick/PushT-v2/success",
        cameras=["front_lower_camera"],
        horizon=16,
        img_size=512,
        first_frame_only=False,
    )
    # dataset_full[0]['obs']['image'].shape == (16, 1, 3, 512, 512)
"""

import copy
from typing import List, Optional, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datalib.dataset import ManiSkillTrajectoryDataset
from diffusion_policy.common.sampler import (
    create_indices,
    get_val_mask,
    downsample_mask,
)



class ManiSkillObservation(TypedDict, total=False):
    image: torch.Tensor
    state: torch.Tensor
    motus_latents: torch.Tensor
    univla_latents: torch.Tensor


class ManiSkillSequenceSample(TypedDict):
    obs: ManiSkillObservation
    action: torch.Tensor


class ManiSkillSequenceDataset(Dataset):
    """
    Wraps ManiSkillTrajectoryDataset in a diffusion-policy-compatible interface.

    - Preloads metadata (qpos, target_qpos) as flat numpy arrays.
    - Builds episode_ends / SequenceSampler indices for O(1) sampling.
    - Lazy-loads and masks images on each __getitem__ call.
        - Returns state/action/latent keys at full ``horizon`` length with
            boundary padding (repeat first/last frame, identical to
            diffusion_policy).
        - Returns images either as the full horizon, the first frame only,
            or the first frame plus sparse in-horizon frames and optional
            future extra frames.
    """

    def __init__(
        self,
        dataset_dir: str,
        cameras: List[str],
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        img_size: int = 512,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        max_val_episodes: Optional[int] = None,
        start_traj_id: int = 0,
        end_traj_id: int = 999,
        use_foreground_mask: bool = True,
        min_episode_length: int = 5,
        load_extra_frame: int = 0,
        first_frame_only: bool = True,
        num_in_horizon_frames_after_first: int = 0,
        pixel_only: bool = False,
        pixel_only_state_dim: int = 0,
        pixel_only_action_dim: int = 0,
        load_motus_latents: bool = False,
        motus_latent_key: str = "motus_latents",
        load_univla_latents: bool = False,
        univla_latent_key: str = "univla_latent_indices",
        univla_codebook_path: str = "",
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.cameras = cameras
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.img_size = img_size
        self.use_foreground_mask = use_foreground_mask
        self.load_extra_frame = max(0, int(load_extra_frame))
        self.first_frame_only = first_frame_only
        self.num_in_horizon_frames_after_first = int(num_in_horizon_frames_after_first)
        self.max_val_episodes = max_val_episodes
        self.seed = seed
        self.pixel_only = pixel_only
        self.load_motus_latents = load_motus_latents
        self.motus_latent_key = motus_latent_key
        self.load_univla_latents = load_univla_latents
        self.univla_latent_key = univla_latent_key

        if self.num_in_horizon_frames_after_first < 0:
            raise ValueError("num_in_horizon_frames_after_first must be >= 0")
        if self.num_in_horizon_frames_after_first > 0 and not self.first_frame_only:
            raise ValueError(
                "num_in_horizon_frames_after_first requires first_frame_only=True"
            )
        if self.num_in_horizon_frames_after_first > max(0, self.horizon - 1):
            raise ValueError(
                "num_in_horizon_frames_after_first must be <= horizon - 1"
            )
        self._in_horizon_image_positions = self._compute_in_horizon_image_positions()

        # Load UniVLA codebook for embedding lookup
        self._univla_codebook: np.ndarray | None = None
        if load_univla_latents:
            if not univla_codebook_path:
                raise ValueError("univla_codebook_path is required when load_univla_latents=True")
            self._univla_codebook = np.load(univla_codebook_path).astype(np.float32)
        # --- Open underlying trajectory store ---
        self.traj_ds = ManiSkillTrajectoryDataset(
            dataset_dir,
            start_traj_id=str(start_traj_id).zfill(6),
            end_traj_id=str(end_traj_id).zfill(6),
        )
        traj_ids = self.traj_ds.list_trajectories()

        # --- Preload metadata into flat arrays ---
        all_qpos: list[np.ndarray] = []
        all_target_qpos: list[np.ndarray] = []
        all_motus_latents: list[np.ndarray] = []
        all_univla_latents: list[np.ndarray] = []
        episode_ends: list[int] = []
        self.traj_id_list: list[str] = []

        total_frames = 0
        for traj_id in tqdm(traj_ids, desc="Preloading metadata" + (" (pixel-only)" if pixel_only else "")):
            if pixel_only:
                # Pixel-only mode: read only video stream lengths, no metadata
                rgb_key = f"{cameras[0]}_rgb"
                traj = self.traj_ds.read_trajectory(
                    traj_id,
                    video_keys=[rgb_key],
                    metadata_keys=[],
                    img_size=None,  # just to get frame count
                )
                T = len(traj.video_streams[rgb_key])
                if T < min_episode_length:
                    continue
                # Dummy zeros for state/action
                all_qpos.append(np.zeros((T, pixel_only_state_dim), dtype=np.float32))
                all_target_qpos.append(np.zeros((T, pixel_only_action_dim), dtype=np.float32))
                if load_motus_latents:
                    traj_ml = self.traj_ds.read_trajectory(
                        traj_id, video_keys=[], metadata_keys=[motus_latent_key],
                    )
                    ml = np.asarray(traj_ml.metadata[motus_latent_key], dtype=np.float32)
                    all_motus_latents.append(ml[:T])
                if load_univla_latents:
                    traj_ul = self.traj_ds.read_trajectory(
                        traj_id, video_keys=[], metadata_keys=[univla_latent_key],
                    )
                    indices = np.asarray(traj_ul.metadata[univla_latent_key], dtype=np.int64)
                    assert self._univla_codebook is not None
                    embedded = self._univla_codebook[indices]  # (T, num_codes, latent_dim)
                    embedded = embedded.reshape(indices.shape[0], -1)  # (T, num_codes * latent_dim)
                    all_univla_latents.append(embedded[:T])
            else:
                metadata_keys_to_load = ["target_qpos", "qpos"]
                if load_univla_latents:
                    metadata_keys_to_load.append(univla_latent_key)
                if load_motus_latents:
                    metadata_keys_to_load.append(motus_latent_key)
                traj = self.traj_ds.read_trajectory(
                    traj_id,
                    video_keys=[],
                    metadata_keys=metadata_keys_to_load,
                )
                qpos = np.asarray(traj.metadata["qpos"], dtype=np.float32)
                target_qpos = np.asarray(traj.metadata["target_qpos"], dtype=np.float32)

                # Squeeze robot dim if present: (T, 1, J) -> (T, J)
                if qpos.ndim == 3 and qpos.shape[1] == 1:
                    qpos = qpos.squeeze(1)
                if target_qpos.ndim == 3 and target_qpos.shape[1] == 1:
                    target_qpos = target_qpos.squeeze(1)

                T = qpos.shape[0]
                if T < min_episode_length:
                    continue

                all_qpos.append(qpos)
                all_target_qpos.append(target_qpos)
                if load_motus_latents:
                    ml = np.asarray(traj.metadata[motus_latent_key], dtype=np.float32)
                    all_motus_latents.append(ml[:T])
                if load_univla_latents:
                    indices = np.asarray(traj.metadata[univla_latent_key], dtype=np.int64)
                    assert self._univla_codebook is not None
                    embedded = self._univla_codebook[indices]  # (T, num_codes, latent_dim)
                    embedded = embedded.reshape(indices.shape[0], -1)  # (T, num_codes * latent_dim)
                    all_univla_latents.append(embedded[:T])

            total_frames += T
            episode_ends.append(total_frames)
            self.traj_id_list.append(traj_id)

        self.qpos = np.concatenate(all_qpos, axis=0)           # (N, state_dim)
        self.target_qpos = np.concatenate(all_target_qpos, axis=0)  # (N, action_dim)
        self.motus_latents: np.ndarray | None = (
            np.concatenate(all_motus_latents, axis=0) if load_motus_latents else None
        )  # (N, latent_dim) or None
        self.univla_latents: np.ndarray | None = (
            np.concatenate(all_univla_latents, axis=0) if load_univla_latents else None
        )  # (N, num_codes * latent_dim) or None
        self.episode_ends = np.array(episode_ends, dtype=np.int64)

        # --- Train / val split (episode-level) ---
        n_episodes = len(self.episode_ends)
        val_mask = get_val_mask(n_episodes, val_ratio, seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_train_episodes, seed)

        # --- Sequence indices (identical to diffusion_policy SequenceSampler) ---
        self.indices = create_indices(
            self.episode_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask

        # Episode starts for reverse mapping
        self.episode_starts = np.zeros(n_episodes, dtype=np.int64)
        if n_episodes > 1:
            self.episode_starts[1:] = self.episode_ends[:-1]

        print(
            f"ManiSkillSequenceDataset: {n_episodes} episodes, "
            f"{total_frames} frames, {len(self.indices)} samples "
            f"(horizon={horizon}, pad_before={pad_before}, pad_after={pad_after})"
        )

    # ------------------------------------------------------------------
    # Diffusion-policy API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def get_validation_dataset(self) -> "ManiSkillSequenceDataset":
        """Return a shallow copy selecting only validation episodes."""
        val_set = copy.copy(self)
        val_mask = ~self.train_mask
        val_mask = downsample_mask(val_mask, self.max_val_episodes, self.seed)
        val_set.indices = create_indices(
            self.episode_ends,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
        )
        val_set.train_mask = ~val_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """Fit and return a diffusion_policy LinearNormalizer over train data."""
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import (
            get_image_range_normalizer,
        )

        data = {
            "action": self.target_qpos,
            "state": self.qpos,
        }
        if self.motus_latents is not None:
            data["motus_latents"] = self.motus_latents
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buffer_idx_to_episode(self, buffer_idx: int) -> tuple[int, int]:
        """Map a flat buffer index to (episode_index, frame_offset_within_episode)."""
        ep_idx = int(np.searchsorted(self.episode_ends, buffer_idx, side="right"))
        ep_start = int(self.episode_starts[ep_idx])
        return ep_idx, buffer_idx - ep_start

    def _pad_sequence(self, data: np.ndarray, sample_start: int, sample_end: int) -> np.ndarray:
        """Pad a raw slice into a full (horizon, ...) array with boundary replication."""
        out = np.zeros((self.horizon,) + data.shape[1:], dtype=data.dtype)
        if sample_start > 0:
            out[:sample_start] = data[0]
        if sample_end < self.horizon:
            out[sample_end:] = data[-1]
        out[sample_start:sample_end] = data
        return out

    def _compute_in_horizon_image_positions(self) -> np.ndarray:
        """Return logical horizon positions used for first-frame image sampling."""
        total_in_horizon_frames = 1 + self.num_in_horizon_frames_after_first
        if total_in_horizon_frames == 1:
            return np.array([0], dtype=np.int64)
        return np.linspace(
            0,
            self.horizon - 1,
            total_in_horizon_frames,
            dtype=np.int64,
        )

    def _map_horizon_positions_to_episode_indices(
        self,
        horizon_positions: np.ndarray,
        buffer_start: int,
        buffer_end: int,
        sample_start: int,
        ep_start: int,
    ) -> np.ndarray:
        """Map logical horizon positions to episode-relative frame indices."""
        first_rel = buffer_start - ep_start
        last_rel = buffer_end - ep_start - 1
        frame_indices = first_rel + (horizon_positions - sample_start)
        return np.clip(frame_indices, first_rel, last_rel).astype(np.int64)

    def _get_image_frame_indices(
        self,
        buffer_start: int,
        buffer_end: int,
        sample_start: int,
        ep_start: int,
        ep_len: int,
    ) -> np.ndarray:
        """Return episode-relative frame indices to load for image observations."""
        if self.first_frame_only:
            frame_indices = self._map_horizon_positions_to_episode_indices(
                self._in_horizon_image_positions,
                buffer_start=buffer_start,
                buffer_end=buffer_end,
                sample_start=sample_start,
                ep_start=ep_start,
            )
        else:
            frame_indices = np.arange(
                buffer_start - ep_start,
                buffer_end - ep_start,
                dtype=np.int64,
            )

        if self.load_extra_frame > 0:
            last_rel_idx = buffer_end - ep_start - 1
            future_offsets = np.arange(1, self.load_extra_frame + 1, dtype=np.int64)
            future_indices = np.minimum(last_rel_idx + future_offsets, ep_len - 1)
            frame_indices = np.concatenate([frame_indices, future_indices], axis=0)

        return frame_indices

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> ManiSkillSequenceSample:
        buffer_start, buffer_end, sample_start, sample_end = self.indices[index]

        # --- 1. Metadata (from preloaded flat arrays) ---
        qpos_slice = self.qpos[buffer_start:buffer_end]
        action_slice = self.target_qpos[buffer_start:buffer_end]

        qpos_out = self._pad_sequence(qpos_slice, sample_start, sample_end)
        action_out = self._pad_sequence(action_slice, sample_start, sample_end)

        # --- 2. Images (lazy-loaded per call) ---
        ep_idx, _ = self._buffer_idx_to_episode(buffer_start)
        traj_id = self.traj_id_list[ep_idx]
        ep_start = int(self.episode_starts[ep_idx])
        ep_len = int(self.episode_ends[ep_idx]) - ep_start

        frame_indices = self._get_image_frame_indices(
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            sample_start=sample_start,
            ep_start=ep_start,
            ep_len=ep_len,
        )
        base_num_frames = len(frame_indices) - self.load_extra_frame

        # Read RGB at target img_size; read masks WITHOUT resize (boolean
        # arrays are incompatible with cv2.resize) and resize manually.
        rgb_keys = [f"{cam}_rgb" for cam in self.cameras]
        traj_rgb = self.traj_ds.read_trajectory(
            traj_id,
            video_keys=rgb_keys,
            metadata_keys=[],
            frame_indices=frame_indices.tolist(),
            img_size=self.img_size,
        )

        if self.use_foreground_mask:
            mask_keys = [f"{cam}_foreground_mask" for cam in self.cameras]
            traj_mask = self.traj_ds.read_trajectory(
                traj_id,
                video_keys=mask_keys,
                metadata_keys=[],
                frame_indices=frame_indices.tolist(),
                img_size=None,  # read at native resolution
            )

        num_cams = len(self.cameras)
        first_rgb = np.asarray(traj_rgb.video_streams[f"{self.cameras[0]}_rgb"])
        T_read, H, W, _ = first_rgb.shape

        # Build per-frame images: (T_read, num_cam, 3, H, W)
        images_raw = np.zeros((T_read, num_cams, 3, H, W), dtype=np.float32)
        for cam_idx, cam in enumerate(self.cameras):
            rgb = np.asarray(traj_rgb.video_streams[f"{cam}_rgb"]).astype(np.float32) / 255.0
            rgb = np.transpose(rgb, (0, 3, 1, 2))  # (T, 3, H, W)

            if self.use_foreground_mask:
                fg = np.asarray(traj_mask.video_streams[f"{cam}_foreground_mask"])
                fg = fg.astype(np.float32)
                # Resize mask to match RGB if dimensions differ
                if fg.shape[1] != H or fg.shape[2] != W:
                    import cv2
                    fg_resized = np.zeros((fg.shape[0], H, W), dtype=np.float32)
                    for t in range(fg.shape[0]):
                        fg_resized[t] = cv2.resize(
                            fg[t], (W, H), interpolation=cv2.INTER_NEAREST
                        )
                    fg = fg_resized
                rgb = rgb * fg[:, None]  # broadcast mask over C

            images_raw[:, cam_idx] = rgb

        if self.first_frame_only:
            # images_raw already contains the selected in-horizon image frames
            # plus any requested future extra frames.
            images_out = images_raw
        else:
            # Keep original horizon behavior for the base segment, then append
            # extra future frame(s) if requested.
            images_base = images_raw[:base_num_frames]
            images_extra = images_raw[base_num_frames:]

            # Pad images identically to metadata
            images_out = np.zeros((self.horizon, num_cams, 3, H, W), dtype=np.float32)
            if sample_start > 0:
                images_out[:sample_start] = images_base[0]
            if sample_end < self.horizon:
                images_out[sample_end:] = images_base[-1]
            images_out[sample_start:sample_end] = images_base

            if self.load_extra_frame > 0:
                images_out = np.concatenate([images_out, images_extra], axis=0)

        # --- 3. Assemble output ---
        obs: ManiSkillObservation = {
            "image": torch.from_numpy(images_out),      # (N_img, C_cam, 3, H, W)
            "state": torch.from_numpy(qpos_out),        # (H, state_dim)
        }
        if self.motus_latents is not None:
            ml_slice = self.motus_latents[buffer_start:buffer_end]
            ml_out = self._pad_sequence(ml_slice, sample_start, sample_end)
            obs["motus_latents"] = torch.from_numpy(ml_out)  # (H, latent_dim)

        if self.univla_latents is not None:
            ul_slice = self.univla_latents[buffer_start:buffer_end]
            ul_out = self._pad_sequence(ul_slice, sample_start, sample_end)
            obs["univla_latents"] = torch.from_numpy(ul_out)  # (H, num_codes * latent_dim)

        sample: ManiSkillSequenceSample = {
            "obs": obs,
            "action": torch.from_numpy(action_out),          # (H, action_dim)
        }
        return sample
