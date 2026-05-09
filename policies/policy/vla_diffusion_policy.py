"""
Diffusion Policy for v4world.

Mirrors diffusion_policy's DiffusionUnetImagePolicy but uses our
MultiCameraEncoder + StateEncoder for multi-camera RGB + low-dim state.
No language conditioning.

Diffusion-policy-style interface:
  - Uses LinearNormalizer (from diffusion_policy) for obs/action normalization.
  - Input: first ``n_obs_steps`` frames of obs (image + state).
  - Output: ``horizon``-length denoised action trajectory.
  - Only ``n_action_steps`` actions (starting from obs step) are executed.

Architecture:
  MultiCameraEncoder → img_feat  (D_img × n_obs_steps)
  StateEncoder        → state_feat (D_state × n_obs_steps)

  global_cond = [img_feat; state_feat]
  ConditionalUnet1D denoises action trajectory conditioned on global_cond.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply

from policies.policy.utils import MultiCameraEncoder, StateEncoder


class DiffusionPolicy(nn.Module):
    """
    Vision + state diffusion policy for v4world.

    Follows the same interface as DiffusionUnetImagePolicy:
      - ``set_normalizer(normalizer)`` to load a fitted LinearNormalizer.
      - ``compute_loss(batch)`` normalizes input, computes diffusion loss.
      - ``predict_action(obs_dict)`` normalizes input, denoises, unnormalizes output.
    """

    def __init__(
        self,
        # action / state dims
        action_dim: int,
        state_dim: int,
        # architecture
        num_cameras: int = 1,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        per_camera_dim: int = 512,
        use_group_norm: bool = True,
        share_backbone: bool = False,
        resize_shape: tuple | None = None,
        crop_shape: tuple | None = None,
        random_crop: bool = True,
        imagenet_norm: bool = False,
        img_size: int | None = None,
        state_hidden_dim: int = 256,
        state_output_dim: int = 256,
        # UNet
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # noise scheduler (built externally via hydra, passed in)
        noise_scheduler=None,
        num_inference_steps: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # --- normalizer (loaded via set_normalizer) ---
        self.normalizer = LinearNormalizer()

        # --- encoders ---
        self.img_encoder = MultiCameraEncoder(
            num_cameras, per_camera_dim, use_group_norm,
            share_backbone=share_backbone,
            resize_shape=tuple(resize_shape) if resize_shape is not None else None,
            crop_shape=tuple(crop_shape) if crop_shape is not None else None,
            random_crop=random_crop,
            imagenet_norm=imagenet_norm,
            input_shape=(img_size, img_size) if img_size is not None else None,
        )
        self.state_encoder = StateEncoder(state_dim, state_hidden_dim, state_output_dim)

        # --- UNet ---
        img_cond_dim = self.img_encoder.output_dim * n_obs_steps
        state_cond_dim = state_output_dim * n_obs_steps
        global_cond_dim = img_cond_dim + state_cond_dim

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None and noise_scheduler is not None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------ #
    # Normalizer
    # ------------------------------------------------------------------ #

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode normalized observations from the first n_obs_steps into a global cond vector."""
        B = nobs['state'].shape[0]
        To = self.n_obs_steps

        img = nobs['image'][:, :To]    # (B, To, C, 3, H, W)
        state = nobs['state'][:, :To]  # (B, To, D)

        BTo = B * To
        C_cam = img.shape[2]
        img_flat = img.reshape(BTo, C_cam, *img.shape[3:])
        state_flat = state.reshape(BTo, -1)

        img_feat = self.img_encoder(img_flat)
        state_feat = self.state_encoder(state_flat)

        img_feat = img_feat.reshape(B, -1)
        state_feat = state_feat.reshape(B, -1)
        return torch.cat([img_feat, state_feat], dim=-1)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise from random noise to produce an action chunk.

        Args:
            obs_dict: {"image": (B,T,C,3,H,W), "state": (B,T,D)}
        Returns:
            dict with "action": (B, n_action_steps, action_dim) in raw space
                  and "action_pred": (B, horizon, action_dim)
        """
        nobs = self.normalizer.normalize(obs_dict)
        B = nobs['state'].shape[0]
        global_cond = self._encode_obs(nobs)

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        # Start from pure noise
        naction = torch.randn(
            (B, self.horizon, self.action_dim),
            device=self.device, dtype=self.dtype,
        )

        for t in scheduler.timesteps:
            model_output = self.model(naction, t, global_cond=global_cond)
            naction = scheduler.step(model_output, t, naction).prev_sample

        # Unnormalize
        action_pred = self.normalizer['action'].unnormalize(naction)

        # Slice to executable action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute diffusion training loss (predict noise) in normalized space.

        Args:
            batch: {"obs": {"image": ..., "state": ...}, "action": (B, T, Da)}
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]

        global_cond = self._encode_obs(nobs)

        # Sample noise
        noise = torch.randn_like(nactions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        pred = self.model(noisy_actions, timesteps, global_cond=global_cond)

        # Loss target depends on prediction_type
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = nactions
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        return F.mse_loss(pred, target)

    # ------------------------------------------------------------------ #
    # Stubs
    # ------------------------------------------------------------------ #

    def reset(self):
        pass
