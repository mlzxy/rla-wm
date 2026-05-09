"""
Behavioral Cloning Policy for v4world.

Diffusion-policy-style interface:
  - Uses LinearNormalizer (from diffusion_policy) for obs/action normalization.
  - Input: first ``n_obs_steps`` frames of obs (image + state).
  - Output: ``horizon``-length action trajectory.
  - Only ``n_action_steps`` actions (starting from obs step) are executed.

Architecture:
  MultiCameraEncoder → img_feat  (D_img × n_obs_steps)
  StateEncoder        → state_feat (D_state × n_obs_steps)

  global_cond = [img_feat; state_feat]
  ActionMLPDecoder maps global_cond → (B, horizon, action_dim)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from policies.policy.utils import MultiCameraEncoder, StateEncoder


class ActionMLPDecoder(nn.Module):
    """MLP that maps a global condition vector to a full action chunk."""

    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple = (512, 512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        layers = []
        in_dim = cond_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim * horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond: (B, cond_dim)
        Returns: (B, horizon, action_dim)
        """
        out = self.net(cond)
        return out.reshape(-1, self.horizon, self.action_dim)


class VLABCPolicy(nn.Module):
    """
    Vision-Language-Action behavioral cloning policy for v4world.

    Follows the diffusion_policy interface:
      - ``set_normalizer(normalizer)`` to load a fitted LinearNormalizer.
      - ``compute_loss(batch)`` normalizes input, computes loss in normalized space.
      - ``predict_action(obs_dict)`` normalizes input, predicts, unnormalizes output.
    """

    def __init__(
        self,
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
        use_state: bool = True,
        state_hidden_dim: int = 256,
        state_output_dim: int = 256,
        # MLP decoder
        mlp_hidden_dims: tuple = (1024, 512, 256),
        mlp_dropout: float = 0.1,
        # loss
        loss_type: str = "mse",  # "mse" | "l1" | "smooth_l1"
        # RL heads (additive, off by default)
        enable_rl_heads: bool = False,
        critic_latent_dim: int = 2048,
        init_logstd: float = -2.0,
        value_hidden_dims: tuple = (512, 256),
        **kwargs,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.loss_type = loss_type
        self.use_state = use_state

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
        if self.use_state:
            self.state_encoder = StateEncoder(state_dim, state_hidden_dim, state_output_dim)
        else:
            self.state_encoder = None

        # --- MLP decoder ---
        # obs features are flattened over n_obs_steps
        img_cond_dim = self.img_encoder.output_dim * n_obs_steps
        state_cond_dim = state_output_dim * n_obs_steps if self.use_state else 0
        cond_dim = img_cond_dim + state_cond_dim

        self.decoder = ActionMLPDecoder(
            cond_dim=cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
        )

        # --- RL heads (additive, optional) ---
        self.enable_rl_heads = enable_rl_heads
        self._cond_dim = cond_dim
        if enable_rl_heads:
            # Value head: critic_latent -> scalar V(s)
            # Input is a frozen token-difference latent from the inv-dynamics encoder.
            v_layers = []
            in_dim = critic_latent_dim
            for h_dim in value_hidden_dims:
                v_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(inplace=True),
                ])
                in_dim = h_dim
            v_layers.append(nn.Linear(in_dim, 1))
            self.value_head = nn.Sequential(*v_layers)

            # Actor logstd parameter over the executable action chunk
            # (only n_action_steps are actually fed to the environment).
            flat_act_dim = n_action_steps * action_dim
            self.actor_logstd = nn.Parameter(
                torch.ones(1, flat_act_dim) * init_logstd
            )
        else:
            self.value_head = None
            self.actor_logstd = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def n_act_dim(self) -> int:
        """Flat dimensionality of the executable action chunk (n_action_steps * action_dim)."""
        return self.n_action_steps * self.action_dim

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

    def _encode_obs(
        self, nobs: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode normalized observations from the first n_obs_steps into a global cond vector."""
        # nobs['image']: (B, T, C, 3, H, W),  nobs['state']: (B, T, D)
        B = nobs['state'].shape[0]
        To = self.n_obs_steps

        # Slice to obs steps and flatten batch×time
        img = nobs['image'][:, :To]    # (B, To, C, 3, H, W)
        state = nobs['state'][:, :To]  # (B, To, D)

        BTo = B * To
        C_cam = img.shape[2]
        img_flat = img.reshape(BTo, C_cam, *img.shape[3:])  # (B*To, C, 3, H, W)
        state_flat = state.reshape(BTo, -1)                  # (B*To, D)

        img_feat = self.img_encoder(img_flat)        # (B*To, D_img)

        # Reshape back and flatten over obs steps
        img_feat = img_feat.reshape(B, -1)    # (B, To * D_img)

        state_encoder = self.state_encoder
        if self.use_state and state_encoder is not None:
            state_feat = state_encoder(state_flat)  # (B*To, D_state)
            state_feat = state_feat.reshape(B, -1)  # (B, To * D_state)
        else:
            state_feat = None

        return img_feat, state_feat

    def _build_global_cond(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        img_feat, state_feat = self._encode_obs(nobs)
        if state_feat is not None:
            return torch.cat([img_feat, state_feat], dim=-1)
        return img_feat

    # ------------------------------------------------------------------ #
    # RL heads (additive, used only by BCRLAgent)
    # ------------------------------------------------------------------ #

    def forward_features(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the global conditioning vector used by the decoder and RL heads.

        Expects already-normalized observations (same contract as
        ``_build_global_cond``).
        """
        return self._build_global_cond(nobs)

    def forward_value(self, critic_latent: torch.Tensor) -> torch.Tensor:
        """Return V(s) of shape (B,) from a frozen critic latent.

        Args:
            critic_latent: ``(B, critic_latent_dim)`` — flattened token-difference
                encoding produced by the frozen inv-dynamics encoder.
        """
        if self.value_head is None:
            raise RuntimeError(
                "VLABCPolicy.forward_value called but enable_rl_heads=False"
            )
        return self.value_head(critic_latent).squeeze(-1)

    def forward_action_dist(
        self, global_cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of the action-chunk distribution.

        mean has shape (B, n_action_steps * action_dim) and lives in the same
        *normalized* action space as ``compute_loss``. std is broadcast from
        ``actor_logstd`` to match.
        """
        if self.actor_logstd is None:
            raise RuntimeError(
                "VLABCPolicy.forward_action_dist called but enable_rl_heads=False"
            )
        # (B, horizon, action_dim) in normalized action space
        naction_pred = self.decoder(global_cond)

        # Slice to executable action chunk (same indexing as predict_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        nmean_chunk = naction_pred[:, start:end]  # (B, K, A)
        mean = nmean_chunk.reshape(nmean_chunk.shape[0], -1)  # (B, K*A)

        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        return mean, std

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict action chunk from observations (diffusion_policy interface).

        Args:
            obs_dict: dict with key "obs" containing {"image": (B,T,C,3,H,W), "state": (B,T,D)}
        Returns:
            dict with "action": (B, n_action_steps, action_dim) in raw (unnormalized) space
        """
        nobs = self.normalizer.normalize(obs_dict)
        global_cond = self._build_global_cond(nobs)
        naction_pred = self.decoder(global_cond)  # (B, horizon, action_dim)

        # Unnormalize
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Slice to executable action steps (starting from n_obs_steps - 1)
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
        Compute BC loss in normalized space (diffusion_policy interface).

        Args:
            batch: {"obs": {"image": (B,T,C,3,H,W), "state": (B,T,D)}, "action": (B,T,Da)}
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        global_cond = self._build_global_cond(nobs)
        pred = self.decoder(global_cond)  # (B, horizon, action_dim)

        if self.loss_type == "l1":
            return F.l1_loss(pred, nactions)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, nactions)
        return F.mse_loss(pred, nactions)

    # ------------------------------------------------------------------ #
    # Stubs
    # ------------------------------------------------------------------ #

    def reset(self):
        pass
