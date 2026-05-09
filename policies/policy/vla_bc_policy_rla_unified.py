"""
Unified Spatial-Attention BC Policy with Robot Latent Action (RLA) for v4world.

All predictions (raw actions + latent actions) are routed through a single
spatial-attention pathway with learnable output queries, followed by a shared
wide MLP and two separate linear heads.

For few-shot mixed training:
  - Samples with robot obs/actions: state encoding is projected to a token
    and added with a learnable ``proprio_token``.
  - Samples without robot obs/actions (pixel-only): the ``proprio_token``
    alone replaces the state token.

Architecture:
  MultiCameraEncoder (spatial) → img spatial tokens  (B, L, D_spatial)
  StateEncoder → state_proj → state token  (B, To, H)
  proprio_token logic → state tokens  (B, To, H)

  Compose: [spatial_proj(img_spatial), state_tokens, output_queries]
  → N × AttentionBlock (self-attention)
  → extract output_queries  (B, N_q, H)
  → flatten → shared wide MLP → shared_feat
  → action_head → (B, horizon, action_dim)
  → latent_head → (B, num_tokens, token_dim)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from diffusion_policy.model.common.normalizer import LinearNormalizer
from policies.policy.utils import MultiCameraEncoder, StateDictMixin, StateEncoder
from src.models.attention_block import AttentionBlock
from src.models.simple_token_transformer import SimpleTokenTransformer
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels
from utils.misc import fetch_state_dict


# ------------------------------------------------------------------ #
# Helper: load frozen inverse-dynamics encoder
# ------------------------------------------------------------------ #


def _load_encoder_from_work_dir(
    work_dir: str,
    device: str = "cpu",
) -> tuple[SimpleTokenTransformer, int]:
    """Load an inverse-dynamics encoder from a training run directory."""
    import os
    from omegaconf import OmegaConf

    config_path = os.path.join(work_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)

    dino_channels = int(OmegaConf.select(cfg, "vars.dino_channels", default=1024))

    enc_cfg = cfg.models.encoder.args
    encoder = SimpleTokenTransformer(
        in_channels=int(enc_cfg.in_channels),
        model_channels=int(enc_cfg.model_channels),
        out_channels=int(enc_cfg.out_channels),
        num_blocks=int(enc_cfg.num_blocks),
        num_heads=int(enc_cfg.num_heads),
        mlp_ratio=float(enc_cfg.get("mlp_ratio", 4.0)),
        use_fp16=bool(enc_cfg.get("use_fp16", False)),
        num_tokens=int(enc_cfg.num_tokens),
    )

    state_dict = fetch_state_dict("encoder", work_dir, device)
    encoder.load_state_dict(state_dict, strict=True)
    print(f"[green]Latent encoder loaded from {work_dir}[/green]")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, dino_channels


# ------------------------------------------------------------------ #
# Policy
# ------------------------------------------------------------------ #


class VLABCPolicyRLAUnified(StateDictMixin, nn.Module):
    """
    Unified spatial-attention BC policy with Robot Latent Action (RLA).

    All predictions flow through a single spatial-attention trunk followed by
    a shared wide MLP.  Raw actions and latent actions are predicted by
    separate linear heads on top of the shared representation.

    Follows the diffusion_policy interface:
      - ``set_normalizer(normalizer)``
      - ``compute_loss(batch)`` → dict of losses
      - ``predict_action(obs_dict)`` → raw actions (latent discarded at eval)
    """

    _LEARNABLE_STATE_MODULE_NAMES = (
        "img_encoder",
        "state_encoder",
        "state_proj",
        "learnable_tokens",
        "spatial_proj",
        "attn_blocks",
        "output_norm",
        "shared_mlp",
        "action_head",
        "latent_head",
        "normalizer",
    )
    _IGNORED_LOAD_PREFIXES = (
        "latent_encoder.",
        "dino_extractor.",
    )

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
        state_hidden_dim: int = 256,
        state_output_dim: int = 256,
        # spatial attention
        spatial_hidden_dim: int = 256,
        spatial_num_blocks: int = 6,
        spatial_num_heads: int = 8,
        spatial_mlp_ratio: float = 4.0,
        # output queries & shared MLP
        num_output_queries: int = 16,
        shared_mlp_dim: int = 2048,
        mlp_dropout: float = 0.1,
        # loss
        loss_type: str = "mse",
        # latent-action (RLA) params
        use_latent_head: bool = True,
        latent_encoder_work_dir: str = "",
        lambda_latent_l1: float = 1.0,
        lambda_latent_mse: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.loss_type = loss_type
        self.use_latent_head = use_latent_head
        self.lambda_latent_l1 = lambda_latent_l1
        self.lambda_latent_mse = lambda_latent_mse

        # --- normalizer ---
        self.normalizer = LinearNormalizer()

        # --- image encoder (trainable, spatial mode) ---
        self.img_encoder = MultiCameraEncoder(
            num_cameras, per_camera_dim, use_group_norm,
            share_backbone=share_backbone,
            resize_shape=tuple(resize_shape) if resize_shape is not None else None,
            crop_shape=tuple(crop_shape) if crop_shape is not None else None,
            random_crop=random_crop,
            imagenet_norm=imagenet_norm,
            input_shape=(img_size, img_size) if img_size is not None else None,
            return_spatial=True,
        )

        # --- state encoder (trainable) ---
        self.state_encoder = StateEncoder(state_dim, state_hidden_dim, state_output_dim)
        self.state_proj = nn.Linear(state_output_dim, spatial_hidden_dim)

        # --- learnable tokens ---
        self.learnable_tokens = nn.ParameterDict({
            "proprio_token": nn.Parameter(torch.randn(1, 1, spatial_hidden_dim) * 0.02),
            "output_queries": nn.Parameter(torch.randn(1, num_output_queries, spatial_hidden_dim) * 0.02),
        })

        # --- spatial attention trunk ---
        self.spatial_proj = nn.Linear(per_camera_dim, spatial_hidden_dim)
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(
                channels=spatial_hidden_dim,
                num_heads=spatial_num_heads,
                mlp_ratio=spatial_mlp_ratio,
            )
            for _ in range(spatial_num_blocks)
        ])
        self.output_norm = nn.LayerNorm(spatial_hidden_dim)

        # --- shared wide MLP ---
        query_flat_dim = num_output_queries * spatial_hidden_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(query_flat_dim, shared_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(shared_mlp_dim, shared_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
        )

        # --- output heads ---
        self.action_head = nn.Linear(shared_mlp_dim, horizon * action_dim)
        self._num_output_queries = num_output_queries

        # ============================================================== #
        # Frozen DINO extractor + inverse-dynamics encoder (for GT)
        # Only created when use_latent_head=True
        # ============================================================== #
        if use_latent_head:
            if not latent_encoder_work_dir:
                raise ValueError(
                    "latent_encoder_work_dir is required when use_latent_head=True "
                    "— point it at a training run directory."
                )

            self.latent_encoder, dino_channels = _load_encoder_from_work_dir(
                latent_encoder_work_dir
            )
            self.dino_channels = dino_channels

            dino_model_name = get_dinov3_model_for_channels(dino_channels)
            self.dino_extractor = DINOv3FeatureExtractor(model_name=dino_model_name)
            self.dino_extractor.eval()
            for p in self.dino_extractor.parameters():
                p.requires_grad = False

            latent_num_tokens = self.latent_encoder.num_tokens
            latent_token_dim = self.latent_encoder.out_channels

            self.latent_head = nn.Linear(shared_mlp_dim, latent_num_tokens * latent_token_dim)
            self._latent_num_tokens = latent_num_tokens
            self._latent_token_dim = latent_token_dim
        else:
            self.latent_encoder = None
            self.dino_extractor = None
            self.latent_head = None
            self._latent_num_tokens = 0
            self._latent_token_dim = 0

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
    # DINO helpers (frozen, no grad)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _extract_dino_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extract flattened DINO patch tokens from multi-camera images.

        Args:
            images: (B, Cam, 3, H, W) float, [0, 1].
        Returns:
            tokens: (B, Cam * Lp, C)
        """
        B, Cam = images.shape[:2]
        imgs = images.float()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
        flat = imgs.reshape(B * Cam, *imgs.shape[2:])
        _, patch_grid = self.dino_extractor(flat, return_spatial_grid=True)
        _, C, pH, pW = patch_grid.shape
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)  # (B*Cam, Lp, C)
        tokens = patch_tokens.reshape(B, Cam * pH * pW, C)
        return tokens

    @torch.no_grad()
    def _compute_gt_latent_action(
        self,
        img_first: torch.Tensor,
        img_last: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ground-truth latent action from first and last frames.

        Args:
            img_first: (B, Cam, 3, H, W) — frame at t=0.
            img_last:  (B, Cam, 3, H, W) — frame at t=T.
        Returns:
            enc_tokens: (B, num_tokens, C)
        """
        x0 = self._extract_dino_tokens(img_first)
        xT = self._extract_dino_tokens(img_last)
        enc_input = xT - x0
        enc_tokens, _ = self.latent_encoder(enc_input)
        return enc_tokens

    # ------------------------------------------------------------------ #
    # Obs encoding + attention trunk
    # ------------------------------------------------------------------ #

    def _forward_trunk(
        self,
        nobs: Dict[str, torch.Tensor],
        has_robot_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations and run through the spatial attention trunk.

        Args:
            nobs: Normalized observations with 'image' and 'state' keys.
            has_robot_data: (B,) bool tensor or None. When provided, pixel-only
                samples use the learnable proprio_token instead of encoded state.

        Returns:
            shared_feat: (B, shared_mlp_dim) — output of the shared MLP.
        """
        B = nobs['state'].shape[0]
        To = self.n_obs_steps

        img = nobs['image'][:, :To]    # (B, To, Cam, 3, H, W)
        state = nobs['state'][:, :To]  # (B, To, D)

        BTo = B * To
        C_cam = img.shape[2]
        img_flat = img.reshape(BTo, C_cam, *img.shape[3:])
        state_flat = state.reshape(BTo, -1)

        # Image: pooled + spatial
        _, img_spatial = self.img_encoder(img_flat)  # (BTo, L, D_spatial)
        img_spatial = img_spatial.reshape(B, -1, img_spatial.shape[-1])  # (B, To*L, D_spatial)
        img_tokens = self.spatial_proj(img_spatial)  # (B, To*L, H)

        # State: encode → project → tokens
        state_feat = self.state_encoder(state_flat)  # (BTo, state_output_dim)
        state_tokens = self.state_proj(state_feat)   # (BTo, H)
        state_tokens = state_tokens.reshape(B, To, -1)  # (B, To, H)

        # Proprio token logic
        proprio = self.learnable_tokens["proprio_token"]  # (1, 1, H)
        proprio_expanded = proprio.expand(B, To, -1)      # (B, To, H)

        if has_robot_data is not None and not has_robot_data.all():
            # mask: (B, 1, 1) for broadcasting
            robot_mask = has_robot_data.bool().view(B, 1, 1)
            state_tokens = torch.where(
                robot_mask,
                state_tokens + proprio_expanded,
                proprio_expanded,
            )
        else:
            state_tokens = state_tokens + proprio_expanded

        # Compose sequence: [img_tokens, state_tokens, output_queries]
        output_queries = self.learnable_tokens["output_queries"].expand(B, -1, -1)
        seq = torch.cat([img_tokens, state_tokens, output_queries], dim=1)

        # Self-attention blocks
        for block in self.attn_blocks:
            seq = block(seq)

        # Extract output query positions
        out_tokens = seq[:, -self._num_output_queries:]  # (B, N_q, H)
        out_tokens = self.output_norm(out_tokens)

        # Shared MLP
        flat = out_tokens.reshape(B, -1)        # (B, N_q * H)
        shared_feat = self.shared_mlp(flat)      # (B, shared_mlp_dim)
        return shared_feat

    # ------------------------------------------------------------------ #
    # Decode heads
    # ------------------------------------------------------------------ #

    def _decode_action(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """(B, shared_mlp_dim) → (B, horizon, action_dim)"""
        return self.action_head(shared_feat).reshape(-1, self.horizon, self.action_dim)

    def _decode_latent(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """(B, shared_mlp_dim) → (B, num_tokens, token_dim)"""
        return self.latent_head(shared_feat).reshape(
            -1, self._latent_num_tokens, self._latent_token_dim
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Predict action chunk (latent output discarded at eval)."""
        nobs = self.normalizer.normalize(obs_dict)
        shared_feat = self._forward_trunk(nobs)
        naction_pred = self._decode_action(shared_feat)

        action_pred = self.normalizer['action'].unnormalize(naction_pred)

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

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute BC loss + latent-action loss.

        Returns a dict:
            - ``loss``: total scalar for backward().
            - ``action_loss``: raw-action reconstruction loss.
            - ``latent_action_l1``: L1 on latent action.
            - ``latent_action_mse``: MSE on latent action.

        When ``batch["has_robot_data"]`` is present, action loss is computed
        only on robot-data samples; latent losses on all samples.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        has_robot = batch.get("has_robot_data", None)

        # Forward through unified trunk
        shared_feat = self._forward_trunk(nobs, has_robot_data=has_robot)

        # Decode action head
        pred_action = self._decode_action(shared_feat)

        # --- action loss (robot-data samples only) ---
        if has_robot is not None and not has_robot.all():
            robot_mask = has_robot.bool()
            if robot_mask.any():
                pred_r = pred_action[robot_mask]
                tgt_r = nactions[robot_mask]
                if self.loss_type == "l1":
                    action_loss = F.l1_loss(pred_r, tgt_r)
                elif self.loss_type == "smooth_l1":
                    action_loss = F.smooth_l1_loss(pred_r, tgt_r)
                else:
                    action_loss = F.mse_loss(pred_r, tgt_r)
            else:
                action_loss = torch.tensor(0.0, device=self.device)
        else:
            if self.loss_type == "l1":
                action_loss = F.l1_loss(pred_action, nactions)
            elif self.loss_type == "smooth_l1":
                action_loss = F.smooth_l1_loss(pred_action, nactions)
            else:
                action_loss = F.mse_loss(pred_action, nactions)

        # --- latent losses (all samples, only when latent head is active) ---
        if self.use_latent_head:
            pred_latent = self._decode_latent(shared_feat)

            raw_images = batch['obs']['image']  # (B, T, Cam, 3, H, W)
            img_first = raw_images[:, self.n_obs_steps - 1]
            img_last = raw_images[:, -1]
            gt_latent = self._compute_gt_latent_action(img_first, img_last) / 10.0

            latent_l1 = F.l1_loss(pred_latent, gt_latent)
            latent_mse = F.mse_loss(pred_latent, gt_latent)

            total_loss = (
                action_loss
                + self.lambda_latent_l1 * latent_l1
                + self.lambda_latent_mse * latent_mse
            )

            return {
                "loss": total_loss,
                "action_loss": action_loss,
                "latent_action_l1": latent_l1,
                "latent_action_mse": latent_mse,
            }

        return action_loss

    # ------------------------------------------------------------------ #
    # Stubs
    # ------------------------------------------------------------------ #

    def reset(self):
        pass
