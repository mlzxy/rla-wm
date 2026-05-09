"""
Sim-to-Real Behavioral Cloning Policy for v4world.

Extends VLABCPolicy with:
  1. Frozen DINO encode → decode image preprocessing to bridge domain gap
     between world-model-generated images and real simulator images.
  2. RL tuning heads (setup_critic_tuning / setup_rl_tuning) with a simple
     MLP-based residual action branch.

The DINO + image decoder pipeline is frozen and only used as a preprocessing
step: raw images → DINO tokens → decoded images → resize → MultiCameraEncoder.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Mapping, Optional, cast

import torch
import torch.nn as nn
from rich import print
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributions.normal import Normal

from policies.policy.vla_bc_policy import VLABCPolicy
from policies.policy.utils import StateDictMixin
from src.models.dino_to_image_unet_v1 import DinoToImageDecoderV1
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels
from utils.misc import fetch_state_dict


def _require_peft() -> tuple[Any, Any]:
    """Import PEFT lazily so BC-only code paths do not hard-fail without it."""
    try:
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for VLABCS2RPolicy RL LoRA tuning. "
            "Install it with `uv sync` or `uv pip install peft`."
        ) from exc
    return peft.LoraConfig, peft.inject_adapter_in_model


# ------------------------------------------------------------------ #
# Residual MLP for RL fine-tuning
# ------------------------------------------------------------------ #


class ResidualActionMLP(nn.Module):
    """Simple MLP that produces (residual_mean, std) for RL fine-tuning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple = (512, 256),
        residual_mean_scale: float = 0.05,
        std_scale: float = 0.03,
        zero_init: bool = True,
    ):
        super().__init__()
        self.residual_mean_scale = residual_mean_scale
        self.std_scale = std_scale
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim * 2)

        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — global conditioning vector.
        Returns:
            residual_mean: (B, output_dim) — scaled via tanh.
            std:           (B, output_dim) — scaled via softplus.
        """
        h = self.backbone(x)
        h = self.head(h)
        raw_mean, raw_std = h.chunk(2, dim=-1)
        residual_mean = torch.tanh(raw_mean) * self.residual_mean_scale
        std = F.softplus(raw_std) * self.std_scale
        return residual_mean, std


# ------------------------------------------------------------------ #
# Helper: load image decoder from work dir
# ------------------------------------------------------------------ #


def _load_image_decoder_from_work_dir(
    work_dir: str,
    device: str = "cpu",
) -> DinoToImageDecoderV1:
    """Load a frozen DinoToImageDecoderV1 from a training work directory."""
    cfg = OmegaConf.load(os.path.join(work_dir, "config.yaml"))
    dec_cfg = cfg.models.decoder.args
    decoder = DinoToImageDecoderV1(
        in_channels=int(dec_cfg.in_channels),
        model_channels=int(dec_cfg.get("model_channels", 256)),
        out_channels=int(dec_cfg.get("out_channels", 3)),
        use_fp16=bool(dec_cfg.get("use_fp16", False)),
    )
    decoder.load_state_dict(
        fetch_state_dict("decoder", work_dir, device),
        strict=True,
    )
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


# ------------------------------------------------------------------ #
# Policy
# ------------------------------------------------------------------ #


class VLABCS2RPolicy(StateDictMixin, VLABCPolicy):
    """
    VLABCPolicy with frozen DINO encode→decode image preprocessing and
    a deferred residual action MLP for RL fine-tuning (setup_rl_tuning).
    No critic / value network — uses REINFORCE with clipping.
    """

    _LEARNABLE_STATE_MODULE_NAMES = (
        "img_encoder",
        "state_encoder",
        "decoder",
        "residual_net",
        "normalizer",
    )
    _IGNORED_LOAD_PREFIXES = (
        "dino_extractor.",
        "image_decoder.",
    )

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        # DINO preprocessing
        dino_channels: int = 1024,
        image_decoder_work_dir: str = "",
        skip_dino_preprocess: bool = False,
        # RL head hyperparams (stored for deferred creation)
        residual_hidden_dims: tuple = (512, 256),
        residual_mean_scale: float = 0.05,
        residual_std_scale: float = 0.03,
        enable_rl_lora: bool = False,
        lora_img_encoder: bool = True,
        lora_state_encoder: bool = True,
        lora_decoder: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_adapter_name: str = "rl",
        lora_use_rslora: bool = False,
        # passed through to parent
        **kwargs,
    ):
        # Force enable_rl_heads=False in parent — we manage RL heads ourselves
        kwargs.pop("enable_rl_heads", None)
        # Drop legacy critic kwargs that may come from old configs
        kwargs.pop("critic_latent_dim", None)
        kwargs.pop("value_hidden_dims", None)
        super().__init__(
            action_dim=action_dim,
            state_dim=state_dim,
            enable_rl_heads=False,
            **kwargs,
        )

        # Skip DINO encode→decode when images already come from DINO decode
        self.skip_dino_preprocess = skip_dino_preprocess

        # Store RL hyperparams for deferred construction
        self._residual_hidden_dims = tuple(residual_hidden_dims)
        self._residual_mean_scale = residual_mean_scale
        self._residual_std_scale = residual_std_scale
        self.enable_rl_lora = bool(enable_rl_lora)
        self.lora_img_encoder = bool(lora_img_encoder)
        self.lora_state_encoder = bool(lora_state_encoder)
        self.lora_decoder = bool(lora_decoder)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_adapter_name = lora_adapter_name
        self.lora_use_rslora = bool(lora_use_rslora)

        # --- Frozen DINO feature extractor ---
        dino_model_name = get_dinov3_model_for_channels(dino_channels)
        self.dino_extractor = DINOv3FeatureExtractor(
            model_name=dino_model_name,
            use_compile=False,
        )
        self.dino_extractor.eval()
        for p in self.dino_extractor.parameters():
            p.requires_grad = False
        self.dino_channels = dino_channels

        # --- Frozen image decoder ---
        if image_decoder_work_dir:
            self.image_decoder: Optional[DinoToImageDecoderV1] = (
                _load_image_decoder_from_work_dir(image_decoder_work_dir, device="cpu")
            )
        else:
            self.image_decoder = None

        # --- RL head (created lazily via setup_rl_tuning) ---
        self.residual_net: Optional[ResidualActionMLP] = None

    # ------------------------------------------------------------------ #
    # DINO encode → decode image preprocessing (frozen)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run normalized images through frozen DINO encoder → image decoder → resize,
        then re-normalize.

        Args:
            images: (B, Cam, 3, H, W) in **normalized** space (from LinearNormalizer).
        Returns:
            (B, Cam, 3, H, W) decoded images, re-normalized back to normalizer space.
        """
        if self.skip_dino_preprocess or self.image_decoder is None:
            # import pudb; pudb.set_trace()
            return images

        B, Cam, C, H, W = images.shape

        # Unnormalize: normalized → raw [0, 1] images
        # normalizer expects (B, ...) so flatten Cam into batch
        raw = self.normalizer["image"].unnormalize(images.reshape(B * Cam, C, H, W))
        raw = raw.reshape(B, Cam, C, H, W)

        # Flatten cameras into batch
        flat = raw.reshape(B * Cam, C, H, W).float()
        if flat.max() > 1.5:
            flat = flat / 255.0

        # Extract DINO patch tokens: (B*Cam, D, pH, pW)
        _, patch_grid = self.dino_extractor(flat, return_spatial_grid=True)
        _, channels, patch_h, patch_w = patch_grid.shape

        # Reshape to decoder input: (B, Cam, Lp, D)
        Lp = patch_h * patch_w
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)  # (B*Cam, Lp, D)
        patch_tokens = patch_tokens.reshape(B, Cam, Lp, channels)

        # Decode to images: (B, Cam, 3, pH*16, pW*16)
        decoded = self.image_decoder(
            patch_tokens.contiguous(),
            patch_hw=(patch_h, patch_w),
        ).clamp(0.0, 1.0)

        # Resize back to original resolution
        if decoded.shape[-2] != H or decoded.shape[-1] != W:
            decoded_flat = decoded.reshape(B * Cam, 3, decoded.shape[-2], decoded.shape[-1])
            decoded_flat = F.interpolate(
                decoded_flat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            decoded = decoded_flat.reshape(B, Cam, 3, H, W)

        # Re-normalize: raw [0, 1] → normalized space
        result = self.normalizer["image"].normalize(decoded.reshape(B * Cam, C, H, W))
        return result.reshape(B, Cam, C, H, W)

    # ------------------------------------------------------------------ #
    # Override _encode_obs to use DINO-preprocessed images
    # ------------------------------------------------------------------ #

    def _encode_obs(
        self, nobs: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observations with DINO-decoded image preprocessing."""
        B = nobs['state'].shape[0]
        To = self.n_obs_steps

        img = nobs['image'][:, :To]    # (B, To, Cam, 3, H, W)
        state = nobs['state'][:, :To]  # (B, To, D)

        BTo = B * To
        C_cam = img.shape[2]

        # Flatten batch×time for preprocessing
        img_flat = img.reshape(BTo, C_cam, *img.shape[3:])  # (BTo, Cam, 3, H, W)

        # DINO encode → decode preprocessing (frozen, no grad)
        img_flat = self._preprocess_images(img_flat)

        # Reshape to (BTo, Cam, 3, H, W) for parent's MultiCameraEncoder
        state_flat = state.reshape(BTo, -1)

        img_feat = self.img_encoder(img_flat)        # (BTo, D_img)

        img_feat = img_feat.reshape(B, -1)

        state_encoder = self.state_encoder
        if self.use_state and state_encoder is not None:
            state_feat = state_encoder(state_flat)  # (BTo, D_state)
            state_feat = state_feat.reshape(B, -1)
        else:
            state_feat = None

        return img_feat, state_feat

    # ------------------------------------------------------------------ #
    # Override predict_action for residual sampling
    # ------------------------------------------------------------------ #

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        deterministic: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict action chunk, optionally adding residual from RL head.

        If ``residual_net`` is None, falls back to the parent deterministic
        prediction.  Otherwise builds a Normal distribution from the base
        decoder mean + residual and either samples or returns the mean.

        ``deterministic`` defaults to the ``_deterministic_eval`` attribute
        (set by the training loop) when not passed explicitly.
        """
        if self.residual_net is None:
            return super().predict_action(obs_dict)

        if deterministic is None:
            deterministic = getattr(self, "_deterministic_eval", False) or False

        nobs = cast(Dict[str, torch.Tensor], self.normalizer.normalize(obs_dict))
        global_cond = self._build_global_cond(nobs)
        mean, std = self.forward_action_dist(global_cond)

        if deterministic:
            action_flat = mean
        else:
            dist = Normal(mean, std)
            action_flat = dist.sample()

        # Reshape to (B, chunk, action_dim) and unnormalize
        action_norm = action_flat.view(-1, self.n_action_steps, self.action_dim)
        action = self.normalizer["action"].unnormalize(action_norm)

        return {"action": action}

    # ------------------------------------------------------------------ #
    # RL phase setup (deferred construction)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        """Enable or disable gradients for a whole module subtree."""
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def _set_lora_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        """Toggle gradients only for PEFT LoRA parameters in a module subtree."""
        if module is None:
            return
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.requires_grad = requires_grad

    @staticmethod
    def _module_has_lora(module: Optional[nn.Module]) -> bool:
        """Return True when a module subtree already contains LoRA parameters."""
        if module is None:
            return False
        return any("lora_" in name for name, _ in module.named_parameters())

    @staticmethod
    def _collect_target_modules(
        module: nn.Module,
        layer_types: tuple[type[nn.Module], ...],
    ) -> list[str]:
        """Collect target module names relative to a subtree root for PEFT."""
        return [
            name
            for name, child in module.named_modules()
            if name and isinstance(child, layer_types)
        ]

    def _iter_img_backbones(self) -> list[nn.Module]:
        """Return the concrete ResNet backbone modules from the image encoder."""
        if getattr(self.img_encoder, "share_backbone", False):
            backbone = getattr(self.img_encoder, "backbone", None)
            return [backbone] if backbone is not None else []
        backbones = getattr(self.img_encoder, "backbones", None)
        if backbones is None:
            return []
        return list(backbones)

    def _build_lora_config(self, target_modules: list[str]) -> Any:
        """Create a LoRA config for a custom nn.Module subtree."""
        LoraConfig, _ = _require_peft()
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_rslora=self.lora_use_rslora,
            inference_mode=False,
        )

    def _inject_lora_into_module(
        self,
        module: Optional[nn.Module],
        target_modules: list[str],
    ) -> None:
        """Inject LoRA adapters into a subtree once, in-place."""
        if module is None or not target_modules or self._module_has_lora(module):
            return

        _, inject_adapter_in_model = _require_peft()
        inject_adapter_in_model(
            self._build_lora_config(target_modules),
            module,
            adapter_name=self.lora_adapter_name,
        )

    def _setup_decoder_lora(self) -> None:
        """Inject LoRA into the action decoder MLP."""
        target_modules = self._collect_target_modules(self.decoder, (nn.Linear,))
        self._inject_lora_into_module(self.decoder, target_modules)

    def _setup_state_encoder_lora(self) -> None:
        """Inject LoRA into the low-dimensional state encoder."""
        if self.state_encoder is None:
            return
        target_modules = self._collect_target_modules(self.state_encoder, (nn.Linear,))
        self._inject_lora_into_module(self.state_encoder, target_modules)

    def _setup_img_encoder_lora(self) -> None:
        """Inject LoRA into each concrete ResNet backbone in the image encoder."""
        for backbone in self._iter_img_backbones():
            target_modules = self._collect_target_modules(backbone, (nn.Conv2d,))
            self._inject_lora_into_module(backbone, target_modules)

    def _ensure_rl_tuning_modules(self) -> None:
        """Create RL-only modules and inject LoRA adapters if configured."""
        if self.residual_net is None:
            self.residual_net = self._create_residual_net().to(
                device=self.device,
                dtype=self.dtype,
            )

        if not self.enable_rl_lora:
            return
        else:
            print("[red]VLABCS2RPolicy: Enabling RL LoRA tuning[/red]")

        if self.lora_decoder:
            self._setup_decoder_lora()
        if self.lora_state_encoder:
            self._setup_state_encoder_lora()
        if self.lora_img_encoder:
            self._setup_img_encoder_lora()

    @staticmethod
    def _state_dict_has_rl_tuning_state(state_dict: Mapping[str, Any]) -> bool:
        """Detect whether an incoming checkpoint contains RL tuning state."""
        return any(
            key.startswith("residual_net.") or ".lora_" in key
            for key in state_dict.keys()
        )

    def _configure_rl_trainable_parameters(self) -> None:
        """Freeze base weights while keeping RL adapters and residual head trainable."""
        self._set_module_requires_grad(self.decoder, False)
        self._set_module_requires_grad(self.state_encoder, False)
        self._set_module_requires_grad(self.img_encoder, False)

        if self.enable_rl_lora:
            if self.lora_decoder:
                self._set_lora_requires_grad(self.decoder, True)
            if self.lora_state_encoder:
                self._set_lora_requires_grad(self.state_encoder, True)
            if self.lora_img_encoder:
                self._set_lora_requires_grad(self.img_encoder, True)

        self._set_module_requires_grad(self.residual_net, True)

    def _create_residual_net(self) -> ResidualActionMLP:
        """Build residual action MLP."""
        output_dim = self.n_action_steps * self.action_dim
        return ResidualActionMLP(
            input_dim=self._cond_dim,
            output_dim=output_dim,
            hidden_dims=self._residual_hidden_dims,
            residual_mean_scale=self._residual_mean_scale,
            std_scale=self._residual_std_scale,
        )

    def setup_rl_tuning(self) -> None:
        """Inject RL LoRA adapters and train only adapters plus residual head."""
        self._ensure_rl_tuning_modules()
        self._configure_rl_trainable_parameters()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if self._state_dict_has_rl_tuning_state(state_dict):
            self._ensure_rl_tuning_modules()
            self._configure_rl_trainable_parameters()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------ #
    # RL heads (used by BCRLAgent)
    # ------------------------------------------------------------------ #

    def forward_features(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the global conditioning vector."""
        return self._build_global_cond(nobs)

    def forward_action_dist(
        self, global_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of the action-chunk distribution.

        mean: (B, n_action_steps * action_dim) in normalized action space.
        std:  (B, n_action_steps * action_dim).
        """
        if self.residual_net is None:
            raise RuntimeError(
                "VLABCS2RPolicy.forward_action_dist called but residual_net not set up. "
                "Call setup_rl_tuning() first."
            )

        # Base mean from frozen decoder
        naction_pred = self.decoder(global_cond)  # (B, horizon, action_dim)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        base_chunk = naction_pred[:, start:end]  # (B, K, A)
        base_mean = base_chunk.reshape(base_chunk.shape[0], -1)  # (B, K*A)

        # Residual from trainable MLP
        residual_mean, std = self.residual_net(global_cond)
        mean = base_mean + residual_mean
        return mean, std
