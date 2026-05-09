"""
Shared encoder modules used by VLA policies.

- MultiCameraEncoder: shared ResNet-18 backbone for multi-camera RGB inputs
- CLIPTextEncoder: frozen CLIP text encoder with learned projection
- StateEncoder: MLP for low-dim robot state embedding
"""

from typing import List, Mapping, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision

from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


# ---------------------------------------------------------------------------
# Image encoder
# ---------------------------------------------------------------------------

def _make_resnet18_encoder(out_dim: int = 512, use_group_norm: bool = True) -> nn.Module:
    """Return a ResNet-18 backbone with optional GroupNorm and global avg pool."""
    import torchvision.models as models
    from diffusion_policy.common.pytorch_util import replace_submodules

    resnet = models.resnet18(weights=None)
    if use_group_norm:
        resnet = replace_submodules(
            root_module=resnet,
            predicate=lambda m: isinstance(m, nn.BatchNorm2d),
            func=lambda m: nn.GroupNorm(num_groups=m.num_features // 16, num_channels=m.num_features),
        )
    # Remove the final FC layer; we only want features
    resnet.fc = nn.Identity()
    return resnet  # output: (B, 512)


class _SpatialResNet18(nn.Module):
    """Wrapper around a ResNet-18 that returns both pooled and spatial features
    in a single forward pass (no duplicated computation).

    When ``return_spatial=False`` (the default at forward time) it behaves
    identically to the original resnet.

    When ``return_spatial=True`` it returns a tuple:
        (pooled (B, 512), spatial (B, H'*W', 512))
    """

    def __init__(self, resnet: nn.Module):
        super().__init__()
        # Steal layers from the resnet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, H', W')

        pooled = self.avgpool(x)  # (B, 512, 1, 1)
        pooled = torch.flatten(pooled, 1)  # (B, 512)

        if return_spatial:
            B, C, H, W = x.shape
            spatial = x.flatten(2).transpose(1, 2)  # (B, H'*W', 512)
            return pooled, spatial
        return pooled


class MultiCameraEncoder(nn.Module):
    """Encode multi-camera RGBs with ResNet backbones then concatenate features.

    When ``share_backbone=False`` (default), each camera gets its own
    independent ResNet-18 — matching the behaviour of diffusion_policy's
    ``MultiImageObsEncoder`` with ``share_rgb_model=False``.

    Supports the same image augmentation pipeline as
    ``MultiImageObsEncoder``: resize → random/center crop → imagenet norm.
    """

    def __init__(
        self,
        num_cameras: int,
        per_camera_dim: int = 512,
        use_group_norm: bool = True,
        share_backbone: bool = False,
        # image augmentation (applied before the backbone)
        resize_shape: tuple[int, int] | None = None,
        crop_shape: tuple[int, int] | None = None,
        random_crop: bool = True,
        imagenet_norm: bool = False,
        # input_shape: used by CropRandomizer when resize_shape is None
        input_shape: tuple[int, int] | None = None,
        # spatial feature output
        return_spatial: bool = False,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.output_dim = per_camera_dim * num_cameras
        self.share_backbone = share_backbone
        self.return_spatial = return_spatial

        def _make_backbone():
            resnet = _make_resnet18_encoder(per_camera_dim, use_group_norm)
            if return_spatial:
                return _SpatialResNet18(resnet)
            return resnet

        if share_backbone:
            self.backbone = _make_backbone()
        else:
            self.backbones = nn.ModuleList([
                _make_backbone()
                for _ in range(num_cameras)
            ])

        # --- build per-image transform: resize → crop → normalise ---
        transform_parts: list[nn.Module] = []

        if resize_shape is not None:
            transform_parts.append(torchvision.transforms.Resize(size=resize_shape))

        if crop_shape is not None:
            h, w = crop_shape
            if random_crop:
                ch_in = 3
                if resize_shape is not None:
                    h_in, w_in = resize_shape[0], resize_shape[1]
                elif input_shape is not None:
                    h_in, w_in = input_shape[0], input_shape[1]
                else:
                    raise ValueError(
                        "CropRandomizer requires input_shape or resize_shape to be set")
                transform_parts.append(CropRandomizer(
                    input_shape=(ch_in, h_in, w_in),
                    crop_height=h,
                    crop_width=w,
                    num_crops=1,
                    pos_enc=False,
                ))
            else:
                transform_parts.append(torchvision.transforms.CenterCrop(size=(h, w)))

        if imagenet_norm:
            transform_parts.append(torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = nn.Sequential(*transform_parts) if transform_parts else None

    def forward(self, rgbs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        rgbs: (B, C, 3, H, W)  — C cameras
        Returns:
            When return_spatial=False: (B, C * per_camera_dim)
            When return_spatial=True:  tuple of
                pooled  (B, C * per_camera_dim),
                spatial (B, C * H' * W', per_camera_dim)
        """
        if rgbs.ndim == 4:
            rgbs = rgbs[:, None]
        B, C = rgbs.shape[:2]

        if self.transform is not None:
            x = rgbs.reshape(B * C, *rgbs.shape[2:])
            x = self.transform(x)
            rgbs = x.reshape(B, C, *x.shape[1:])

        if self.return_spatial:
            return self._forward_spatial(rgbs, B, C)

        if self.share_backbone:
            x = rgbs.reshape(B * C, *rgbs.shape[2:])
            feat = self.backbone(x)  # (B*C, D)
            feat = feat.reshape(B, C * feat.shape[-1])
        else:
            feats = []
            for i, backbone in enumerate(self.backbones):
                feats.append(backbone(rgbs[:, i]))  # (B, D)
            feat = torch.cat(feats, dim=-1)  # (B, C*D)
        return feat

    def _forward_spatial(
        self, rgbs: torch.Tensor, B: int, C: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that returns both pooled and spatial features."""
        if self.share_backbone:
            x = rgbs.reshape(B * C, *rgbs.shape[2:])
            pooled, spatial = self.backbone(x, return_spatial=True)
            # pooled: (B*C, D), spatial: (B*C, L, D)
            pooled = pooled.reshape(B, C * pooled.shape[-1])
            spatial = spatial.reshape(B, C * spatial.shape[1], spatial.shape[-1])
        else:
            pooled_list, spatial_list = [], []
            for i, backbone in enumerate(self.backbones):
                p, s = backbone(rgbs[:, i], return_spatial=True)
                pooled_list.append(p)    # (B, D)
                spatial_list.append(s)   # (B, L, D)
            pooled = torch.cat(pooled_list, dim=-1)    # (B, C*D)
            spatial = torch.cat(spatial_list, dim=1)    # (B, C*L, D)
        return pooled, spatial


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder with a learned projection head."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 128,
        freeze: bool = True,
        max_len: int = 77,
    ):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.freeze = freeze
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip_text = CLIPTextModel.from_pretrained(model_name)
        self.max_len = max_len
        clip_hidden = self.clip_text.config.hidden_size
        self.proj = nn.Linear(clip_hidden, output_dim)

        if self.freeze:
            for p in self.clip_text.parameters():
                p.requires_grad = False
            self.clip_text.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze", False):
            self.clip_text.eval()
        return self

    def forward(self, text_tokens: dict) -> torch.Tensor:
        outputs = self.clip_text(
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
        )
        return self.proj(outputs.pooler_output)

    def tokenize(self, prompts: List[str], device: torch.device) -> dict:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tokens.items()}


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """MLP that embeds the low-dim robot state (arm joints + gripper)."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)



class StateDictMixin:
    _LEARNABLE_STATE_MODULE_NAMES = (
        # "state_token_encoder",
        # "post_flow_decoder",
        # "action_decoder",
        # "normalizer",
    )
    _IGNORED_LOAD_PREFIXES = (
        # "latent_encoder.",
        # "flow_model.",
        # "dino_extractor.",
        # "goal_decoder.",
        # "goal_image_decoder.",
    )
    
    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()  # type: ignore[attr-defined]

        metadata = getattr(destination, "_metadata", None)
        if isinstance(metadata, OrderedDict):
            metadata[prefix[:-1]] = {"version": self._version}

        for module_name in self._LEARNABLE_STATE_MODULE_NAMES:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            module.state_dict(
                destination=destination,
                prefix=f"{prefix}{module_name}.",
                keep_vars=keep_vars,
            )
        return destination

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict).__name__}"
            )

        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        learnable_prefixes = tuple(
            f"{module_name}." for module_name in self._LEARNABLE_STATE_MODULE_NAMES
        )

        for key in state_dict.keys():
            if key.startswith(learnable_prefixes) or key in self._LEARNABLE_STATE_MODULE_NAMES:
                continue
            if key.startswith(self._IGNORED_LOAD_PREFIXES):
                continue
            unexpected_keys.append(key)

        for module_name in self._LEARNABLE_STATE_MODULE_NAMES:
            module_prefix = f"{module_name}."
            module = getattr(self, module_name, None)
            module_state_dict = OrderedDict(
                (key[len(module_prefix) :], value)
                for key, value in state_dict.items()
                if key.startswith(module_prefix)
            )
            if module is None:
                unexpected_keys.extend(
                    f"{module_name}.{key}" for key in module_state_dict.keys()
                )
                continue
            incompatible = module.load_state_dict(
                module_state_dict,
                strict=strict,
                assign=assign,
            )
            missing_keys.extend(
                f"{module_name}.{key}" for key in incompatible.missing_keys
            )
            unexpected_keys.extend(
                f"{module_name}.{key}" for key in incompatible.unexpected_keys
            )

        if strict and (missing_keys or unexpected_keys):
            error_msgs = []
            if unexpected_keys:
                error_msgs.append(
                    "Unexpected key(s) in state_dict: "
                    + ", ".join(f'\"{key}\"' for key in unexpected_keys)
                    + "."
                )
            if missing_keys:
                error_msgs.append(
                    "Missing key(s) in state_dict: "
                    + ", ".join(f'\"{key}\"' for key in missing_keys)
                    + "."
                )
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                + "\n\t".join(error_msgs)
            )

        return nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)