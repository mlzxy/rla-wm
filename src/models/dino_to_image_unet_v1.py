from typing import Optional

import torch
import torch.nn as nn
from src.modules.utils import convert_module_to_f16, convert_module_to_f32


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + res)


class UpsampleBlock(nn.Module):
    """
    A basic convolutional upsampling block with additional capacity via ResBlocks.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor
        )
        self.resblock = nn.Sequential(
            ResBlock(out_channels),
            ResBlock(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.resblock(x)
        return x


class DinoToImageDecoderV1(nn.Module):
    """
    Decodes DINOv3 features back to an RGB image.
    We upsample the spatial grid by 16x via four progressive 2x blocks.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int = 256,
        out_channels: int = 3,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.GELU(),
            ResBlock(model_channels),
            ResBlock(model_channels)
        )

        # Four progressive 2x upsample stages (16x total).
        self.up1 = UpsampleBlock(model_channels, model_channels // 2, scale_factor=2)
        self.up2 = UpsampleBlock(model_channels // 2, model_channels // 4, scale_factor=2)
        self.up3 = UpsampleBlock(model_channels // 4, model_channels // 8, scale_factor=2)
        self.up4 = UpsampleBlock(model_channels // 8, model_channels // 16, scale_factor=2)

        self.out_conv = nn.Sequential(
            ResBlock(model_channels // 16),
            nn.Conv2d(model_channels // 16, out_channels, kernel_size=3, padding=1)
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def convert_to_fp16(self) -> None:
        """Convert model to fp16."""
        self.dtype = torch.float16
        self.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert model to fp32."""
        self.dtype = torch.float32
        self.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        mod: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        patch_hw: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, Cam, Lp, C)
            mod: Dummy argument for compatibility
            cond: Dummy argument for compatibility
            patch_hw: Original patch grid dimensions (pH, pW)
        Returns:
            (B, Cam, 3, pH*16, pW*16)
        """
        bsz, cams, lp, ch = x.shape
        
        # Determine patch spatial dimensions
        if patch_hw is not None:
            ph, pw = patch_hw
        else:
            # Assume square if not provided
            ph = pw = int(lp ** 0.5)
            assert ph * pw == lp, "Patch sequence length must be a perfect square if patch_hw is not provided."

        # Reshape to (B*Cam, C, pH, pW)
        h = x.view(bsz * cams, ph, pw, ch).permute(0, 3, 1, 2)
        h = h.type(self.dtype)

        # Progressive upsampling
        h = self.input_proj(h)         # (B*Cam, M, pH, pW)
        h = self.up1(h)                # (B*Cam, M/2, pH*2, pW*2)
        h = self.up2(h)                # (B*Cam, M/4, pH*4, pW*4)
        h = self.up3(h)                # (B*Cam, M/8, pH*8, pW*8)
        h = self.up4(h)                # (B*Cam, M/16, pH*16, pW*16)
        out = self.out_conv(h)         # (B*Cam, 3, pH*16, pW*16)

        # Output shape is (B, Cam, 3, H, W)
        _, _, H, W = out.shape
        out = out.view(bsz, cams, self.out_channels, H, W)
        
        return out.float()
