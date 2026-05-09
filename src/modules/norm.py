import torch
import torch.nn as nn


class RMSNorm32(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)


class BatchNorm32(nn.BatchNorm2d):
    """A BatchNorm2d layer that computes in fp32 and returns input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)


class ChannelRMSNorm32(RMSNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        # N, C, D, H, W -> N, D, H, W, C
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        # N, D, H, W, C -> N, C, D, H, W
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x
