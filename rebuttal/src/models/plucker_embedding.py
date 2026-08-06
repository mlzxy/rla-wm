"""Plücker-ray positional encoding for patch tokens.

A learned positional embedding only tells a transformer *which slot* a token sits in. Plücker
coordinates instead tell it *where in space the token is looking from*: each patch is turned into
the camera ray through its centre, encoded as ``(d, o x d)`` where ``d`` is the unit world-space
direction and ``o`` the camera centre. That representation is unique per ray, invariant to where
the origin sits along the ray, and directly comparable across cameras — so tokens from two views
that observe the same region get similar codes, which a slot index can never express.

Requires real camera intrinsics and camera-to-world extrinsics. See
`scripts/convert_libero_to_trajectory.py`: LIBERO's RLDS export ships no camera poses, so its
extrinsics are identity placeholders and Plücker encoding carries no signal on that dataset until
they are recovered from the simulator.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def compute_plucker_rays(
    intrinsics: Tensor,
    cam2world: Tensor,
    patch_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Tensor:
    """Plücker coordinates of the ray through each patch centre.

    Args:
        intrinsics: ``[B, Cam, 3, 3]`` pinhole intrinsics for the *resized* image.
        cam2world: ``[B, Cam, 4, 4]`` camera-to-world transforms. Note TrajectoryDataset stores
            these under the key ``w2c`` but does not invert them, so they are camera-to-world.
        patch_hw: ``(ph, pw)`` patch grid of the ViT.
        image_hw: ``(H, W)`` pixel size the intrinsics refer to.

    Returns:
        ``[B, Cam, ph * pw, 6]`` Plücker coordinates ``(direction, moment)``.
    """
    bsz, cams = intrinsics.shape[:2]
    ph, pw = patch_hw
    height, width = image_hw
    device, dtype = intrinsics.device, intrinsics.dtype

    # Pixel coordinates of patch centres.
    ys = (torch.arange(ph, device=device, dtype=dtype) + 0.5) * (height / ph)
    xs = (torch.arange(pw, device=device, dtype=dtype) + 0.5) * (width / pw)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), torch.ones(ph * pw, device=device, dtype=dtype)],
        dim=-1,
    )  # [Lp, 3]

    # Camera-space directions, then rotate into the world.
    inv_k = torch.linalg.inv(intrinsics.float()).to(dtype)  # [B, Cam, 3, 3]
    dirs_cam = torch.einsum("bcij,lj->bcli", inv_k, pixels)  # [B, Cam, Lp, 3]
    rotation = cam2world[..., :3, :3]  # [B, Cam, 3, 3]
    dirs_world = torch.einsum("bcij,bclj->bcli", rotation, dirs_cam)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    origins = cam2world[..., :3, 3].unsqueeze(2).expand_as(dirs_world)  # [B, Cam, Lp, 3]
    moments = torch.cross(origins, dirs_world, dim=-1)
    return torch.cat([dirs_world, moments], dim=-1)


class PluckerEmbedding(nn.Module):
    """Encode ``[..., 6]`` Plücker rays into ``out_channels`` transformer features.

    The raw 6-D coordinates are low-frequency, so they are lifted through a fixed sinusoidal
    basis (the standard NeRF-style encoding) before a learned linear projection.
    """

    def __init__(
        self,
        out_channels: int,
        num_frequencies: int = 8,
        include_input: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(num_frequencies).float(), persistent=False
        )

        in_channels = 6 * (2 * num_frequencies + (1 if include_input else 0))
        self.proj = nn.Linear(in_channels, out_channels)
        if zero_init:
            # Start as a no-op so adding the module cannot destabilise an existing recipe.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, rays: Tensor) -> Tensor:
        # rays: [..., 6]
        scaled = rays.unsqueeze(-1) * self.frequencies  # [..., 6, F]
        features = [torch.sin(scaled), torch.cos(scaled)]
        encoded = torch.cat(features, dim=-1).flatten(-2)  # [..., 6 * 2F]
        if self.include_input:
            encoded = torch.cat([rays, encoded], dim=-1)
        return self.proj(encoded)
