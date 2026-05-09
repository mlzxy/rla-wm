import torch
from torch import Tensor
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from src.representations import Gaussian


@torch.compile
def flat_grid_sample(
    input: torch.Tensor,
    coords: torch.Tensor,
    batch_inds: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Samples from a batch of images using arbitrary (flat) coordinates and batch indices.
    Equivalent to F.grid_sample but allows a variable number of points per batch item
    via a flat coordinate list.

    Args:
        input (Tensor): Source batch of images. Shape (B, C, H, W).
        coords (Tensor): 2D normalized coordinates in range [-1, 1]. Shape (N, 2).
                         Format is (x, y).
        batch_inds (Tensor): Batch index for each point in `coords`. Shape (N,).
                             Values must be in range [0, B-1].
        align_corners (bool): Geometrically, we consider the pixels of the input as
                              squares rather than points. If set to True, the extrema
                              (-1 and 1) are considered as referring to the center
                              points of the input's corner pixels. If False, they
                              refer to the corner points of the input's corner pixels.
                              Default: False.

    Returns:
        Tensor: Sampled values. Shape (N, C).
    """
    B, C, H, W = input.shape

    # Unpack x and y coordinates
    x_norm = coords[:, 0]
    y_norm = coords[:, 1]

    # Convert normalized coordinates [-1, 1] to pixel coordinates
    if align_corners:
        x = ((x_norm + 1) / 2) * (W - 1)
        y = ((y_norm + 1) / 2) * (H - 1)
    else:
        # grid_sample standard: -1 -> -0.5, 1 -> W-0.5
        x = ((x_norm + 1) * W - 1) / 2
        y = ((y_norm + 1) * H - 1) / 2

    # Get the corner pixel coordinates (top-left, top-right, etc.)
    # We use floor() to get the top-left integer coordinate
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp coordinates to be within image bounds
    # Note: This effectively repeats the border pixels (padding_mode='border')
    x0_c = torch.clamp(x0, 0, W - 1)
    x1_c = torch.clamp(x1, 0, W - 1)
    y0_c = torch.clamp(y0, 0, H - 1)
    y1_c = torch.clamp(y1, 0, H - 1)

    # Compute bilinear weights
    # We detach gradients for indices, but weights allow backprop to flow to input coords
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # Advanced Indexing to gather pixel values
    # We want shape (N, C).
    # input[...] gives (N, C) directly because we index the B, H, and W dims
    # with tensors of size N, while keeping dim 1 (C) as a slice.
    Ia = input[batch_inds, :, y0_c, x0_c]
    Ib = input[batch_inds, :, y1_c, x0_c]
    Ic = input[batch_inds, :, y0_c, x1_c]
    Id = input[batch_inds, :, y1_c, x1_c]

    # Weighted sum
    # Weights are (N,), Inputs are (N, C). Unsqueeze weights for broadcasting.
    out = (
        wa.unsqueeze(1) * Ia
        + wb.unsqueeze(1) * Ib
        + wc.unsqueeze(1) * Ic
        + wd.unsqueeze(1) * Id
    )

    return out


def depth_to_point_cloud_torch(
    depth: Tensor,
    rgb: Optional[Tensor] = None,
    intrinsics: Optional[Tensor] = None,
    w2c: Optional[Tensor] = None,
    max_depth: float = 5.0,
    foreground_mask: Optional[Tensor] = None,
    attrs: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None,
) -> Tuple[
    Tensor, Optional[Tensor], Tensor, Tensor, Union[Tensor, Tuple[Tensor, ...], None]
]:
    """
    Differentiable batched RGBD -> point cloud in PyTorch.

    Args:
        depth: Tensor of shape (B, H, W) or (H, W), depth in meters.
        rgb: Optional tensor of shape (B, H, W, 3), (H, W, 3), (B, 3, H, W), or (3, H, W).
        intrinsics: Tensor of shape (B, 3, 3) or (3, 3).
        w2c: Optional tensor of shape (B, 4, 4) or (4, 4), world-to-camera.
        max_depth: Maximum valid depth in meters.
        foreground_mask: Optional tensor of shape (B, H, W) or (H, W) (boolean/float).
        attrs: Optional tensor or list/tuple of tensors of shape (B, H, W, ...) or (H, W, ...).
               If provided, the corresponding attributes for the valid points will be returned.

    Returns:
        A tuple of:
            pts: (N, 3) points in world coordinates (or camera if
                 w2c is None).
            rgbs: (N, 3) colors in [0, 1] or None.
            batch_inds: (N,) batch index for each point, in [0, B-1].
            coords_2d: (N, 2) 2D normalized image coordinates in the
                       format expected by grid_sample (x, y), each
                       in the range [-1, 1].
            attrs_out: The sampled attributes. If 'attrs' input was a single tensor,
                       returns (N, ...) tensor. If 'attrs' was a list/tuple, returns
                       a tuple of (N, ...) tensors. Returns None if attrs is None.
    """
    # Normalize depth shape to (B, H, W)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    elif depth.ndim != 3:
        raise ValueError(
            f"depth must have shape (H, W) or (B, H, W), got {depth.shape}"
        )

    B, H, W = depth.shape

    device = depth.device
    depth = depth.float()

    # Normalize intrinsics to (B, 3, 3)
    if intrinsics is None:
        raise ValueError("intrinsics must be provided")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)
    elif intrinsics.ndim == 3:
        if intrinsics.shape[0] != B:
            raise ValueError(
                f"intrinsics batch size {intrinsics.shape[0]} does not match depth batch {B}"
            )
    else:
        raise ValueError(
            f"intrinsics must have shape (3, 3) or (B, 3, 3), got {intrinsics.shape}"
        )
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)

    # Normalize w2c to (B, 4, 4) if provided
    if w2c is not None:
        if w2c.ndim == 2:
            w2c = w2c.unsqueeze(0).expand(B, -1, -1)
        elif w2c.ndim == 3:
            if w2c.shape[0] != B:
                raise ValueError(
                    f"w2c batch size {w2c.shape[0]} does not match depth batch {B}"
                )
        else:
            raise ValueError(
                f"w2c must have shape (4, 4) or (B, 4, 4), got {w2c.shape}"
            )
        w2c = w2c.to(device=device, dtype=torch.float32)

    # Normalize rgb to (B, H, W, 3) if provided
    if rgb is not None:
        if rgb.ndim == 3:
            # Could be (H, W, 3) or (3, H, W)
            if rgb.shape[0] == 3 and rgb.shape[2] != 3:
                # (3, H, W) -> (H, W, 3)
                rgb = rgb.permute(1, 2, 0)
            # Now it's (H, W, 3), unsqueeze to (1, H, W, 3)
            rgb = rgb.unsqueeze(0)
        elif rgb.ndim == 4:
            # Could be (B, H, W, 3) or (B, 3, H, W)
            if rgb.shape[1] == 3 and rgb.shape[3] != 3:
                # (B, 3, H, W) -> (B, H, W, 3)
                rgb = rgb.permute(0, 2, 3, 1)
            # Now it's (B, H, W, 3)
        else:
            raise ValueError(
                f"rgb must have shape (H, W, 3), (B, H, W, 3), (3, H, W), or (B, 3, H, W), got {rgb.shape}"
            )
        if rgb.shape[0] != B or rgb.shape[1] != H or rgb.shape[2] != W:
            raise ValueError(
                f"rgb shape {rgb.shape} is inconsistent with depth shape {depth.shape}"
            )
        rgb = rgb.to(device=device, dtype=torch.float32)

    # Normalize foreground_mask to (B, H, W) if provided
    if foreground_mask is not None:
        if foreground_mask.ndim == 2:
            foreground_mask = foreground_mask.unsqueeze(0)
        elif foreground_mask.ndim != 3:
            raise ValueError(
                "foreground_mask must have shape (H, W) or (B, H, W), "
                f"got {foreground_mask.shape}"
            )
        if (
            foreground_mask.shape[0] != B
            or foreground_mask.shape[1] != H
            or foreground_mask.shape[2] != W
        ):
            raise ValueError(
                f"foreground_mask shape {foreground_mask.shape} is inconsistent "
                f"with depth shape {depth.shape}"
            )
        foreground_mask = foreground_mask.to(device=device)

    # Normalize attrs
    attr_list: List[Tensor] = []
    is_attr_list = False
    if attrs is not None:
        if isinstance(attrs, (list, tuple)):
            is_attr_list = True
            raw_attrs = list(attrs)
        else:
            is_attr_list = False
            raw_attrs = [attrs]

        for i, attr in enumerate(raw_attrs):
            # Normalize to (B, H, W, ...)
            if attr.ndim == 2:  # (H, W)
                attr = attr.unsqueeze(0)  # (1, H, W)

            if attr.shape[0] != B:
                if B != 1 and attr.shape[0] == 1:
                    attr = attr.expand(B, *attr.shape[1:])
                elif B == 1 and attr.shape[0] == 1:
                    pass  # match
                elif attr.shape[0] != B:
                    # Attempt to unsqueeze if (H, W, C) provided and matches B=1?
                    # But (B, H, W, C) is safer.
                    # If (H, W), we already unsqueezed.
                    raise ValueError(f"attr[{i}] batch dim {attr.shape[0]} != B={B}")

            if attr.shape[1] != H or attr.shape[2] != W:
                raise ValueError(
                    f"attr[{i}] shape {attr.shape} incompatible with (B, H, W) = {(B, H, W)}"
                )

            attr_list.append(attr)

    # Pixel grid (shared across batch), in pixel coordinates
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    pts_list: List[Tensor] = []
    rgbs_list: List[Tensor] = []
    coords_list: List[Tensor] = []
    batch_inds_list: List[Tensor] = []
    out_attrs_list: List[List[Tensor]] = [[] for _ in range(len(attr_list))]

    for b in range(B):
        depth_b = depth[b]
        intr_b = intrinsics[b]
        w2c_b = w2c[b] if w2c is not None else None
        rgb_b = rgb[b] if rgb is not None else None
        fg_b = foreground_mask[b] if foreground_mask is not None else None

        # Valid depth
        valid_mask = (depth_b > 0) & (depth_b <= max_depth) & torch.isfinite(depth_b)

        if fg_b is not None:
            fg_b = fg_b.bool()
            if fg_b.shape != depth_b.shape:
                raise ValueError(
                    f"foreground_mask[{b}] shape {fg_b.shape} does not match "
                    f"depth[{b}] shape {depth_b.shape}"
                )
            valid_mask = valid_mask & fg_b

        if not torch.any(valid_mask):
            continue

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_b[valid_mask]

        fx = intr_b[0, 0]
        fy = intr_b[1, 1]
        cx = intr_b[0, 2]
        cy = intr_b[1, 2]

        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid

        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, 3)

        if w2c_b is not None:
            # world-to-camera -> camera-to-world
            T_cam_to_world = torch.linalg.inv(w2c_b)
            ones = torch.ones(
                (points_cam.shape[0], 1),
                device=points_cam.device,
                dtype=points_cam.dtype,
            )
            points_cam_h = torch.cat([points_cam, ones], dim=-1)  # (N, 4)
            points_world_h = (T_cam_to_world @ points_cam_h.T).T  # (N, 4)
            pts_b = points_world_h[:, :3]
        else:
            pts_b = points_cam

        # Colors
        if rgb_b is not None:
            cols_b = rgb_b[valid_mask]  # (N, 3)
            if cols_b.numel() > 0 and cols_b.max() > 1.0:
                cols_b = cols_b / 255.0
        else:
            cols_b = None

        # Attrs
        for i, attr in enumerate(attr_list):
            attr_b = attr[b]
            # attr_b is (H, W, ...)
            # valid_mask is (H, W)
            # We select using boolean mask which handles trailing dims automatically in PyTorch
            # attr_b[valid_mask] returns (N_valid, ...)
            val_attr = attr_b[valid_mask]
            out_attrs_list[i].append(val_attr)

        # 2D pixel coordinates (u, v) for valid pixels
        # u in [0, W-1], v in [0, H-1]
        # Convert to normalized grid_sample coordinates in [-1, 1]
        # Using the default align_corners=False convention:
        #   x = ((x_norm + 1) * W - 1) / 2
        #   y = ((y_norm + 1) * H - 1) / 2
        # -> x_norm = (2 * x + 1 - W) / W, similarly for y
        x_norm = (2.0 * u_valid + 1.0 - W) / W
        y_norm = (2.0 * v_valid + 1.0 - H) / H
        coords_2d_b = torch.stack([x_norm, y_norm], dim=-1)  # (N, 2)

        # Batch indices for these points
        batch_inds_b = torch.full(
            (depth_valid.shape[0],),
            b,
            device=device,
            dtype=torch.long,
        )

        pts_list.append(pts_b)
        coords_list.append(coords_2d_b)
        batch_inds_list.append(batch_inds_b)

        if cols_b is not None:
            rgbs_list.append(cols_b)

    if len(pts_list) == 0:
        # No valid points at all
        pts = depth.new_zeros((0, 3))
        rgbs = depth.new_zeros((0, 3)) if rgb is not None else None
        batch_inds = depth.new_zeros((0,), dtype=torch.long)
        coords_2d = depth.new_zeros((0, 2))

        if attrs is None:
            return pts, rgbs, batch_inds, coords_2d, None
        else:
            ret_attrs = []
            for attr in attr_list:
                # attr: (B, H, W, ...)
                trailing = attr.shape[3:]
                ret_attrs.append(attr.new_zeros((0, *trailing)))
            if is_attr_list:
                return pts, rgbs, batch_inds, coords_2d, tuple(ret_attrs)
            else:
                return pts, rgbs, batch_inds, coords_2d, ret_attrs[0]

    pts = torch.cat(pts_list, dim=0)
    coords_2d = torch.cat(coords_list, dim=0)
    batch_inds = torch.cat(batch_inds_list, dim=0)

    rgbs: Optional[Tensor]
    if rgb is not None and len(rgbs_list) > 0:
        rgbs = torch.cat(rgbs_list, dim=0)
    else:
        rgbs = None

    final_attrs = []
    if attrs is not None:
        for lst in out_attrs_list:
            if len(lst) > 0:
                final_attrs.append(torch.cat(lst, dim=0))
            else:
                # If valid points exist but no attrs captured? Should be impossible if points exist.
                # But just in case
                pass  # Logic above guarantees consistent appending

    if attrs is None:
        return pts, rgbs, batch_inds, coords_2d, None
    elif is_attr_list:
        return pts, rgbs, batch_inds, coords_2d, tuple(final_attrs)
    else:
        return pts, rgbs, batch_inds, coords_2d, final_attrs[0]


def pool_features_from_views(
    voxel_centers: Tensor,
    voxel_batch_inds: Tensor,
    w2c: Tensor,
    intrinsics: Tensor,
    depths: Tensor,
    features: Tensor,
    foreground_masks: Optional[Tensor] = None,
    occlusion_margin: float = 0.1,
    check_foreground: bool = True,
) -> Tensor:
    """
    Pools features from multiple views to voxel centers.

    Args:
        voxel_centers: [P, 3] Voxel world centers.
        voxel_batch_inds: [P] Batch index for each voxel (0 to B*T-1).
        w2c: [B*T*CAM, 4, 4] World-to-Camera matrices.
        intrinsics: [B*T*CAM, 3, 3] Camera intrinsics.
        depths: [B*T*CAM, 1, H, W] Ground truth depth maps.
        features: [B*T*CAM, D, H', W'] Feature maps.
        foreground_masks: [B*T*CAM, 1, H, W] Optional foreground masks (optional).
        occlusion_margin: Margin for occlusion check in meters.

    Returns:
        Tensor: [P, D] Pooled features.
    """
    P = voxel_centers.shape[0]
    BTC = w2c.shape[0]

    # Derive BT from voxel batch indices to infer CAM
    BT = int(voxel_batch_inds.max().item()) + 1 if P > 0 else 1
    CAM = BTC // BT if BT > 0 else 1

    _, D, H_feat, W_feat = features.shape
    _, _, H, W = depths.shape

    # 1. Expand for all cameras
    # [P, CAM, 3] repeating centers
    voxel_centers_expanded = voxel_centers.unsqueeze(1).expand(-1, CAM, -1)
    voxel_centers_flat = voxel_centers_expanded.reshape(-1, 3)  # [P*CAM, 3]

    # Construct batch indices for these expanded points
    # voxel_batch_inds -> [P] (values 0 to BT-1)
    # [P, CAM]
    voxel_btc_idx = voxel_batch_inds.long().unsqueeze(1) * CAM + torch.arange(
        CAM, device=voxel_centers.device
    ).unsqueeze(0)
    voxel_btc_idx_flat = voxel_btc_idx.flatten()  # [P*CAM]

    # 2. Project to Image Planes
    curr_w2c = w2c[voxel_btc_idx_flat]  # [P*CAM, 4, 4]
    curr_K = intrinsics[voxel_btc_idx_flat]  # [P*CAM, 3, 3]

    # Transform to camera space
    pts_h = torch.cat(
        [voxel_centers_flat, torch.ones_like(voxel_centers_flat[:, :1])], dim=-1
    )
    pts_cam = (curr_w2c @ pts_h.unsqueeze(-1)).squeeze(-1)
    pts_z = pts_cam[:, 2]  # Depth

    # Project to pixels
    pts_img = (curr_K @ pts_cam[:, :3].unsqueeze(-1)).squeeze(-1)
    pts_uv_h = pts_img[:, :2] / (pts_img[:, 2:3] + 1e-6)  # [P*CAM, 2] (u, v)

    # 3. Validity Checks
    u, v = pts_uv_h[:, 0], pts_uv_h[:, 1]
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pts_z > 0)

    # Normalized coords for grid_sample
    u_norm = (2.0 * u + 1.0 - W) / W
    v_norm = (2.0 * v + 1.0 - H) / H
    coords_norm = torch.stack([u_norm, v_norm], dim=-1)  # [P*CAM, 2]

    # Pre-filter for Depth Sampling (Optional optimization)
    # We must sample depth for occlusion check.
    # Can we skip sampling depth for points out of bounds? Yes.

    # Occlusion check
    # Initialize valid_mask with bounds check
    valid_mask = in_bounds

    # Only proceed with expensive checks for potentially valid points
    if valid_mask.any():
        valid_indices_depth = torch.nonzero(valid_mask).squeeze(1)

        # Sample GT depth only for in-bound points
        sampled_depth_valid = flat_grid_sample(
            depths,
            coords_norm[valid_indices_depth],
            voxel_btc_idx_flat[valid_indices_depth],
            align_corners=False,
        ).squeeze(-1)  # [N_valid]

        pts_z_valid = pts_z[valid_indices_depth]
        not_occluded_valid = pts_z_valid < (sampled_depth_valid + occlusion_margin)

        # Update valid_mask
        # We need to map back to full size.
        # Easier: create a temp mask for just valid subset or update full mask?
        # Let's update full mask.
        # Scatter the results back? Or just keep filtering indices.

        # Actually, let's refine valid_indices
        valid_indices = valid_indices_depth[not_occluded_valid]
    else:
        valid_indices = torch.zeros(0, device=voxel_centers.device, dtype=torch.long)

    # Foreground check
    if check_foreground and foreground_masks is not None and valid_indices.numel() > 0:
        # Ensure masks have channel dim
        if foreground_masks.ndim == 3:
            foreground_masks = foreground_masks.unsqueeze(1)

        sampled_fg_valid = flat_grid_sample(
            foreground_masks.float(),
            coords_norm[valid_indices],
            voxel_btc_idx_flat[valid_indices],
            align_corners=False,
        ).squeeze(-1)

        in_foreground_valid = sampled_fg_valid > 0.5
        valid_indices = valid_indices[in_foreground_valid]

    # 4. Sample Features (Only for valid points)
    if valid_indices.numel() > 0:
        sampled_feats_valid = flat_grid_sample(
            features,
            coords_norm[valid_indices],
            voxel_btc_idx_flat[valid_indices],
            align_corners=False,
        )  # [N_final, D]

        # 5. Aggregate (Average)
        # We need to sum these features back to their corresponding voxels.
        # voxel_indices: map each of P*CAM points to P voxel index.
        # [P, CAM] -> [P*CAM] with values 0..P-1
        voxel_id_flat = (
            torch.arange(P, device=voxel_centers.device)
            .unsqueeze(1)
            .expand(-1, CAM)
            .reshape(-1)
        )
        valid_voxel_ids = voxel_id_flat[valid_indices]

        # Sum features
        sum_feats = torch.zeros(
            (P, D), device=voxel_centers.device, dtype=features.dtype
        )
        sum_feats.index_add_(0, valid_voxel_ids, sampled_feats_valid)

        # Count per voxel
        valid_counts = torch.zeros(
            (P, 1), device=voxel_centers.device, dtype=features.dtype
        )
        ones = torch.ones(
            (valid_indices.shape[0], 1),
            device=voxel_centers.device,
            dtype=features.dtype,
        )
        valid_counts.index_add_(0, valid_voxel_ids, ones)

        pooled_features = sum_feats / (valid_counts + 1e-6)
    else:
        pooled_features = torch.zeros(
            (P, D), device=voxel_centers.device, dtype=features.dtype
        )

    return pooled_features
