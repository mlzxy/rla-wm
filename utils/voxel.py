import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional, Dict

# Expected libraries based on user context
try:
    from torch_geometric.nn.pool import voxel_grid
except ImportError:
    # Dummy for documentation if not installed in environment
    def voxel_grid(pos, size, batch=None, start=None, end=None):
        raise ImportError("torch_geometric is required.")


try:
    from torch_scatter import segment_csr
except ImportError:

    def segment_csr(*args, **kwargs):
        raise ImportError("torch_scatter is required.")


def hash_coords(coords, res=128):
    coords = coords.long()
    return (
        coords[:, 0] * (res**3)
        + coords[:, 1] * (res**2)
        + coords[:, 2] * res
        + coords[:, 3]
    )


class VoxelizationLayer(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int, int] = (64, 64, 64),
        min_cell_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    ):
        """
        A differentiable layer for robustly converting continuous point clouds into a regular voxel grid.

        This layer acts like a Fixed-Resolution 3D Camera or Crop Window:
        1.  It determines a robust center for the point cloud (via Median).
        2.  It defines a physical crop window around that center. The window size is fixed
            as `resolution * min_cell_size`.
        3.  Any points falling outside this window are considered outliers and dropped.
        4.  Points inside are mapped to integer grid indices [0, resolution-1].

        The normalized coordinate space is consistently [0.0, 1.0].

        Args:
            resolution: (D, H, W) The resolution of the voxel grid (the sensor resolution).
            min_cell_size: (x, y, z) The physical size of one voxel in meters.
        """
        super().__init__()
        # Register buffers so they are saved with the model state_dict and move to GPU automatically
        self.register_buffer("resolution", torch.tensor(resolution, dtype=torch.long))
        self.register_buffer(
            "min_cell_size", torch.tensor(min_cell_size, dtype=torch.float)
        )

    def _split_pts_batch(
        self, pts: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal helper to normalize input shapes: handles [N, 4] vs [N, 3] + batch."""
        if pts.shape[-1] == 4:
            batch = pts[:, 0].long()
            pts = pts[:, 1:]

        if batch is None:
            batch = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.long)

        return pts, batch

    def estimate_voxel_parameters(
        self,
        pts: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        resolution: Optional[Tuple[int, int, int]] = None,
        min_cell_size: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        """
        Calculates the "Crop Window" parameters (Center, Min Bound, Max Bound) for the voxelization.

        Uses **Robust Centering (Median)** for stability against outliers.
        The physical size of the bounds is determined by `resolution * min_cell_size`.

        Args:
            pts: [N, 3] Point coordinates or [N, 4] (batch, x, y, z).
            batch: [N] Batch indices (optional if pts is [N, 4]).
            resolution: Optional override for the grid resolution.
            min_cell_size: Optional override for the voxel size.

        Returns:
            norm_params: [B, 3, 3] Tensor containing normalization parameters per batch.
                          - norm_params[:, 0, :] -> Center (x, y, z)
                          - norm_params[:, 1, :] -> Min Bound (x, y, z)
                          - norm_params[:, 2, :] -> Max Bound (x, y, z)
        """
        pts, batch = self._split_pts_batch(pts, batch)
        device = pts.device

        g_res = (
            torch.tensor(resolution, device=device)
            if resolution is not None
            else self.resolution
        )
        c_size = (
            torch.tensor(min_cell_size, device=device)
            if min_cell_size is not None
            else self.min_cell_size
        )

        # Calculate the fixed physical span of our "Camera"
        target_span = g_res.float() * c_size  # [3]
        half_span = target_span / 2.0

        num_batches = int(batch.max().item()) + 1
        centers = torch.zeros((num_batches, 3), device=device, dtype=pts.dtype)

        # Robust Centering (Median)
        for b in range(num_batches):
            mask = batch == b
            if mask.any():
                centers[b] = torch.median(pts[mask], dim=0)[0]

        # Construct Bounds centered around the robust center
        b_min = centers - half_span
        b_max = centers + half_span

        # Stack into [B, 3, 3]
        return torch.stack([centers, b_min, b_max], dim=1)

    def augment(
        self,
        norm_params: torch.Tensor,
        pts: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the efficient shift (augmentation parameters) to maximize the usage of the voxel grid
        without cropping any valid points (if possible).

        Returns:
            aug_params: [B, 3] shift vector to be added to the center/bounds.
        """
        pts, batch = self._split_pts_batch(pts, batch)

        # Unpack norm params
        b_min_orig = norm_params[:, 1, :]  # [B, 3]
        b_max_orig = norm_params[:, 2, :]  # [B, 3]

        num_batches = norm_params.shape[0]
        shift = torch.zeros_like(b_min_orig)

        for b in range(num_batches):
            mask = batch == b
            if not mask.any():
                continue

            pts_b = pts[mask]
            # Calculate tight bounds of the points
            p_min = pts_b.min(dim=0)[0]
            p_max = pts_b.max(dim=0)[0]

            # Calculate the bounds of the voxel grid for this batch
            b_min = b_min_orig[b]
            b_max = b_max_orig[b]

            # Calculate the valid range for the shift `s`
            # We want:
            #   b_min + s <= p_min  ==>  s <= p_min - b_min
            #   b_max + s >= p_max  ==>  s >= p_max - b_max

            s_max = p_min - b_min
            s_min = p_max - b_max

            # Check if a valid range exists (i.e., s_min <= s_max)
            # This means the point cloud extent is smaller than or equal to the voxel grid extent
            valid_dims = s_min <= s_max

            # For dimensions where it fits, sample a random shift
            if valid_dims.any():
                # Uniform sample in [s_min, s_max]
                rand_val = torch.rand(3, device=shift.device, dtype=shift.dtype)
                s = s_min + rand_val * (s_max - s_min)

                # Apply only to valid dimensions. For invalid dimensions, we keep shift as 0 (center aligned)
                # or we could try to center it optimally, but 0 (median center) is a safe default from estim_params.
                shift[b, valid_dims] = s[valid_dims]

        return shift

    def voxelize(
        self,
        pts: torch.Tensor,
        norm_params: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        resolution: Optional[Tuple[int, int, int]] = None,
        aug_params: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Performs the core voxelization process: Normalize, Crop, Discretize, Cluster, Sort.

        The normalization maps world coordinates to the [0.0, 1.0) range.

        Args:
            pts: [N, 3] or [N, 4] input points.
            batch: [N] batch indices (optional).
            norm_params: [B, 3, 3] The crop window parameters.
            resolution: Optional override.
            aug_params: [B, 3] Optional shift vectors to apply to the crop window.

        Returns:
            voxel_output: The geometry of the resulting voxels.
                - If input [N, 4]: Returns [P, 4] Tensor (batch, x, y, z).
                - If input [N, 3]: Returns ([P, 3], [P]) Tuple ((x,y,z), batch).
            pooling_meta: (sorted_cluster_indices, idx_ptr)
                - sorted_cluster_indices: [Valid_N] Indices mapping valid points back to the
                  Original input tensor. Used to pool features.
                - idx_ptr: [P+1] CSR row pointers indicating start/end of points per voxel.
        """
        pts_raw, batch_raw = self._split_pts_batch(pts, batch)
        device = pts_raw.device
        g_res = (
            torch.tensor(resolution, device=device)
            if resolution is not None
            else self.resolution
        )

        # 1. Unpack & Normalize (Maps to [0, 1] range)
        b_min_batch = norm_params[batch_raw, 1, :]
        b_max_batch = norm_params[batch_raw, 2, :]

        if aug_params is not None:
            b_min_batch = b_min_batch + aug_params[batch_raw]
            b_max_batch = b_max_batch + aug_params[batch_raw]

        b_span = b_max_batch - b_min_batch + 1e-6
        norm_pts = (pts_raw - b_min_batch) / b_span

        # 2. Filter Outliers (Strict Crop)
        in_bound_mask = (norm_pts >= 0).all(dim=1) & (norm_pts < 1).all(dim=1)
        valid_indices = torch.nonzero(in_bound_mask).squeeze(1)

        if valid_indices.numel() == 0:
            return self._return_empty(pts.shape[-1] == 4, device)

        valid_norm_pts = norm_pts[valid_indices]
        valid_batch = batch_raw[valid_indices]

        # 3. Discretize to 3D Integer Indices (used for final output)
        raw_voxel_indices_3d = torch.floor(
            valid_norm_pts * g_res.unsqueeze(0).float()
        ).long()

        # 4. Use `voxel_grid` for Clustering (finds unique 1D hash for points in the same cell)
        cluster_1d = voxel_grid(
            pos=valid_norm_pts, size=1.0 / g_res.float(), batch=valid_batch
        )

        # 5. Unique & Sort (PyG Pattern)
        unique_cluster_ids, inverse_indices, counts = torch.unique(
            cluster_1d, sorted=True, return_inverse=True, return_counts=True
        )

        _, sorted_compact_arg = torch.sort(inverse_indices, stable=True)
        sorted_cluster_indices = valid_indices[
            sorted_compact_arg
        ]  # Map valid subset back to original N
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        # 6. Retrieve 3D Voxel Indices & Batch from Clusters (using min reduction over sorted indices)
        raw_voxels_sorted = raw_voxel_indices_3d[sorted_compact_arg]
        raw_batch_sorted = valid_batch[sorted_compact_arg]

        final_voxels = segment_csr(raw_voxels_sorted, idx_ptr, reduce="min")
        final_batch = segment_csr(raw_batch_sorted, idx_ptr, reduce="min")

        # 8. Deduplicate (Merges clusters that map to the same integer voxel)
        # Sometiems voxel_grid produces multiple clusters for the same voxel.
        # We need to merge them.
        coords_4d = torch.cat([final_batch.unsqueeze(1), final_voxels], dim=1)
        # Hashes of the K initial clusters
        cluster_hashes = hash_coords(coords_4d, res=g_res.max().item() + 1)
        # Map K clusters -> M unique voxels
        _, meta_inverse = torch.unique(cluster_hashes, sorted=True, return_inverse=True)

        # Map N points -> K clusters -> M unique voxels
        final_point_ids = meta_inverse[inverse_indices]

        # Re-sort points based on the new unique voxel IDs
        _, sorted_compact_arg = torch.sort(final_point_ids, stable=True)
        sorted_cluster_indices = valid_indices[sorted_compact_arg]

        # Re-compute idx_ptr for the merged clusters
        _, counts = torch.unique(final_point_ids, sorted=True, return_counts=True)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        # Re-compute the final voxel coordinates (now unique)
        raw_voxels_sorted = raw_voxel_indices_3d[sorted_compact_arg]
        raw_batch_sorted = valid_batch[sorted_compact_arg]
        final_voxels = segment_csr(raw_voxels_sorted, idx_ptr, reduce="min")
        final_batch = segment_csr(raw_batch_sorted, idx_ptr, reduce="min")

        # 9. Output Format
        pooling_meta = (sorted_cluster_indices, idx_ptr)

        if pts.shape[-1] == 4:
            # Return [P, 4] -> (batch, x, y, z)
            return torch.cat([final_batch.unsqueeze(1), final_voxels], dim=1).to(
                torch.int32
            ), pooling_meta
        else:
            # Return ([P, 3], [P])
            return (final_voxels, final_batch), pooling_meta

    def feature_voxel_pool(
        self,
        pool_indices: Tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """
        Pools a single feature tensor (e.g., RGB) from points to voxels using Max Pooling.
        """
        sorted_indices, idx_ptr = pool_indices
        return segment_csr(features[sorted_indices], idx_ptr, reduce=reduce)

    def devoxelize(
        self,
        voxels: torch.Tensor,
        batch: Optional[torch.Tensor],
        norm_params: torch.Tensor,
        aug_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstructs real-world physical coordinates from integer voxel indices.
        Essentially: `World_Pos = Min_Bound + (Voxel_Index + 0.5) * Voxel_Size`.

        Args:
            voxels: [P, 3] or [P, 4] Integer voxel indices.
            batch: [P] Batch indices (optional if voxels is [P, 4]).
            norm_params: [B, 3, 3] The parameters used to create these voxels.
            aug_params: [B, 3] Optional shift vectors used during voxelization.

        Returns:
            real_pts: [P, 3] The physical center coordinates of each voxel.
        """
        if voxels.shape[-1] == 4:
            batch = voxels[:, 0].long()
            voxels = voxels[:, 1:]

        if batch is None:
            batch = torch.zeros(voxels.shape[0], device=voxels.device, dtype=torch.long)

        # Unpack Bounds
        b_min = norm_params[batch, 1, :]
        if aug_params is not None:
            b_min = b_min + aug_params[batch]

        b_max = norm_params[batch, 2, :]
        # Note: b_max isn't strictly needed for devoxelize if we use b_min and resolution,
        # but if we used it for span calculation we would shift it too.
        # Actually in devoxelize step is calculated from resolution.
        # But wait, original code used norm_params to get span?
        # Orig Code: step = (b_max - b_min) / g_res

        # If we shifted both, span is same.
        if aug_params is not None:
            b_max = b_max + aug_params[batch]

        g_res = self.resolution.to(voxels.device).float()
        step = (b_max - b_min) / g_res

        # Reconstruction: Min + (Index + 0.5) * Step
        real_pts = b_min + (voxels.float() + 0.5) * step
        return real_pts

    def normalize_cameras_to_voxel_space(
        self,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        norm_params: torch.Tensor,
        aug_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms camera extrinsics (w2c) from world coordinates to the normalized voxel space [0.0, 1.0].
        Intrinsics remain unchanged.

        E_norm @ P_norm = P_cam
        P_cam = E_world @ P_world
        P_norm = T_norm @ P_world => P_world = T_norm^-1 @ P_norm

        P_cam = E_world @ (T_norm^-1 @ P_norm)
        => E_norm = E_world @ T_norm^-1

        Args:
            w2c: [B, *, 4, 4] or [B, 4, 4] World-to-Camera (E_world).
            intrinsics: [B, *, 3, 3] or [B, 3, 3] Intrinsics (K).
            norm_params: [B, 3, 3] Voxel bounds (Center, Min, Max) used for normalization.
            aug_params: [B, 3] Optional shift vectors.

        Returns:
            normed_w2c: [B, *, 4, 4] or [B, 4, 4] Extrinsics mapping NVS [0, 1] to Camera.
            normed_intrinsics: [B, *, 3, 3] or [B, 3, 3] Identical to input intrinsics.
        """
        B = w2c.shape[0]
        # Store original shape for reshaping back
        shape_in = w2c.shape
        w2c_flat = w2c.reshape(-1, 4, 4)

        # 1. Extract Min and Span
        b_min = norm_params[:, 1, :]  # [B, 3]
        b_max = norm_params[:, 2, :]  # [B, 3]

        if aug_params is not None:
            b_min = b_min + aug_params
            b_max = b_max + aug_params

        b_span = b_max - b_min + 1e-6  # [B, 3]

        # 2. Construct T_norm: World to Normalized Voxel Space [0, 1] transformation
        # T_norm = [ S | T ]
        #          [ 0 | 1 ] where S = diag(1/B_span), T = -B_min / B_span

        # scale_diag = 1.0 / b_span  # [B, 3]
        # translation = -b_min / b_span  # [B, 3]

        # Build T_norm [B, 4, 4]
        # T_norm = (
        #     torch.eye(4, device=w2c.device, dtype=w2c.dtype)
        #     .unsqueeze(0)
        #     .repeat(B, 1, 1)
        # )

        # T_norm[:, 0, 0] = scale_diag[:, 0]
        # T_norm[:, 1, 1] = scale_diag[:, 1]
        # T_norm[:, 2, 2] = scale_diag[:, 2]

        # T_norm[:, 0, 3] = translation[:, 0]
        # T_norm[:, 1, 3] = translation[:, 1]
        # T_norm[:, 2, 3] = translation[:, 2]

        # Construct T_denorm = T_norm^-1
        # Scale = b_span, Trans = b_min
        T_denorm = (
            torch.eye(4, device=w2c.device, dtype=w2c.dtype)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        T_denorm[:, 0, 0] = b_span[:, 0]
        T_denorm[:, 1, 1] = b_span[:, 1]
        T_denorm[:, 2, 2] = b_span[:, 2]
        T_denorm[:, 0, 3] = b_min[:, 0]
        T_denorm[:, 1, 3] = b_min[:, 1]
        T_denorm[:, 2, 3] = b_min[:, 2]

        # 3. Apply T_denorm to Extrinsics (W2C)
        # E_norm = E_world @ T_denorm

        # Repeat T_denorm to match all cameras B*C
        num_cameras = w2c_flat.shape[0] // B
        T_denorm_expanded = (
            T_denorm.unsqueeze(1).repeat(1, num_cameras, 1, 1).reshape(-1, 4, 4)
        )

        # E_norm = E_world @ T_denorm
        normed_w2c_flat = w2c_flat @ T_denorm_expanded

        # 4. Reshape back
        normed_w2c = normed_w2c_flat.reshape(shape_in)
        normed_intrinsics = intrinsics

        return normed_w2c, normed_intrinsics

    def _return_empty(self, is_4d: bool, device: torch.device):
        """Internal helper to create empty tensors when all points are filtered out."""
        empty_idx = torch.zeros((0), dtype=torch.long, device=device)
        empty_ptr = torch.zeros((1), dtype=torch.long, device=device)
        if is_4d:
            return torch.zeros((0, 4), dtype=torch.long, device=device), (
                empty_idx,
                empty_ptr,
            )
        else:
            return (torch.zeros((0, 3), dtype=torch.long, device=device), empty_idx), (
                empty_idx,
                empty_ptr,
            )


# ==============================================================================
# Testing Code
# ==============================================================================
if __name__ == "__main__":
    print("--- Testing VoxelizationLayer with voxel_grid ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init (resolution is now used)
    LAYER_RESOLUTION = (32, 32, 32)
    layer = VoxelizationLayer(
        resolution=LAYER_RESOLUTION,
        min_cell_size=(0.1, 0.1, 0.1),
    ).to(device)

    # 1. Test batch=None (Single Point Cloud)
    print("\n[Test 1] Single Batch (batch=None)")
    pts_single = torch.randn(1000, 3, device=device) * 0.5

    params = layer.estimate_voxel_parameters(pts_single, batch=None)
    print(f"Params Shape: {params.shape} (Expect [1, 3, 3])")

    b_voxels, meta = layer.voxelize(pts_single, params, batch=None)
    if isinstance(b_voxels, tuple):
        voxels, v_batch = b_voxels
    else:
        voxels, v_batch = b_voxels[:, 1:], b_voxels[:, 0]
    print(f"Voxels: {voxels.shape}, BatchOut: {v_batch.shape}")

    # 2. Test Multi-Batch with Outliers
    print("\n[Test 2] Multi-Batch with Outliers & Pooling")
    N = 2000
    pts_multi = torch.cat(
        [
            torch.randn(N // 2, 3, device=device) * 0.2,  # Batch 0
            torch.randn(N // 2, 3, device=device) * 0.2 + 10.0,  # Batch 1
        ]
    )
    # Add outlier to Batch 0
    pts_multi[0] = torch.tensor([100.0, 100.0, 100.0], device=device)

    batch_multi = torch.cat(
        [torch.zeros(N // 2, device=device), torch.ones(N // 2, device=device)]
    ).long()

    params = layer.estimate_voxel_parameters(pts_multi, batch_multi)
    aug_params = layer.augment(params, pts_multi, batch=batch_multi)

    # Voxelize
    (voxels_m, batch_m), (indices, ptr) = layer.voxelize(
        pts_multi, params, batch=batch_multi, aug_params=aug_params
    )

    print(f"Voxel Count: {voxels_m.shape[0]}")

    # Check if outlier (index 0) was dropped
    # indices contains indices into original pts_multi
    if 0 in indices:
        print("!! Fail: Outlier at index 0 was included.")
    else:
        print(">> Success: Outlier at index 0 was dropped.")

    # 3. Test New Functions (Camera & Mask)
    print("\n[Test 3] Camera Normalization & 2D Masks")

    # Define mock camera parameters (B=2, C=3 cameras per batch element)
    B, C = 2, 3
    H_img, W_img = 256, 512

    # E_world (Camera to World) for batch 0 near origin, batch 1 translated far away
    E_world = torch.eye(4, device=device).unsqueeze(0).repeat(B, C, 1, 1)
    E_world[1, :, :3, 3] = (
        10.0  # B1 is far away in world space (within a reasonable range)
    )

    # Intrinsics: Simple pinhole
    K = torch.zeros(3, 3, device=device)
    K[0, 0] = 500
    K[1, 1] = 500
    K[0, 2] = W_img / 2
    K[1, 2] = H_img / 2
    K[2, 2] = 1
    K = K.unsqueeze(0).repeat(B, C, 1, 1)

    # Use the parameters from Test 2 (aug_params has B=2)

    # 3a. Camera Normalization (Already verified)
    norm_E, norm_K = layer.normalize_cameras_to_voxel_space(
        E_world, K, params, aug_params=aug_params
    )

    print(f"Normed Extrinsics Shape: {norm_E.shape}")
    # mask_2d_raycast = layer.create_2d_voxel_masks_raycast(
    #     aug_params, K, E_world, (H_img, W_img), mask_reduction=8
    # )

    # print(
    #     f"Efficient Raycast Mask Sum (B0/C0): {mask_2d_raycast[0, 0].sum()} active pixels."
    # )
