"""
Robot Anchor Points for differentiable kinematic tracking.

This module provides RobotAnchorSDF, an extension to RobotSDF that enables:
1. Adaptive sampling of anchor points on robot meshes
2. Differentiable forward kinematics for these anchor points
3. Optional voxelization of the tracked points
"""

import typing
import logging
from dataclasses import dataclass
from loguru import logger


import torch
from utils.voxel import VoxelizationLayer
import third_party.pytorch_kinematics as pk
from third_party.pytorch_volumetric.model_to_sdf_cuda import RobotSDF
from third_party.pytorch_kinematics.transforms.rotation_conversions import (
    quaternion_to_matrix,
)


@dataclass
class AnchorState:
    """State containing anchor point information for kinematic tracking.

    Attributes:
        points_local: (N, 3) Points in their respective link's local frame.
        link_indices: (N,) Index into sdf_to_link_name for each point.
        face_indices: (N,) Face index on the link's mesh for each point.
        barycentric_weights: (N, 3) Barycentric coordinates for each point on its face.
        link_sdf_indices: (N,) Index into self.sdf.sdfs for each point (maps to mesh data).
    """

    points_local: torch.Tensor
    link_indices: torch.Tensor  # Maps to sdf_to_link_name
    face_indices: torch.Tensor
    barycentric_weights: torch.Tensor
    link_sdf_indices: torch.Tensor  # Maps to self.sdf.sdfs


class RobotAnchorSDF(RobotSDF):
    """Extended RobotSDF with adaptive anchor point tracking.

    This class adds two key methods:
    - initialize_anchors: Sample N points adaptively across robot mesh surfaces
    - forward_kinematic_anchors: Compute world-frame positions of anchors given
      joint configurations and base transforms (differentiable)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute link name to index mapping for efficiency
        self._link_name_to_idx = {
            name: idx for idx, name in enumerate(self.sdf_to_link_name)
        }

    @torch.no_grad()
    def initialize_anchors(
        self, N: int, seed: int = 42, min_points_per_link: int = 0
    ) -> AnchorState:
        """Sample N anchor points adaptively across the robot mesh surfaces.

        Points are distributed proportionally based on the surface area of each link,
        while ensuring each link has at least min_points_per_link points.

        Args:
            N: Total number of anchor points to sample.
            seed: Random seed for reproducibility.
            min_points_per_link: Minimum number of points per link.

        Returns:
            AnchorState containing all anchor point information.
        """
        if seed > 0:
            torch.manual_seed(seed)
        dtype = self.dtype

        # Collect mesh data from all links
        all_vertices = []  # List of (V_i, 3) tensors
        all_faces = []  # List of (F_i, 3) tensors
        link_face_counts = []

        vertex_offset = 0
        device = None  # Will be set from first mesh
        for i, mesh_sdf in enumerate(self.sdf.sdfs):
            # Handle different SDF wrappers (MeshSDF, CachedSDF, etc.)
            if hasattr(mesh_sdf, "obj_factory"):
                obj_factory = mesh_sdf.obj_factory
            elif hasattr(mesh_sdf, "gt_sdf") and hasattr(
                mesh_sdf.gt_sdf, "obj_factory"
            ):
                obj_factory = mesh_sdf.gt_sdf.obj_factory
            else:
                logger.warning(
                    f"SDF {i} on link {self.sdf_to_link_name[i]} does not have an obj_factory; skipping anchor sampling"
                )
                continue

            obj_factory.precompute_sdf()  # Ensure mesh is loaded

            # Get vertices and faces from Kaolin-precomputed data (stored on obj_factory in sdf_cuda)
            if not hasattr(obj_factory, "verts_cuda") or not hasattr(
                obj_factory, "faces_cuda"
            ):
                logger.warning(
                    f"SDF {i} on link {self.sdf_to_link_name[i]} does not have verts_cuda/faces_cuda; skipping anchor sampling"
                )
                continue

            verts = obj_factory.verts_cuda.to(dtype=dtype)  # (V, 3)
            faces = obj_factory.faces_cuda  # (F, 3)

            if device is None:
                device = verts.device  # Use device from mesh data

            all_vertices.append(verts)
            all_faces.append(faces + vertex_offset)  # Offset face indices
            link_face_counts.append(faces.shape[0])
            vertex_offset += verts.shape[0]

        # Concatenate all meshes into one big mesh
        vertices = torch.cat(all_vertices, dim=0)  # (V_total, 3)
        faces = torch.cat(all_faces, dim=0)  # (F_total, 3)

        # Create link index for each face
        face_to_link_idx = torch.cat(
            [
                torch.full((count,), i, dtype=torch.long, device=device)
                for i, count in enumerate(link_face_counts)
            ]
        )  # (F_total,)

        # Compute triangle areas using cross product
        v0 = vertices[faces[:, 0]]  # (F, 3)
        v1 = vertices[faces[:, 1]]  # (F, 3)
        v2 = vertices[faces[:, 2]]  # (F, 3)

        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = torch.cross(edge1, edge2, dim=1)
        areas = 0.5 * torch.linalg.norm(cross, dim=1)  # (F,)

        # Ensure each link has at least min_points_per_link
        num_links = len(link_face_counts)
        link_total_areas = torch.zeros(num_links, device=device, dtype=dtype)
        link_total_areas.scatter_add_(0, face_to_link_idx, areas)

        # Calculate target probability per link
        # Traditional area-based probability: link_total_areas / link_total_areas.sum()
        # Target probability: max(area_prob, min_points_per_link / N)
        if min_points_per_link > 0:
            total_area = link_total_areas.sum()
            if total_area == 0:
                raise ValueError("Robot mesh has zero surface area")

            # Initial probabilities based on area
            area_probs = link_total_areas / total_area
            min_prob = min_points_per_link / N

            # Identify links that need boosting
            links_to_boost = area_probs < min_prob
            num_boosted = links_to_boost.sum()

            # If we can satisfy the minimum for all boosted links
            if num_boosted * min_prob <= 1.0:
                target_link_probs = torch.zeros_like(area_probs)
                target_link_probs[links_to_boost] = min_prob

                remaining_prob = 1.0 - (num_boosted * min_prob)
                other_area_sum = area_probs[~links_to_boost].sum()

                if other_area_sum > 0:
                    target_link_probs[~links_to_boost] = (
                        area_probs[~links_to_boost] / other_area_sum * remaining_prob
                    )
                elif not links_to_boost.all():
                    # If other links have zero area but exist, distribute equally
                    target_link_probs[~links_to_boost] = (
                        remaining_prob / (~links_to_boost).sum()
                    )
            else:
                # Impossible to satisfy strict minimum, fallback to soft boost
                target_link_probs = torch.maximum(
                    area_probs, torch.tensor(min_prob, device=device, dtype=dtype)
                )
                target_link_probs = target_link_probs / target_link_probs.sum()

            # Adjust triangle areas to match target link probabilities
            # For each link i, we want sum(adjusted_areas[link_i]) = target_link_probs[link_i]
            # triangle_weight = area / link_total_area * target_link_prob

            # Pre-calculate area-within-link normalization
            # Prevent division by zero for links with no faces or zero area (though we checked total_area)
            safe_link_total_areas = torch.where(
                link_total_areas > 0,
                link_total_areas,
                torch.ones_like(link_total_areas),
            )

            # probability of each triangle is triangle_area / link_total_area * target_link_prob
            face_weights = (
                areas / safe_link_total_areas[face_to_link_idx]
            ) * target_link_probs[face_to_link_idx]
        else:
            total_area = areas.sum()
            if total_area == 0:
                raise ValueError("Robot mesh has zero surface area")
            face_weights = areas / total_area

        # Sample N faces proportionally to their area
        # torch.multinomial samples indices with replacement based on weights
        sampled_face_indices = torch.multinomial(
            face_weights, N, replacement=True
        )  # (N,)

        # Get vertices for sampled faces
        sampled_faces = faces[sampled_face_indices]  # (N, 3)
        sampled_v0 = vertices[sampled_faces[:, 0]]  # (N, 3)
        sampled_v1 = vertices[sampled_faces[:, 1]]  # (N, 3)
        sampled_v2 = vertices[sampled_faces[:, 2]]  # (N, 3)

        # Generate random barycentric coordinates
        # Using the standard formula for uniform sampling on a triangle:
        # sqrt(r1) for the first barycentric coordinate ensures uniform distribution
        r1 = torch.rand(N, device=device, dtype=dtype)
        r2 = torch.rand(N, device=device, dtype=dtype)

        sqrt_r1 = torch.sqrt(r1)
        bary_u = 1 - sqrt_r1
        bary_v = sqrt_r1 * (1 - r2)
        bary_w = sqrt_r1 * r2

        barycentric_weights = torch.stack([bary_u, bary_v, bary_w], dim=1)  # (N, 3)

        # Compute point coordinates from barycentric weights
        points_local = (
            bary_u.unsqueeze(1) * sampled_v0
            + bary_v.unsqueeze(1) * sampled_v1
            + bary_w.unsqueeze(1) * sampled_v2
        )  # (N, 3)

        # Get link indices for sampled points
        link_indices = face_to_link_idx[sampled_face_indices]  # (N,)

        # Compute local face indices (within each link's mesh)
        # First compute the cumulative face count offset per link
        face_offsets = torch.zeros(
            len(link_face_counts) + 1, dtype=torch.long, device=device
        )
        face_offsets[1:] = torch.cumsum(
            torch.tensor(link_face_counts, device=device), dim=0
        )

        # Local face index = global face index - offset for that link
        local_face_indices = sampled_face_indices - face_offsets[link_indices]

        return AnchorState(
            points_local=points_local,
            link_indices=link_indices,
            face_indices=local_face_indices,
            barycentric_weights=barycentric_weights,
            link_sdf_indices=link_indices.clone(),
        )

    def forward_kinematic_anchors(
        self,
        state: AnchorState,
        joint_configurations: torch.Tensor,
        base_transformations: torch.Tensor,
        voxel_layer: typing.Optional["VoxelizationLayer"] = None,
        voxel_params: typing.Optional[torch.Tensor] = None,
        aug_params: typing.Optional[torch.Tensor] = None,
        return_voxelized: bool = False,
        joint_names: typing.Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute world-frame positions of anchor points.

        This method is fully differentiable with respect to joint_configurations
        and base_transformations.

        Args:
            state: AnchorState from initialize_anchors
            joint_configurations: (B, T, R, J) Joint angles in radians
            base_transformations: (B, T, R, 7) Base poses as [x, y, z, qw, qx, qy, qz]
            voxel_layer: Optional VoxelizationLayer instance for voxelization
            voxel_params: Optional (B, 3, 3) voxelization parameters from VoxelizationLayer
            aug_params: Optional (B, 3) augmentation shift for voxelization
            return_voxelized: If True, return voxelized coordinates instead of continuous

        Returns:
            (B, T, N, 3) World-frame coordinates of anchor points
            or voxelized output if return_voxelized is True
        """
        B, T, R, J = joint_configurations.shape
        device = joint_configurations.device
        dtype = joint_configurations.dtype
        N = state.points_local.shape[0]

        # Flatten batch dimensions for FK: (B*T*R, J)
        joint_flat = joint_configurations.reshape(-1, J)
        base_flat = base_transformations.reshape(-1, 7)  # (B*T*R, 7)

        # Compute forward kinematics for all configurations
        # Returns dict of link_name -> Transform3d with batch size B*T*R
        fk_results = {}
        current_idx = 0
        if joint_names is not None:
            assert len(self.chains) == len(joint_names)
        for chain_id, chain in enumerate(self.chains):
            n_dof = len(chain.get_joint_parameter_names())
            _joint_names = joint_names[chain_id] if joint_names is not None else None
            chain_config = joint_flat[:, current_idx : current_idx + n_dof]
            current_idx += n_dof
            if _joint_names is not None:
                chain_config = {name: chain_config[:, i] for i, name in enumerate(_joint_names)}

            fk = chain.forward_kinematics(chain_config)
            for k, v in fk.items():
                fk_results[k] = v

        # Construct base transform from the 7-vector (pos + quat)
        base_pos = base_flat[:, :3]  # (B*T*R, 3)
        base_quat = base_flat[:, 3:]  # (B*T*R, 4) as [qw, qx, qy, qz]
        base_rot = quaternion_to_matrix(base_quat)  # (B*T*R, 3, 3)

        # Create base Transform3d (world_to_base)
        base_matrices = (
            torch.eye(4, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(B * T * R, -1, -1)
            .clone()
        )
        base_matrices[:, :3, :3] = base_rot
        base_matrices[:, :3, 3] = base_pos
        base_transform = pk.Transform3d(matrix=base_matrices)

        # Transform anchor points to world frame
        # For each anchor point, get its link transform and compose with base
        # Move anchor points to input device for consistency
        points_local = state.points_local.to(device=device, dtype=dtype)  # (N, 3)
        link_indices = state.link_indices.to(device=device)  # (N,)

        # Gather link transforms for each anchor point
        # First, get unique link names
        unique_links = torch.unique(link_indices)

        # Initialize output points
        points_world = torch.zeros(B * T * R, N, 3, device=device, dtype=dtype)

        for link_idx in unique_links:
            link_name = self.sdf_to_link_name[link_idx.item()]
            link_mask = link_indices == link_idx
            link_points = points_local[link_mask]  # (M, 3)

            # Get the link-to-base transform
            if link_name in fk_results:
                link_transform = fk_results[link_name]  # Transform3d with batch B*T*R

                # Apply offset transform (visual frame offset)
                sdf_idx = link_idx.item()
                offset = self.offset_transforms[sdf_idx : sdf_idx + 1]

                # Compose: world = base @ link @ offset
                full_transform = base_transform.compose(link_transform).compose(offset)

                # Transform points
                transformed = full_transform.transform_points(
                    link_points
                )  # (B*T*R, M, 3)

                # Scatter back to output
                mask_indices = torch.where(link_mask)[0]
                points_world[:, mask_indices, :] = transformed

        # Reshape to (B, T, N, 3)
        points_world = points_world.reshape(B, T, R, N, 3)

        # Sum over robots dimension (R) - they share the same anchor state
        # Actually, if R robots each have N anchors, output should be (B, T, R*N, 3) or (B, T, N, 3)
        # Based on the signature, it seems like R is folded into the anchor dimension
        # Let's keep it as (B, T, R, N, 3) for now and let the caller handle it
        # Actually the spec says output is (B, T, N, 3), so we need to handle R
        # If there are R robots, each has N anchors, so total is R*N anchors
        points_world = points_world.reshape(B, T, R * N, 3)

        if return_voxelized and voxel_layer is not None and voxel_params is not None:
            # Flatten to (B*T, R*N, 3) and voxelize
            points_flat = points_world.reshape(B * T, -1, 3)
            batch_indices = (
                torch.arange(B * T, device=device)
                .unsqueeze(1)
                .expand(-1, R * N)
                .reshape(-1)
            )
            points_flat = points_flat.reshape(-1, 3)

            voxel_output, _ = voxel_layer.voxelize(
                points_flat,
                voxel_params.repeat(T, 1, 1)
                if voxel_params.shape[0] == B
                else voxel_params,
                batch=batch_indices,
                aug_params=aug_params.repeat(T, 1)
                if aug_params is not None and aug_params.shape[0] == B
                else aug_params,
            )

            return voxel_output

        return points_world
