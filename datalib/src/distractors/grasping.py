"""
Grasp Generator for pre-calculating valid grasps for distractor objects.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from datalib.src.play.geometry import GeometryLib
from datalib.src.play.logging_util import get_logger
logger = get_logger(__name__)
from datalib.src.play.utils import sapien_pose_to_numpy, pose_to_matrix


class GraspGenerator:
    """
    Generates valid antipodal grasps for distractor actors in their LOCAL frame.

    Constraints:
    - Width < 0.07m
    - Orientation: Approach within 20 degrees of [0, 0, -1] in local frame.
    """

    def __init__(
        self,
        max_width: float = 0.07,
        max_angle_deg: float = 20.0,
        n_surface_samples: int = 500,
    ):
        self.max_width = max_width
        self.max_angle_rad = np.radians(max_angle_deg)
        self.n_surface_samples = n_surface_samples
        self.down_local = np.array([0, 0, -1.0])

    def generate(self, actor, shape_type: str) -> List[Tuple[float, np.ndarray]]:
        """
        Generate valid grasps for an actor.

        Args:
            actor: SAPIEN Actor
            shape_type: Type of shape (cube, stick, sphere, etc.)

        Returns:
            List of (width, local_pose_matrix)
        """
        # 1. Get meshes in LOCAL frame
        # GeometryLib.get_collision_meshes returns world frame meshes.
        # We need local frame for pre-generation.

        sapien_actor = actor._objs[0] if hasattr(actor, "_objs") else actor
        # Use Sapien pose directly to avoid ManiSkill CUDA initialization issues during _load_scene
        actor_pose = sapien_actor.get_pose()
        actor_pos, actor_quat = sapien_pose_to_numpy(actor_pose)
        actor_matrix = pose_to_matrix(actor_pos, actor_quat)
        actor_inv_matrix = np.linalg.inv(actor_matrix)

        world_meshes = GeometryLib.get_collision_meshes(actor)
        if not world_meshes:
            logger.warning(f"[Grasp Gen] No meshes found for {actor.name}")
            return []

        local_meshes = []
        for mesh in world_meshes:
            local_mesh = mesh.copy()
            local_mesh.apply_transform(actor_inv_matrix)
            local_meshes.append(local_mesh)

        combined = trimesh.util.concatenate(local_meshes)

        # 2. Sample surface points
        points, face_indices = combined.sample(
            self.n_surface_samples, return_index=True
        )
        normals = combined.face_normals[face_indices]

        # 3. Find antipodal pairs
        grasps = []
        n_points = len(points)

        rejected_width = 0
        rejected_normal = 0
        rejected_orientation = 0

        # Optimize: use pairs that are potentially opposing
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Check normal opposition
                dot_normals = np.dot(normals[i], normals[j])
                threshold = -0.4 if shape_type == "triangle" else -0.8
                if dot_normals > threshold:
                    rejected_normal += 1
                    continue

                # Check width
                diff = points[j] - points[i]
                dist = np.linalg.norm(diff)
                if dist > self.max_width or dist < 0.005:
                    rejected_width += 1
                    continue

                # Compute grasp pose
                grasp_pose = self._compute_local_grasp_pose(
                    points[i], points[j], normals[i], normals[j], shape_type
                )

                if grasp_pose is not None:
                    grasps.append((dist, grasp_pose))
                else:
                    rejected_orientation += 1

        if not grasps:
            logger.warning(
                f"[Grasp Gen] {actor.name}: 0 grasps. Rejected: normal={rejected_normal}, width={rejected_width}, orientation={rejected_orientation}"
            )

        # Sort by width
        grasps.sort(key=lambda x: x[0])

        # Keep a reasonable number of diverse grasps
        # Maybe top 50 or so
        return grasps[:50]

    def _compute_local_grasp_pose(
        self, p1, p2, n1, n2, shape_type
    ) -> Optional[np.ndarray]:
        """Compute grasp matrix in local frame."""
        center = (p1 + p2) / 2

        # X-axis: Finger direction
        x_axis = p2 - p1
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Natural approach (avg negated normal)
        approach = -(n1 + n2) / 2

        # Target: [0, 0, -1] (top down in local frame)
        down = self.down_local

        # For sticks, we want to ensure we aren't grasping the long axis
        # (Already handled by width constraint usually, but we can be explicit)

        # Project down onto plane perpendicular to x_axis
        down_proj = down - np.dot(down, x_axis) * x_axis
        down_norm = np.linalg.norm(down_proj)

        if down_norm > 0.6:  # If x_axis is horizontal, we can point down
            final_approach = down_proj / down_norm
        else:
            # If x_axis is vertical, we rely on normals
            if np.linalg.norm(approach) < 1e-3:
                # Opposing normals cancel out? Pick any perpendicular
                if abs(x_axis[2]) < 0.9:
                    final_approach = np.cross(x_axis, [0, 0, 1])
                else:
                    final_approach = np.cross(x_axis, [0, 1, 0])
            else:
                final_approach = approach / np.linalg.norm(approach)

        # Angle check: approach must be within 20 deg of Down
        # (Relaxed for triangles to allow grasping from more orientations)
        dot_down = np.dot(final_approach, down)
        angle_threshold = 60.0 if shape_type == "triangle" else 20.0
        if dot_down < np.cos(np.radians(angle_threshold)):
            return None

        # Orthonormalize
        z_axis = final_approach
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            return None
        y_axis = y_axis / y_norm
        x_axis = np.cross(y_axis, z_axis)

        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[:, 1] *= -1

        grasp_matrix = np.eye(4)
        grasp_matrix[:3, :3] = rot_matrix
        grasp_matrix[:3, 3] = center

        return grasp_matrix
