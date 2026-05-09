"""
Geometry library for grasp sampling and surface analysis.

Provides antipodal grasp sampling using trimesh for mesh analysis.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from typing import Optional
import sapien

from .utils import (
    sapien_pose_to_numpy,
    numpy_to_sapien_pose,
    pose_to_matrix,
    matrix_to_pose,
)


class GeometryLib:
    """Utilities for geometric analysis of actors."""

    @staticmethod
    def get_collision_meshes(actor) -> list[trimesh.Trimesh]:
        """
        Extract collision meshes from a SAPIEN/ManiSkill actor.

        Args:
            actor: SAPIEN Actor or ManiSkill Actor wrapper

        Returns:
            List of trimesh.Trimesh objects in world frame
        """
        meshes = []

        # Get the underlying SAPIEN entity
        if hasattr(actor, "_objs"):
            # ManiSkill wrapped actor (batched)
            sapien_actor = actor._objs[0]
        elif hasattr(actor, "entity"):
            sapien_actor = actor.entity
        else:
            sapien_actor = actor

        # Get actor world pose
        # Use underlying sapien actor to avoid premature CUDA access issues in ManiSkill
        actor_pose = sapien_actor.get_pose()

        actor_pos, actor_quat = sapien_pose_to_numpy(actor_pose)
        actor_matrix = pose_to_matrix(actor_pos, actor_quat)

        # Get collision shapes from components
        components = sapien_actor.components
        for comp in components:
            if hasattr(comp, "collision_shapes"):
                for shape in comp.collision_shapes:
                    mesh = GeometryLib._shape_to_trimesh(shape, actor_matrix)
                    if mesh is not None:
                        meshes.append(mesh)

        return meshes

    @staticmethod
    def _shape_to_trimesh(shape, actor_matrix: np.ndarray) -> Optional[trimesh.Trimesh]:
        """Convert a SAPIEN collision shape to trimesh.

        Handles SAPIEN's collision shape types:
        - PhysxCollisionShapeBox: has half_size
        - PhysxCollisionShapeSphere: has radius
        - PhysxCollisionShapeCapsule: has radius, half_length
        - PhysxCollisionShapeCylinder: has radius, half_length
        - PhysxCollisionShapeConvexMesh: has vertices, triangles
        """
        # Get shape's local pose
        local_pose = shape.local_pose
        local_pos, local_quat = sapien_pose_to_numpy(local_pose)
        local_matrix = pose_to_matrix(local_pos, local_quat)

        # Combined transform
        world_matrix = actor_matrix @ local_matrix

        mesh = None
        shape_type = type(shape).__name__

        # Check for box shape (has half_size)
        if hasattr(shape, "half_size"):
            half = np.array(shape.half_size)
            mesh = trimesh.primitives.Box(extents=half * 2)

        # Check for sphere shape (has radius but not half_length)
        elif hasattr(shape, "radius") and not hasattr(shape, "half_length"):
            mesh = trimesh.primitives.Sphere(radius=shape.radius)

        # Check for cylinder/capsule (has radius and half_length)
        elif hasattr(shape, "radius") and hasattr(shape, "half_length"):
            mesh = trimesh.primitives.Cylinder(
                radius=shape.radius, height=shape.half_length * 2
            )

        # Check for convex mesh (has vertices/triangles)
        elif hasattr(shape, "vertices") and hasattr(shape, "triangles"):
            vertices = np.array(shape.vertices)
            faces = np.array(shape.triangles).reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Alternative: check for scale (convex mesh indicator)
        elif hasattr(shape, "scale") and hasattr(shape, "vertices"):
            vertices = np.array(shape.vertices) * np.array(shape.scale)
            if hasattr(shape, "triangles"):
                faces = np.array(shape.triangles).reshape(-1, 3)
            else:
                # Create convex hull if no faces
                mesh = trimesh.Trimesh(vertices=vertices)
                mesh = mesh.convex_hull

        if mesh is None:
            # Fallback: try to create a small box as placeholder
            print(f"Warning: Unsupported shape type {shape_type}, using placeholder")
            mesh = trimesh.primitives.Box(extents=[0.05, 0.05, 0.05])

        # Apply world transform
        mesh.apply_transform(world_matrix)
        return mesh

    @staticmethod
    def sample_surface_points(
        actor, n_samples: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample points and normals from an actor's collision surface.

        Args:
            actor: SAPIEN/ManiSkill actor
            n_samples: Number of points to sample

        Returns:
            Tuple of (points [n, 3], normals [n, 3]) in world frame
        """
        meshes = GeometryLib.get_collision_meshes(actor)

        if not meshes:
            raise ValueError("Actor has no collision meshes")

        # Combine all meshes
        combined = trimesh.util.concatenate(meshes)

        # Sample points with normals
        points, face_indices = combined.sample(n_samples, return_index=True)
        normals = combined.face_normals[face_indices]

        return points.astype(np.float32), normals.astype(np.float32)

    @staticmethod
    def check_grasp_orientation(
        quat: np.ndarray,
        reference_vector: np.ndarray = np.array([0, 0, -1]),
        threshold_degrees: float = 45.0,
    ) -> bool:
        """
        Check if a grasp orientation aligns with a reference vector.

        Args:
            quat: Grasp quaternion [4] xyzw
            reference_vector: Vector to align with (default: DOWN [0, 0, -1])
            threshold_degrees: Maximum angle deviation in degrees

        Returns:
            True if grasp approach axis is within threshold of reference
        """
        # Extract approach axis (Z-axis of grasp frame)
        rot = R.from_quat(quat)
        z_axis = rot.as_matrix()[:, 2]  # 3rd column is Z

        # Normalize vectors
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        ref = reference_vector / (np.linalg.norm(reference_vector) + 1e-8)

        # Compute angle
        dot = np.dot(z_axis, ref)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        return angle_deg <= threshold_degrees
