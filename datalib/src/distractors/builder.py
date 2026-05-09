"""
Distractor Builder Module

Provides shape builders for all distractor primitives using ManiSkill actors 
and trimesh for complex geometry.
"""
from .color import ColorSampler
import os
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Tuple, List
import sapien


# Cache directory for generated meshes
MESH_CACHE_DIR = Path(os.path.expanduser("~/.cache/v4world/shapes"))


def _ensure_cache_dir():
    """Ensure the mesh cache directory exists."""
    MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DistractorBuilder:
    """
    Builder for distractor objects supporting both native ManiSkill primitives
    and custom trimesh-generated geometry.
    """
    
    def __init__(self, scene):
        """
        Args:
            scene: The SAPIEN/ManiSkill scene to add actors to
        """
        self.scene = scene
        _ensure_cache_dir()
        self.color_sampler = ColorSampler()
    
    # =========================================================================
    # Standard Primitives (using ManiSkill native actors)
    # =========================================================================
    
    def build_cube(
        self,
        half_size: float = 0.025,
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
        name: str = "cube",
    ):
        """Build a cube distractor using native ManiSkill builder."""
        from mani_skill.utils.building import actors
        return actors.build_cube(
            self.scene,
            half_size=half_size,
            color=color,
            name=name,
        )
    
    def build_sphere(
        self,
        radius: float = 0.025,
        color: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0),
        name: str = "sphere",
    ):
        """Build a sphere distractor using native ManiSkill builder."""
        from mani_skill.utils.building import actors
        return actors.build_sphere(
            self.scene,
            radius=radius,
            color=color,
            name=name,
        )
    
    def build_cylinder(
        self,
        radius: float = 0.02,
        half_length: float = 0.04,
        color: Tuple[float, float, float, float] = (0.2, 0.2, 0.8, 1.0),
        name: str = "cylinder",
    ):
        """Build a cylinder distractor using native ManiSkill builder."""
        from mani_skill.utils.building import actors
        return actors.build_cylinder(
            self.scene,
            radius=radius,
            half_length=half_length,
            color=color,
            name=name,
        )
    
    def build_box(
        self,
        half_sizes: Tuple[float, float, float] = (0.04, 0.02, 0.015),
        color: Tuple[float, float, float, float] = (0.8, 0.8, 0.2, 1.0),
        name: str = "box",
    ):
        """Build a rectangular box (rect-cube) using native ManiSkill builder."""
        from mani_skill.utils.building import actors
        return actors.build_box(
            self.scene,
            half_sizes=half_sizes,
            color=color,
            name=name,
        )
    
    def build_stick(
        self,
        radius: float = 0.008,
        half_length: float = 0.06,
        color: Tuple[float, float, float, float] = (0.5, 0.3, 0.1, 1.0),
        name: str = "stick",
    ):
        """Build a stick (thin box/square prism) distractor to prevent rolling."""
        from mani_skill.utils.building import actors
        return actors.build_box(
            self.scene,
            half_sizes=(half_length, radius, radius),
            color=color,
            name=name,
        )
    
    # =========================================================================
    # Complex Primitives (using trimesh + OBJ export)
    # =========================================================================
    
    def build_triangle(
        self,
        base: float = 0.05,
        height: float = 0.04,
        depth: float = 0.03,
        color: Tuple[float, float, float, float] = (0.9, 0.5, 0.1, 1.0),
        name: str = "triangle",
    ):
        """
        Build a triangular prism distractor.
        
        Creates a prism with triangular cross-section.
        """
        cache_key = f"triangle_{base:.4f}_{height:.4f}_{depth:.4f}"
        obj_path = MESH_CACHE_DIR / f"{cache_key}.obj"
        
        if not obj_path.exists():
            # Create triangular prism vertices
            # Triangle in XY plane, extruded along Z
            half_base = base / 2
            half_depth = depth / 2
            
            vertices = np.array([
                # Bottom triangle (z = -half_depth)
                [-half_base, 0, -half_depth],
                [half_base, 0, -half_depth],
                [0, height, -half_depth],
                # Top triangle (z = +half_depth)
                [-half_base, 0, half_depth],
                [half_base, 0, half_depth],
                [0, height, half_depth],
            ])
            
            faces = np.array([
                # Bottom face
                [0, 2, 1],
                # Top face
                [3, 4, 5],
                # Side faces
                [0, 1, 4], [0, 4, 3],  # bottom edge
                [1, 2, 5], [1, 5, 4],  # right edge
                [2, 0, 3], [2, 3, 5],  # left edge
            ])
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(str(obj_path))
        
        return self._build_from_obj(obj_path, color, name)
    
    def build_polyhedron(
        self,
        radius: float = 0.03,
        subdivisions: int = 1,
        color: Tuple[float, float, float, float] = (0.6, 0.2, 0.8, 1.0),
        name: str = "polyhedron",
    ):
        """
        Build a polyhedron (icosphere) distractor.
        
        Uses trimesh icosphere for low-poly appearance.
        """
        cache_key = f"polyhedron_{radius:.4f}_{subdivisions}"
        obj_path = MESH_CACHE_DIR / f"{cache_key}.obj"
        
        if not obj_path.exists():
            mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
            mesh.export(str(obj_path))
        
        return self._build_from_obj(obj_path, color, name)
    
    def build_number(
        self,
        digit: int,
        size: float = 0.03,
        depth: float = 0.01,
        color: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 1.0),
        name: Optional[str] = None,
    ):
        """
        Build a 3D number shape (0-9).
        
        Uses voxelized (block-based) representation as shapely may not be available.
        """
        if name is None:
            name = f"number_{digit}"
        
        cache_key = f"number_{digit}_{size:.4f}_{depth:.4f}"
        obj_path = MESH_CACHE_DIR / f"{cache_key}.obj"
        
        if not obj_path.exists():
            mesh = self._generate_number_mesh(digit, size, depth)
            mesh.export(str(obj_path))
        
        return self._build_from_obj(obj_path, color, name)
    
    def _generate_number_mesh(self, digit: int, size: float, depth: float) -> trimesh.Trimesh:
        """
        Generate a voxelized number mesh.
        
        Uses a 3x5 pixel grid for each digit.
        """
        # 3x5 pixel patterns for digits 0-9
        # Each row is bottom to top
        patterns = {
            0: [
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            1: [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 1],
            ],
            2: [
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
            ],
            3: [
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
            ],
            4: [
                [1, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
                [0, 0, 1],
            ],
            5: [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
            ],
            6: [
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            7: [
                [1, 1, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ],
            8: [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            9: [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
            ],
        }
        
        pattern = patterns.get(digit, patterns[0])
        
        # Size of each voxel
        voxel_size = size / 5  # 5 rows
        half_voxel = voxel_size / 2
        half_depth = depth / 2
        
        meshes = []
        for row_idx, row in enumerate(pattern):
            for col_idx, val in enumerate(row):
                if val == 1:
                    # Create a box for this voxel
                    box = trimesh.creation.box(
                        extents=[voxel_size, voxel_size, depth]
                    )
                    # Position: center of grid is origin
                    x = (col_idx - 1) * voxel_size  # col 0,1,2 -> -1,0,1
                    y = (row_idx - 2) * voxel_size  # row 0,1,2,3,4 -> -2,-1,0,1,2
                    box.apply_translation([x, y, 0])
                    meshes.append(box)
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            return combined
        else:
            # Fallback: single box
            return trimesh.creation.box(extents=[size, size, depth])
    
    def _build_from_obj(
        self,
        obj_path: Path,
        color: Tuple[float, float, float, float],
        name: str,
    ):
        """
        Build a Sapien actor from an OBJ file.
        """
        builder = self.scene.create_actor_builder()
        
        # Add visual mesh
        builder.add_visual_from_file(
            filename=str(obj_path),
            material=sapien.render.RenderMaterial(base_color=color),
        )
        
        # Add collision mesh (convex hull for physics)
        builder.add_convex_collision_from_file(
            filename=str(obj_path),
        )
        
        return builder.build(name=name)
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def build_random(
        self,
        shape_type: str,
        name: str,
        color: Optional[Tuple[float, float, float, float]] = None,
        size_scale: float = 1.0,
        graspable_prob: float = 0.7,
    ) -> Tuple:
        """
        Build a distractor of the given type with randomized parameters.
        
        Args:
            shape_type: One of 'cube', 'sphere', 'box', 'stick',
                       'triangle', 'polyhedron', 'number'
            name: Name for the actor
            color: Optional color, random if None
            size_scale: Scale factor for size. Use 1.0-3.0 for varied sizes.
                       Larger values = larger objects.
            graspable_prob: Probability of generating a graspable object (max dim < 7.5cm).
                           Default 0.7 (70% graspable, 30% large for pushing).
                       
        Returns:
            Tuple of (actor, footprint_dict) where footprint_dict contains:
                - 'type': 'circle' or 'aabb'
                - 'radius': float (for circle)
                - 'half_extents': (hx, hy) tuple (for aabb)
                - 'z_offset': float (distance from center to bottom, for table seating)
        """
        if color is None:
            # Use the advanced ColorSampler for better distribution
            rgb, _ = self.color_sampler.sample_color()
            color = (*rgb, 1.0)
        
        # Determine if this object should be graspable
        is_graspable = np.random.random() < graspable_prob
        
        # Graspability constraints (max dim < 7.5cm for safe robot grasping)
        # For graspable: use base ranges. For large: scale up 1.5-2x.
        SAFE_GRASP_HALF = 0.0375  # 3.75cm half = 7.5cm full width
        MIN_SIZE = 0.015  # 1.5cm minimum (no "too small" objects)
        
        # Track footprint for each shape type
        footprint = {'type': 'circle', 'radius': 0.05, 'z_offset': 0.05}  # Default
        
        if shape_type == 'cube':
            if is_graspable:
                half_size = np.random.uniform(MIN_SIZE, SAFE_GRASP_HALF) * size_scale
            else:
                half_size = np.random.uniform(0.04, 0.06) * size_scale  # Large cube
            actor = self.build_cube(half_size=half_size, color=color, name=name)
            footprint = {'type': 'aabb', 'half_extents': (half_size, half_size), 'z_offset': half_size}
            
        elif shape_type == 'sphere':
            if is_graspable:
                radius = np.random.uniform(MIN_SIZE, SAFE_GRASP_HALF) * size_scale
            else:
                radius = np.random.uniform(0.04, 0.06) * size_scale  # Large sphere
            actor = self.build_sphere(radius=radius, color=color, name=name)
            footprint = {'type': 'circle', 'radius': radius, 'z_offset': radius}
            
        elif shape_type == 'cylinder':
            # Cylinder is disabled from default sampling but still buildable
            if is_graspable:
                radius = np.random.uniform(MIN_SIZE, 0.035) * size_scale
                half_length = np.random.uniform(0.03, 0.05) * size_scale
            else:
                radius = np.random.uniform(0.03, 0.05) * size_scale
                half_length = np.random.uniform(0.04, 0.08) * size_scale
            actor = self.build_cylinder(radius=radius, half_length=half_length, color=color, name=name)
            footprint = {'type': 'circle', 'radius': radius, 'z_offset': half_length}
            
        elif shape_type == 'box':
            if is_graspable:
                # At least one dimension must be graspable
                hx = np.random.uniform(MIN_SIZE, SAFE_GRASP_HALF) * size_scale
                hy = np.random.uniform(MIN_SIZE, SAFE_GRASP_HALF) * size_scale
                hz = np.random.uniform(MIN_SIZE, 0.03) * size_scale
            else:
                hx = np.random.uniform(0.05, 0.08) * size_scale
                hy = np.random.uniform(0.03, 0.05) * size_scale
                hz = np.random.uniform(0.02, 0.04) * size_scale
            actor = self.build_box(half_sizes=(hx, hy, hz), color=color, name=name)
            footprint = {'type': 'aabb', 'half_extents': (hx, hy), 'z_offset': hz}
            
        elif shape_type == 'stick':
            # Sticks are thin boxes (square prisms)
            if is_graspable:
                radius = np.random.uniform(0.008, 0.015) * size_scale
                half_length = np.random.uniform(0.06, 0.12) * size_scale
            else:
                # Large stick: thicker and longer
                radius = np.random.uniform(0.025, 0.04) * size_scale
                half_length = np.random.uniform(0.15, 0.25) * size_scale
            actor = self.build_stick(radius=radius, half_length=half_length, color=color, name=name)
            footprint = {'type': 'aabb', 'half_extents': (half_length, radius), 'z_offset': radius}
            
        elif shape_type == 'triangle':
            if is_graspable:
                base = np.random.uniform(0.04, 0.07) * size_scale
                height = np.random.uniform(0.03, 0.06) * size_scale
                depth = np.random.uniform(0.02, 0.04) * size_scale
            else:
                base = np.random.uniform(0.07, 0.10) * size_scale
                height = np.random.uniform(0.05, 0.08) * size_scale
                depth = np.random.uniform(0.03, 0.06) * size_scale
            actor = self.build_triangle(base=base, height=height, depth=depth, color=color, name=name)
            footprint = {'type': 'aabb', 'half_extents': (base/2, depth/2), 'z_offset': 0.0}
            
        elif shape_type == 'polyhedron':
            if is_graspable:
                radius = np.random.uniform(0.02, SAFE_GRASP_HALF) * size_scale
            else:
                radius = np.random.uniform(0.04, 0.06) * size_scale
            subdivisions = np.random.choice([0, 1])
            actor = self.build_polyhedron(radius=radius, subdivisions=subdivisions, color=color, name=name)
            footprint = {'type': 'circle', 'radius': radius, 'z_offset': radius}
            
        elif shape_type == 'number':
            if is_graspable:
                size = np.random.uniform(0.06, 0.10) * size_scale
                depth = np.random.uniform(0.015, 0.030) * size_scale
            else:
                # Large number: significantly bigger
                size = np.random.uniform(0.12, 0.18) * size_scale
                depth = np.random.uniform(0.04, 0.06) * size_scale
            digit = np.random.randint(0, 10)
            actor = self.build_number(digit=digit, size=size, depth=depth, color=color, name=name)
            footprint = {'type': 'circle', 'radius': size * 0.6, 'z_offset': depth / 2}
            
        else:
            raise ValueError(f"Unknown shape type: {shape_type}. Available: cube, sphere, cylinder, box, stick, triangle, polyhedron, number")
        
        return actor, footprint



