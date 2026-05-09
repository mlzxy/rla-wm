import numpy as np
import sapien
import torch
import random
import os
from pathlib import Path
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from datalib.src.table import TableSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.render.shaders import (
    ShaderConfig,
    rt_texture_names,
    rt_texture_transforms,
)
from mani_skill.utils.registration import register_env
from datalib.src import robots
from datalib.src.distractors.placement import UniformFrontSampler, CollisionAwareSampler
from datalib.src.distractors.builder import DistractorBuilder
from datalib.src.distractors.grasping import GraspGenerator


# New shape types supported by DistractorBuilder
NEW_SHAPE_TYPES = [
    "cube",
    "sphere",
    "box",
    "stick",
    "triangle",
    "polyhedron",
    "number",
    "cylinder",
]
# Legacy types that use old Objaverse logic
LEGACY_TYPES = ["objaverse", "legacy_objaverse"]


# --- Camera Config: cinematic_camera_0 ---
# eye: [0.8000, 0.0000, 0.6000]
# target: [-0.0480, 0.0000, 0.0700]
# fov: 1.0000
# -------------------------------------------
# --- Camera Config: cinematic_camera_1 ---
# eye: [0.5016, 1.2687, 0.7413]
# target: [0.0307, 0.4532, 0.4049]
# fov: 0.8560
# -------------------------------------------
# --- Camera Config: cinematic_camera_2 ---
# eye: [-0.5635, 0.7759, 0.6371]
# target: [-0.1395, 0.0415, 0.1071]
# fov: 1.1650
# -------------------------------------------
# --- Camera Config: cinematic_camera_3 ---
# eye: [-1.3752, 0.0000, 0.8034]
# target: [-0.4717, -0.0000, 0.3747]
# fov: 0.9340
# -------------------------------------------
# --- Camera Config: cinematic_camera_4 ---
# eye: [-0.6571, -0.8381, 0.7590]
# target: [-0.2331, -0.1037, 0.2290]
# fov: 0.9410
# -------------------------------------------
# --- Camera Config: cinematic_camera_5 ---
# eye: [0.5449, -1.2437, 0.7413]
# target: [0.0740, -0.4282, 0.4049]
# fov: 0.7920
# -------------------------------------------


class UnifiedWorkspaceEnv(BaseEnv):
    """
    Base environment for all v2 tasks with a unified workspace.
    Features:
    - Standard TableSceneBuilder (Table top at z=0)
    - 7 Cinematic Cameras for full scene capture
    - Standardized cinematic lighting
    - Enhanced distractor generation with diverse primitives
    """

    # Default workspace bounds for placement (front of robot)
    DEFAULT_X_BOUNDS = (-0.1, 0.1)  # Forward from robot
    DEFAULT_Y_BOUNDS = (-0.2, 0.2)  # Left-right
    MIN_DISTRACTOR_DISTANCE = 0.08  # Minimum spacing between objects

    def __init__(
        self,
        *args,
        robot_uids="panda_stick",
        robot_init_qpos_noise=0.02,
        robot_init_high=False,
        num_distractors=0,
        distractor_types=["cube", "sphere", "box"],
        distractor_density=None,
        collision_free_placement=True,
        workspace_x_bounds=None,
        workspace_y_bounds=None,
        drop_on_table=False,
        z_stagger=0.0,
        random_rotation=False,
        include_all_cameras=False,
        camera_names=None,
        distractor_scale_min=1.0,
        distractor_scale_max=1.0,
        camera_width=256,
        camera_height=256,
        shader_dir="default",
        **kwargs,
    ):
        """
        Args:
            robot_init_high: If True, pose robot arm straight up during init.
            distractor_density: If set (float, objects/m²), overrides num_distractors.
                                Calculated as: workspace_area * density.
            collision_free_placement: If True, use CollisionAwareSampler for no-overlap placement.
            workspace_x_bounds: Custom (x_min, x_max) bounds. Default: (-0.1, 0.3).
            workspace_y_bounds: Custom (y_min, y_max) bounds. Default: (-0.25, 0.25).
            drop_on_table: If True, use z_offset from footprints to sit objects on table.
            z_stagger: If > 0, stagger Z positions by this amount per object (for drop effect).
            random_rotation: If True, apply random Z-axis rotation to each distractor.
            distractor_scale_min: Minimum scale factor for distractors.
            distractor_scale_max: Maximum scale factor for distractors.
            camera_width: Width of all cameras.
            camera_height: Height of all cameras.
            shader_dir: Shader pack name/path.
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.shader_dir = shader_dir
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.robot_init_high = robot_init_high
        self.distractor_density = distractor_density
        self.collision_free_placement = collision_free_placement
        self.distractor_types = distractor_types
        self.distractors = []
        self.distractor_footprints = []  # Store footprints for collision-aware placement
        self.distractor_grasps = {}  # Store pre-generated grasps: {actor_name: [(width, local_pose), ...]}
        self.camera_names = camera_names
        if camera_names:
            include_all_cameras = True
        self.include_all_cameras = include_all_cameras

        # Scale randomization
        self.distractor_scale_min = distractor_scale_min
        self.distractor_scale_max = distractor_scale_max

        # Configurable workspace bounds
        self.workspace_x_bounds = workspace_x_bounds or self.DEFAULT_X_BOUNDS
        self.workspace_y_bounds = workspace_y_bounds or self.DEFAULT_Y_BOUNDS

        # Drop-on-table and rotation settings
        self.drop_on_table = drop_on_table
        self.z_stagger = z_stagger
        self.random_rotation = random_rotation

        # Calculate num_distractors from density if provided
        if distractor_density is not None:
            x_range = self.workspace_x_bounds[1] - self.workspace_x_bounds[0]
            y_range = self.workspace_y_bounds[1] - self.workspace_y_bounds[0]
            workspace_area = x_range * y_range
            self.num_distractors = int(workspace_area * distractor_density)
        else:
            self.num_distractors = num_distractors

        # Initialize placement samplers with configurable bounds
        self._placement_sampler = UniformFrontSampler(
            x_bounds=self.workspace_x_bounds,
            y_bounds=self.workspace_y_bounds,
            min_distance=self.MIN_DISTRACTOR_DISTANCE,
        )
        self._collision_sampler = CollisionAwareSampler(
            x_bounds=self.workspace_x_bounds,
            y_bounds=self.workspace_y_bounds,
            margin=0.01,
            warn_on_collision=True,
        )

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def sensors(self):
        return self._sensors

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            if self.robot_init_high:
                qpos = self.agent.keyframes["rest_high"].qpos
            elif "rest" in self.agent.keyframes:
                qpos = self.agent.keyframes["rest"].qpos
            elif "zeros" in self.agent.keyframes:
                qpos = self.agent.keyframes["zeros"].qpos
            else:
                qpos = np.zeros(self.agent.robot.dof)

            qpos = common.to_tensor(qpos, device=self.device)
            # Apply initialization noise
            if self.robot_init_qpos_noise > 0:
                qpos = (
                    qpos
                    + torch.randn((b, len(qpos)), device=self.device)
                    * self.robot_init_qpos_noise
                )

            self.agent.reset(qpos)
            # Standardize mount at reset to prevent drifting
            if self.robot_uids not in [
                "xarm6_allegro_left",
                "xarm6_allegro_right",
                "xarm6_robotiq",
                "xarm6_nogripper",
            ]:
                self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import GPUMemoryConfig

        config = super()._default_sim_config
        config.spacing = 50
        config.gpu_memory_config = GPUMemoryConfig(
            found_lost_pairs_capacity=2**25,
            max_rigid_patch_count=2**21,
            max_rigid_contact_count=2**23,
        )
        config.scene_config.enable_ccd = True
        return config

    @property
    def shader_kwargs(self):
        if self.shader_dir == "rt-clean":
            shader_kwargs = {
                "shader_config": ShaderConfig(
                    shader_pack="rt",
                    texture_names=rt_texture_names,
                    shader_pack_config={
                        "ray_tracing_samples_per_pixel": 64,  # 32
                        "ray_tracing_path_depth": 32,
                        "ray_tracing_denoiser": "none",
                    },
                    texture_transforms=rt_texture_transforms,
                )
            }
        else:
            shader_kwargs = {"shader_pack": self.shader_dir}
        return shader_kwargs

    @property
    def _default_sensor_configs(self):
        """Camera setup - base camera + cinematic cameras."""
        configs = []

        # 1. Base Camera
        pose = sapien_utils.look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        configs.append(
            CameraConfig(
                "base_camera",
                pose,
                self.camera_width,
                self.camera_height,
                np.pi / 2,
                0.01,
                100,
                **self.shader_kwargs,
            )
        )

        if self.include_all_cameras:
            # for i in range(num_cinematic_cameras):
            #     cam_name = f"cinematic_camera_{i}"
            #     angle = (2 * np.pi * i) / 6
            #     radius = 0.8 if i % 2 == 0 else 1.4
            #     height = 0.6
            #     eye = [radius * np.cos(angle), radius * np.sin(angle), height]
            #     pose = sapien_utils.look_at(eye, [0, 0, 0.1])
            cinematic_configs = [
                (
                    "front_lower_camera",
                    [0.8000, 0.0000, 0.6000],
                    [-0.0480, 0.0000, 0.0700],
                    1.0000,
                ),
                (
                    "front_right_camera",
                    [0.5016, 1.2687, 0.7413],
                    [0.0307, 0.4532, 0.4049],
                    0.8560,
                ),
                (
                    "rear_left_camera",
                    [-0.5635, 0.7759, 0.6371],
                    [-0.1395, 0.0415, 0.1071],
                    1.1650,
                ),
                (
                    "rear_camera",
                    [-1.3752, 0.0000, 0.8034],
                    [-0.4717, -0.0000, 0.3747],
                    0.9340,
                ),
                (
                    "rear_right_camera",
                    [-0.6571, -0.8381, 0.7590],
                    [-0.2331, -0.1037, 0.2290],
                    0.9410,
                ),
                (
                    "front_left_camera",
                    [0.5449, -1.2437, 0.7413],
                    [0.0740, -0.4282, 0.4049],
                    0.7920,
                ),
            ]

            for cam_name, eye, target, fov in cinematic_configs:
                pose = sapien_utils.look_at(eye, target)
                configs.append(
                    CameraConfig(
                        cam_name,
                        pose,
                        self.camera_width,
                        self.camera_height,
                        fov,
                        0.01,
                        100,
                        **self.shader_kwargs,
                    )
                )

        if self.camera_names:
            configs = [c for c in configs if c.uid in self.camera_names]

        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.2, 1.2, 1.0], target=[0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera",
            pose,
            512,
            512,
            1,
            0.01,
            100,
            **self.shader_kwargs,
        )

    def _load_agent(self, options: dict):
        # Mount at z=0 by default, ensuring fix_root_link is handled by agent class
        BaseEnv._load_agent(self, options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict, skip_table=False):
        BaseEnv._load_scene(self, options)
        if not skip_table:
            if not hasattr(self, "table_scene") or self.table_scene is None:
                self.table_scene = TableSceneBuilder(self)
            if not hasattr(self.table_scene, "table"):
                self.table_scene.build()

        self._load_distractors()
        self._generate_main_object_grasps()

    def _load_lighting(self, *args):
        self.scene.set_ambient_light([0.6, 0.6, 0.6])
        # self.scene.add_directional_light(
        #     direction=[-1, -1, -1],
        #     color=[1.0, 1.0, 1.0],
        #     shadow=shadow,
        #     shadow_scale=2.0,
        #     shadow_map_size=2048,
        # )
        light_color = [0.75, 0.75, 0.75]
        self.scene.add_point_light(
            position=[0.5, 0.5, 1.0], color=light_color, shadow=False
        )
        self.scene.add_point_light(
            position=[0.5, -0.5, 1.0], color=light_color, shadow=False
        )
        # self.scene.set_ambient_light([0.3, 0.3, 0.3])
        # self.scene.add_directional_light(
        #     [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        # )
        # self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_distractors(self):
        """Load distractor objects using DistractorBuilder, storing footprints."""
        if isinstance(self.num_distractors, (list, tuple)):
            num_to_load = random.randint(
                self.num_distractors[0], self.num_distractors[1]
            )
        else:
            num_to_load = self.num_distractors

        if num_to_load <= 0:
            return

        # Clear internal state for distractors during reconfiguration
        self.distractors = []
        self.distractor_footprints = []
        self.distractor_grasps = {}

        # Initialize builder and generator
        builder = DistractorBuilder(self.scene)
        grasp_generator = GraspGenerator()

        # Separate new types from legacy types
        new_types = [t for t in self.distractor_types if t in NEW_SHAPE_TYPES]
        legacy_types = [t for t in self.distractor_types if t in LEGACY_TYPES]

        # Prepare legacy objects if needed
        objaverse_objects = []
        if legacy_types:
            objaverse_objects = self._get_objaverse_object_paths()

        for i in range(num_to_load):
            # Choose between new and legacy types
            available_types = new_types.copy()
            if legacy_types and objaverse_objects:
                available_types.extend(legacy_types)

            if not available_types:
                available_types = ["cube"]  # Fallback

            dtype = random.choice(available_types)
            # color = tuple(np.random.uniform(0.2, 0.9, 3).tolist()) + (1.0,)
            color = None
            name = f"distractor_{i}_{dtype}"
            footprint = {"type": "circle", "radius": 0.05}  # Default footprint

            if dtype in NEW_SHAPE_TYPES:
                # Random scale between scale_min and scale_max
                scale = np.random.uniform(
                    self.distractor_scale_min, self.distractor_scale_max
                )

                # Use new DistractorBuilder (returns actor, footprint tuple)
                distractor, footprint = builder.build_random(
                    shape_type=dtype,
                    name=name,
                    color=color,
                    size_scale=scale,
                )
            elif dtype in LEGACY_TYPES and objaverse_objects:
                # Use legacy Objaverse logic
                obj_info = random.choice(objaverse_objects)
                scale = 0.15 * random.uniform(0.8, 1.2)
                distractor = self._build_legacy_objaverse(obj_info, scale, name=name)
                footprint = {"type": "circle", "radius": scale * 0.5}  # Approximate
            else:
                # Fallback to cube
                half_size = np.random.uniform(0.02, 0.04)
                distractor = builder.build_cube(
                    half_size=half_size,
                    color=color,
                    name=name,
                )
                footprint = {"type": "aabb", "half_extents": (half_size, half_size)}

            self.distractors.append(distractor)
            self.distractor_footprints.append(footprint)

            # Pre-generate grasps
            grasps = grasp_generator.generate(distractor, dtype)
            self.distractor_grasps[distractor.name] = grasps

    def remove_off_table_objects(self):
        """Remove distractor objects that have fallen off the table."""
        if not self.distractors:
            return

        # Get table bounds from TableSceneBuilder
        # Note: table top is at z=0. table_height is depth of the table block.
        # table_length is along x, table_width is along y (after pi/2 rotation in builder)
        # Actually TableSceneBuilder.build() sets self.table_length, width, height from AABB.
        # In TableOnly-v2, the table is centered at [-0.12, 0, -TABLE_DIM[2]] but the top is at 0.

        # We use a simple heuristic: if z < -0.05, it's definitely fallen or falling.
        # Also check if it's far outside the x/y bounds of the workspace.
        h_buffer = 0.02  # 10cm buffer outside workspace
        x_min, x_max = (
            self.workspace_x_bounds[0] - h_buffer,
            self.workspace_x_bounds[1] + h_buffer,
        )
        y_min, y_max = (
            self.workspace_y_bounds[0] - h_buffer,
            self.workspace_y_bounds[1] + h_buffer,
        )

        # We iterate backwards to safely remove from list
        removed_count = 0
        new_distractors = []
        new_footprints = []

        for i in range(len(self.distractors)):
            distractor = self.distractors[i]
            footprint = self.distractor_footprints[i]

            pos = distractor.pose.p.cpu().numpy().reshape(-1)

            # Check if fallen or outside reachable workspace
            is_off = (
                pos[2] < -0.05
                or pos[0] < x_min
                or pos[0] > x_max
                or pos[1] < y_min
                or pos[1] > y_max
            )

            if is_off:
                # "Remove" by teleporting far away and hiding
                distractor.set_pose(sapien.Pose(p=[0, 0, -10]))
                if distractor.name in self.distractor_grasps:
                    del self.distractor_grasps[distractor.name]
                removed_count += 1
            else:
                new_distractors.append(distractor)
                new_footprints.append(footprint)

        self.distractors = new_distractors
        self.distractor_footprints = new_footprints

        if removed_count > 0:
            print(
                f"  [Cleanup] Removed {removed_count} objects that fell off the table."
            )

    def _generate_main_object_grasps(self):
        """Generate grasps for standard task objects if not already present."""
        if not self.distractors:
            return
        grasp_generator = GraspGenerator()
        # Common ManiSkill task object attributes
        for attr in ["obj", "cube", "tool", "peg", "goal", "box"]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                # Handle both single actors and lists
                actors_to_check = val if isinstance(val, (list, tuple)) else [val]
                for actor in actors_to_check:
                    # Some attributes might be None or not actors
                    if actor is None or not hasattr(actor, "name"):
                        continue
                    if actor.name not in self.distractor_grasps:
                        # Determine dtype (rough guess based on name)
                        dtype = "cube"
                        name_lower = actor.name.lower()
                        if "tool" in name_lower or "peg" in name_lower:
                            dtype = "stick"
                        elif "sphere" in name_lower:
                            dtype = "sphere"

                        grasps = grasp_generator.generate(actor, dtype)
                        self.distractor_grasps[actor.name] = grasps
                        print(
                            f"  [Grasp Info] Generated {len(grasps)} grasps for task object: {actor.name}"
                        )

    # =========================================================================
    # Legacy Objaverse Support (isolated)
    # =========================================================================

    def _get_objaverse_object_paths(self):
        """Legacy: Get available Objaverse object paths."""
        objaverse_root = os.path.expanduser("~/.cache/v4-world/objaverse")
        objaverse_path = Path(objaverse_root)
        if not objaverse_path.exists():
            return []
        object_paths = []
        for category_dir in objaverse_path.iterdir():
            if not category_dir.is_dir():
                continue
            for obj_dir in category_dir.iterdir():
                if not obj_dir.is_dir():
                    continue
                visual_file = obj_dir / "visual" / "model_normalized_0.obj"
                collision_dir = obj_dir / "collision"
                if visual_file.exists() and collision_dir.exists():
                    collision_files = sorted(
                        collision_dir.glob("model_normalized_collision_*.obj")
                    )
                    if len(collision_files) > 0:
                        object_paths.append(
                            {
                                "visual": str(visual_file),
                                "collision_files": [str(f) for f in collision_files],
                            }
                        )
        return object_paths

    def _build_legacy_objaverse(self, obj_path_info, scale, name):
        """Legacy: Build an Objaverse object."""
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=obj_path_info["visual"], scale=[scale] * 3
        )
        try:
            builder.add_nonconvex_collision_from_file(
                filename=obj_path_info["visual"], scale=[scale] * 3
            )
        except Exception:
            try:
                builder.add_nonconvex_collision_from_file(
                    filename=obj_path_info["collision_files"][0], scale=[scale] * 3
                )
            except Exception:
                pass
        return builder.build(name=name)

    # =========================================================================
    # Episode Initialization
    # =========================================================================

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if hasattr(self, "table_scene") and self.table_scene is not None:
            self.table_scene.initialize(env_idx)
        self._initialize_agent(env_idx)
        self._initialize_distractors(env_idx)

    def _initialize_distractors(self, env_idx: torch.Tensor):
        """Place distractors using collision-aware or uniform sampling."""
        # Print grasp information for each object
        for name, grasps in self.distractor_grasps.items():
            print(f"  {name}: {len(grasps)} valid grasps found")

        with torch.device(self.device):
            b = len(env_idx)
            if isinstance(self.num_distractors, (list, tuple)):
                # Random number of distractors for each environment in the batch
                low, high = self.num_distractors
                active_counts = torch.randint(low, high + 1, (b,), device=self.device)
            else:
                active_counts = torch.full(
                    (b,), self.num_distractors, device=self.device
                )

            # Sample positions
            max_active = (
                int(active_counts.max().item()) if len(active_counts) > 0 else 0
            )
            sampled_positions = []

            if max_active > 0 and len(self.distractors) > 0:
                num_to_place = min(max_active, len(self.distractors))

                if (
                    self.collision_free_placement
                    and len(self.distractor_footprints) >= num_to_place
                ):
                    # Use collision-aware placement with footprints
                    footprints_to_place = self.distractor_footprints[:num_to_place]
                    sampled_positions = self._collision_sampler.place_objects(
                        footprints_to_place,
                        z_height=0.05,
                        use_z_offset=self.drop_on_table,
                        z_stagger=self.z_stagger,
                    )
                    sampled_positions = torch.tensor(
                        np.array(sampled_positions),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    # Fallback to uniform Poisson Disk sampling
                    sampled_positions = self._placement_sampler.sample_with_z(
                        num_to_place, z_height=0.05
                    )
                    sampled_positions = torch.tensor(
                        sampled_positions, device=self.device, dtype=torch.float32
                    )

            for i, distractor in enumerate(self.distractors):
                # Only place if index < active_count for that env
                mask = (i < active_counts).float()

                if i < len(sampled_positions):
                    # Use sampled position
                    xyz = sampled_positions[i].unsqueeze(0).expand(b, -1).clone()
                else:
                    # Fallback to random if more distractors than samples
                    xy = torch.rand((b, 2), device=self.device)
                    xy[:, 0] = (
                        xy[:, 0]
                        * (self.workspace_x_bounds[1] - self.workspace_x_bounds[0])
                        + self.workspace_x_bounds[0]
                    )
                    xy[:, 1] = (
                        xy[:, 1]
                        * (self.workspace_y_bounds[1] - self.workspace_y_bounds[0])
                        + self.workspace_y_bounds[0]
                    )
                    z = torch.ones((b, 1), device=self.device) * 0.05
                    xyz = torch.cat([xy, z], dim=1)

                # If inactive, teleport far under table
                xyz[mask == 0] = torch.tensor([0, 0, -10.0], device=self.device)

                # Apply random rotation around Z-axis if enabled
                if self.random_rotation:
                    # Random angle per environment
                    angles = torch.rand(b, device=self.device) * 2 * np.pi
                    # Quaternion for Z-axis rotation: [cos(a/2), 0, 0, sin(a/2)]
                    q = torch.stack(
                        [
                            torch.cos(angles / 2),
                            torch.zeros(b, device=self.device),
                            torch.zeros(b, device=self.device),
                            torch.sin(angles / 2),
                        ],
                        dim=1,
                    )
                else:
                    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                        b, 1
                    )

                distractor.set_pose(torch.cat([xyz, q], dim=1))

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_static(self, threshold: float = 0.01) -> torch.Tensor:
        """Check if all objects in the scene (excluding table/ground/robot) are static."""
        is_static = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for actor in self.scene.actors.values():
            if actor.px_body_type == "dynamic":
                # Skip robot links (usually handled via articulations but check just in case)
                if (
                    "robot" in actor.name
                    or "panda" in actor.name
                    or "xarm" in actor.name
                ):
                    continue
                # Skip environment structures
                if any(x in actor.name for x in ["table", "ground", "workspace"]):
                    continue

                is_static = torch.logical_and(
                    is_static,
                    actor.is_static(lin_thresh=threshold, ang_thresh=threshold * 10),
                )

        for articulation in self.scene.articulations.values():
            # Skip robot
            if articulation == self.agent.robot:
                continue

            # Check qvel
            qvel_static = torch.all(torch.abs(articulation.qvel) <= threshold, dim=1)
            # Check root velocity
            root_static = torch.logical_and(
                torch.linalg.norm(articulation.root_linear_velocity, axis=1)
                <= threshold,
                torch.linalg.norm(articulation.root_angular_velocity, axis=1)
                <= threshold * 10,
            )
            is_static = torch.logical_and(
                is_static, torch.logical_and(qvel_static, root_static)
            )

        return is_static

    def step_wo_obs(self, action=None):
        self._step_action(action)
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self._get_obs_state_dict(info)
        obs = self._flatten_raw_obs(obs)
        self._last_obs = obs
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        return (
            obs,
            None,
            terminated,
            None,
            info,
        )

    def state_dict(self) -> dict:
        """Snapshot the complete state of the environment."""
        return self.get_state_dict()

    def load_state_dict(self, state: dict):
        """Restore the state of the environment."""
        self.set_state_dict(state)

    def _clear(self):
        super()._clear()
        self.distractors.clear()
        self.table_scene = None

    def get_object_poses(self) -> np.ndarray:
        """
        Returns the poses of all task-relevant objects as a numpy array.
        Format: (num_objects, 7) where each pose is xyz + xyzw.
        If batched, returns (b, num_objects, 7).
        Maintains a consistent order by sorting the actor names.
        """
        poses = []
        for name in sorted(self.scene.actors.keys()):
            if any(
                x in name.lower()
                for x in [
                    "table",
                    "ground",
                    "robot",
                    "workspace",
                    "goal_region",
                    "camera",
                    "panda",
                    "xarm",
                ]
            ):
                continue

            actor = self.scene.actors[name]

            if hasattr(actor.pose, "p") and hasattr(actor.pose.p, "cpu"):
                p = actor.pose.p.cpu().numpy()
            else:
                p = np.array(actor.pose.p)

            if hasattr(actor.pose, "q") and hasattr(actor.pose.q, "cpu"):
                q = actor.pose.q.cpu().numpy()
            else:
                q = np.array(actor.pose.q)

            if q.ndim == 1:
                xyzw = q[[1, 2, 3, 0]]
                pose_7d = np.concatenate([p, xyzw], axis=0)
            else:
                xyzw = q[:, [1, 2, 3, 0]]
                pose_7d = np.concatenate([p, xyzw], axis=1)

            poses.append(pose_7d)

        if not poses:
            num_envs = getattr(self, "num_envs", 1)
            b = num_envs if isinstance(num_envs, int) else 1
            if b > 1:
                return np.zeros((b, 0, 7))
            else:
                return np.zeros((0, 7))

        if poses[0].ndim == 2:
            return np.stack(poses, axis=1)
        else:
            return np.stack(poses, axis=0)
    
    def set_object_poses(self, object_poses):
        """
        Set the poses of all task-relevant objects.
        Format: (num_objects, 7) or (b, num_objects, 7) where each pose is xyz + xyzw.
        Must match the order and count returned by get_object_poses.
        """
        object_poses = np.asarray(object_poses)
        batched = object_poses.ndim == 3

        obj_idx = 0
        for name in sorted(self.scene.actors.keys()):
            if any(
                x in name.lower()
                for x in [
                    "table",
                    "ground",
                    "robot",
                    "workspace",
                    "goal_region",
                    "camera",
                    "panda",
                    "xarm",
                ]
            ):
                continue

            actor = self.scene.actors[name]

            if batched:
                pose_7d = object_poses[:, obj_idx, :]  # (b, 7)
                p = pose_7d[:, :3]
                xyzw = pose_7d[:, 3:]
                q = xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
            else:
                pose_7d = object_poses[obj_idx, :]  # (7,)
                p = pose_7d[:3]
                xyzw = pose_7d[3:]
                q = xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz

            pose = sapien.Pose(p=p, q=q)
            actor.set_pose(pose)
            obj_idx += 1
    

    @property
    def scene_objects(self):
        """Return poses and meshes of all task-relevant objects."""
        objects = []
        for name, actor in self.scene.actors.items():
            # Filter out non-task objects
            if any(
                x in name.lower()
                for x in [
                    "table",
                    "ground",
                    "robot",
                    "workspace",
                    "goal_region",
                    "camera",
                ]
            ):
                continue

            # Get meshes
            meshes = []
            for obj in actor._objs:
                for comp in obj.components:
                    if isinstance(comp, sapien.render.RenderBodyComponent):
                        for shape in comp.render_shapes:
                            meshes.append(shape)

            objects.append({"name": name, "pose": actor.pose, "meshes": meshes})

        for name, articulation in self.scene.articulations.items():
            if articulation == self.agent.robot:
                continue

            objects.append(
                {
                    "name": name,
                    "pose": articulation.root_pose,
                    "qpos": articulation.qpos,
                    "meshes": [],  # Articulation meshes are complex, omitting for now or can add link-by-link
                }
            )
        return objects


@register_env("TableOnly-v2", max_episode_steps=100)
class TableOnlyEnvV2(UnifiedWorkspaceEnv):
    """Empty table environment for pure teleoperation/exploration."""

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

    def evaluate(self) -> dict:
        """Default evaluation - returns empty success dict."""
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }

    def compute_dense_reward(self, obs, action, info) -> torch.Tensor:
        """Default dense reward - returns zeros."""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info) -> torch.Tensor:
        """Default normalized dense reward - returns zeros."""
        return torch.zeros(self.num_envs, device=self.device)
