import gymnasium as gym
import numpy as np
import sapien
import os
import random
from pathlib import Path
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.common import to_numpy
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from mani_skill.envs.sapien_env import BaseEnv
from datalib.cameras import CinematicCameraSystem
from mani_skill.agents.multi_agent import MultiAgent


def _ensure_rng_initialized(env_unwrapped):
    """Ensure _batched_episode_rng is initialized for the environment."""
    if (
        not hasattr(env_unwrapped, "_batched_episode_rng")
        or env_unwrapped._batched_episode_rng is None
    ):
        num_envs = getattr(env_unwrapped, "num_envs", 1)
        # Initialize with random seeds if not already initialized
        seeds = [random.randint(0, 2**31 - 1) for _ in range(num_envs)]
        env_unwrapped._batched_episode_rng = BatchedRNG.from_seeds(
            seeds, backend="numpy:random_state"
        )
        env_unwrapped._episode_seed = np.array(seeds, dtype=np.int64)


def make_enriched_env(
    task_id: str,
    num_distractors=5,
    num_cameras=3,
    camera_res=(250, 250),
    shader_pack="rt",
    objaverse_scale=0.15,  # Scale factor for objaverse objects (default 0.1 = 10% of original size)
    distractor_offset=(0.0, 0.0),
    **kwargs,
):
    """
    Factory function that creates a ManiSkill environment (any task)
    and dynamically enriches it with distractors and cinematic cameras.

    Args:
        camera_res: Tuple of (width, height) for camera resolution. Affects both base camera and cinematic cameras.
        shader_pack: Shader pack configuration for cameras (e.g., "rt" for raytracing).
        objaverse_scale: Scale factor for objaverse objects. Default 0.1 means objects are 10% of original size.
                         Adjust this if objects appear too large or too small.
        distractor_offset: Tuple of (x, y) offset for spawning distractors.
    """

    # 2. Initialize Environment with sensor_configs for base camera
    # Pass camera_res and shader_pack to affect the builtin base camera
    sensor_configs = dict(
        width=camera_res[0], height=camera_res[1], shader_pack=shader_pack
    )
    # Merge with any existing sensor_configs in kwargs
    if "sensor_configs" in kwargs:
        sensor_configs.update(kwargs["sensor_configs"])
        kwargs = {k: v for k, v in kwargs.items() if k != "sensor_configs"}
    kwargs["sensor_configs"] = sensor_configs
    env = gym.make(task_id, **kwargs)

    # 1. Define Cinematic Camera Configs (as CameraConfig objects)
    # All cameras start at default pose (will be moved by cinematic system)
    cinematic_camera_configs = []
    for i in range(num_cameras):
        cam_name = f"cinematic_cam_{i}"
        cinematic_camera_configs.append(
            CameraConfig(
                uid=cam_name,
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=camera_res[0],
                height=camera_res[1],
                fov=1.0,
                near=0.1,
                far=10.0,
                mount=None,
                shader_pack=shader_pack,
            )
        )

    # 3. Manually add cinematic cameras to the environment
    # _setup_sensors was already called during __init__, so we add cameras directly
    from mani_skill.sensors.camera import parse_camera_configs, Camera

    # Parse and add to _sensor_configs
    parsed_cinematic = parse_camera_configs(cinematic_camera_configs)
    env.unwrapped._sensor_configs.update(parsed_cinematic)

    # Create the actual sensor objects for our cameras
    for uid, sensor_config in parsed_cinematic.items():
        if uid not in env.unwrapped._sensors:
            # Create sensor object (articulation=None for non-agent cameras)
            sensor = Camera(sensor_config, env.unwrapped.scene, articulation=None)
            env.unwrapped._sensors[uid] = sensor

    # Update scene.sensors to include our new cameras
    env.unwrapped.scene.sensors = env.unwrapped._sensors

    # 4. Attach State
    env.num_distractors = num_distractors
    env.num_cinematic_cameras = num_cameras
    env.cinematic_system = None
    env.objaverse_scale = objaverse_scale  # Store scale parameter
    env.distractor_offset = distractor_offset  # Store offset parameter

    # 5. Monkey Patch _load_scene
    # original_load_scene = env.unwrapped._load_scene

    # def enriched_load_scene(options: dict):
    #     # Load Original Task
    #     original_load_scene(options)

    #     # Spawn Distractors
    #     if env.num_distractors > 0:
    #         spawn_distractors(env)

    # env.unwrapped._load_scene = enriched_load_scene

    # 6. If scene is already loaded (from __init__), manually add our enrichments
    # Check if scene has been initialized by checking if agent exists
    if hasattr(env, "agent") and env.agent is not None:
        # Spawn distractors if not already spawned
        if env.num_distractors > 0:
            distractor_count = sum(
                1
                for actor in env.scene.actors.values()
                if actor is not None and "distractor" in actor.name.lower()
            )
            if distractor_count == 0:
                spawn_distractors(env, offset=env.distractor_offset)

    # 7. Monkey Patch _setup_sensors to re-add cinematic cameras on reconfigure and capture base camera pose
    # (in case _reconfigure is called, which calls _setup_sensors again)
    original_setup_sensors = env.unwrapped._setup_sensors

    def enriched_setup_sensors(options: dict):
        # Call original to set up default sensors
        original_setup_sensors(options)

        # Re-add our cinematic cameras (in case they were cleared)
        parsed_cinematic = parse_camera_configs(cinematic_camera_configs)
        env.unwrapped._sensor_configs.update(parsed_cinematic)

        # Create sensor objects if they don't exist
        for uid, sensor_config in parsed_cinematic.items():
            if uid not in env.unwrapped._sensors:
                sensor = Camera(sensor_config, env.unwrapped.scene, articulation=None)
                env.unwrapped._sensors[uid] = sensor

        # Update scene.sensors
        env.unwrapped.scene.sensors = env.unwrapped._sensors

    env.unwrapped._setup_sensors = enriched_setup_sensors

    # 8. Monkey Patch reset to Init Cinematic System
    # Note: _setup_sensors runs in __init__, which is already done.
    # However, we need to initialize the CinematicSystem object for `step` to use.
    # We can do this right now or inside the first step/reset.
    # Best place is typically after reset because robot needs to exist.
    # We'll hook into `reset` (via a wrapper or patch).

    original_reset = env.unwrapped.reset

    def enriched_reset(seed=None, options=None):
        # Ensure enriched_load_scene is called if reconfigure is True
        if options is None:
            options = {}

        # If reconfigure is True or not specified (default behavior), _load_scene will be called
        # Our monkey patch will ensure enriched_load_scene is used
        obs, info = original_reset(seed=seed, options=options)

        # Init System if not exists (Robot exists now)
        if env.cinematic_system is None:
            init_cinematic_system(env)
        return obs, info

    env.unwrapped.reset = enriched_reset

    # 9. Monkey Patch step for Camera Update
    original_step = env.unwrapped.step

    def enriched_step(action):
        # Update Cameras
        if env.cinematic_system:
            # Use internal step if available, else 0
            step_count = (
                env.unwrapped.elapsed_steps
                if hasattr(env.unwrapped, "elapsed_steps")
                else 0
            )
            for i, cam in enumerate(env.cinematic_system.cameras):
                config = env.cinematic_system.configs[i]
                mode = config["mode"]
                # Pass all config parameters to update_pose
                kwargs = {k: v for k, v in config.items() if k not in ["name", "mode"]}
                env.cinematic_system.update_pose(cam, mode, step_count, 200, **kwargs)

        return original_step(action)

    env.unwrapped.step = enriched_step

    return env


def init_cinematic_system(env):
    """Initializes the camera controller logic."""

    cam_configs = []

    # "Fixed" mode for all cameras to ensure temporal consistency
    mode = "fixed"

    # Configuration for circular placement
    # Cameras surrounding the workspace circle, with varying distances
    for i in range(env.num_cinematic_cameras):
        # Evenly spaced angles around the circle
        angle_offset = (2 * np.pi * i) / env.num_cinematic_cameras

        # Alternating distances (close / further away) to reduce partial observability
        # Close: 0.8m, Far: 1.4m (slightly larger range to cover more)
        radius = 0.8 if i % 2 == 0 else 1.4

        # Fixed height to ensure consistent perspective
        height = 0.6

        # Unused parameters for fixed mode (set to defaults)
        omega = 0.0
        phase = 0.0
        vertical_speed = 0.0
        orbit_direction = 1.0

        cam_configs.append(
            {
                "name": f"cinematic_cam_{i}",
                "mode": mode,
                "radius": radius,
                "height": height,
                "angle_offset": angle_offset,
                "omega": omega,
                "phase": phase,
                "vertical_speed": vertical_speed,
                "orbit_direction": orbit_direction,
            }
        )

    # Find robot actor (usually agent.robot or first articulation)
    robot = None
    if hasattr(env.agent, "robot"):
        # Multi-link robot
        robot = env.agent.robot.links[0]
    else:
        # Fallback
        robot = env.agent.robot

    env.cinematic_system = CinematicCameraSystem(env.scene, robot, cam_configs)

    # Link actual sensors
    for i in range(env.num_cinematic_cameras):
        cam_name = f"cinematic_cam_{i}"
        # ManiSkill stores sensors in _sensors dict
        if cam_name in env.unwrapped._sensors:
            cam = env.unwrapped._sensors[cam_name]
            env.cinematic_system.cameras.append(cam)


def get_objaverse_object_paths(objaverse_root=None):
    """Get all available objaverse object paths."""
    if objaverse_root is None:
        # Try to find objaverse folder relative to this file
        current_dir = Path(__file__).parent.parent
        objaverse_root = current_dir / "runs/cache" / "objaverse"
        if not objaverse_root.exists():
            # Fallback to absolute path
            objaverse_root = os.path.expanduser("~/.cache/v4-world/objaverse")

    objaverse_path = Path(objaverse_root)
    if not objaverse_path.exists():
        return []

    object_paths = []
    # Iterate through category folders
    for category_dir in objaverse_path.iterdir():
        if not category_dir.is_dir():
            continue

        # Iterate through object instances (e.g., apple_0, apple_1, ...)
        for obj_dir in category_dir.iterdir():
            if not obj_dir.is_dir():
                continue

            visual_file = obj_dir / "visual" / "model_normalized_0.obj"
            collision_dir = obj_dir / "collision"

            if visual_file.exists() and collision_dir.exists():
                # Get all collision files
                collision_files = sorted(
                    collision_dir.glob("model_normalized_collision_*.obj")
                )
                if len(collision_files) > 0:
                    object_paths.append(
                        {
                            "category": category_dir.name,
                            "instance": obj_dir.name,
                            "visual": str(visual_file),
                            "collision_files": [str(f) for f in collision_files],
                            "base_dir": str(obj_dir),
                        }
                    )

    return object_paths


def load_objaverse_object(env, obj_path_info, scale=None):
    """Load an objaverse object into the scene.

    Args:
        env: The environment (to get objaverse_scale parameter)
        obj_path_info: Dictionary with object path information
        scale: Optional scale override. If None, uses env.objaverse_scale
    """
    if scale is None:
        scale = getattr(env, "objaverse_scale", 0.1)  # Default to 0.1 if not set

    builder = env.scene.create_actor_builder()

    # Add visual mesh
    visual_file = obj_path_info["visual"]
    if not os.path.exists(visual_file):
        raise FileNotFoundError(f"Visual file not found: {visual_file}")

    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    # Add collision meshes
    collision_files = obj_path_info["collision_files"]
    if len(collision_files) > 0:
        # Try to use multiple convex collisions from the collision directory
        # We'll use the first collision file - if it's a single mesh, we can use non-convex
        # For better performance with multiple collision files, we could combine them
        # For now, use non-convex collision from visual mesh (simpler and works for all cases)
        try:
            # Try using the visual mesh for non-convex collision (works but slower)
            builder.add_nonconvex_collision_from_file(
                filename=visual_file, scale=[scale] * 3
            )
        except Exception:
            # Fallback: try using first collision file
            try:
                builder.add_nonconvex_collision_from_file(
                    filename=collision_files[0], scale=[scale] * 3
                )
            except Exception as e:
                print(
                    f"Warning: Failed to add collision for {obj_path_info.get('instance', 'unknown')}: {e}"
                )
                # Continue without collision (visual only)

    return builder


def find_tables_and_surfaces(env):
    """Find table-like surfaces in the scene where we can place objects."""
    tables = []

    # Search through all actors in the scene
    for actor in env.scene.actors.values():
        if actor is None:
            continue

        name_lower = actor.name.lower() if hasattr(actor, "name") else ""

        # Check if it's a table or surface
        if "table" in name_lower or "counter" in name_lower or "surface" in name_lower:
            try:
                pose = actor.pose
                if hasattr(pose, "p"):
                    pos = to_numpy(pose.p)
                    if isinstance(pos, np.ndarray) and pos.ndim > 1:
                        pos = pos[0]  # Take first element for batched
                    pos = np.array(pos).flatten()[:3]

                    # Get table bounds using AABB if possible
                    table_bounds = None
                    table_surface_height = pos[2]

                    try:
                        from mani_skill.utils.geometry.geometry import (
                            get_axis_aligned_bbox_for_actor,
                        )

                        # Get the underlying sapien entity
                        sapien_entity = (
                            actor._objs[0]
                            if hasattr(actor, "_objs") and len(actor._objs) > 0
                            else None
                        )
                        if sapien_entity is not None:
                            mins, maxs = get_axis_aligned_bbox_for_actor(sapien_entity)
                            if (
                                mins is not None
                                and maxs is not None
                                and not np.any(np.isinf(mins))
                                and not np.any(np.isinf(maxs))
                            ):
                                # Table surface is at the top of the bounding box
                                table_surface_height = maxs[2]
                                # Get table surface bounds (x, y bounds at the top)
                                table_bounds = {
                                    "x_min": mins[0],
                                    "x_max": maxs[0],
                                    "y_min": mins[1],
                                    "y_max": maxs[1],
                                    "z": table_surface_height,
                                }
                    except Exception as e:
                        # Fallback: use pose position and estimate bounds
                        # Assume a reasonable table size if we can't get bounds
                        table_bounds = {
                            "x_min": pos[0] - 0.5,
                            "x_max": pos[0] + 0.5,
                            "y_min": pos[1] - 0.5,
                            "y_max": pos[1] + 0.5,
                            "z": table_surface_height,
                        }

                    # Only add if it's at a reasonable height (0 to 1.5m)
                    if 0 <= table_surface_height <= 1.5:
                        tables.append(
                            {
                                "actor": actor,
                                "position": pos,
                                "height": table_surface_height,
                                "bounds": table_bounds,
                                "name": name_lower,
                            }
                        )
            except Exception as e:
                continue

    return tables


def get_robot_position(env: BaseEnv):
    """Get the robot's base position."""
    try:
        if hasattr(env, "agent") and hasattr(env.agent, "robot"):
            if isinstance(env.agent, MultiAgent):
                robot = env.agent.agents[0].robot
            else:
                robot = env.agent.robot
            if hasattr(robot, "links") and len(robot.links) > 0:
                base_link = robot.links[0]
                pose = base_link.pose
                if hasattr(pose, "p"):
                    pos = to_numpy(pose.p)
                    if isinstance(pos, np.ndarray) and pos.ndim > 1:
                        pos = pos[0]
                    pos = np.array(pos).flatten()[:3]
                    return pos
    except Exception as e:
        pass

    # Fallback: return origin
    return np.array([0.0, 0.0, 0.0])


def get_robot_gripper_position(env):
    """Get the robot's gripper/end-effector position (tool center point)."""
    try:
        if hasattr(env, "agent"):
            agent = env.agent
            # Try to get TCP (Tool Center Point) position - this is where the gripper operates
            if hasattr(agent, "tcp_pose"):
                tcp_pose = agent.tcp_pose
                if hasattr(tcp_pose, "p"):
                    pos = to_numpy(tcp_pose.p)
                    if isinstance(pos, np.ndarray) and pos.ndim > 1:
                        pos = pos[0]
                    pos = np.array(pos).flatten()[:3]
                    return pos
            elif hasattr(agent, "tcp_pos"):
                tcp_pos = agent.tcp_pos
                pos = to_numpy(tcp_pos)
                if isinstance(pos, np.ndarray) and pos.ndim > 1:
                    pos = pos[0]
                pos = np.array(pos).flatten()[:3]
                return pos
            # Fallback: try to find end-effector link by name
            elif hasattr(agent, "robot") and hasattr(agent.robot, "links_map"):
                ee_link_names = [
                    "ee_link",
                    "end_effector_link",
                    "gripper_link",
                    "tcp",
                    "hand",
                    "panda_hand",
                ]
                for ee_name in ee_link_names:
                    if ee_name in agent.robot.links_map:
                        ee_link = agent.robot.links_map[ee_name]
                        pose = ee_link.pose
                        if hasattr(pose, "p"):
                            pos = to_numpy(pose.p)
                            if isinstance(pos, np.ndarray) and pos.ndim > 1:
                                pos = pos[0]
                            pos = np.array(pos).flatten()[:3]
                            return pos
    except Exception as e:
        pass

    # Fallback: use robot base position
    return get_robot_position(env)


def spawn_distractors(env, offset=(0.0, 0.0)):
    """Spawns dynamic objects in workspace using objaverse objects."""
    # Get available objaverse objects
    objaverse_objects = get_objaverse_object_paths()
    assert len(objaverse_objects) > 0, "No objaverse objects found"

    # Get robot base position for reference (more stable than gripper position)
    robot_base_pos = get_robot_position(env)
    robot_height = robot_base_pos[2] if len(robot_base_pos) >= 3 else 0.0

    # Find tables/surfaces
    tables = find_tables_and_surfaces(env)

    # Get objaverse scale parameter
    objaverse_scale = getattr(env, "objaverse_scale", 0.1)

    # Randomly sample objaverse objects
    selected_objects = random.sample(
        objaverse_objects, min(env.num_distractors, len(objaverse_objects))
    )

    for i, obj_info in enumerate(selected_objects):
        try:
            # Use the objaverse_scale parameter (objects will be much smaller)
            # Add small random variation (±20%) for variety
            scale_variation = random.uniform(0.8, 1.2)
            scale = objaverse_scale * scale_variation
            builder = load_objaverse_object(env, obj_info, scale=scale)

            # Determine placement position
            # Priority: Place on tables when available (robots operate on tables)
            if (
                len(tables) > 0 and random.random() > 0.2
            ):  # 80% chance to place on table
                # Place directly on table surface (not high above)
                table = random.choice(tables)
                table_height = table["height"]
                table_bounds = table.get("bounds")

                if table_bounds is not None:
                    # Place within table bounds with some margin from edges
                    margin = 0.1  # 10cm margin from table edges
                    x_min = table_bounds["x_min"] + margin
                    x_max = table_bounds["x_max"] - margin
                    y_min = table_bounds["y_min"] + margin
                    y_max = table_bounds["y_max"] - margin

                    # Ensure valid bounds
                    if x_max > x_min and y_max > y_min:
                        # Random placement on table (uniform distribution)
                        x = random.uniform(x_min, x_max)
                        y = random.uniform(y_min, y_max)

                        # Place directly on table surface with small offset to avoid penetration
                        z = (
                            table_height + 0.01 + scale * 0.05
                        )  # Small offset based on object scale
                        pos = np.array([x + offset[0], y + offset[1], z])
                    else:
                        # Fallback: use table position with random offset
                        offset_xy = (
                            np.random.rand(2) - 0.5
                        ) * 0.3  # Random offset within 0.3m
                        pos = np.array(
                            [
                                table["position"][0] + offset_xy[0] + offset[0],
                                table["position"][1] + offset_xy[1] + offset[1],
                                table_height + 0.01 + scale * 0.05,
                            ]
                        )
                else:
                    # No bounds available: use table position with random offset
                    offset_xy = (
                        np.random.rand(2) - 0.5
                    ) * 0.3  # Random offset within 0.3m
                    pos = np.array(
                        [
                            table["position"][0] + offset_xy[0] + offset[0],
                            table["position"][1] + offset_xy[1] + offset[1],
                            table_height + 0.01 + scale * 0.05,  # Directly on table
                        ]
                    )
            else:
                # No tables or 20% chance: Place near robot base but HIGH above it so it falls
                # This ensures objects don't penetrate the robot initially
                min_height_above_base = 0.5  # Minimum 0.5m above robot base
                spawn_height = (
                    robot_height + min_height_above_base + random.uniform(0.1, 0.3)
                )
                # Place objects in a radius around the robot base (within working range)
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0.2, 0.6)  # 20-60cm from robot base
                offset_xy = np.array([np.cos(angle) * radius, np.sin(angle) * radius])
                pos = np.array(
                    [
                        robot_base_pos[0] + offset_xy[0] + offset[0],
                        robot_base_pos[1] + offset_xy[1] + offset[1],
                        spawn_height,  # High above robot base
                    ]
                )

            # Random orientation
            quat = sapien.Pose().q  # Default quaternion, or randomize if needed

            builder.set_initial_pose(sapien.Pose(p=pos, q=quat))
            actor = builder.build(name=f"distractor_obj_{i}")

        except Exception as e:
            print(
                f"Warning: Failed to load objaverse object {obj_info.get('instance', 'unknown')}: {e}"
            )
            continue
