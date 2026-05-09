import torch
import numpy as np
import sapien
from typing import List, Optional, Union
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.utils.randomization.samplers import UniformPlacementSampler
from datalib.src.unified_workspace import UnifiedWorkspaceEnv

@register_env("PlayEnv-v0", max_episode_steps=200)
class PlayEnv(UnifiedWorkspaceEnv):
    """
    Infinite Playroom Environment for Autonomous Interaction Discovery.
    Automatically spawns random objects and provides utilities for interaction discovery.
    """
    
    def __init__(self, 
                 *args, 
                 num_play_objects: int = 5,
                 play_object_types: List[str] = ["cube", "objaverse"],
                 workspace_radius: float = 0.4,
                 workspace_offset: List[float] = [0.0, 0.0],
                 workspace_scale: List[float] = [0.4, 1.0],
                 **kwargs):
        self.num_play_objects = num_play_objects
        self.play_object_types = play_object_types
        self.workspace_radius = workspace_radius
        self.workspace_offset = workspace_offset
        self.workspace_scale = workspace_scale
        self.play_objects = []
        self._play_actor_metadata = {}
        
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        # Objects will be built/reset in _initialize_episode or during initialization
        # depending on standard ManiSkill patterns. 
        # For AID, we want objects to be persistent or refreshed.
        self._play_actors = []

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.table_scene.initialize(env_idx)
        
        # Build objects if they don't exist
        # Note: ManiSkill best practice is to build once and re-pose.
        if len(self._play_actors) == 0:
            self._setup_play_objects()
            
        # Randomize object poses
        self._randomize_play_objects(env_idx)

    def _get_obj_extents(self, path):
        """Quickly parse OBJ file to get bounding box extents."""
        try:
            pts = []
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        pts.append([float(x) for x in line.split()[1:4]])
            if not pts:
                return np.array([0.1, 0.1, 0.1])
            pts = np.array(pts)
            return np.max(pts, axis=0) - np.min(pts, axis=0)
        except Exception:
            return np.array([0.1, 0.1, 0.1])

    def _setup_play_objects(self):
        """Build the pool of objects used for play."""
        for i in range(self.num_play_objects):
            obj_type = np.random.choice(self.play_object_types)
            name = f"play_obj_{i}"
            
            if obj_type == "cube":
                # Absolute size: 3cm to 10cm half-size -> 6cm to 20cm full size? 
                # Let's target 4cm to 12cm full size (0.02 to 0.06 half-size)
                half_size = np.random.uniform(0.02, 0.06) 
                color = np.random.random(3).tolist() + [1.0]
                actor = actors.build_cube(self.scene, half_size=half_size, color=color, name=name)
            elif obj_type == "objaverse":
                # Leverages UnifiedWorkspaceEnv's objaverse loader
                obj_path_info = np.random.choice(self._get_objaverse_object_paths())
                
                # Target max dimension: 5cm to 15cm
                target_max_dim = np.random.uniform(0.05, 0.15)
                
                # Get intrinsic extents by parsing the visual OBJ file
                extents = self._get_obj_extents(obj_path_info["visual"])
                intrinsic_max_dim = np.max(extents)
                scale = target_max_dim / intrinsic_max_dim if intrinsic_max_dim > 0 else 0.1
                
                # Build final actor with correct scale
                actor = self._build_objaverse_object(obj_path_info, scale, name)
            else:
                # Default to cube
                actor = actors.build_cube(self.scene, half_size=0.04, color=[0, 0, 1, 1], name=name)
            
            self._play_actors.append(actor)
            
            # Extract geometry metadata from mesh
            # Use local frame (to_world_frame=False) to get the intrinsic size
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is not None:
                bounds = mesh.bounds # (2, 3) min/max
                extents = mesh.extents # (3,) width, depth, height
                # Radius for XY collision: use a slightly buffered maximum half-extent
                radius = np.linalg.norm(extents[:2]) / 2.0 + 0.01
                # Z-offset to sit on table: the center is at 0, so if min_z is -0.05, 
                # we need to lift it by +0.05 to sit on Z=0.
                z_offset = -bounds[0, 2]
            else:
                # Fallback for actors without collision meshes (should be rare)
                radius = 0.05
                z_offset = 0.05
            
            self._play_actor_metadata[actor.name] = {
                "radius": radius,
                "z_offset": z_offset
            }

    def _randomize_play_objects(self, env_idx: torch.Tensor):
        """Spawn objects randomly within the reachable workspace without collisions."""
        b = len(env_idx)
        
        # Define safe bounds for placement based on offset and scale
        offset = np.array(self.workspace_offset)
        scale = np.array(self.workspace_scale)
        low = offset - scale / 2.0
        high = offset + scale / 2.0
        
        placement_bounds = (low.tolist(), high.tolist())
        sampler = UniformPlacementSampler(
            bounds=placement_bounds,
            batch_size=b,
            device=self.device
        )
        
        for actor in self._play_actors:
            meta = self._play_actor_metadata[actor.name]
            
            # Sample collision-free XY positions
            # We sample for the entire batch at once
            xy = sampler.sample(radius=meta["radius"], max_trials=100)
            
            # Higher-fidelity Z placement
            z = torch.ones(b, device=self.device) * meta["z_offset"]
            
            # Random rotation around Z-axis
            qs = torch.zeros((b, 4), device=self.device)
            # Standard ManiSkill randomization utility for quats
            from mani_skill.envs.utils import randomization
            qs = randomization.random_quaternions(
                b, device=self.device, lock_x=True, lock_y=True, lock_z=False
            )
            
            actor.set_pose(Pose.create_from_pq(p=torch.stack([xy[:, 0], xy[:, 1], z], dim=-1), q=qs))

    @property
    def interactive_objects(self):
        """Returns all actors currently designated as play objects."""
        return self._play_actors

    def is_on_table(self, actor, buffer=0.05):
        """Checks if an actor is within the table bounds."""
        pos = actor.pose.p
        # Table is at Z=0. Standard TableSceneBuilder is ~[0.6, 0.6] half size center at [0,0]?
        # Actual bounds depend on TableSceneBuilder config. 
        # For now, use relative distance to center or workspace radius.
        dist = torch.linalg.norm(pos[..., :2], axis=-1)
        return dist < (self.workspace_radius + buffer)

if __name__ == "__main__":
    import gymnasium as gym
    import mani_skill.envs
    
    # Register the environment if not already registered (though @register_env does it)
    # We can use the registered name directly
    env = gym.make(
        "PlayEnv-v0",
        num_play_objects=8,
        render_mode="human",
        robot_uids="panda",
        control_mode="pd_ee_delta_pose",
        workspace_offset=[0.0, 0.0], # Moved slightly closer to robot (default is 0.3)
        workspace_scale=[0.4, 1.0]    # Adjusted scale
    )
    
    print("Environment initialized. Starting visualization loop...")
    print("Close the window to exit.")
    
    obs, _ = env.reset()
    try:
        while True:
            # Sample random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                obs, _ = env.reset()
    except Exception as e:
        print(e)
