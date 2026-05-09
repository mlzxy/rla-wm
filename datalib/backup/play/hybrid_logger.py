import os
import json
import zarr
import torch
import numpy as np
from typing import Dict, Any, List

class HybridLogger:
    """
    Records interaction data in two formats:
    1. Action Trace (JSON): seed, metadata, and the sequence of actions for replay.
    2. State Cache (Zarr): 10Hz snapshot of (pos, rot, vel) for fast training.
    """
    
    def __init__(self, log_dir: str, num_envs: int):
        self.log_dir = log_dir
        self.num_envs = num_envs
        os.makedirs(log_dir, exist_ok=True)
        
        # Action Trace Storage
        self.action_history = []
        self.metadata = {}
        
        # State Cache Storage (Zarr)
        # We'll initialize the Zarr group on the first 'record_state' call 
        # once we know the dimensions of the objects.
        self.zarr_root = None
        self.state_step = 0
        
    def log_metadata(self, metadata: Dict[str, Any]):
        """Logs initial environment setup (seed, object IDs, initial poses)."""
        self.metadata.update(metadata)
        with open(os.path.join(self.log_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

    def log_action(self, action: torch.Tensor):
        """Records an action step for the trace."""
        # Convert torch to list for JSON serialization
        if torch.is_tensor(action):
            action_np = action.cpu().numpy().tolist()
        else:
            action_np = action.tolist()
        self.action_history.append(action_np)

    def record_state(self, env: Any):
        """
        Snapshots object and robot states at high frequency (called usually at 10Hz).
        Stored in Zarr for fast sequential access during model training.
        """
        if self.zarr_root is None:
            self._init_zarr(env)
            
        # Extract states
        # Format for each object: [pos(3), rot(4), vel(3), ang_vel(3)] = 13 floats
        state_data = []
        
        # Robot segments
        qpos = env.agent.robot.get_qpos().cpu().numpy()
        qvel = env.agent.robot.get_qvel().cpu().numpy()
        
        # Actors (Play Objects)
        for actor in env.interactive_objects:
            p = actor.pose.p.cpu().numpy()
            q = actor.pose.q.cpu().numpy()
            v = actor.linear_velocity.cpu().numpy()
            av = actor.angular_velocity.cpu().numpy()
            state_data.append(np.concatenate([p, q, v, av], axis=-1))
            
        # Append to Zarr (Zarr arrays are chunked and can be appended to)
        # Note: In a production environment, we might use a larger buffer before writing to disk.
        self.zarr_root["robot_qpos"][self.state_step] = qpos
        self.zarr_root["robot_qvel"][self.state_step] = qvel
        self.zarr_root["object_states"][self.state_step] = np.stack(state_data, axis=1)
        
        self.state_step += 1

    def _init_zarr(self, env: Any):
        """Initialize Zarr arrays based on environment dimensions."""
        store = zarr.DirectoryStore(os.path.join(self.log_dir, "state_cache.zarr"))
        self.zarr_root = zarr.group(store=store, overwrite=True)
        
        # Estimate max steps (e.g., episode_len * frequency_ratio)
        max_steps = 1000 # Buffer size
        
        robot_q_dim = env.agent.robot.get_qpos().shape[-1]
        num_objs = len(env.interactive_objects)
        
        self.zarr_root.create_dataset("robot_qpos", shape=(max_steps, self.num_envs, robot_q_dim), chunks=(100, 1, robot_q_dim), dtype='f4')
        self.zarr_root.create_dataset("robot_qvel", shape=(max_steps, self.num_envs, robot_q_dim), chunks=(100, 1, robot_q_dim), dtype='f4')
        self.zarr_root.create_dataset("object_states", shape=(max_steps, self.num_envs, num_objs, 13), chunks=(100, 1, num_objs, 13), dtype='f4')

    def save(self):
        """Finalize and save the action trace."""
        with open(os.path.join(self.log_dir, "action_trace.json"), "w") as f:
            json.dump(self.action_history, f)
        print(f"Log saved to {self.log_dir}")
