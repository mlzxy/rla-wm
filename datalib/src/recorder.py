import os
import numpy as np
from typing import Dict, Any
from rich import print
try:
    from datalib.dataset import ManiSkillTrajectoryDataset, TrajectoryData, RobotInfo
except ImportError:
    print("[yellow]Warning: datalib not found. TeleopRecorder will not be available.[/yellow]")

class TeleopRecorder:
    def __init__(self, output_dir: str, robot_uid: str):
        self.output_dir = output_dir
        self.dataset = ManiSkillTrajectoryDataset(output_dir)
        self.robot_uid = robot_uid
        self.current_episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "images": {} # {cam_name: [frames]}
        }
        self.episode_count = 0
        self._load_metadata()

    def _load_metadata(self):
        # Scan output_dir to get current episode count
        trajs = self.dataset.list_trajectories()
        if trajs:
            self.episode_count = max([int(t) for t in trajs]) + 1
        else:
            self.episode_count = 0

    def record_step(self, obs, action, reward, done, info, images: Dict[str, np.ndarray]):
        self.current_episode_data["observations"].append(obs)
        self.current_episode_data["actions"].append(action)
        self.current_episode_data["rewards"].append(reward)
        self.current_episode_data["dones"].append(done)
        self.current_episode_data["infos"].append(info)
        
        for cam_name, img in images.items():
            if cam_name not in self.current_episode_data["images"]:
                self.current_episode_data["images"][cam_name] = []
            self.current_episode_data["images"][cam_name].append(img)

    def save_episode(self, success: bool):
        if not self.current_episode_data["actions"]:
            return
            
        traj_id = f"{self.episode_count:06d}"
        
        # Prepare TrajectoryData
        video_streams = {}
        for cam_name, frames in self.current_episode_data["images"].items():
            video_streams[cam_name] = np.array(frames)
            
        metadata = {
            "actions": np.array(self.current_episode_data["actions"]),
            "rewards": np.array(self.current_episode_data["rewards"]),
            "success": success
        }
        
        # Add observations if they are state-based
        # If they are images, they are already in video_streams
        # For ManiSkill, we often store qpos/qvel in metadata
        obs_list = self.current_episode_data["observations"]
        if isinstance(obs_list[0], dict):
            for key in obs_list[0].keys():
                if key != "sensor_data":
                    metadata[key] = np.array([o[key] for o in obs_list])
        
        traj_data = TrajectoryData(
            success=success,
            video_streams=video_streams,
            metadata=metadata
        )
        
        self.dataset.write_trajectory(traj_id, traj_data)
        print(f"Episode {traj_id} saved. Success: {success}")
        
        self.episode_count += 1
        self.reset()

    def reset(self):
        self.current_episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "images": {}
        }
