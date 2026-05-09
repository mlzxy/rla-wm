import numpy as np
import sapien
import torch
from mani_skill.sensors.camera import Camera
from mani_skill.utils.common import to_numpy
from mani_skill.utils.sapien_utils import look_at
from typing import List, Literal


class CinematicCameraSystem:
    def __init__(
        self, env_scene, robot_actor: sapien.Entity, camera_configs: List[dict]
    ):
        """
        Manage multiple cinematic cameras for data collection.

        Args:
            env_scene: The ManiSkill/Sapien scene.
            robot_actor: The target actor (robot base or EE) to look at.
            camera_configs: List of dicts defining config for each cam.
                            e.g. {'name': 'cam_0', 'mode': 'anchor', 'dist': 1.5, ...}
        """
        self.scene = env_scene
        self.robot = robot_actor
        self.cameras: List[Camera] = []
        self.configs = camera_configs

        # Initialize cameras
        # Note: In ManiSkill, sensors are usually created during setup.
        # If created dynamically here, ensure they are registered if needed.
        # Ideally, these are passed in from the EnrichedEnv.
        pass

    def update_pose(
        self,
        camera: Camera,
        mode: str,
        step_idx: int,
        total_steps: int,
        radius=1.0,
        height=0.5,
        **kwargs,
    ):
        """
        Update a single camera's pose based on its mode and time.
        Additional kwargs can include: angle_offset, omega, phase, vertical_speed, etc.
        """
        # Convert pose.p to numpy array (handles both torch tensors and numpy arrays)
        # For batched environments, take the first element (env_idx=0)
        target_pos_raw = self.robot.pose.p

        # Handle batched case: if shape is [batch_size, 3], take first element
        if torch.is_tensor(target_pos_raw):
            if target_pos_raw.ndim > 1:
                target_pos_raw = target_pos_raw[0]
        elif isinstance(target_pos_raw, np.ndarray):
            if target_pos_raw.ndim > 1:
                target_pos_raw = target_pos_raw[0]

        # Convert to numpy and ensure it's a flat 3-element array
        target_pos = to_numpy(target_pos_raw)
        # Ensure it's a 1D array of exactly 3 elements
        target_pos = np.asarray(target_pos, dtype=np.float64).flatten()
        if target_pos.size != 3:
            # Take first 3 elements or pad if needed
            if target_pos.size > 3:
                target_pos = target_pos[:3]
            else:
                target_pos = np.pad(
                    target_pos, (0, 3 - target_pos.size), mode="constant"
                )

        # Final check: ensure shape is (3,) and all elements are scalars
        target_pos = np.array(
            [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
            dtype=np.float64,
        )

        t = float(step_idx) / max(1, float(total_steps))

        # Extract optional parameters from kwargs
        angle_offset = kwargs.get("angle_offset", 0.0)
        omega = kwargs.get("omega", 2 * np.pi)
        phase = kwargs.get("phase", 0.0)
        vertical_speed = kwargs.get("vertical_speed", 0.5)
        orbit_direction = kwargs.get("orbit_direction", 1.0)  # 1.0 or -1.0

        if mode == "anchor":
            # Small jitter around a fixed point relative to robot
            angle = np.pi / 4 + angle_offset  # Use provided angle offset
            offset = np.array(
                [np.cos(angle) * radius, np.sin(angle) * radius, height],
                dtype=np.float64,
            )

            # Add very slow breathing motion
            noise = float(np.sin(t * np.pi * 2) * 0.05)
            eye_pos = (
                target_pos + offset + np.array([0.0, 0.0, noise], dtype=np.float64)
            )

        elif mode == "spiral":
            # Spiraling up and around with customizable parameters
            theta = omega * t * orbit_direction + phase
            z = float(height + (t * vertical_speed))
            x = float(radius * np.cos(theta))
            y = float(radius * np.sin(theta))
            eye_pos = target_pos + np.array([x, y, z], dtype=np.float64)

        elif mode == "orbit":
            # Circular orbit at fixed height
            theta = omega * t * orbit_direction + phase
            x = float(radius * np.cos(theta))
            y = float(radius * np.sin(theta))
            eye_pos = target_pos + np.array([x, y, height], dtype=np.float64)

        elif mode == "figure8":
            # Figure-8 pattern
            theta = omega * t * orbit_direction + phase
            x = float(radius * np.sin(theta))
            y = float(radius * np.sin(theta) * np.cos(theta))
            z = float(height + np.sin(t * np.pi * 2) * 0.2)
            eye_pos = target_pos + np.array([x, y, z], dtype=np.float64)

        elif mode == "pendulum":
            # Pendulum-like motion
            theta = np.sin(omega * t + phase) * (np.pi / 3)  # Swing ±60 degrees
            x = float(radius * np.cos(theta))
            y = float(radius * np.sin(theta))
            z = float(height + np.sin(t * np.pi * 2) * 0.1)
            eye_pos = target_pos + np.array([x, y, z], dtype=np.float64)

        elif mode == "fixed":
            # Fixed position relative to robot/target
            angle = angle_offset
            x = float(radius * np.cos(angle))
            y = float(radius * np.sin(angle))
            # Fixed height
            z = float(height)
            eye_pos = target_pos + np.array([x, y, z], dtype=np.float64)

        else:
            # Static fallback
            return

        # Calculate LookAt Pose using ManiSkill's utility function
        # Convert numpy arrays to lists/tensors for look_at function
        eye_tensor = torch.tensor(eye_pos, dtype=torch.float32)
        target_tensor = torch.tensor(target_pos, dtype=torch.float32)
        up_tensor = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        # Use ManiSkill's look_at utility which returns a Pose object
        pose = look_at(eye_tensor, target_tensor, up_tensor)

        # Convert to sapien.Pose (since RenderCamera.set_local_pose expects sapien.Pose)
        # The look_at function returns a ManiSkill Pose, so we need to convert it
        sapien_pose = pose.sp  # Get sapien.Pose from ManiSkill Pose

        # Update SAPIEN camera pose
        # Note: ManiSkill cameras might be attached to mounts.
        # For cinematic cams, we assume they are 'floating' (mount=None).
        # Camera object has a .camera attribute which is a RenderCamera
        camera.camera.set_local_pose(sapien_pose)

    def step(self, step_idx: int, total_steps: int = 200):
        """Called every env step to update all camera poses."""
        # This assumes self.cameras is populated by the Env
        pass
