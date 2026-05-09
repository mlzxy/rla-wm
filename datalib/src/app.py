import os
from contextlib import nullcontext

# Ensure hidapi is found on macOS
os.environ["DYLD_LIBRARY_PATH"] = (
    "/opt/homebrew/Cellar/hidapi/0.15.0/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
)
import cv2
import numpy as np
import gymnasium as gym
import torch
import argparse
import json
import time
from typing import Tuple, Dict, Optional, List
from scipy.spatial.transform import Rotation

import pyspacemouse
from pyspacemouse.callbacks import ButtonCallback
from pyspacemouse.types import DeviceInfo, AxisSpec
from .recorder import TeleopRecorder
from .tasks import *  # Registration
from . import robots
from .dashboard import TeleopDashboard
from mani_skill.sensors.camera import Camera
from mani_skill.utils import sapien_utils
import sapien

# Baseline scaling for all robots
BASE_SCALE_XYZ = 0.2
BASE_SCALE_ROT = 0.2


def get_custom_device_spec(
    base_spec: DeviceInfo, axis_map: List[int], axis_signs: List[float]
) -> DeviceInfo:
    """Create a custom DeviceInfo with permuted and inverted axes for pyspacemouse."""
    new_mappings = {}
    axes = ["x", "y", "z"]
    for i, axis in enumerate(axes):
        # mapped_idx is which physical axis (0=x, 1=y, 2=z) maps to this software 'axis'
        mapped_idx = axis_map[i]
        mapped_axis = axes[mapped_idx]
        spec = base_spec.mappings[mapped_axis]
        new_mappings[axis] = AxisSpec(
            channel=spec.channel,
            byte1=spec.byte1,
            byte2=spec.byte2,
            scale=int(spec.scale * axis_signs[i]),
        )

    # Keep rotation axes as they are (usually mapped separately or fixed)
    for rot in ["roll", "pitch", "yaw"]:
        new_mappings[rot] = base_spec.mappings[rot]

    return DeviceInfo(
        name=f"{base_spec.name} (Custom Mapping)",
        vendor_id=base_spec.vendor_id,
        product_id=base_spec.product_id,
        led_id=base_spec.led_id,
        axis_scale=base_spec.axis_scale,
        mappings=new_mappings,
        button_specs=base_spec.button_specs,
        button_names=base_spec.button_names,
    )


class TeleopManager:
    """Manages teleoperation state and handles callbacks."""

    def __init__(self, is_saving_enabled: bool, recorder: Optional[TeleopRecorder]):
        self.gripper_action = -1.0  # Start Open
        self.is_saving_enabled = is_saving_enabled
        self.recorder = recorder
        self.running = True
        self.reward = 0.0
        self.success = False
        self.rotation_lock = False
        self.z_lock = False
        self.waiting_for_reset = False
        self.scale_xyz = 1.0  # Default relative scale
        self.scale_rot = 1.0
        self.waiting_for_skip_confirmation = False
        self.gym_episode_id = -1  # Will be incremented to 0 on first reset

    def toggle_gripper(self, state, buttons, pressed_buttons):
        self.gripper_action *= -1.0
        print(f"Gripper: {'Closed' if self.gripper_action > 0 else 'Open'}")

    def toggle_recording(self, state, buttons, pressed_buttons):
        if self.is_saving_enabled:
            # Toggling OFF: Save the current episode
            if self.recorder:
                self.recorder.save_episode(success=self.success)
            self.is_saving_enabled = False
            print("Data Saving: Disabled. Episode saved.")
        else:
            # Toggling ON: Start a new episode
            self.is_saving_enabled = True
            if self.recorder:
                self.recorder.reset()
            print("Data Saving: Enabled. New episode started.")


def compute_robot_action(state, config):
    """Convert SpaceMouse state to robot action vector using global scales."""
    if state is None:
        return np.zeros(3), np.zeros(3)

    # pyspacemouse already applies axis_map and axis_signs via Custom DeviceInfo
    mapped_pos = (
        np.array([state.x, state.y, state.z]) * config["scale_xyz"] * BASE_SCALE_XYZ
    )

    # Rotation mapping: pyspacemouse attributes are .pitch, .roll, .yaw
    delta_euler = (
        np.array([state.roll, state.pitch, state.yaw * 2])
        * config["scale_rot"]
        * BASE_SCALE_ROT
    )

    return mapped_pos, delta_euler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PushT-v2",
        choices=[
            "PushT-v2",
            "PokeCube-v2",
            "PullCube-v2",
            "StackCube-v1",
            "RollBall-v1",
            "PegInsertionSide-v1",
        ],
    )
    parser.add_argument(
        "-r",
        "--robot-uid",
        type=str,
        default="panda_closed",
        choices=[
            "panda",
            "xarm6_robotiq",
            "ur10e_stick",
            "panda_closed",
            "xarm6_robotiq_closed",
        ],
    )
    parser.add_argument("-o", "--output-dir", type=str, default="data/teleop")
    parser.add_argument("--scale-xyz", type=float, default=0.3)
    parser.add_argument("--scale-rot", type=float, default=1.0)
    parser.add_argument("--num-distractors", type=int, default=0)
    parser.add_argument(
        "--save", action="store_true", help="Enable saving episodes by default"
    )
    parser.add_argument(
        "--axis-map",
        type=int,
        nargs=3,
        default=[1, 0, 2],
        help="Axis permutation (e.g. 0 1 2)",
    )
    parser.add_argument(
        "--axis-signs",
        type=float,
        nargs=3,
        default=[-1.0, 1.0, 1.0],
        help="Axis signs (e.g. 1 -1 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Environment
    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uid,
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        obs_mode="state+rgb+depth+segmentation",
        max_episode_steps=1e24,
        num_distractors=args.num_distractors,
    )

    # 2. Discover SpaceMouse and Setup Custom Specification
    print("Discovering SpaceMouse...")
    connected = pyspacemouse.get_connected_devices()
    assert len(connected) > 0
    device_name = connected[0]
    print(f"Found: {device_name}")
    base_spec = pyspacemouse.get_device_specs()[device_name]
    custom_spec = get_custom_device_spec(base_spec, args.axis_map, args.axis_signs)

    # 3. Setup Teleop Manager and Callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    recorder = TeleopRecorder(args.output_dir, args.robot_uid) if args.save else None
    manager = TeleopManager(args.save, recorder)

    btn_callbacks = [
        ButtonCallback(0, manager.toggle_gripper),  # Left Click
        ButtonCallback(1, manager.toggle_recording),  # Right Click
    ]

    config = {"scale_xyz": args.scale_xyz, "scale_rot": args.scale_rot}
    manager.scale_xyz = args.scale_xyz
    manager.scale_rot = args.scale_rot

    print(f"Teleop System Ready: Task={args.env_id}, Robot={args.robot_uid}")
    print(
        "Controls: [SpaceMouse] 6DoF movement, [Left Click] Toggle Gripper, [Right Click] Toggle Saving"
    )

    obs, info = env.reset()

    # 4. Setup Visual Enhancements (Dashboard & Camera)
    viewer = env.unwrapped.viewer
    dashboard = None
    if viewer:
        # Set default camera view facing the robot
        # Robot is at [-0.615, 0, 0], Workspace at [0, 0, 0]
        eye = [0.8, 0.0, 0.5]
        target = [-0.1, 0.0, 0.1]
        ms_pose = sapien_utils.look_at(eye, target)
        viewer.set_camera_pose(
            sapien.Pose(ms_pose.p[0].cpu().numpy(), ms_pose.q[0].cpu().numpy())
        )

        # Register Dashboard Plugin
        dashboard = TeleopDashboard(env, manager)
        viewer.plugins.append(dashboard)
        viewer.init_plugins([dashboard])

    # Use pyspacemouse context manager if device is available
    mouse_context = pyspacemouse.open(
        button_callbacks=btn_callbacks, device_spec=custom_spec
    )

    try:
        with mouse_context as dev:
            while manager.running:
                # A. Read Input
                delta_pos = np.zeros(3)
                delta_euler = np.zeros(3)

                if dev:
                    state = dev.read()
                    if state:
                        # Use dynamic scales from manager
                        cur_config = {
                            "scale_xyz": manager.scale_xyz,
                            "scale_rot": manager.scale_rot,
                        }
                        delta_pos, delta_euler = compute_robot_action(state, cur_config)

                        # Apply locks
                        if manager.z_lock:
                            delta_pos[2] = 0.0
                        if manager.rotation_lock:
                            delta_euler = np.zeros(3)

                # B. Handle Skip Confirmation
                if manager.waiting_for_skip_confirmation:
                    # Inhibit robot movement during confirmation
                    delta_pos = np.zeros(3)
                    delta_euler = np.zeros(3)

                    if viewer and viewer.window:
                        if viewer.window.key_press("y"):
                            if manager.recorder:
                                manager.recorder.save_episode(success=False)
                            obs, info = env.reset()
                            manager.gym_episode_id += 1
                            manager.success = False
                            manager.waiting_for_reset = False
                            manager.waiting_for_skip_confirmation = False
                            print("Episode saved and reset.")
                        elif viewer.window.key_press("n"):
                            if manager.recorder:
                                manager.recorder.reset()
                            obs, info = env.reset()
                            manager.gym_episode_id += 1
                            manager.success = False
                            manager.waiting_for_reset = False
                            manager.waiting_for_skip_confirmation = False
                            print("Episode discarded and reset.")
                        elif viewer.window.key_press("c"):
                            manager.waiting_for_skip_confirmation = False
                            print("Skip cancelled.")

                    # C. Handle Keyboard Inputs
                    if viewer.window.key_down("q"):
                        manager.running = False

                    # Lock Toggles
                    if viewer.window.key_press("1"):
                        manager.rotation_lock = not manager.rotation_lock
                        print(
                            f"Rotation Lock: {'Enabled' if manager.rotation_lock else 'Disabled'}"
                        )
                    if viewer.window.key_press("2"):
                        manager.z_lock = not manager.z_lock
                        print(
                            f"Z-Axis Lock: {'Enabled' if manager.z_lock else 'Disabled'}"
                        )

                    # Sensitivity Adjustments
                    if viewer.window.key_press("up"):
                        manager.scale_xyz = round(manager.scale_xyz + 0.01, 2)
                        print(f"XYZ Scale: {manager.scale_xyz}")
                    if viewer.window.key_press("down"):
                        manager.scale_xyz = max(
                            0.01, round(manager.scale_xyz - 0.01, 2)
                        )
                        print(f"XYZ Scale: {manager.scale_xyz}")
                    if viewer.window.key_press("right"):
                        manager.scale_rot = round(manager.scale_rot + 0.01, 2)
                        print(f"Rot Scale: {manager.scale_rot}")
                    if viewer.window.key_press("left"):
                        manager.scale_rot = max(
                            0.01, round(manager.scale_rot - 0.01, 2)
                        )
                        print(f"Rot Scale: {manager.scale_rot}")

                    # Skip Request
                    if (
                        viewer.window.key_press("r")
                        and not manager.waiting_for_skip_confirmation
                    ):
                        if (
                            manager.recorder
                            and manager.recorder.current_episode_data["actions"]
                        ):
                            manager.waiting_for_skip_confirmation = True
                        else:
                            obs, info = env.reset()
                            manager.gym_episode_id += 1
                            manager.success = False
                            manager.waiting_for_reset = False
                            print("Episode reset.")

                    # Reset Logic (only when waiting)
                    if manager.waiting_for_reset and viewer.window.key_press("space"):
                        obs, info = env.reset()
                        manager.gym_episode_id += 1
                        manager.waiting_for_reset = False
                        manager.success = False
                        print("Environment Reset. Resuming...")

                action_vec = np.concatenate(
                    [delta_pos, delta_euler, [manager.gripper_action]]
                )
                action_vec = action_vec[: env.action_space.shape[0]]

                action = (
                    torch.from_numpy(action_vec).float().to(env.device).unsqueeze(0)
                )
                obs, reward, terminated, truncated, info = env.step(action)

                # Update dashboard data
                manager.reward = (
                    reward.item() if isinstance(reward, torch.Tensor) else reward
                )
                new_success = env.unwrapped.evaluate().get("success", False)

                # Check for success transition
                if new_success and not manager.success:
                    print("Task Successful! Press SPACE to reset.")
                    manager.waiting_for_reset = True
                    # Auto-save episode if recording is enabled
                    if manager.is_saving_enabled and manager.recorder:
                        manager.recorder.save_episode(success=True)
                        manager.is_saving_enabled = (
                            False  # Disable until manual toggle or reset
                        )

                manager.success = new_success

                if dashboard:
                    dashboard.update_state(manager.reward, manager.success)

                if (
                    manager.is_saving_enabled
                    and manager.recorder
                    and not manager.waiting_for_reset
                ):
                    images = {}
                    if "sensor_data" in obs:
                        for cam_name, cam_data in obs["sensor_data"].items():
                            if "rgb" in cam_data:
                                images[cam_name] = cam_data["rgb"].cpu().numpy()[0]
                    manager.recorder.record_step(
                        obs, action_vec, reward, terminated | truncated, info, images
                    )

                if (terminated or truncated) and not manager.waiting_for_reset:
                    obs, info = env.reset()
                    manager.gym_episode_id += 1

                env.render()
                time.sleep(0.005)
    except Exception as e:
        print(e)
    finally:
        env.close()


if __name__ == "__main__":
    main()
