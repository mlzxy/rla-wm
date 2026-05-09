import argparse
import gc
import torch
import numpy as np
import os
import random
from typing import Dict, Union, TypeVar
from datalib.enriched_env import make_enriched_env
from datalib.dataset import ManiSkillTrajectoryDataset, TrajectoryData, RobotInfo
from datalib.ppo import Agent
from datalib.controller_utils import extract_target_qpos_from_controller
from mani_skill.agents.multi_agent import MultiAgent, BaseAgent
from mani_skill.utils.common import to_numpy
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
import gymnasium as gym
from tqdm import tqdm
from PIL import Image
from mani_skill import PACKAGE_ASSET_DIR

T = TypeVar("T")


def get_task_description(env_id: str) -> str:
    """Get language description of the task from env_id."""
    # Dictionary lookup
    task_descriptions: Dict[str, str] = {
        # Tabletop Manipulation Tasks (from previous turns)
        "PickCube-v1": "Grasp a red cube with the Panda robot and move it to a target goal position.",
        "PickCubeSO100-v1": "Grasp a red cube with the SO100 robot and move it to a target goal position.",
        "PickCubeWidowXAI-v1": "Grasp a red cube with the WidowXAI robot and move it to a target goal position.",
        "PushT-v1": "Precisely push the T-shaped block into the target region.",
        "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
        "PegInsertionSide-v1": "Pick up an orange-white peg and insert the orange end into the box with a hole in it.",
        "PullCube-v1": "Pull a cube onto a target region.",
        "PlugCharger-v1": "Pick up the charger and insert it into the receptacle.",
        # Mobile Manipulation Tasks (from previous turns)
        "OpenCabinetDrawer-v1": "Use the Fetch mobile manipulation robot to move towards a target cabinet and open the target drawer out.",
        "OpenCabinetDoor-v1": "Use the Fetch mobile manipulation robot to move towards a target cabinet and open the target door.",
        # Dexterity/TriFinger Tasks (from previous turns)
        "TriFingerRotateCubeLevel0-v1": "Rotate the cube to match the random goal position on the table (no orientation constraint).",
        "TriFingerRotateCubeLevel1-v1": "Rotate the cube to match the random goal position on the table (with yaw orientation constraint).",
        "TriFingerRotateCubeLevel2-v1": "Rotate the cube to match the fixed goal position in the air (no orientation constraint).",
        "TriFingerRotateCubeLevel3-v1": "Rotate the cube to match the random goal position in the air (no orientation constraint).",
        "TriFingerRotateCubeLevel4-v1": "Rotate the cube to match the random goal pose (position and orientation) in the air.",
        # Humanoid Tasks (newly added)
        "UnitreeG1PlaceAppleInBowl-v1": "Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side.",  #
        "UnitreeH1Stand-v1": "Make the Unitree H1 robot stand without falling.",  # Derived from HumanoidStandEnv logic
        "UnitreeG1Stand-v1": "Make the Unitree G1 robot stand without falling.",  # Derived from HumanoidStandEnv logic
        "UnitreeG1TransportBox-v1": "A G1 humanoid robot must find a box on a table and transport it to the other table and place it there.",  #
        # Bimanual/Multi-Robot Tasks (newly added)
        "TwoRobotStackCube-v1": "A collaborative task where two robot arms need to work together to stack two cubes.",  # Derived from docstring
        "TwoRobotPickCube-v1": "The goal is to pick up a red cube and lift it to a goal location, requiring two robots to cooperatively move the cube due to reach limitations.",  # Derived from docstring
    }

    if env_id not in task_descriptions:
        raise ValueError(f"Task description not found for env_id: {env_id}")

    return task_descriptions[env_id]


def process_camera_data(
    env: BaseEnv,
    cam_name: str,
    cam_data: Dict,
    sensor_params: Dict,
    video_streams: Dict,
    metadata_arrays: Dict,
):
    """
    Process camera data for a single camera (RGB, depth, segmentation, parameters).

    Args:
        cam_name: Name of the camera
        cam_data: Camera sensor data dictionary
        sensor_params: Sensor parameters dictionary
        video_streams: Dictionary to append video frames to
        metadata_arrays: Dictionary to append metadata to
    """
    rgb = cam_data["rgb"].cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    video_streams[f"{cam_name}_rgb"].append(rgb)

    depth = cam_data["depth"].cpu().numpy()
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    elif depth.ndim == 4:
        depth = depth[0, :, :, 0]
    video_streams[f"{cam_name}_depth"].append(depth)

    seg = cam_data["segmentation"].cpu().numpy()
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    elif seg.ndim == 4:
        seg = seg[0, :, :, 0]

    # Extract robot mask (robot typically has ID 1 or can be identified)
    seg_id2name = {k: v.name for k, v in env.unwrapped.segmentation_id_map.items()}
    background_ids = [0]
    static_ids = []
    robot_ids = []
    for seg_id, seg_name in seg_id2name.items():
        if seg_name == "ground":
            background_ids.append(seg_id)
        elif seg_name in [
            "table-workspace"
        ]:  # TODO: this may need to be extended to other scenes
            static_ids.append(seg_id)
        else:
            if seg_name in env.unwrapped.robot_link_names:
                robot_ids.append(seg_id)

    robot_mask = np.isin(seg, robot_ids).astype(np.uint8) * 255
    video_streams[f"{cam_name}_robot_mask"].append(robot_mask)

    # Foreground mask (non-background, i.e., non-zero)
    foreground_mask = (~np.isin(seg, background_ids)).astype(np.uint8) * 255
    video_streams[f"{cam_name}_foreground_mask"].append(foreground_mask)

    static_mask = np.isin(seg, static_ids).astype(np.uint8) * 255
    video_streams[f"{cam_name}_static_mask"].append(static_mask)

    # Camera parameters
    if cam_name in sensor_params:
        cam_params = sensor_params[cam_name]
        if "intrinsic_cv" in cam_params:
            metadata_arrays[f"{cam_name}_intrinsics"].append(
                cam_params["intrinsic_cv"].cpu().numpy()
            )
        if "extrinsic_cv" in cam_params:
            metadata_arrays[f"{cam_name}_extrinsics"].append(
                cam_params["extrinsic_cv"].cpu().numpy()
            )


def as_list(x: Union[T, list[T]]) -> list[T]:
    if isinstance(x, list):
        return x
    else:
        return [x]


def collect_trajectory(
    env: BaseEnv,
    agent: Agent,
    device: torch.device,
    env_id: str,
    max_steps: int = 200,
    random_action_prob: float = 0.0,
    deterministic: bool = False,
    stabilization_time: float = 5.0,
    control_freq: float = 20.0,
    random_action_strength: float = 5.0,
    verbose: bool = True,
) -> tuple[TrajectoryData, list[str]]:
    """
    Collect a single trajectory using the agent.

    Args:
        stabilization_time: Time in simulation seconds to wait before data collection (for physics to settle)
        control_freq: Control frequency in Hz (default 20Hz = 0.05s per step)

    Returns:
        TrajectoryData with all collected information
    """
    # Reset environment
    obs, info = env.reset()
    obs0 = obs

    # Stabilization period: let physics settle before data collection
    if stabilization_time > 0:
        stabilization_steps = int(stabilization_time * control_freq)
        if verbose:
            print(
                f"  Stabilizing scene for {stabilization_time:.2f}s ({stabilization_steps} steps)..."
            )

        # Create no-op action (zeros) as numpy array
        noop_action = np.zeros(env.action_space.shape, dtype=np.float32)

        # Step environment during stabilization (data will be discarded)
        for _ in range(stabilization_steps):
            obs, reward, terminated, truncated, info = env.step(noop_action)
            assert not (terminated or truncated)

        if verbose:
            print("  Stabilization complete, starting data collection...")

    # Get task description and URDF (only need once per trajectory)
    task_description = get_task_description(env_id)
    robot_infos = [
        RobotInfo(
            uid=a.uid,
            urdf_path=a.urdf_path[len(str(PACKAGE_ASSET_DIR)) + 1 :],
            urdf_config=a.urdf_config,
            joint_names=[j.name for j in a.controller.joints],
            action_space=[
                [float(v) for v in a.controller.action_space.low],
                [float(v) for v in a.controller.action_space.high],
                a.controller.action_space.shape,
            ],
            action_mapping=a.controller.action_mapping,
        )
        for a in as_list(env.agent)
    ]

    # Storage for trajectory data
    video_streams = {}
    metadata_arrays = {}

    # Initialize lists for each camera
    base_camera_names = ["base_camera", "render_camera", "camera"]
    base_cam_name = None
    for name in base_camera_names:
        if name in obs.get("sensor_data", {}):
            base_cam_name = name
            break

    if base_cam_name:
        video_streams[f"{base_cam_name}_rgb"] = []
        video_streams[f"{base_cam_name}_depth"] = []
        video_streams[f"{base_cam_name}_robot_mask"] = []
        video_streams[f"{base_cam_name}_foreground_mask"] = []
        video_streams[f"{base_cam_name}_static_mask"] = []

    # Get cinematic cameras
    num_cinematic_cams = getattr(env, "num_cinematic_cameras", 0)
    for i in range(num_cinematic_cams):
        cam_name = f"cinematic_cam_{i}"
        video_streams[f"{cam_name}_rgb"] = []
        video_streams[f"{cam_name}_depth"] = []
        video_streams[f"{cam_name}_robot_mask"] = []
        video_streams[f"{cam_name}_foreground_mask"] = []
        video_streams[f"{cam_name}_static_mask"] = []

    # Initialize metadata arrays
    metadata_arrays["actions"] = []
    if base_cam_name:
        metadata_arrays[f"{base_cam_name}_intrinsics"] = []
        metadata_arrays[f"{base_cam_name}_extrinsics"] = []
    for i in range(num_cinematic_cams):
        cam_name = f"cinematic_cam_{i}"
        metadata_arrays[f"{cam_name}_intrinsics"] = []
        metadata_arrays[f"{cam_name}_extrinsics"] = []

    # Collect trajectory
    success = False
    step_count = 0

    # Progress bar for episode steps
    # Progress bar for episode steps
    headless = not verbose
    pbar = tqdm(
        range(max_steps),
        desc="Episode progress",
        leave=False,
        unit="step",
        disable=headless,
    )

    for step in pbar:
        # Convert initial observation to tensor if needed
        # Agent expects state observation, not sensor data
        obs_state = obs["state"]
        obs_tensor = obs_state.float().to(device)

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        if random_action_prob > 0 and random.random() < random_action_prob:
            # Probabilistic random action intervention
            # Sample uniform random action from environment space (usually [-1, 1])
            # This creates large, meaningful diverse movements
            # tqdm.write(f"[{step}/{max_steps}] Random action intervention")
            action_np = env.action_space.sample() * random_action_strength
            action = torch.from_numpy(action_np).float().to(device)
        else:
            # Get action from agent
            action = agent.get_action(obs_tensor, deterministic=deterministic)
            if action.ndim > 1:
                action = action[0]  # Take first element if batched

        # action_np = to_numpy(action).flatten()
        # metadata_arrays['actions'].append(action_np)

        # Get robot states before step
        env.agent.set_action(
            (action[None] if action.ndim == 1 else action).cpu()
        )  # NOTE: the goal of this is to update controller state, to get virtual targets
        metadata_arrays.setdefault("qpos", []).append(
            env.agent.controller.qpos.cpu().numpy()
        )
        metadata_arrays.setdefault("target_qpos", []).append(
            extract_target_qpos_from_controller(env.agent.controller)
        )
        # NOTE: even for mobile-manipulation (e.g., Fetch), we need to rewrite the controller to support position control
        # as this will make "sim-to-real" easier

        # the raw pose format shall be [x, y, z, qw, qx, qy, qz]
        metadata_arrays.setdefault("root_poses", []).append(
            np.concat(
                [a.robot.root_pose.raw_pose.cpu().numpy() for a in as_list(env.agent)],
                axis=0,
            )
        )

        # Collect camera data
        sensor_data = obs.get("sensor_data", {})
        sensor_params = obs.get("sensor_param", {})

        # Process base camera
        if base_cam_name and base_cam_name in sensor_data:
            process_camera_data(
                env,
                base_cam_name,
                sensor_data[base_cam_name],
                sensor_params,
                video_streams,
                metadata_arrays,
            )

        # Process cinematic cameras
        for i in range(num_cinematic_cams):
            cam_name = f"cinematic_cam_{i}"
            if cam_name in sensor_data:
                process_camera_data(
                    env,
                    cam_name,
                    sensor_data[cam_name],
                    sensor_params,
                    video_streams,
                    metadata_arrays,
                )

        # Step environment (convert action to proper format - numpy array)
        action_np_for_step = to_numpy(action).flatten()
        obs, reward, terminated, truncated, info = env.step(action_np_for_step)

        # Check for success
        if info.get("success", False):
            success = True

        step_count += 1

        # Update progress bar with current status
        pbar.set_postfix({"success": "✓" if success else "✗", "steps": step_count})

        if headless and step_count % 2 == 0:
            print(f"PROGRESS_STEP: {step_count}/{max_steps}", flush=True)

        # Stop if done
        if terminated or truncated:
            break

    # Close progress bar
    pbar.close()

    # Convert lists to numpy arrays
    video_streams_np = {}
    for key, frames in video_streams.items():
        if len(frames) > 0:
            # Ensure all frames have the same shape
            try:
                video_streams_np[key] = np.array(frames)
            except ValueError:
                # If shapes don't match, skip this stream
                print(f"Warning: Skipping {key} due to shape mismatch")

    metadata_np = {}
    for key, values in metadata_arrays.items():
        if len(values) > 0:
            try:
                metadata_np[key] = np.array(values)
            except ValueError:
                print(f"Warning: Skipping metadata {key} due to shape mismatch")

    metadata_np["task_description"] = task_description
    metadata_np["success"] = success
    metadata_np["num_steps"] = step_count

    # Create TrajectoryData
    trajectory_data = TrajectoryData(
        success=success, video_streams=video_streams_np, metadata=metadata_np
    )

    # Log trajectory result
    status = "SUCCESS" if success else "FAILED"
    if verbose:
        print(f"  Trajectory {status} | Steps: {step_count}/{max_steps}")

    return trajectory_data, robot_infos


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectories from trained checkpoints"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint paths to load",
    )
    parser.add_argument("--env_id", type=str, default="PushT-v1", help="Environment ID")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/trajectories",
        help="Output directory for trajectories",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to collect",
    )
    parser.add_argument(
        "--num_cameras", type=int, default=6, help="Number of cinematic cameras"
    )
    parser.add_argument(
        "--num_distractors", type=int, default=0, help="Number of distractors"
    )
    parser.add_argument(
        "--camera_resolution",
        type=str,
        default="256,256",
        help="Camera resolution as 'width,height' (e.g., '256,256' or '512,512')",
    )
    parser.add_argument(
        "--distractor_offset",
        type=str,
        default="0.0,0.0",
        help="Offset for spawning distractors as 'x,y' (e.g., '0.5,-0.5')",
    )
    parser.add_argument(
        "--max_steps", type=int, default=200, help="Maximum steps per trajectory"
    )

    parser.add_argument(
        "--random_action_prob",
        type=float,
        default=0.0,
        help="Probability of taking a random action at each step (0-1)",
    )

    parser.add_argument(
        "--stabilization_time",
        type=float,
        default=5.0,
        help="Time in simulation seconds to wait before data collection (for physics to settle)",
    )
    parser.add_argument(
        "--control_freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz (default 20Hz = 0.05s per step)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)",
    )
    parser.add_argument(
        "--random_action_strength",
        type=float,
        default=5.0,
        help="Scaling factor for random actions (default: 5.0)",
    )
    parser.add_argument(
        "--checkpoint_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for sampling checkpoints (should match number of checkpoints)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a test episode and save demo video for each checkpoint",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable progress bars and logging for cleaner subprocess output",
    )

    args = parser.parse_args()

    # Parse camera resolution
    try:
        camera_res_parts = args.camera_resolution.split(",")
        if len(camera_res_parts) != 2:
            raise ValueError("Camera resolution must be in format 'width,height'")
        camera_res = (
            int(camera_res_parts[0].strip()),
            int(camera_res_parts[1].strip()),
        )
        if camera_res[0] <= 0 or camera_res[1] <= 0:
            raise ValueError(
                "Camera resolution width and height must be positive integers"
            )
    except ValueError as e:
        raise ValueError(
            f"Invalid camera resolution format '{args.camera_resolution}': {e}"
        )

    # Parse distractor offset
    try:
        offset_parts = args.distractor_offset.split(",")
        if len(offset_parts) != 2:
            raise ValueError("Distractor offset must be in format 'x,y'")
        distractor_offset = (
            float(offset_parts[0].strip()),
            float(offset_parts[1].strip()),
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid distractor offset format '{args.distractor_offset}': {e}"
        )

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device(args.device)

    # Load checkpoints and create agents
    print(f"Loading {len(args.checkpoints)} checkpoints...")
    agents = []
    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Create a dummy env to get observation/action space (we'll recreate it later)
        # For now, we'll create the env first
        temp_env = gym.make(args.env_id, obs_mode="state", render_mode="rgb_array")
        if isinstance(temp_env.action_space, gym.spaces.Dict):
            from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

            temp_env = FlattenActionSpaceWrapper(temp_env)
        # Create agent
        agent = Agent(temp_env).to(device)
        agent.load_state_dict(torch.load(ckpt_path, map_location=device))
        agent.eval()

        # eval an episode
        if args.test_run:
            print(f"  Evaluating checkpoint {ckpt_path}...")
            # Create test environment
            test_env = make_enriched_env(
                task_id=args.env_id,
                num_distractors=args.num_distractors,
                num_cameras=args.num_cameras,
                camera_res=camera_res,
                distractor_offset=distractor_offset,
                obs_mode="state+rgb+depth+segmentation",
                render_mode="rgb_array",
                shader_pack="rt" if torch.cuda.is_available() else "default",
                max_episode_steps=args.max_steps
                + int(args.stabilization_time * args.control_freq),
                reconfiguration_freq=1,
            )

            # Wrap action space if needed
            if isinstance(test_env.action_space, gym.spaces.Dict):
                from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

                test_env = FlattenActionSpaceWrapper(test_env)

            # Wrap with RecordEpisode if test-run flag is set
            checkpoint_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            demo_output_dir = os.path.join(
                os.path.dirname(ckpt_path), "demo_videos", checkpoint_name
            )
            os.makedirs(demo_output_dir, exist_ok=True)
            test_env = RecordEpisode(
                test_env,
                output_dir=demo_output_dir,
                save_trajectory=False,
                max_steps_per_video=args.max_steps,
                video_fps=30,
            )

            # Run evaluation episode
            obs, info = test_env.reset()
            if isinstance(obs, dict):
                obs_state = obs["state"]
            else:
                obs_state = obs
            obs_tensor = obs_state.float().to(device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            success = False
            step_count = 0
            for step in range(args.max_steps):
                action = agent.get_action(obs_tensor, deterministic=False)
                if action.ndim > 1:
                    action = action[0]
                action_np = to_numpy(action).flatten()
                obs, reward, terminated, truncated, info = test_env.step(action_np)

                if info.get("success", False):
                    success = True

                step_count += 1
                if terminated or truncated:
                    break

                # Update observation for next step
                if isinstance(obs, dict):
                    obs_state = obs["state"]
                else:
                    obs_state = obs
                obs_tensor = obs_state.float().to(device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)

            # Print success as sanity check
            print(f"  Test episode success: {success}, Steps: {step_count}")
            print(f"  Demo video saved to: {demo_output_dir}")

            test_env.close()

        agents.append(agent)
        temp_env.close()
        print(f"  Loaded: {ckpt_path}")

    # Setup checkpoint sampling weights
    if args.checkpoint_weights is None:
        checkpoint_weights = [1.0 / len(args.checkpoints)] * len(args.checkpoints)
    else:
        if len(args.checkpoint_weights) != len(args.checkpoints):
            raise ValueError(
                "Number of checkpoint weights must match number of checkpoints"
            )
        total_weight = sum(args.checkpoint_weights)
        checkpoint_weights = [w / total_weight for w in args.checkpoint_weights]

    # Create dataset
    dataset = ManiSkillTrajectoryDataset(args.output_dir)

    # Collect trajectories
    print(f"Collecting {args.num_trajectories} trajectories...")

    collected = 0
    successful_trajectories = 0

    # Progress bar for all trajectories
    # If headless, we disable the tqdm bar but print structured progress
    headless = args.headless
    verbose = not headless

    traj_pbar = tqdm(
        total=args.num_trajectories,
        desc="Collecting trajectories",
        unit="traj",
        disable=headless,
    )

    try:
        while collected < args.num_trajectories:
            # Sample checkpoint
            checkpoint_idx = np.random.choice(len(agents), p=checkpoint_weights)
            agent = agents[checkpoint_idx]

            # Update trajectory progress bar
            traj_pbar.set_description(
                f"Trajectory {collected + 1}/{args.num_trajectories}"
            )
            checkpoint_info = f"ckpt:{checkpoint_idx + 1}/{len(agents)}"
            traj_pbar.set_postfix({"checkpoint": checkpoint_info})

            # Create a new environment for this trajectory
            env = make_enriched_env(
                task_id=args.env_id,
                num_distractors=args.num_distractors,
                num_cameras=args.num_cameras,
                camera_res=camera_res,
                distractor_offset=distractor_offset,
                obs_mode="state+rgb+depth+segmentation",
                render_mode="rgb_array",
                shader_pack="rt" if torch.cuda.is_available() else "default",
                max_episode_steps=args.max_steps
                + int(args.stabilization_time * args.control_freq),
            )

            # Wrap action space if needed
            if isinstance(env.action_space, gym.spaces.Dict):
                from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

                env = FlattenActionSpaceWrapper(env)

            try:
                # Collect trajectory
                trajectory_data, robot_infos = collect_trajectory(
                    env=env,
                    agent=agent,
                    device=device,
                    env_id=args.env_id,
                    max_steps=args.max_steps,
                    random_action_prob=args.random_action_prob,
                    deterministic=args.deterministic,
                    stabilization_time=args.stabilization_time,
                    control_freq=args.control_freq,
                    random_action_strength=args.random_action_strength,
                    verbose=verbose,
                )
            finally:
                # Close the environment after collecting the trajectory
                env.close()

            # Save trajectory
            traj_id = f"{collected:06d}"
            # if not trajectory_data.success:
            #     import pudb
            #     pudb.set_trace()
            dataset.write_trajectory(traj_id, trajectory_data)
            dataset.save_robot_infos(robot_infos)

            # Track overall success
            if trajectory_data.success:
                successful_trajectories += 1

            collected += 1

            # Update trajectory progress bar with success info
            success_rate = successful_trajectories / collected if collected > 0 else 0.0
            postfix_dict = {
                "checkpoint": checkpoint_info,
                "success": f"{successful_trajectories}/{collected} ({success_rate:.1%})",
                "last": "✓" if trajectory_data.success else "✗",
            }

            del trajectory_data
            gc.collect()

            if not headless:
                traj_pbar.set_postfix(postfix_dict)
                traj_pbar.update(1)
            else:
                traj_pbar.update(
                    1
                )  # Still update the internal counter if needed, but display is disabled
                # Emit machine readable progress
                # Format: PROGRESS: <collected>/<total> | success=<success_count>
                print(
                    f"PROGRESS: {collected}/{args.num_trajectories} | success={successful_trajectories}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        traj_pbar.close()
        success_rate = successful_trajectories / collected if collected > 0 else 0.0
        print(f"\nCollected {collected} trajectories")
        print(f"Successful: {successful_trajectories}/{collected} ({success_rate:.1%})")
        print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
