from __future__ import annotations

from utils.vis import to_pil
from copy import deepcopy
import os
import sys
import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Union, TypeVar, Callable, Any, Tuple

# When run as script (python datalib/collect_all.py), ensure project root is on path
if __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import cv2
import gymnasium as gym

from rich import print
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
import logging
from collections import deque
import numpy as np
import torch
import glob
import tyro

# Register tasks and robots so gym.make(env_id) resolves
from datalib.src import tasks, robots  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from datalib.src import unified_workspace
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.common import to_numpy
import os.path as osp
from datalib.agent_respeed_wrapper import AgentRespeedWrapper
from datalib.dataset import (
    ManiSkillTrajectoryDataset,
    TrajectoryData,
    RobotInfo,
    VideoEncoding,
)
from datalib.ppo import Agent, SpecificObservationMaskWrapper
from datalib.controller_utils import extract_target_qpos_from_controller
from datalib.src.play.engine import PlannedAction, TrajectoryConfig, TrajectoryEngine
from datalib.src.play.primitives import AtomicPrimitives

ROOT_DIR = osp.dirname(osp.dirname(__file__))
logger = logging.getLogger("datalib")

T = TypeVar("T")


@dataclass
class Args:
    """Arguments for unified data collection (SPEC.md)."""

    mode: Literal["ppo", "play"] = "play"
    """Collection mode: ppo or play."""
    env_id: Literal[
        "PushT-v2",
        "RollBall-v1",
        "PokeCube-v2",
        "PullCubeTool-v1",
        "PullCube-v2",  # bad
        "PegInsertionSide-v1",
    ] = "PushT-v2"
    """Task or scene ID (e.g. PushCube-v1, PushT-v2)."""
    robot: Literal[
        "panda", "ur10e_stick", "xarm6_robotiq", "panda_closed", "xarm6_robotiq_closed"
    ] = "panda"
    num_trajectories: int = 1
    """Target number of success trajectories (PPO) or total trajectories (play when num_trajectories_per_action is not set)."""
    num_fail_trajectories: Optional[int] = None
    """Target number of failure trajectories (PPO). If None, defaults to half of num_trajectories. Set to num_trajectories for equal success/fail (legacy)."""
    num_trajectories_per_action: Optional[int] = None
    """Play mode: target per action category. If None, each category gets num_trajectories (default). When set, each category gets this many. Use play_total_cap_only=True for legacy total cap = num_trajectories."""
    play_total_cap_only: bool = False
    """Play mode: if True, ignore num_trajectories_per_action and collect until total trajectories = num_trajectories (legacy behavior)."""
    play_actions: Optional[str] = "push_only"
    """Play mode: comma-separated action types to collect (e.g. pick_and_place,push_only). If None, collect all. Valid: pick_and_place, tool_push, push_only. Stick/closed robots only support push_only."""
    respeed: bool = True
    """Enable agent respeeding (PPO only)."""
    checkpoint: Optional[str] = None
    """Path to checkpoint file or runs/Pool dir (for PPO)."""
    out_dir: Optional[str] = "data/ood_patch"
    """Output directory (default: data/ppo or data/play)."""

    # New args matching collect_trajectories.py where relevant
    max_steps: int = 150
    """Maximum steps per trajectory."""
    sys_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use (cuda/cpu)."""
    seed: int = -1
    """Random seed."""
    include_all_cameras: bool = True
    """Whether to include all cinematic cameras (default: True)."""
    render_resolution: str = "512,512"
    """Render resolution as 'width,height' (e.g., '1024,1024')."""
    depth_encoding: VideoEncoding = VideoEncoding.H264_LOSSLESS
    """Depth encoding format."""
    mask_encoding: VideoEncoding = VideoEncoding.H264_CRF28
    """Mask encoding format."""
    mask_obs: bool = False
    """whether to mask the 6th and 12th element of the observation"""

    min_success_rate: float = 0.5
    """Minimum success rate for checkpoints in the pool (default: 0.5)."""

    max_success_attempts: int = 100
    """PPO pool: max attempts with same env state (different models) before giving up on a success slot (default: 20)."""

    num_distractors: tuple[int, int] = (5, 25)
    """Number of distractors to spawn."""

    waypoint_interval: float = 0.03
    """Distance between waypoints (meters)"""
    x_bounds: Tuple[float, float] = (-0.4, 0.2)
    """X workspace bounds"""
    y_bounds: Tuple[float, float] = (-0.5, 0.5)
    """Y workspace bounds"""
    time_warp_speed_bounds: Optional[Tuple[float, float]] = (0.5, 3.0)
    """(v_min, v_max) to time-warp resample trajectories for speed diversity; None = no warp"""
    interpolate_steps: int = 3
    """Number of interpolated points between waypoints"""

    force_headless: bool = False
    """Force headless mode"""

    def __post_init__(self) -> None:
        self.out_dir = osp.join(self.out_dir, self.mode, f"{self.robot}")


# --- Duplicated Helper Functions from collect_trajectories.py ---

task_descriptions: Dict[str, str] = {
    "PushT-v2": "Precisely push the T-shaped block into the target region.",
    "RollBall-v1": "Roll the ball to the target goal position.",
    "PokeCube-v2": "Pick up the peg and use it to push the cube to the target goal position.",
    "PullCubeTool-v1": "Pick up the tool and use it to pull the cube to the target goal position.",
    "PullCube-v2": "Pull a cube onto a target region.",
    "PegInsertionSide-v1": "Pick up an orange-white peg and insert the orange end into the box with a hole in it.",
}


def ensure_dir(dir: str):
    os.makedirs(dir, exist_ok=True)


def unpack_state_dict(state_dict: Dict) -> Dict:
    if "model" in state_dict:
        return state_dict["model"]
    elif "model_state_dict" in state_dict:
        return state_dict["model_state_dict"]
    else:
        return state_dict


def process_camera_data(
    args: Args,
    env: BaseEnv,
    cam_name: str,
    cam_data: Dict,
    sensor_params: Dict,
    video_streams: Dict,
    metadata_arrays: Dict,
):
    """
    Process camera data for a single camera (RGB, depth, segmentation, parameters).
    Assumes images are already rendered at args.render_resolution.
    """
    # 1. RGB
    rgb = cam_data["rgb"].cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    video_streams[f"{cam_name}_rgb"].append(rgb)

    # 2. Depth
    depth = cam_data["depth"].cpu().numpy()
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    elif depth.ndim == 4:
        depth = depth[0, :, :, 0]
    video_streams[f"{cam_name}_depth"].append(depth)

    # 3. Segmentation
    seg = cam_data["segmentation"].cpu().numpy()
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    elif seg.ndim == 4:
        seg = seg[0, :, :, 0]

    # Extract robot mask
    seg_id2name = {k: v.name for k, v in env.unwrapped.segmentation_id_map.items()}
    background_ids = [0]
    static_ids = []
    robot_ids = []
    for seg_id, seg_name in seg_id2name.items():
        if seg_name == "ground":
            background_ids.append(seg_id)
        elif seg_name in ["table-workspace"]:
            static_ids.append(seg_id)
        else:
            if seg_name in env.unwrapped.robot_link_names:
                robot_ids.append(seg_id)

    robot_mask = np.isin(seg, robot_ids).astype(np.uint8) * 255
    video_streams[f"{cam_name}_robot_mask"].append(robot_mask)

    # Foreground mask (non-background)
    foreground_mask = (~np.isin(seg, background_ids)).astype(np.uint8) * 255
    video_streams[f"{cam_name}_foreground_mask"].append(foreground_mask)

    static_mask = np.isin(seg, static_ids).astype(np.uint8) * 255
    video_streams[f"{cam_name}_static_mask"].append(static_mask)

    # 4. Camera parameters
    if cam_name in sensor_params:
        cam_params = sensor_params[cam_name]
        if "intrinsic_cv" in cam_params:
            K = cam_params["intrinsic_cv"].cpu().numpy()
            metadata_arrays[f"{cam_name}_intrinsics"].append(K)

        if "extrinsic_cv" in cam_params:
            metadata_arrays[f"{cam_name}_extrinsics"].append(
                cam_params["extrinsic_cv"].cpu().numpy()
            )


def as_list(x: Union[T, list[T]]) -> list[T]:
    if isinstance(x, list):
        return x
    else:
        return [x]


def find_pool_directory(env_id: str, robot: str) -> Optional[str]:
    """Find the checkpoint pool directory for a given env and robot."""
    # Pool structure: runs/Pool/{id}_{env_id}_{robot}
    # We search for any directory matching *_{env_id}_{robot}
    pool_root = "runs/Pool"
    if not os.path.exists(pool_root):
        return None

    # Search pattern
    pattern = os.path.join(pool_root, f"*_{env_id}_{robot}")
    matches = glob.glob(pattern)

    if not matches:
        return None

    # If multiple, maybe pick the one with highest ID? Or just the first one.
    # Usually there should be only one active pool per config.
    # We sort to be deterministic.
    matches.sort()
    return matches[-1]  # Return the last one (likely highest ID if numbered)


def load_pool_checkpoints(pool_dir: str) -> List[Dict]:
    """
    Load all valid checkpoints from the pool directory structure.
    Expected layout: checkpoints/{success_rate}/*.pt
    Returns a list of dicts with keys: 'path', 'success_rate', 'reward'.
    """
    checkpoints = []
    pattern = os.path.join(pool_dir, "checkpoints", "*", "*.pt")
    ckpt_paths = glob.glob(pattern)

    for ckpt_path in ckpt_paths:
        success_rate_dir = os.path.basename(os.path.dirname(ckpt_path))
        try:
            success_rate = float(success_rate_dir)
        except ValueError:
            print(
                f"[yellow]Skipping checkpoint with non-numeric success rate folder: {ckpt_path}[/yellow]"
            )
            continue

        checkpoints.append(
            {
                "path": ckpt_path,
                "success_rate": success_rate,
                # Keep key for compatibility; not encoded in folder layout.
                "reward": None,
            }
        )

    checkpoints.sort(key=lambda x: (x["success_rate"], x["path"]))
    return checkpoints


def count_trajectories_in_root(root: str) -> int:
    """Return the number of trajectory directories (traj_*) under root. Root may not exist."""
    if not os.path.isdir(root):
        return 0
    pattern = os.path.join(root, "traj_*")
    matches = glob.glob(pattern)
    return len([m for m in matches if os.path.isdir(m)])


def ask_resume_confirm(summary_lines: List[str]) -> bool:
    """Print resume summary and ask user y/n. Returns True to proceed, False to abort."""
    from rich.panel import Panel as RichPanel

    console = Console()
    console.print(
        RichPanel(
            "\n".join(summary_lines),
            title="[bold yellow]Resume collection[/bold yellow]",
            border_style="yellow",
        )
    )
    while True:
        try:
            ans = input("Proceed with resume? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        console.print("[yellow]Please enter y or n.[/yellow]")


# --- Logging setup ---
class DashboardHandler(logging.Handler):
    def __init__(self, logs_deque):
        super().__init__()
        self.logs_deque = logs_deque

    def emit(self, record):
        try:
            msg = self.format(record)
            # Apply color based on level
            if record.levelno >= logging.ERROR:
                msg = f"[red]{msg}[/red]"
            elif record.levelno >= logging.WARNING:
                msg = f"[yellow]{msg}[/yellow]"
            elif record.levelno >= logging.INFO:
                msg = f"[white]{msg}[/white]"
            else:
                msg = f"[grey]{msg}[/grey]"
            self.logs_deque.append(msg)
        except Exception:
            self.handleError(record)


def setup_dashboard_logging(logs_deque):
    logger = logging.getLogger("datalib")
    logger.setLevel(logging.INFO)
    # Remove existing handlers if any to avoid duplicates
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler = DashboardHandler(logs_deque)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


# --- Main Collection Logic ---


def collect_ppo(args: Args) -> None:
    """PPO collection: load agent, run episodes, save success/fail datasets."""
    # if not args.checkpoint or not os.path.isfile(args.checkpoint):
    #     raise FileNotFoundError(
    #         f"PPO checkpoint file not found: {args.checkpoint}. "
    #         "Provide a path to a .pt file (e.g. from runs/Pool or runs/PPO)."
    #     )
    if args.robot == "ur10e_stick":
        print(
            "Masking the 6th and 12th element of the observation for ur10e_stick (for reusing old PPO checkpoints)"
        )
        args.mask_obs = True

    # 1. Setup Device
    device = torch.device(args.sys_device)

    # 2. Setup Temporary Env for Agent Initialization
    # Agent expects a flat state observation space.
    # We create a dummy env with obs_mode="state" to initialize the Agent correctly.
    temp_env_kwargs = dict(
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        robot_uids=args.robot,
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    # Note: we should avoid passing many complex args to simple temp_env
    temp_env = gym.make(args.env_id, **temp_env_kwargs)

    if isinstance(temp_env.action_space, gym.spaces.Dict):
        temp_env = FlattenActionSpaceWrapper(temp_env)

    print(f"DEBUG: args.robot = {args.robot}")
    print(f"DEBUG: temp_env.observation_space = {temp_env.observation_space}")

    if args.mask_obs:
        # Note: temp_env is used to initialize Agent. Agent network input size depends on observation space.
        # So we MUST wrap temp_env too.
        temp_env = SpecificObservationMaskWrapper(temp_env)

    # 3. Load Agent
    # 3. Load Agent Structure
    # We initialize the agent here. If using a single file checkpoint, we load it now.
    # If using a pool, we will load weights dynamically in the loop.
    agent = Agent(temp_env).to(device)
    logger = logging.getLogger("datalib")

    # Checkpoint Handling (File vs Pool)
    pool_checkpoints = []
    using_pool = False

    # 1. Determine if we are using a pool
    if args.checkpoint is None:
        # Try to find a pool
        pool_dir = find_pool_directory(args.env_id, args.robot)
        if pool_dir:
            print(f"Auto-discovered checkpoint pool: {pool_dir}")
            args.checkpoint = pool_dir  # Set it for consistency
        else:
            raise FileNotFoundError(
                f"No checkpoint provided and no pool found for {args.env_id} + {args.robot}"
            )

    if os.path.isdir(args.checkpoint):
        # It's a directory -> Assume functionality of a Pool
        using_pool = True
        pool_checkpoints = load_pool_checkpoints(args.checkpoint)
        if not pool_checkpoints:
            raise ValueError(f"No valid checkpoints found in pool: {args.checkpoint}")

        # Apply filtering
        num_before = len(pool_checkpoints)
        pool_checkpoints = [
            ckpt
            for ckpt in pool_checkpoints
            if ckpt["success_rate"] >= args.min_success_rate
        ]
        if not pool_checkpoints:
            raise ValueError(
                f"No checkpoints found with success_rate >= {args.min_success_rate} "
                f"(Total was {num_before} before filtering)"
            )

        print(
            f"Loaded {num_before} checkpoints from pool. Filtered to {len(pool_checkpoints)} with SR >= {args.min_success_rate}."
        )
        logger.info(
            f"Using {len(pool_checkpoints)} checkpoints (filtered from {num_before})"
        )
    else:
        # It's a file -> Load normally
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        agent.load_state_dict(unpack_state_dict(state_dict))
        print(f"Loaded single checkpoint: {args.checkpoint}")

    agent.eval()
    temp_env.close()

    if args.respeed:
        agent = AgentRespeedWrapper(agent, min_speed=0.25, max_speed=1.0)

    # Parse render resolution (for env creation)
    try:
        parts = args.render_resolution.split(",")
        render_w, render_h = int(parts[0]), int(parts[1])
    except Exception:
        print(f"Invalid render resolution '{args.render_resolution}', using 1024,1024")
        render_w, render_h = 1024, 1024

    # 4. Setup Real Data Collection Env
    # We use "state_dict" obs_mode to get access to sensor data (cameras) and state
    # Plan requires "state+rgb+depth+segmentation" or similar
    headless = not (sys.platform == "darwin" and not args.force_headless)
    env_kwargs = dict(
        obs_mode="state+rgb+depth+segmentation",
        render_mode="rgb_array" if headless else "human",
        sim_backend="physx_cpu",
        robot_uids=args.robot,
        control_mode="pd_joint_delta_pos",
        include_all_cameras=args.include_all_cameras,
        camera_width=render_w,
        camera_height=render_h,
        shader_dir="rt-clean" if torch.cuda.is_available() else "default",
        max_episode_steps=args.max_steps,
    )

    env = gym.make(
        args.env_id,
        reconfiguration_freq=0,  # NOTE: if set to 1, the camera will be dead after the first reset, do not know why. This does not happen on tableonly env
        **env_kwargs,
    )

    # Wrap action space if needed
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    if args.mask_obs:
        env = SpecificObservationMaskWrapper(env)

    # We DO NOT use ManiSkillVectorEnv wrapping here if we can avoid it,
    # to keep direct access to the env simpler for data extraction (mirroring collect_trajectories.py)
    # But Agent expects vectorized-like input (dim 0).

    # 4. Setup Output Datasets
    out_root = os.path.join(args.out_dir, args.env_id)
    success_root = os.path.join(out_root, "success")
    fail_root = os.path.join(out_root, "fail")
    success_dataset = ManiSkillTrajectoryDataset(
        success_root,
        depth_encoding=args.depth_encoding,
        mask_encoding=args.mask_encoding,
    )
    fail_dataset = ManiSkillTrajectoryDataset(
        fail_root, depth_encoding=args.depth_encoding, mask_encoding=args.mask_encoding
    )

    existing_success = count_trajectories_in_root(success_root)
    existing_fail = count_trajectories_in_root(fail_root)
    success_count = existing_success
    fail_count = existing_fail

    if existing_success > 0 or existing_fail > 0:
        summary = [
            f"Output path: [bold]{out_root}[/bold]",
            f"Existing: [green]success={existing_success}[/green], [red]fail={existing_fail}[/red] (total={existing_success + existing_fail})",
            f"Target: success={args.num_trajectories}, fail={args.num_fail_trajectories if args.num_fail_trajectories is not None else args.num_trajectories // 2}",
            f"Will collect [green]{max(0, args.num_trajectories - existing_success)}[/green] more success, [red]{max(0, (args.num_fail_trajectories if args.num_fail_trajectories is not None else args.num_trajectories // 2) - existing_fail)}[/red] more fail.",
        ]
        if not ask_resume_confirm(summary):
            print("[yellow]Aborted.[/yellow]")
            sys.exit(0)
        print("[green]Resuming collection.[/green]")
    current_attempt = 0
    max_attempt = 0
    success_target = args.num_trajectories
    fail_target = (
        args.num_fail_trajectories
        if args.num_fail_trajectories is not None
        else (args.num_trajectories // 2)
    )

    # Setup Rich Progress Bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        expand=True,
    )
    success_task = progress.add_task(
        "[green]Success Trajectories", total=success_target
    )
    fail_task = progress.add_task("[red]Fail Trajectories", total=fail_target)
    episode_task = progress.add_task(
        "[cyan]Active Episode", total=args.max_steps, visible=False
    )

    # Logging Queue
    logs = deque(maxlen=10)
    setup_dashboard_logging(logs)

    console = Console()

    def make_dashboard():
        table = Table.grid(expand=True)
        table.add_column()
        header = f"[bold blue]PPO Collection[/bold blue] | Env: [green]{args.env_id if args.mode == 'ppo' else 'TableOnly-v2'}[/green] | Robot: [green]{args.robot}[/green] | Device: [yellow]{args.sys_device}[/yellow]"

        total = success_count + fail_count
        sr = (success_count / total * 100) if total > 0 else 0
        attempt_str = (
            f" | [magenta]Attempt: {current_attempt}/{max_attempt}[/magenta]"
            if current_attempt > 0
            else ""
        )
        stats = f"[green]Success: {success_count}[/green] | [red]Fail: {fail_count}[/red] | [white]Total: {total}[/white] | [bold cyan]SR: {sr:.1f}%[/bold cyan]{attempt_str}"

        table.add_row(Panel(Group(header, stats), border_style="blue"))
        table.add_row(progress)
        log_panel = Panel(
            "\n".join(logs), title="[bold]Logs[/bold]", border_style="white", height=12
        )
        table.add_row(log_panel)
        return table

    def run_one_episode(obs: Dict) -> Tuple[bool, TrajectoryData, List[RobotInfo]]:
        """Run a single episode from the given obs; return success, traj_data, robot_infos."""
        robot_infos = [
            RobotInfo(
                uid=a.uid,
                urdf_path=a.urdf_path[len(str(ROOT_DIR)) + 1 :],
                urdf_config=a.urdf_config,
                joint_names=[j.name for j in a.controller.joints],
                action_space=[
                    [float(v) for v in a.controller.action_space.low],
                    [float(v) for v in a.controller.action_space.high],
                    a.controller.action_space.shape,
                ],
                action_mapping=a.controller.action_mapping,
            )
            for a in as_list(env.unwrapped.agent)
        ]
        video_streams: Dict[str, List] = {}
        metadata_arrays: Dict[str, List] = {}
        sensor_data = obs.get("sensor_data", {})
        sensor_params = obs.get("sensor_param", {})
        for name in sensor_data.keys():
            for stream in [
                "rgb",
                "depth",
                "robot_mask",
                "foreground_mask",
                "static_mask",
            ]:
                video_streams[f"{name}_{stream}"] = []
        metadata_arrays["actions"] = []
        metadata_arrays["qpos"] = []
        metadata_arrays["full_qpos"] = []
        metadata_arrays["target_qpos"] = []
        metadata_arrays["root_poses"] = []
        metadata_arrays["object_poses"] = []
        metadata_arrays["eef_pose"] = []
        for name in sensor_data.keys():
            metadata_arrays[f"{name}_intrinsics"] = []
            metadata_arrays[f"{name}_extrinsics"] = []

        done = False
        step_count = 0
        success = False
        max_steps = args.max_steps
        while not done and step_count < max_steps:
            obs_state = obs["state"]
            sensor_data = obs.get("sensor_data", {})
            sensor_params = obs.get("sensor_param", {})
            obs_tensor = torch.as_tensor(
                obs_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_tensor, deterministic=False)
                if action.ndim > 1:
                    action = action[0]
            env.unwrapped.agent.set_action(action.cpu())
            metadata_arrays["qpos"].append(
                deepcopy(env.unwrapped.agent.controller.qpos.cpu().numpy())
            )
            metadata_arrays["full_qpos"].append(
                deepcopy(env.agent.get_proprioception()["qpos"].cpu().numpy())
            )
            metadata_arrays["target_qpos"].append(
                deepcopy(
                    extract_target_qpos_from_controller(env.unwrapped.agent.controller)
                )
            )
            metadata_arrays["root_poses"].append(
                np.concatenate(
                    [
                        deepcopy(a.robot.root_pose.raw_pose.cpu().numpy())
                        for a in as_list(env.unwrapped.agent)
                    ],
                    axis=0,
                )
            )
            metadata_arrays["eef_pose"].append(
                deepcopy(env.unwrapped.agent.tcp.pose.raw_pose.cpu().numpy()[0])
            )
            for cam_name in sensor_data.keys():
                process_camera_data(
                    args,
                    env,
                    cam_name,
                    sensor_data[cam_name],
                    sensor_params,
                    video_streams,
                    metadata_arrays,
                )
            metadata_arrays["actions"].append(to_numpy(deepcopy(action)).flatten())
            # Record object poses
            object_poses = env.unwrapped.get_object_poses()
            if object_poses.ndim == 3:
                object_poses = object_poses[0]
            metadata_arrays["object_poses"].append(deepcopy(object_poses))

            progress.update(episode_task, advance=1, visible=True)
            action_np = to_numpy(action).flatten()
            obs, reward, term, trunc, info = env.step(action_np)  # STEP HERE
            if not headless:
                env.render()
            if info.get("success", False):
                success = True
                max_steps = min(args.max_steps, 10 + step_count)
            done = term or trunc
            step_count += 1
        progress.update(episode_task, visible=False, completed=0)

        video_streams_np = {
            k: np.array(v) for k, v in video_streams.items() if len(v) > 0
        }
        metadata_np = {k: np.array(v) for k, v in metadata_arrays.items() if len(v) > 0}
        metadata_np["task_description"] = task_descriptions[args.env_id]
        metadata_np["success"] = success
        metadata_np["num_steps"] = step_count
        traj_data = TrajectoryData(
            success=success, video_streams=video_streams_np, metadata=metadata_np
        )
        return success, traj_data, robot_infos

    # 5. Collection Loop
    with Live(make_dashboard(), refresh_per_second=4) as live:
        while success_count < success_target or fail_count < fail_target:
            need_success = success_count < success_target
            need_fail = fail_count < fail_target
            # Alternate success and failure when both are needed (like play collection).
            if need_success and need_fail:
                desired_success = (success_count + fail_count) % 2 == 0
            elif need_success:
                desired_success = True
            else:
                desired_success = False

            if using_pool and desired_success:
                # Success slot: same env state, try different models until success or max attempts.
                max_sr = max(ckpt["success_rate"] for ckpt in pool_checkpoints)
                top_ckpts = [
                    ckpt for ckpt in pool_checkpoints if ckpt["success_rate"] == max_sr
                ]
                episode_seed = random.randint(0, 2**31 - 1)
                success_saved = False
                max_attempt = args.max_success_attempts
                for attempt in range(args.max_success_attempts):
                    current_attempt = attempt + 1
                    live.update(make_dashboard())
                    obs, _ = env.reset(seed=episode_seed)
                    selected_ckpt = random.choice(top_ckpts)
                    state_dict = torch.load(selected_ckpt["path"], map_location=device)
                    (agent.agent if args.respeed else agent).load_state_dict(
                        unpack_state_dict(state_dict)
                    )
                    success, traj_data, robot_infos = run_one_episode(obs)
                    if success:
                        traj_data.metadata["num_attempts"] = attempt + 1
                        success_dataset.write_trajectory(
                            f"{success_count:06d}", traj_data
                        )
                        success_dataset.save_robot_infos(robot_infos)
                        success_count += 1
                        progress.update(success_task, advance=1)
                        logs.append(
                            f"[green]✔[/green] Saved Success Trajectory {success_count}/{success_target} (attempt {attempt + 1})"
                        )
                        success_saved = True
                        break
                current_attempt = 0
                if not success_saved:
                    logs.append(
                        f"[yellow]Gave up success slot after {args.max_success_attempts} attempts (same state)[/yellow]"
                    )
                live.update(make_dashboard())
                continue

            # Single-episode path: failure slot or single-checkpoint mode.
            if using_pool and not desired_success:
                weights = [
                    (1.0 - ckpt["success_rate"]) + 0.01 for ckpt in pool_checkpoints
                ]
                selected_ckpt = random.choices(pool_checkpoints, weights=weights, k=1)[
                    0
                ]
                state_dict = torch.load(selected_ckpt["path"], map_location=device)
                (agent.agent if args.respeed else agent).load_state_dict(
                    unpack_state_dict(state_dict)
                )

            obs, _ = env.reset(seed=int(torch.randint(0, 2**31, (1,)).item()))
            success, traj_data, robot_infos = run_one_episode(obs)

            traj_data.metadata["num_attempts"] = 1
            if success:
                if success_count < success_target:
                    success_dataset.write_trajectory(f"{success_count:06d}", traj_data)
                    success_dataset.save_robot_infos(robot_infos)
                    success_count += 1
                    progress.update(success_task, advance=1)
                    logs.append(
                        f"[green]✔[/green] Saved Success Trajectory {success_count}/{success_target}"
                    )
            else:
                if fail_count < fail_target:
                    fail_dataset.write_trajectory(f"{fail_count:06d}", traj_data)
                    fail_dataset.save_robot_infos(robot_infos)
                    fail_count += 1
                    progress.update(fail_task, advance=1)
                    logs.append(
                        f"[red]✘[/red] Saved Fail Trajectory {fail_count}/{fail_target}"
                    )

            live.update(make_dashboard())

    env.close()
    logger.info(f"Collection Complete. Success: {success_count}, Fail: {fail_count}")


@dataclass
class _TrajectoryBuffers:
    """Buffers used by DataCollectionWrapper for a single trajectory."""

    video_streams: Dict[str, List[np.ndarray]]
    metadata_arrays: Dict[str, List[np.ndarray]]


class DataCollectionWrapper(gym.Wrapper):
    """
    Wrapper that records S_t and A_t before each env step.

    This aligns play collection with PPO collection semantics where images and state are
    captured before executing the action.
    """

    def __init__(self, env: gym.Env, args: Args):
        super().__init__(env)
        self._args = args
        self._buffers: Optional[_TrajectoryBuffers] = None
        self._cached_obs: Optional[Dict[str, Any]] = None
        self._last_action: Optional[np.ndarray] = None
        self.headless = not (sys.platform == "darwin" and not args.force_headless)
        self._step = 0

    def set_buffers(self, buffers: _TrajectoryBuffers) -> None:
        self._buffers = buffers

    def set_cached_obs(self, obs: Dict[str, Any]) -> None:
        self._cached_obs = obs

    def _snapshot_obs(self) -> Dict[str, Any]:
        if self._cached_obs is not None:
            return self._cached_obs
        if hasattr(self.env.unwrapped, "get_obs"):
            return self.env.unwrapped.get_obs()
        raise RuntimeError("No cached observation available for data capture")

    @property
    def dumpy_action(self):
        action = np.zeros(self.env.action_space.shape)
        action[:] = (
            self.env.unwrapped.agent.robot.get_qpos()
            .cpu()
            .numpy()
            .reshape(-1)[: len(action)]
        )
        if "stick" not in self._args.robot and "close" not in self._args.robot:
            action[-1] = 0 if "xarm6" in self._args.robot else 1
        return action

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = self.env.reset(*args, **kwargs)
        self._step = 0
        _stab_total = 2500
        with Progress(
            TextColumn("[bold blue]Stabilizing"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            expand=True,
        ) as progress:
            stab_task = progress.add_task("steps", total=_stab_total)
            for _ in range(_stab_total):
                self.env.step_wo_obs(self.dumpy_action)
                # if not self.headless:
                #     self.env.render()
                progress.update(stab_task, advance=1)

        self.env.unwrapped.remove_off_table_objects()
        for _ in range(10):
            self.env.step_wo_obs(self.dumpy_action)
            # if not self.headless:
            #     self.env.render()
        obs = self.env.unwrapped.get_obs()
        self._cached_obs = deepcopy(obs)
        return obs, info

    def step(self, action):
        if self._buffers is None:
            raise RuntimeError(
                "Trajectory buffers were not initialized before stepping"
            )

        obs_t = self._snapshot_obs()
        sensor_data = obs_t.get("sensor_data", {})
        sensor_params = obs_t.get("sensor_param", {})

        if hasattr(action, "cpu"):
            action_np = action.detach().cpu().numpy().reshape(-1)
        else:
            action_np = np.asarray(action).reshape(-1)

        # Capture state and camera data at S_t (before stepping).
        self._buffers.metadata_arrays["qpos"].append(
            deepcopy(self.env.unwrapped.agent.controller.qpos.cpu().numpy())
        )
        self._buffers.metadata_arrays["full_qpos"].append(
            deepcopy(
                self.env.unwrapped.agent.get_proprioception()["qpos"].cpu().numpy()
            )
        )

        self.env.unwrapped.agent.set_action(torch.from_numpy(action))
        target_qpos = extract_target_qpos_from_controller(
            self.env.unwrapped.agent.controller
        )  # NOTE: target_qpos could be just partial qpos, and cause problems in pytorch_kinematics!
        self._buffers.metadata_arrays["target_qpos"].append(deepcopy(target_qpos))

        self._buffers.metadata_arrays["root_poses"].append(
            np.concatenate(
                [
                    deepcopy(a.robot.root_pose.raw_pose.cpu().numpy())
                    for a in as_list(self.env.unwrapped.agent)
                ],
                axis=0,
            )
        )
        self._buffers.metadata_arrays["eef_pose"].append(
            deepcopy(self.env.unwrapped.agent.tcp.pose.raw_pose.cpu().numpy()[0])
        )

        for cam_name in sensor_data.keys():
            process_camera_data(
                self._args,
                self.env,
                cam_name,
                sensor_data[cam_name],
                sensor_params,
                self._buffers.video_streams,
                self._buffers.metadata_arrays,
            )

        # Record object poses
        object_poses = self.env.unwrapped.get_object_poses()
        if object_poses.ndim == 3:
            object_poses = object_poses[0]
        self._buffers.metadata_arrays["object_poses"].append(deepcopy(object_poses))

        self._buffers.metadata_arrays["actions"].append(action_np)
        obs, reward, term, trunc, info = self.env.step(action)
        self._cached_obs = deepcopy(obs)
        self._last_action = action

        # DEBUG
        # ensure_dir('runs/debug/')
        # to_pil(self._buffers.video_streams['base_camera_rgb'][-1]).save(f'runs/debug/frame_{self._step:05d}.jpg')
        self._step += 1
        return obs, reward, term, trunc, info

    def save_robot_mesh(self, index=-1):
        from datalib.robot_geometry import (
            DifferentiableRobotGeometry,
            to_o3d_mesh,
            mesh_to_arrays,
            save_o3d_mesh,
        )

        test_robot_geom = DifferentiableRobotGeometry(
            urdf_path=self.env.unwrapped.agent.urdf_path,
            base_dir=osp.dirname(self.env.unwrapped.agent.urdf_path),
        )
        test_robot_geom.set_pose(
            torch.tensor(self._buffers.metadata_arrays["qpos"][index]),
            torch.tensor(self._buffers.metadata_arrays["root_poses"][index]),
        )
        robot_mesh = to_o3d_mesh(test_robot_geom.sdf)
        save_o3d_mesh(robot_mesh, "runs/debug/robot_mesh.ply")


def _init_buffers_from_obs(obs: Dict[str, Any]) -> _TrajectoryBuffers:
    sensor_data = obs.get("sensor_data", {})
    video_streams: Dict[str, List[np.ndarray]] = {}
    metadata_arrays: Dict[str, List[np.ndarray]] = {
        "actions": [],
        "qpos": [],
        "full_qpos": [],
        "target_qpos": [],
        "root_poses": [],
        "object_poses": [],
        "eef_pose": [],
    }

    for name in sensor_data.keys():
        for stream in ["rgb", "depth", "robot_mask", "foreground_mask", "static_mask"]:
            video_streams[f"{name}_{stream}"] = []
        metadata_arrays[f"{name}_intrinsics"] = []
        metadata_arrays[f"{name}_extrinsics"] = []

    return _TrajectoryBuffers(
        video_streams=video_streams, metadata_arrays=metadata_arrays
    )


def _extract_robot_infos(env: gym.Env) -> List[RobotInfo]:
    try:
        return [
            RobotInfo(
                uid=a.uid,
                urdf_path=a.urdf_path[len(str(ROOT_DIR)) + 1 :],
                urdf_config=a.urdf_config,
                joint_names=[j.name for j in a.controller.joints],
                action_space=[
                    [float(v) for v in a.controller.action_space.low],
                    [float(v) for v in a.controller.action_space.high],
                    a.controller.action_space.shape,
                ],
                action_mapping=a.controller.action_mapping,
            )
            for a in as_list(env.unwrapped.agent)
        ]
    except Exception:
        return []


class PlayCollector:
    """Collects play trajectories using a callback-driven supervisor."""

    def __init__(
        self,
        args: Args,
        supervisor_callback: Optional[
            Callable[
                [gym.Env, AtomicPrimitives, TrajectoryEngine], Optional[PlannedAction]
            ]
        ] = None,
        env_factory: Optional[Callable[[Args], gym.Env]] = None,
        dataset_factory: Optional[Callable[[str], ManiSkillTrajectoryDataset]] = None,
        primitives_factory: Callable[..., AtomicPrimitives] = AtomicPrimitives,
        engine_factory: Callable[..., TrajectoryEngine] = TrajectoryEngine,
    ):
        self.args = args
        self.supervisor_callback = supervisor_callback or self.default_supervisor
        self._env_factory = env_factory or self._make_env
        self._dataset_factory = dataset_factory or (
            lambda path: ManiSkillTrajectoryDataset(
                path,
                depth_encoding=args.depth_encoding,
                mask_encoding=args.mask_encoding,
            )
        )
        self._primitives_factory = primitives_factory
        self._engine_factory = engine_factory
        self.headless = not (sys.platform == "darwin" and not args.force_headless)

    def _make_env(self, args: Args) -> gym.Env:
        try:
            render_w, render_h = [int(p) for p in args.render_resolution.split(",")]
        except Exception:
            render_w, render_h = 1024, 1024

        env = gym.make(
            "TableOnly-v2",
            obs_mode="state+rgb+depth+segmentation",
            control_mode="pd_joint_pos_6d"
            if args.robot == "ur10e_stick"
            else "pd_joint_pos",
            render_mode="human" if not self.headless else "rgb_array",
            sim_backend="physx_cpu",
            robot_uids=args.robot,
            include_all_cameras=args.include_all_cameras,
            camera_width=render_w,
            camera_height=render_h,
            # max_episode_steps=args.max_steps,
            robot_init_high=True,
            collision_free_placement=True,
            num_distractors=args.num_distractors,
            random_rotation=True,
            workspace_x_bounds=args.x_bounds,
            workspace_y_bounds=args.y_bounds,
            distractor_types=[
                "cube",
                "sphere",
                "box",
                "stick",
                "triangle",
                "polyhedron",
                "number",
            ],
            distractor_scale_min=1.0,
            distractor_scale_max=1.0,
            max_episode_steps=1e6,
            shader_dir="rt-clean" if torch.cuda.is_available() else "default",
            # z_stagger=0.1
        )
        # env = gym.make(
        #     "TableOnly-v2",
        #     obs_mode="state+rgb+depth+segmentation",
        #     control_mode="pd_joint_pos_6d" if args.robot == "ur10e_stick" else "pd_joint_pos",
        #     sim_backend="physx_cuda" if args.sys_device == "cuda" else "physx_cpu",
        #     render_mode="human",
        #     robot_uids=args.robot,
        #     num_distractors=(4, 10),
        #     distractor_types=["cube", "sphere", "box", "stick", "triangle", "polyhedron", "number"],
        #     distractor_scale_min=1.0,
        #     distractor_scale_max=1.0,
        #     reconfiguration_freq=1,
        #     robot_init_high=True,
        #     random_rotation=True,
        #     workspace_x_bounds=args.x_bounds,
        #     workspace_y_bounds=args.y_bounds,
        #     collision_free_placement=True,
        #     include_all_cameras=args.include_all_cameras,
        #     # extra flags
        #     max_episode_steps=1e6
        # )
        # env.reset()
        # while True:
        #     env.render()
        #     action = np.zeros(env.action_space.shape)
        #     action[:] = env.agent.robot.get_qpos().reshape(-1)[:len(action)]
        #     if 'stick' not in args.robot and 'close' not in args.robot:
        #         action[-1] = -1 if 'xarm6' in args.robot else 1
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     if terminated or truncated:
        #         env.reset()
        return env

    def _make_engine(self, env: gym.Env) -> Tuple[AtomicPrimitives, TrajectoryEngine]:
        initial_qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy()
        primitives = self._primitives_factory(
            env,
            robot_name=self.args.robot,
            initial_qpos=initial_qpos,
            render_callback=env.render if not self.headless else None,
            interpolate_steps=self.args.interpolate_steps,
        )

        place_bounds = (
            self.args.x_bounds[0],
            self.args.x_bounds[1],
            self.args.y_bounds[0],
            self.args.y_bounds[1],
        )
        cfg = TrajectoryConfig(
            waypoint_interval=self.args.waypoint_interval,
            place_bounds=place_bounds,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
        )
        engine = self._engine_factory(env, primitives=primitives, config=cfg)
        return primitives, engine

    @staticmethod
    def default_supervisor(
        env: gym.Env, primitives: AtomicPrimitives, engine: TrajectoryEngine
    ) -> Optional[PlannedAction]:
        plans = engine.plan_episode(steps=1)
        if not plans:
            return None
        return plans[0]

    def _run_push_only(
        self,
        engine: TrajectoryEngine,
        primitives: AtomicPrimitives,
        action_type_sequence: List[str],
        progress: Optional[Any] = None,
        active_task_id: Optional[Any] = None,
    ) -> bool:
        # Sample a target object (distractor)
        target = engine._sample_target_actor()
        if target is None:
            return False

        params = primitives.sample_push_parameters(target)
        trajectory = primitives.generate_push_trajectory(params)
        result = primitives.execute_trajectory(
            trajectory,
            engine.kinematics,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            target_actor=None,
            progress=progress,
            task_id=active_task_id,
        )
        action_type_sequence.append("push")
        episode_success = result.success and result.steps_taken > 0
        if not episode_success:
            logger.warning(
                f"    [Fail, keeping anyway] Push Only: success={result.success}, steps={result.steps_taken}, msg={result.message}"
            )
        return episode_success

    def _run_pick_and_place(
        self,
        engine: TrajectoryEngine,
        primitives: AtomicPrimitives,
        action_type_sequence: List[str],
        progress: Optional[Any] = None,
        active_task_id: Optional[Any] = None,
    ) -> bool:
        # Step 1: Pick
        target = engine._sample_target_actor()
        if target is None:
            return False

        # PICK
        params = primitives.sample_pick_parameters(target)
        if params is None:
            return False

        trajectory = primitives.generate_pick_trajectory(params)
        primitives.monitor.enabled = False
        result = primitives.execute_trajectory(
            trajectory,
            engine.kinematics,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            target_actor=target,
            progress=progress,
            task_id=active_task_id,
        )
        primitives.monitor.enabled = True
        action_type_sequence.append("pick")
        pick_success = result.success and result.steps_taken > 0
        if not pick_success:
            print(
                f"    [Fail] Pick and Place (Pick): success={result.success}, steps={result.steps_taken}, msg={result.message}"
            )

        # Step 2: Place (execute even if pick failed)
        primitives._held_object = target
        place_params = primitives.sample_place_parameters()
        trajectory = primitives.generate_place_trajectory(place_params)
        result = primitives.execute_trajectory(
            trajectory,
            engine.kinematics,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            target_actor=None,
            progress=progress,
            task_id=active_task_id,
        )
        action_type_sequence.append("place")
        place_success = result.success and result.steps_taken > 0
        if not place_success:
            logger.warning(f"[Partial Fail] Place failed: msg={result.message}")
        return pick_success and place_success

    def _run_tool_push(
        self,
        engine: TrajectoryEngine,
        primitives: AtomicPrimitives,
        action_type_sequence: List[str],
        progress: Optional[Any] = None,
        active_task_id: Optional[Any] = None,
    ) -> bool:
        # Step 1: Pick Tool
        tool = None
        actors = engine.available_actors
        for actor in actors:
            name = actor.name.lower()
            if "tool" in name or "peg" in name:
                tool = actor
                break

        if tool is None:
            tool = engine._sample_target_actor()

        if tool is None:
            return False

        # PICK TOOL
        params = primitives.sample_pick_parameters(tool)
        if params is None:
            return False

        trajectory = primitives.generate_pick_trajectory(params)
        result = primitives.execute_trajectory(
            trajectory,
            engine.kinematics,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            target_actor=tool,
            progress=progress,
            task_id=active_task_id,
        )
        action_type_sequence.append("pick_tool")
        pick_success = result.success and result.steps_taken > 0
        if not pick_success:
            logger.error(
                f"[Fail] Tool Push (Pick Tool): success={result.success}, steps={result.steps_taken}, msg={result.message}"
            )

        # Step 2: Push Object with Tool
        primitives._held_object = tool

        target = engine._sample_target_actor(exclude=[tool])
        if target is None:
            return False

        # PUSH TARGET WITH TOOL
        push_params = primitives.sample_push_parameters(target)
        trajectory = primitives.generate_push_trajectory(push_params)
        result = primitives.execute_trajectory(
            trajectory,
            engine.kinematics,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            target_actor=None,
            progress=progress,
            task_id=active_task_id,
        )
        action_type_sequence.append("tool_push")
        push_success = result.success and result.steps_taken > 0
        if not push_success:
            logger.warning(f"[Partial Fail] Push failed: msg={result.message}")
        return pick_success and push_success

    def collect(self) -> None:
        wrapped_env = DataCollectionWrapper(self._env_factory(self.args), self.args)
        out_root = os.path.join(self.args.out_dir, self.args.env_id)
        datasets: Dict[str, ManiSkillTrajectoryDataset] = {}
        per_action_counts: Dict[str, int] = {}
        per_action_successes: Dict[str, int] = {}
        saved = 0

        # Per-action targets: legacy (total cap) vs per-category cap
        is_stick = "stick" in self.args.robot or "closed" in self.args.robot
        all_available_actions = (
            ["push_only"] if is_stick else ["pick_and_place", "tool_push", "push_only"]
        )
        if self.args.play_actions is not None:
            requested = [
                a.strip() for a in self.args.play_actions.split(",") if a.strip()
            ]
            invalid = [a for a in requested if a not in all_available_actions]
            if invalid:
                raise ValueError(
                    f"play_actions contains invalid or unavailable actions for robot {self.args.robot}: {invalid}. "
                    f"Available: {all_available_actions}"
                )
            if not requested:
                raise ValueError(
                    "play_actions must not be empty; use at least one of: "
                    + ", ".join(all_available_actions)
                )
            action_categories = requested
        else:
            action_categories = all_available_actions
        if self.args.play_total_cap_only:
            per_action_target = None
            total_target = self.args.num_trajectories
        else:
            per_action_target = (
                self.args.num_trajectories_per_action
                if self.args.num_trajectories_per_action is not None
                else self.args.num_trajectories
            )
            total_target = len(action_categories) * per_action_target

        # Resume: count existing trajectories per action and optionally confirm
        for action in action_categories:
            action_root = os.path.join(out_root, action)
            n = count_trajectories_in_root(action_root)
            if n > 0:
                per_action_counts[action] = n
                datasets[action] = self._dataset_factory(action_root)
                if action not in per_action_successes:
                    per_action_successes[action] = 0  # unknown for existing
        saved = sum(per_action_counts.get(c, 0) for c in action_categories)

        if saved > 0:
            summary = [
                f"Output path: [bold]{out_root}[/bold]",
                "Existing per action: "
                + ", ".join(
                    f"[cyan]{a}={per_action_counts.get(a, 0)}[/cyan]"
                    for a in action_categories
                )
                + f" (total={saved})",
            ]
            if per_action_target is not None:
                remaining_per = [
                    max(0, per_action_target - per_action_counts.get(c, 0))
                    for c in action_categories
                ]
                summary.append(
                    f"Target per action: {per_action_target}. Will collect: "
                    + ", ".join(
                        f"{c}: {r} more"
                        for c, r in zip(action_categories, remaining_per)
                    )
                )
            else:
                summary.append(
                    f"Target total: {total_target}. Will collect [bold]{max(0, total_target - saved)}[/bold] more."
                )
            if not ask_resume_confirm(summary):
                print("[yellow]Aborted.[/yellow]")
                sys.exit(0)
            print("[green]Resuming collection.[/green]")

        def is_collection_done() -> bool:
            if per_action_target is not None:
                return all(
                    per_action_counts.get(c, 0) >= per_action_target
                    for c in action_categories
                )
            return saved >= total_target

        # Setup Rich Progress Bars
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            expand=True,
        )
        total_task = progress.add_task(
            "[bold cyan]Total Progress", total=total_target, completed=saved
        )
        active_task = progress.add_task(
            "[yellow]Active Execution", total=100, visible=False
        )
        action_tasks: Dict[str, Any] = {}
        for action in action_categories:
            action_total = per_action_target if per_action_target is not None else None
            action_tasks[action] = progress.add_task(
                f"[white]Action: {action}[/white]",
                total=action_total,
                completed=per_action_counts.get(action, 0),
            )

        # Logging Queue
        logs = deque(maxlen=10)
        setup_dashboard_logging(logs)
        current_action: Dict[str, Optional[str]] = {"type": None}

        def make_dashboard():
            table = Table.grid(expand=True)
            table.add_column()
            header = f"[bold magenta]Play Data Collection[/bold magenta] | Env: [green]{self.args.env_id}[/green] | Robot: [green]{self.args.robot}[/green] | Device: [yellow]{self.args.sys_device}[/yellow]"
            current = current_action.get("type") or "—"
            action_line = f"[bold]Current action:[/bold] [yellow]{current}[/yellow]"
            # Action Stats (counts per type)
            stats_parts = []
            for action_type, count in per_action_counts.items():
                successes = per_action_successes.get(action_type, 0)
                sr = (successes / count * 100) if count > 0 else 0
                stats_parts.append(
                    f"[cyan]{action_type}: {count} (SR: {sr:.1f}%)[/cyan]"
                )
            stats_str = (
                " | ".join(stats_parts) if stats_parts else "No trajectories saved yet"
            )
            table.add_row(
                Panel(Group(header, action_line, stats_str), border_style="magenta")
            )
            table.add_row(progress)
            log_panel = Panel(
                "\n".join(logs),
                title="[bold]Logs[/bold]",
                border_style="white",
                height=12,
            )
            table.add_row(log_panel)
            return table

        try:
            with Live(make_dashboard(), refresh_per_second=4) as live:
                prev_saved = 0
                current_attempt = 0
                while not is_collection_done():
                    current_attempt += 1
                    obs, _ = wrapped_env.reset(
                        seed=int(torch.randint(0, 2**31, (1,)).item()),
                        options={
                            "reconfigure": (saved % 3 == 0 and prev_saved != saved)
                            or (current_attempt % 10 == 0)
                        },
                    )
                    # obs, _ = wrapped_env.reset()

                    prev_saved = saved
                    wrapped_env.set_cached_obs(obs)
                    buffers = _init_buffers_from_obs(obs)
                    wrapped_env.set_buffers(buffers)

                    primitives, engine = self._make_engine(wrapped_env)
                    robot_infos = _extract_robot_infos(wrapped_env)

                    # --- 1. Sample Macro-Action ---
                    if per_action_target is not None:
                        need_categories = [
                            c
                            for c in action_categories
                            if per_action_counts.get(c, 0) < per_action_target
                        ]
                        if not need_categories:
                            break
                        macro_action = need_categories[saved % len(need_categories)]
                    else:
                        macro_action = action_categories[saved % len(action_categories)]

                    current_action["type"] = macro_action
                    logs.append(f"[blue]Attempting {macro_action}...[/blue]")
                    live.update(make_dashboard())

                    action_type_sequence: List[str] = []
                    episode_success = True
                    if not self.headless:
                        wrapped_env.env.render()
                    # --- 2. Implement Macro-Action Logic ---
                    try:
                        if macro_action == "push_only":
                            episode_success = self._run_push_only(
                                engine,
                                primitives,
                                action_type_sequence,
                                progress=progress,
                                active_task_id=active_task,
                            )
                        elif macro_action == "pick_and_place":
                            episode_success = self._run_pick_and_place(
                                engine,
                                primitives,
                                action_type_sequence,
                                progress=progress,
                                active_task_id=active_task,
                            )
                        elif macro_action == "tool_push":
                            episode_success = self._run_tool_push(
                                engine,
                                primitives,
                                action_type_sequence,
                                progress=progress,
                                active_task_id=active_task,
                            )
                    except Exception:
                        logs.append(
                            f"[red]✘[/red] Exception during {macro_action}, skipping."
                        )
                        continue

                    if wrapped_env._last_action is not None:
                        for _ in range(3):
                            wrapped_env.step(wrapped_env._last_action)

                    progress.update(active_task, visible=False)

                    if (
                        not action_type_sequence
                        or len(buffers.metadata_arrays["actions"]) == 0
                    ):
                        logs.append(
                            f"[yellow]⚠[/yellow] Empty trajectory for {macro_action}, skipping."
                        )
                        continue

                    primary_action_type = macro_action
                    if primary_action_type not in datasets:
                        datasets[primary_action_type] = self._dataset_factory(
                            os.path.join(out_root, primary_action_type)
                        )
                        per_action_counts[primary_action_type] = 0

                    video_streams_np = {
                        k: np.array(v)
                        for k, v in buffers.video_streams.items()
                        if len(v) > 0
                    }
                    metadata_np = {
                        k: np.array(v)
                        for k, v in buffers.metadata_arrays.items()
                        if len(v) > 0
                    }
                    metadata_np["task_description"] = task_descriptions.get(
                        self.args.env_id, self.args.env_id
                    )
                    metadata_np["success"] = episode_success
                    metadata_np["num_steps"] = int(
                        len(buffers.metadata_arrays["actions"])
                    )
                    metadata_np["action_type"] = primary_action_type
                    metadata_np["action_type_sequence"] = np.array(
                        action_type_sequence, dtype=object
                    )

                    traj_data = TrajectoryData(
                        success=episode_success,
                        video_streams=video_streams_np,
                        metadata=metadata_np,
                    )
                    traj_id = f"{per_action_counts[primary_action_type]:06d}"
                    datasets[primary_action_type].write_trajectory(traj_id, traj_data)
                    if robot_infos:
                        datasets[primary_action_type].save_robot_infos(robot_infos)

                    per_action_counts[primary_action_type] += 1
                    if episode_success:
                        per_action_successes[primary_action_type] = (
                            per_action_successes.get(primary_action_type, 0) + 1
                        )
                    saved += 1
                    current_attempt = 0

                    # Update progress bars
                    if primary_action_type not in action_tasks:
                        action_total = (
                            per_action_target if per_action_target is not None else None
                        )
                        action_tasks[primary_action_type] = progress.add_task(
                            f"[white]Action: {primary_action_type}", total=action_total
                        )
                    progress.update(total_task, advance=1)
                    progress.update(action_tasks[primary_action_type], advance=1)
                    if episode_success:
                        logs.append(
                            f"[green]✔[/green] Saved {primary_action_type} ({per_action_counts[primary_action_type]})"
                        )
                    else:
                        logs.append(
                            f"[red]✘[/red] Saved {primary_action_type} (Fail) ({per_action_counts[primary_action_type]})"
                        )
                    live.update(make_dashboard())
        finally:
            wrapped_env.close()

        if not is_collection_done():
            raise RuntimeError(
                f"Play collection incomplete: saved {saved}/{total_target} trajectories"
            )

        logger.info(
            f"Play collection complete: saved {saved} trajectories across {len(per_action_counts)} action buckets."
        )


def collect_play(args: Args) -> None:
    """Play data collection using callback-based supervisor and absolute joint-space actions."""
    args.env_id = "TableOnly-v2"
    PlayCollector(args).collect()


def main() -> None:
    args = tyro.cli(Args)
    assert args.robot in [
        "ur10e_stick",
        "panda_closed",
        "panda",
        "xarm6_robotiq",
        "xarm6_robotiq_closed",
    ]

    assert args.env_id in task_descriptions

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    console = Console()
    console.print("[bold magenta]Parsed arguments:[/bold magenta]")
    console.print(args)

    if args.mode == "ppo":
        collect_ppo(args)
    elif args.mode == "play":
        collect_play(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    import logging

    logging.getLogger("mani_skill ").setLevel(logging.ERROR)
    main()
