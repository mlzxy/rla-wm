"""
Atomic manipulation primitives for the play module.

Provides high-level Pick, Place, Push actions using geometry-based
grasp sampling and motion control.
"""

from datalib.src.play.mani_kinematics_util import Kinematics
import numpy as np
from .logging_util import get_logger
logger = get_logger(__name__)
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass, field, replace
import gymnasium as gym
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from enum import Enum
import random

import torch
from .geometry import GeometryLib
from .utils import (
    sapien_pose_to_numpy,
    pose_to_matrix,
    matrix_to_pose,
    get_actor_world_pose,
)


class PushType(Enum):
    """Types of push trajectories."""

    STRAIGHT = "straight"
    CURVE = "curve"
    WIGGLE = "wiggle"


@dataclass
class PrimitiveResult:
    """Result of a primitive action."""

    success: bool
    action_name: str
    steps_taken: int
    message: str = ""
    actions: List[np.ndarray] = field(default_factory=list)
    poses: List[np.ndarray] = field(default_factory=list)
    trajectory_steps: List["PrimitiveStep"] = field(
        default_factory=list
    )  # Executed steps with metadata


@dataclass
class PrimitiveStep:
    """
    A single step in a generated trajectory.

    Represents a waypoint with pose, gripper state, and metadata.
    Used for offline trajectory generation and validation before execution.
    """

    position: np.ndarray  # [3] position in world frame
    quaternion: np.ndarray  # [4] quaternion in xyzw format
    gripper: float = 1.0  # 1.0 = open, -1.0 = closed
    phase: str = "move"  # e.g. "approach", "grasp", "lift", "retract"
    is_interaction: bool = False  # True for key waypoints (grasp, contact, etc.)
    joints: Optional[np.ndarray] = (
        None  # Optional joint positions for joint-space control
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotConfig:
    """Robot-specific configuration parameters."""

    name: str
    ee_link_name: str
    approach_height: float = 0.2
    lift_height: float = 0.2
    home_qpos: Optional[np.ndarray] = None
    # Action space config (e.g., gripper index)
    gripper_idx: int = -1  # Index in action vector; -1 = no gripper
    # Gripper convention: trajectory uses 1=open, -1=close. 
    # These fields map trajectory values to robot action values.
    gripper_open_val: float = 1.0
    gripper_close_val: float = -1.0
    # Additive height modifier for push task. Base push Z is 0.05 m; this offset tunes per-robot.
    push_height_offset: float = 0.0
    z_min: float = 0.004
    max_grasp_width: float = 0.0  # 0 means ignore
    # XY workspace bound (x_min, x_max, y_min, y_max) for reachability and reliable execution.
    # All planned/generated trajectory waypoints are clipped to this bound.
    xy_bounds: Tuple[float, float, float, float] = (-0.4, 0.4, -0.4, 0.4)

    def get_gripper_action(self, gripper_value: float) -> float:
        """
        Map trajectory gripper value (1=open, -1=close) to the action value for this robot.
        No gripper (gripper_idx < 0) returns 0.0; otherwise uses gripper_invert for convention.
        """
        if self.gripper_idx < 0:
            return 0.0
        # Linear interpolation from trajectory range [-1, 1] to robot range [close_val, open_val]
        # f(x) = close_val + (x + 1) / 2 * (open_val - close_val)
        return self.gripper_close_val + (gripper_value + 1.0) / 2.0 * (
            self.gripper_open_val - self.gripper_close_val
        )


@dataclass
class ExecutionMonitor:
    """Monitors trajectory execution for deviations."""

    enabled: bool = True
    pos_threshold: float = 0.05  # 5cm
    rot_threshold: float = 0.2  # ~11 degrees (radians)

    def check_deviation(
        self, current_pos, current_quat, target_step: PrimitiveStep
    ) -> bool:
        """
        Check if current pose deviates too much from target step.
        Returns True if deviation is within limits, False if failed.
        """
        if not self.enabled:
            return True

        # Position error
        pos_err = np.linalg.norm(current_pos - target_step.position)
        if pos_err > self.pos_threshold:
            return False

        # Rotation error (quaternion distance)
        # q_diff = q1 * q2_inv
        # angle = 2 * atan2(norm(vec), w)
        # Simpler: 1 - |q1.dot(q2)| is roughly related to angle
        # 1 - |dot| < threshold
        dot = np.abs(np.dot(current_quat, target_step.quaternion))
        if dot < (1.0 - self.rot_threshold):  # Rough approximation
            return False

        return True


def robot_uid_to_short_name(full_uid: str) -> str:
    """
    Convert full robot UID (e.g. "xarm6_robotiq_closed", "ur10e_stick") to short name for config lookup.
    Short names match keys in ROBOT_CONFIGS: "panda", "xarm6", "ur10e", etc.
    """
    if not full_uid or not isinstance(full_uid, str):
        return "panda"
    # Explicit mapping for known agent UIDs (from robots.py / ManiSkill)
    _UID_TO_SHORT = {
        "panda": "panda",
        "panda_closed": "panda",
        "xarm6_robotiq": "xarm6",
        "xarm6_robotiq_closed": "xarm6",
        "ur10e_stick": "ur10e",
    }
    if full_uid in _UID_TO_SHORT:
        return _UID_TO_SHORT[full_uid]
    # Fallback: first segment before underscore (e.g. "xarm6_robotiq_close" -> "xarm6")
    return full_uid.split("_")[0]


# Predefined robot configurations
ROBOT_CONFIGS = {
    "panda": RobotConfig(
        name="panda",
        ee_link_name="panda_hand",
        approach_height=-0.06,  # 0.15,
        lift_height=0.05,
        max_grasp_width=0.06,
        gripper_idx=7,  # 7th element in action is gripper
        xy_bounds=(-0.42, 0.17, -0.5, 0.5),
        push_height_offset=0.01,
    ),
    "xarm6": RobotConfig(
        name="xarm6",
        ee_link_name="link_eef",
        approach_height=-0.06,  # 0.12,
        lift_height=0.0,
        max_grasp_width=0.06,
        gripper_idx=6,
        gripper_open_val=0.0,  # XArm6/Robotiq: -1 = open, 1 = close
        gripper_close_val=0.81,
        xy_bounds=(-0.39, 0.17, -0.5, 0.5),
        push_height_offset=0.01,
    ),
    "ur10e": RobotConfig(
        name="ur10e",
        ee_link_name="tcp",
        approach_height=0.12,
        lift_height=0.10,
        max_grasp_width=0.0,
        gripper_idx=-1,  # no gripper (stick end-effector)
        xy_bounds=(-0.34, 0.42, -1.0, 1.0),
        push_height_offset=0.01,
    ),
}


class AtomicPrimitives:
    """
    High-level manipulation primitives for robot interaction.

    Provides Pick, Place, Push, and Home actions that combine
    geometry-based planning with motion execution.
    """
    

    # Default parameters (Fallbacks if no config)
    DEFAULT_PUSH_DISTANCE = 0.10
    # Radius (m) for selecting a target object when sampling push direction (XY only).
    PUSH_TARGET_RADIUS = 0.2
    # Z height randomness: ± this value (meters) added to trajectory heights for pick/place/push
    Z_RANDOM_RANGE = 0.01
    # Pick: hold at pregrasp with gripper open before closing (reduces spline overshoot vs close)
    PREGRASP_HOLD_OPEN_STEPS = 10
    # Pick: number of in-place steps with gripper closed for a firm grasp before lift
    GRASP_HOLD_STEPS = 20

    # Down orientation: 180 deg around X axis (EEF pointing down in world frame)
    # In xyzw format for scipy: [sin(90), 0, 0, cos(90)] = [1, 0, 0, 0]
    DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(
        self,
        env,
        robot_name: str = "panda",
        initial_qpos: Optional[np.ndarray] = None,
        max_steps_per_waypoint: int = 100,
        convergence_threshold: float = 0.02,
        render_callback: Optional[callable] = None,
        interpolate_steps: int = 5,
        enable_path_deviation_check: bool = True,
    ):
        """
        Initialize the atomic primitives.

        Args:
            env: ManiSkill environment
            robot_name: Name of the robot to load config for
            initial_qpos: Initial joint positions (home pose)
            grasp_sampler: Custom grasp sampler (uses default if None)
            max_steps_per_waypoint: Maximum steps to reach each waypoint
            convergence_threshold: Distance threshold for waypoint convergence
            render_callback: Optional callback to run after each step (e.g., env.render)
        """
        self.env = env
        self.robot_name = robot_name
        self.render_callback = render_callback

        # Resolve full UID to short name for config (e.g. "xarm6_robotiq_closed" -> "xarm6")
        short_name = robot_uid_to_short_name(robot_name)
        if short_name not in ROBOT_CONFIGS:
            logger.warning(
                f"Robot {robot_name} (short: {short_name}) not found in configs. Using default."
            )
            self.config = ROBOT_CONFIGS["panda"]
        else:
            self.config = ROBOT_CONFIGS[short_name]

        if "close" in robot_name:
            self.config = replace(self.config, gripper_idx=-1)

        # Set home_qpos from initial_qpos if provided, else from current env robot state
        if initial_qpos is not None:
            self.config = replace(self.config, home_qpos=initial_qpos)
        else:
            qpos = self.get_qpos()
            self.config = replace(self.config, home_qpos=qpos.copy())
        # Record initial TCP pose for home: trajectory is current pose -> this pose
        self._initial_tcp_pose = self.get_tcp_pose()

        self.points_lib = GeometryLib()

        # Execution Monitor
        self.monitor = ExecutionMonitor()
        # Optional path deviation / execution monitor check
        # When disabled, trajectories will execute without deviation-based early aborts.
        self.monitor.enabled = enable_path_deviation_check
        self.last_execution_result = "SUCCESS"

        # Constants
        self.approach_height = self.config.approach_height
        self.lift_height = self.config.lift_height
        self.push_height_offset = self.config.push_height_offset

        self.max_steps = max_steps_per_waypoint
        self.threshold = convergence_threshold
        self.interpolate_steps = interpolate_steps

        # Track current held object
        self._held_object = None

        # Recording buffers
        self._current_actions: List[np.ndarray] = []
        self._current_poses: List[np.ndarray] = []

        # Track last target Euler for continuity
        self._last_target_euler: Optional[np.ndarray] = None

        # XY workspace bounds from config (for clipping trajectories)
        self._xy_bounds = self.config.xy_bounds  # (x_min, x_max, y_min, y_max)
        self._z_min = self.config.z_min

    def _clip_pos(self, pos: np.ndarray) -> np.ndarray:
        """Clip position to the robot's workspace bound (XY and Z)."""
        x_min, x_max, y_min, y_max = self._xy_bounds
        out = np.array(pos, dtype=np.float32, copy=True)
        out[0] = np.clip(out[0], x_min, x_max)
        out[1] = np.clip(out[1], y_min, y_max)
        out[2] = np.maximum(out[2], self._z_min)
        return out

    def _within_xy_bounds(self, pos: np.ndarray) -> bool:
        """Return True if position is within the robot's XY workspace bound."""
        x_min, x_max, y_min, y_max = self._xy_bounds
        return (x_min <= pos[0] <= x_max) and (y_min <= pos[1] <= y_max)

    def _get_pushable_actors(self) -> List:
        """
        Get all pushable (graspable) actors in the scene.
        Mirrors UnifiedPlay._find_all_objects logic for consistency.
        """
        actors: List[Any] = []
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "obj") and unwrapped.obj is not None:
            actors.append(unwrapped.obj)
        if hasattr(unwrapped, "distractors"):
            actors.extend(unwrapped.distractors)
        if hasattr(unwrapped, "actors"):
            actors.extend(unwrapped.actors)
        if not actors:
            scene = unwrapped.scene
            for actor in scene.get_all_actors():
                name = actor.name.lower()
                if any(
                    x in name
                    for x in [
                        "panda",
                        "xarm",
                        "robot",
                        "ground",
                        "table",
                        "camera",
                        "goal",
                        "workspace",
                    ]
                ):
                    continue
                actors.append(actor)
        return list(set(actors))

    @property
    def is_holding(self) -> bool:
        """Check if robot is currently holding an object."""
        return self._held_object is not None

    def _get_down_with_random_yaw(self, angle_range: float = np.pi / 6) -> np.ndarray:
        """
        Get a downward-facing orientation with random yaw variation.

        Args:
            angle_range: Max random yaw deviation in radians (default ±30 deg)

        Returns:
            Quaternion [4] in xyzw format
        """
        yaw = np.random.uniform(-angle_range, angle_range)

        # Compose: DOWN_QUAT * Z_rotation(yaw)
        # DOWN_QUAT is 180 deg X rotation
        r_down = R.from_quat(self.DOWN_QUAT)
        r_yaw = R.from_euler("z", yaw)
        r_combined = r_down * r_yaw

        return r_combined.as_quat().astype(np.float32)

    def _interpolate_pose(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        end_pos: np.ndarray,
        end_quat: np.ndarray,
        fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate between two poses.

        Args:
            start_pos: Start position [3]
            start_quat: Start quaternion [4]
            end_pos: End position [3]
            end_quat: End quaternion [4]
            fraction: Interpolation fraction [0, 1]

        Returns:
            Tuple of (interpolated_pos, interpolated_quat)
        """
        # Linear interpolation for position
        interp_pos = start_pos + (end_pos - start_pos) * fraction

        # Spherical linear interpolation for rotation
        key_times = [0, 1]
        key_rots = R.from_quat([start_quat, end_quat])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([fraction])
        interp_quat = interp_rot.as_quat()[0]

        return interp_pos, interp_quat

    def _unwrap_euler(
        self, target_euler: np.ndarray, reference_euler: np.ndarray = None
    ) -> np.ndarray:
        """
        Unwrap Euler angles to ensure continuity with reference.

        This prevents sudden jumps when Euler angles cross ±π boundaries.
        For example, if reference is [0.1, 0, 3.0] and target is [0.1, 0, -3.1],
        we unwrap target to [0.1, 0, 3.18] (adding 2π) for smooth motion.

        Args:
            target_euler: Target Euler angles [3] in XYZ order
            reference_euler: Reference Euler angles [3] to be continuous with.
                             If None, uses self._last_target_euler.

        Returns:
            Unwrapped Euler angles [3]
        """
        if reference_euler is None:
            reference_euler = self._last_target_euler

        if reference_euler is None:
            # No previous reference, can't unwrap
            return target_euler

        unwrapped = target_euler.copy()
        for i in range(3):
            diff = target_euler[i] - reference_euler[i]
            # If diff is more than π, subtract 2π
            # If diff is less than -π, add 2π
            if diff > np.pi:
                unwrapped[i] -= 2 * np.pi
            elif diff < -np.pi:
                unwrapped[i] += 2 * np.pi

        return unwrapped

    def _generate_interpolated_path(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        end_pos: np.ndarray,
        end_quat: np.ndarray,
        resolution: float = 0.03,
        gripper: float = 1.0,
        phase: str = "move",
        is_interaction: bool = False,
    ) -> List[PrimitiveStep]:
        """
        Generate a list of interpolated poses between start and end.

        Waypoints are spaced by `resolution` (default 3cm). Uses linear
        interpolation for position and SLERP for rotation.

        Args:
            start_pos: Start position [3]
            start_quat: Start quaternion [4] xyzw
            end_pos: End position [3]
            end_quat: End quaternion [4] xyzw
            resolution: Distance between waypoints in meters (default 0.03)
            gripper: Gripper state for all generated steps
            phase: Phase label for metadata
            is_interaction: Whether these are interaction waypoints

        Returns:
            List of PrimitiveStep from start to end (inclusive)
        """
        distance = np.linalg.norm(end_pos - start_pos)

        # Calculate number of steps (at least 2: start and end)
        num_steps = max(2, int(np.ceil(distance / resolution)) + 1)

        steps = []
        for i in range(num_steps):
            fraction = i / (num_steps - 1) if num_steps > 1 else 1.0
            interp_pos, interp_quat = self._interpolate_pose(
                start_pos, start_quat, end_pos, end_quat, fraction
            )
            interp_pos = self._clip_pos(interp_pos)
            # Mark first and last as interaction if specified only for last
            step_is_interaction = is_interaction and (i == num_steps - 1)

            steps.append(
                PrimitiveStep(
                    position=interp_pos.astype(np.float32),
                    quaternion=interp_quat.astype(np.float32),
                    gripper=gripper,
                    phase=phase,
                    is_interaction=step_is_interaction,
                )
            )

        return steps

    def _interpolate_joint_path(
        self,
        start_qpos: np.ndarray,
        end_qpos: np.ndarray,
        num_steps: int = 150,
    ) -> List[np.ndarray]:
        """
        Interpolate a smooth joint-space path from start to end using cubic spline per joint.

        Args:
            start_qpos: Start joint configuration [dof]
            end_qpos: End joint configuration [dof]
            num_steps: Number of waypoints (default 150 for dense, smooth motion)

        Returns:
            List of qpos arrays from start to end (inclusive).
        """
        start_qpos = np.asarray(start_qpos, dtype=np.float64)
        end_qpos = np.asarray(end_qpos, dtype=np.float64)
        if start_qpos.shape != end_qpos.shape:
            raise ValueError(
                f"start_qpos shape {start_qpos.shape} != end_qpos shape {end_qpos.shape}"
            )
        dof = start_qpos.size
        # Two waypoints: t=0 and t=1
        t_waypoints = np.array([0.0, 1.0])
        q_waypoints = np.stack([start_qpos.ravel(), end_qpos.ravel()], axis=0)
        # Cubic spline per joint over t in [0, 1]
        t_query = np.linspace(0.0, 1.0, num_steps)
        interp_qpos_list = []
        for j in range(dof):
            spline = CubicSpline(t_waypoints, q_waypoints[:, j])
            interp_qpos_list.append(spline(t_query))
        # (num_steps, dof)
        interp_qpos = np.stack(interp_qpos_list, axis=1).astype(np.float32)
        return [interp_qpos[i] for i in range(num_steps)]

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current tool center point pose."""
        agent = self.env.unwrapped.agent
        tcp_pose = agent.tcp.pose
        pos, quat = sapien_pose_to_numpy(tcp_pose)
        return pos.astype(np.float32), quat.astype(np.float32)

    def get_qpos(self) -> np.ndarray:
        """Get current robot joint configuration (qpos)."""
        agent = self.env.unwrapped.agent
        qpos = agent.robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        return np.asarray(qpos, dtype=np.float32).reshape(-1)

    def _get_root_pose_euler(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world pose (pos, quat_xyzw) to root frame pos and Euler angles [XYZ].

        Uses ManiSkill's Pose struct and internal transformation utilities
        for robust coordinate frame conversion.
        """
        from mani_skill.utils.structs import Pose
        from mani_skill.utils.geometry.rotation_conversions import (
            matrix_to_euler_angles,
            quaternion_to_matrix,
        )

        agent = self.env.unwrapped.agent
        base_pose = Pose.create(agent.robot.pose)

        # Convert xyzw to wxyz for Pose.create
        quat_wxyz = world_quat[[3, 0, 1, 2]]
        target_pose_world = Pose.create_from_pq(world_pos, quat_wxyz)

        # Transform to root frame: base_inv * world
        target_pose_root = base_pose.inv() * target_pose_world

        # Extract numpy values
        pos_root = target_pose_root.p[0].cpu().numpy()

        # Convert quat to Euler XYZ
        # Using ManiSkill's utility for consistency with pd_ee_pose expectations
        euler_root = (
            matrix_to_euler_angles(quaternion_to_matrix(target_pose_root.q), "XYZ")[0]
            .cpu()
            .numpy()
        )

        return pos_root, euler_root

    def _step_robot(
        self, delta_pos: np.ndarray, delta_rot: np.ndarray = None, gripper: float = 1.0
    ) -> None:
        """
        Step the robot with delta position and rotation.

        Args:
            delta_pos: Position delta [3]
            delta_rot: Rotation delta as axis-angle [3], or None
            gripper: Gripper action (-1 close, 1 open)
        """
        action_dim = self.env.unwrapped.agent.action_space.shape[0]
        action = np.zeros(action_dim, dtype=np.float32)
        action[0:3] = delta_pos
        if delta_rot is not None:
            action[3 : min(6, action_dim)] = delta_rot[: min(3, action_dim - 3)]

        if action_dim > 6:
            action[6] = gripper

        # Record action and pose BEFORE reshape/step if possible, or just the composed action
        self._current_actions.append(action.copy())
        pos, quat = self.get_tcp_pose()
        self._current_poses.append((pos, quat))

        # Batch for ManiSkill
        action_batch = action.reshape(1, -1)
        self.env.step(action_batch)

        if self.render_callback:
            self.render_callback()

    def _wait_for_grasp(self, steps: int = 30, gripper: float = -1.0) -> None:
        """Wait while closing gripper."""
        for _ in range(steps):
            self._step_robot(np.zeros(3), None, gripper)

    def _wait_for_release(self, steps: int = 20, gripper: float = 1.0) -> None:
        """Wait while opening gripper."""
        for _ in range(steps):
            self._step_robot(np.zeros(3), None, gripper)

    def _get_gripper_width(self) -> float:
        """
        Get current gripper width in meters.
        Assumes 2-finger gripper where width = sum of joint positions (or similar).
        """
        # Get all joint positions
        qpos = self.get_qpos()

        width = 0.0
        if self.robot_name == "panda":
            # Panda has 2 fingers, normally indexes -1 and -2 in qpos
            # Open = 0.04 each -> width 0.08
            # Closed = 0.0
            if len(qpos) >= 9:
                width = qpos[-1] + qpos[-2]
        elif self.robot_name == "xarm6":
            # Many xArm6 gripper configs do not expose a reliable width signal.
            # We keep this branch explicit to avoid accidentally relying on noise.
            width = 0.0
        else:
            # Default/Fallback: assume last 2 joints sum to width
            if len(qpos) >= 2:
                width = qpos[-1] + qpos[-2]

        return width

    def check_grasp_success(self, actor=None) -> bool:
        return self.env.agent.is_grasping(actor)

    def generate_pick_trajectory(
        self,
        params: dict,
        start_pos: np.ndarray = None,
        start_quat: np.ndarray = None,
        resolution: float = 0.03,
    ) -> List[PrimitiveStep]:
        """
        Generate a complete pick trajectory without executing.

        Generates dense waypoints from current position through:
        approach -> pregrasp -> grasp (close gripper) -> lift

        Args:
            params: Parameters from sample_pick_parameters
            start_pos: Starting position (uses current TCP if None)
            start_quat: Starting quaternion (uses current TCP if None)
            resolution: Distance between waypoints (default 3cm)

        Returns:
            List of PrimitiveStep representing the complete trajectory
        """
        # Get start pose if not provided
        if start_pos is None or start_quat is None:
            start_pos, start_quat = self.get_tcp_pose()

        approach_pos = self._clip_pos(
            np.asarray(params["approach_pos"], dtype=np.float32)
        )
        approach_quat = params["approach_quat"]
        pregrasp_pos = self._clip_pos(
            np.asarray(params["pregrasp_pos"], dtype=np.float32)
        )
        pregrasp_quat = params.get("pregrasp_quat", approach_quat)
        obj_pos = params.get("obj_pos")

        # Small Z randomness for this trajectory (±Z_RANDOM_RANGE)
        delta_z = np.random.uniform(-self.Z_RANDOM_RANGE, self.Z_RANDOM_RANGE)
        approach_pos[2] += delta_z
        pregrasp_pos[2] += delta_z

        trajectory = []
        current_pos, current_quat = start_pos.copy(), start_quat.copy()

        # Transit height: high enough to clear approach, but do not force robot higher if already high
        # (approach + margin, or current Z if already above that, with a small floor so we don't transit too low)
        z_transit = max(0.1, approach_pos[2], start_pos[2])

        # Phase 0: Vertical retract only when current pose is below transit height
        # If robot is already decently high (e.g. above object), skip going higher
        if current_pos[2] < z_transit - 0.01:
            retract_pos = self._clip_pos(
                np.array([current_pos[0], current_pos[1], z_transit], dtype=np.float32)
            )
            steps = self._generate_interpolated_path(
                current_pos,
                current_quat,
                retract_pos,
                current_quat,
                resolution=resolution,
                gripper=1.0,
                phase="retract",
                is_interaction=False,
            )
            trajectory.extend(steps)
            current_pos = retract_pos

        # Phase 2: Transit to above approach at z_transit and rotate to approach orientation
        above_approach = self._clip_pos(
            np.array([approach_pos[0], approach_pos[1], z_transit], dtype=np.float32)
        )
        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            above_approach,
            approach_quat,
            resolution=resolution,
            gripper=1.0,
            phase="transit_to_approach",
            is_interaction=False,
        )
        trajectory.extend(steps[1:] if trajectory else steps)
        current_pos, current_quat = above_approach, approach_quat

        # Phase 3: Vertical Descent to approach pose
        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            approach_pos,
            approach_quat,
            resolution=0.005,  # Slower approach (5mm)
            gripper=1.0,
            phase="approach",
            is_interaction=False,
        )
        if steps:
            steps[-1].metadata["is_special"] = True  # Highlight final approach pose
        trajectory.extend(steps[1:])
        current_pos, current_quat = approach_pos, approach_quat

        # Phase 4: Move to pregrasp
        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            pregrasp_pos,
            pregrasp_quat,
            resolution=0.005,  # Slower pregrasp (5mm)
            gripper=1.0,
            phase="pregrasp",
            is_interaction=True,
        )
        if steps:
            steps[-1].metadata["is_special"] = True  # Highlight pregrasp pose
        trajectory.extend(steps[1:])
        current_pos, current_quat = pregrasp_pos, pregrasp_quat

        # Hold at pregrasp with gripper open so interpolator settles before close (no close-while-moving)
        for _ in range(self.PREGRASP_HOLD_OPEN_STEPS):
            trajectory.append(
                PrimitiveStep(
                    position=current_pos.copy(),
                    quaternion=current_quat.copy(),
                    gripper=1.0,
                    phase="pregrasp",
                    is_interaction=False,
                )
            )

        # Phase 5: Grasp (close gripper in place, firm)
        for _ in range(self.GRASP_HOLD_STEPS):
            trajectory.append(
                PrimitiveStep(
                    position=current_pos.copy(),
                    quaternion=current_quat.copy(),
                    gripper=-1.0,
                    phase="grasp",
                    is_interaction=True,
                )
            )

        # Phase 6: Strictly Vertical Lift
        # Ensure we clear the object vertically before moving elsewhere.
        # Add small randomness to lift height for trajectory diversity.
        base_lift_height = max(self.lift_height, 0.13)
        lift_height = base_lift_height + np.random.uniform(
            -self.Z_RANDOM_RANGE, self.Z_RANDOM_RANGE
        ) * 0.5
        lift_pos = self._clip_pos(current_pos.copy())
        lift_pos[2] += lift_height
        # Cap safe height
        lift_pos[2] = max(lift_pos[2], 0.2)

        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            lift_pos,
            current_quat,
            resolution=resolution / 2,
            gripper=-1.0,
            phase="lift",
            is_interaction=False,
        )
        trajectory.extend(steps[1:])

        return trajectory

    def generate_place_trajectory(
        self,
        params: dict,
        start_pos: np.ndarray = None,
        start_quat: np.ndarray = None,
        resolution: float = 0.03,
    ) -> List[PrimitiveStep]:
        """
        Generate a complete place trajectory without executing.

        Generates dense waypoints from current position through:
        above_target -> place -> release (open gripper) -> retract

        Args:
            params: Parameters from sample_place_parameters
            start_pos: Starting position (uses current TCP if None)
            start_quat: Starting quaternion (uses current TCP if None)
            resolution: Distance between waypoints (default 3cm)

        Returns:
            List of PrimitiveStep representing the complete trajectory
        """
        if start_pos is None or start_quat is None:
            start_pos, start_quat = self.get_tcp_pose()

        target_pos = self._clip_pos(np.asarray(params["target_pos"], dtype=np.float32))
        place_quat = params["place_quat"]

        # Small Z randomness for this trajectory (±Z_RANDOM_RANGE)
        target_pos[2] += np.random.uniform(-self.Z_RANDOM_RANGE, self.Z_RANDOM_RANGE)

        trajectory = []
        current_pos, current_quat = start_pos.copy(), start_quat.copy()

        # Phase 1: Move above target
        above_pos = self._clip_pos(target_pos.copy())
        above_pos[2] += self.approach_height

        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            above_pos,
            place_quat,
            resolution=resolution,
            gripper=-1.0,
            phase="above",
            is_interaction=False,
        )
        trajectory.extend(steps)
        current_pos, current_quat = above_pos, place_quat

        # Phase 2: Lower to place position
        place_pos = self._clip_pos(target_pos.copy())
        place_pos[2] += 0.02

        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            place_pos,
            current_quat,
            resolution=0.005,  # Slower place (5mm)
            gripper=-1.0,
            phase="place",
            is_interaction=True,
        )
        trajectory.extend(steps[1:])
        current_pos = place_pos

        # Phase 3: Release (open gripper)
        for i in range(8):
            trajectory.append(
                PrimitiveStep(
                    position=current_pos.copy(),
                    quaternion=current_quat.copy(),
                    gripper=1.0,
                    phase="release",
                    is_interaction=True,
                )
            )

        # Phase 4: Retract
        retract_pos = self._clip_pos(current_pos.copy())
        retract_pos[2] += self.lift_height

        steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            retract_pos,
            current_quat,
            resolution=resolution,
            gripper=1.0,
            phase="retract",
            is_interaction=False,
        )
        trajectory.extend(steps[1:])

        return trajectory

    def generate_push_trajectory(
        self,
        params: dict,
        start_pos: np.ndarray = None,
        start_quat: np.ndarray = None,
        resolution: float = 0.03,
    ) -> List[PrimitiveStep]:
        """
        Generate a complete push trajectory without executing.

        Generates dense waypoints from current position through:
        approach -> contact -> push_through -> retract

        Args:
            params: Parameters from sample_push_parameters
            start_pos: Starting position (uses current TCP if None)
            start_quat: Starting quaternion (uses current TCP if None)
            resolution: Distance between waypoints (default 3cm)

        Returns:
            List of PrimitiveStep representing the complete trajectory
        """
        if start_pos is None or start_quat is None:
            start_pos, start_quat = self.get_tcp_pose()

        push_quat = params["push_quat"]
        push_z = params.get("push_z", 0.05)
        # Support multi-step: list of {direction, distance [, push_type, curve_control]}
        steps_list = params.get("steps")
        if not steps_list:
            steps_list = [
                {
                    "direction": params["direction"],
                    "distance": params.get("distance", self.DEFAULT_PUSH_DISTANCE),
                    "push_type": params.get("push_type", PushType.STRAIGHT),
                    "curve_control": params.get("curve_control"),
                }
            ]

        trajectory = []
        current_pos, current_quat = start_pos.copy(), start_quat.copy()
        current_obj_pos = np.array(params["obj_pos"], dtype=np.float32)

        for step in steps_list:
            direction = np.asarray(step["direction"], dtype=np.float32)
            distance = step["distance"]
            push_type = step.get("push_type", PushType.STRAIGHT)
            curve_control = step.get("curve_control")

            # Approach (behind object along this step's direction)
            approach_dist = 0.08
            approach_pos = current_obj_pos.copy()
            approach_pos[0] -= direction[0] * approach_dist
            approach_pos[1] -= direction[1] * approach_dist
            approach_pos[2] = push_z
            approach_pos = self._clip_pos(approach_pos)

            path_steps = self._generate_interpolated_path(
                current_pos,
                current_quat,
                approach_pos,
                push_quat,
                resolution=0.02,
                gripper=-1.0,
                phase="approach",
                is_interaction=False,
            )
            trajectory.extend(path_steps)
            current_pos, current_quat = approach_pos.copy(), push_quat.copy()

            # Contact at current object position
            contact_pos = self._clip_pos(current_obj_pos.copy())
            contact_pos[2] = push_z
            path_steps = self._generate_interpolated_path(
                current_pos,
                current_quat,
                contact_pos,
                current_quat,
                resolution=0.02,
                gripper=-1.0,
                phase="contact",
                is_interaction=True,
            )
            trajectory.extend(path_steps[1:])
            current_pos = contact_pos.copy()

            # Push through
            if push_type == PushType.CURVE and curve_control is not None:
                control_pt = self._clip_pos(
                    np.asarray(curve_control, dtype=np.float32).copy()
                )
                end_pos = self._clip_pos(current_obj_pos + direction * distance)
                end_pos = np.asarray(end_pos, dtype=np.float32)
                end_pos[2] = push_z
                control_pt[2] = push_z
                curve_len = np.linalg.norm(end_pos - current_pos)

                # Finer resolution for curve push (5mm)
                push_resolution = 0.02
                n_pts = max(2, int(curve_len / push_resolution) + 1)

                for i in range(1, n_pts + 1):
                    t = i / n_pts
                    p0 = current_pos
                    p1 = control_pt
                    p2 = end_pos
                    pos_t = self._clip_pos(
                        (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
                    )
                    trajectory.append(
                        PrimitiveStep(
                            position=pos_t.astype(np.float32),
                            quaternion=current_quat.copy(),
                            gripper=-1.0,
                            phase="push",
                            is_interaction=True,
                        )
                    )
                current_pos = trajectory[-1].position.copy()
            elif push_type == PushType.WIGGLE:
                end_pos = self._clip_pos(current_obj_pos + direction * distance)
                end_pos = np.asarray(end_pos, dtype=np.float32)
                end_pos[2] = push_z

                # Finer resolution for wiggle (5mm)
                push_resolution = 0.02
                n_pts = max(2, int(distance / push_resolution) + 1)

                perp = np.array([-direction[1], direction[0], 0], dtype=np.float32)
                amplitude = 0.05
                freq = 2 * np.pi * 2
                for i in range(1, n_pts + 1):
                    t = i / n_pts
                    linear_pos = current_pos + direction * distance * t
                    offset = perp * amplitude * np.sin(freq * t)
                    pos_t = self._clip_pos(linear_pos + offset)
                    trajectory.append(
                        PrimitiveStep(
                            position=pos_t.astype(np.float32),
                            quaternion=current_quat.copy(),
                            gripper=-1.0,
                            phase="push",
                            is_interaction=True,
                        )
                    )
                current_pos = trajectory[-1].position.copy()
            else:
                push_pos = current_pos.copy()
                push_pos[0] += direction[0] * distance
                push_pos[1] += direction[1] * distance
                push_pos[2] = push_z
                push_pos = self._clip_pos(push_pos)
                path_steps = self._generate_interpolated_path(
                    current_pos,
                    current_quat,
                    push_pos,
                    current_quat,
                    resolution=0.02,
                    gripper=-1.0,
                    phase="push",
                    is_interaction=True,
                )
                trajectory.extend(path_steps[1:])
                current_pos = push_pos.copy()

            # Object moves to end of this push for next step (clip so next step stays in bounds)
            current_obj_pos = self._clip_pos(current_obj_pos + direction * distance)
            current_obj_pos[2] = params["obj_pos"][
                2
            ]  # keep original z for next contact

        # Single retract after all steps
        retract_pos = self._clip_pos(current_pos.copy())
        retract_pos[2] += 0.1
        path_steps = self._generate_interpolated_path(
            current_pos,
            current_quat,
            retract_pos,
            current_quat,
            resolution=resolution,
            gripper=1.0,
            phase="retract",
            is_interaction=False,
        )
        trajectory.extend(path_steps[1:])

        # Per-step Z randomness along push trajectory (±Z_RANDOM_RANGE)
        for step in trajectory:
            step.position[2] += np.random.uniform(
                -self.Z_RANDOM_RANGE, self.Z_RANDOM_RANGE
            )

        return trajectory

    def generate_home_trajectory(
        self,
        params: dict = None,
        start_pos: np.ndarray = None,
        start_quat: np.ndarray = None,
        resolution: float = 0.03,
        num_steps: int = 50,
    ) -> List[PrimitiveStep]:
        """
        Generate a home trajectory to the initial (recorded) configuration.

        When config.home_qpos is set (e.g. initial_qpos was passed at creation),
        generates a dense joint-space path from current qpos to home_qpos using
        cubic spline interpolation for smooth motion. Otherwise uses Cartesian
        interpolation to the initial TCP pose (IK per step at execution).

        Args:
            params: Optional dict with home_pos, home_quat, home_qpos (overrides)
            start_pos: Start position (default: current TCP; used only for Cartesian path)
            start_quat: Start quaternion (default: current TCP; used only for Cartesian path)
            resolution: Distance between waypoints for Cartesian path (default 3cm)
            num_steps: Number of joint-space waypoints when using home_qpos (default 150)

        Returns:
            List of PrimitiveStep
        """
        home_pos, home_quat = (
            self._initial_tcp_pose[0].copy(),
            self._initial_tcp_pose[1].copy(),
        )
        if params:
            if "home_pos" in params:
                home_pos = np.asarray(params["home_pos"], dtype=np.float32)
            if "home_quat" in params:
                home_quat = np.asarray(params["home_quat"], dtype=np.float32)
        home_pos = self._clip_pos(home_pos)
        gripper = -1.0 if self.is_holding else 1.0

        # Joint-space home: return to exact initial joint configuration with dense smooth path
        home_qpos = self.config.home_qpos
        if params and "home_qpos" in params:
            home_qpos = np.asarray(params["home_qpos"], dtype=np.float32)
        if home_qpos is not None:
            start_qpos = self.get_qpos()
            home_qpos = np.asarray(home_qpos, dtype=np.float32).ravel()
            if len(start_qpos) != len(home_qpos):
                raise ValueError(
                    f"get_qpos() length {len(start_qpos)} != home_qpos length {len(home_qpos)}"
                )
            qpos_path = self._interpolate_joint_path(
                start_qpos, home_qpos, num_steps=num_steps
            )
            steps = []
            for qpos in qpos_path:
                steps.append(
                    PrimitiveStep(
                        position=home_pos.copy(),
                        quaternion=home_quat.copy(),
                        gripper=gripper,
                        phase="home",
                        is_interaction=False,
                        joints=qpos,
                    )
                )
            return steps

        # Cartesian fallback when home_qpos was not provided at creation
        if start_pos is None or start_quat is None:
            start_pos, start_quat = self.get_tcp_pose()
        return self._generate_interpolated_path(
            start_pos,
            start_quat,
            home_pos,
            home_quat,
            resolution=resolution,
            gripper=gripper,
            phase="home",
            is_interaction=False,
        )

    def retreat(self, kinematics, distance: float = 0.1) -> PrimitiveResult:
        """
        Retreats the end-effector by a safe distance (usually UP).
        Disables execution monitoring during retreat to prevent recursive failure loops.
        """
        start_pos, start_quat = self.get_tcp_pose()
        target_pos = self._clip_pos(start_pos.copy())
        target_pos[2] = min(target_pos[2] + distance, 0.5)  # Cap height

        # Generate path
        steps = self._generate_interpolated_path(
            start_pos,
            start_quat,
            target_pos,
            start_quat,
            resolution=0.02,
            gripper=(-1.0 if self.is_holding else 1.0),
            phase="recovery",
            is_interaction=False,
        )

        # Execute with monitor disabled
        was_enabled = self.monitor.enabled
        self.monitor.enabled = False
        try:
            res = self.execute_trajectory(steps, kinematics)
        finally:
            self.monitor.enabled = was_enabled

        return res

    def execute_trajectory(
        self,
        trajectory: List[PrimitiveStep],
        kinematics: Kinematics,
        interpolate_steps: Optional[int] = None,
        time_warp_speed_bounds: Optional[Tuple[float, float]] = None,
        target_actor: Any = None,
        action_name: str = "trajectory",
        progress: Optional[Any] = None,
        task_id: Optional[Any] = None,
    ) -> PrimitiveResult:
        """
        Execute a pre-generated trajectory using IK-based joint control.

        Takes a list of PrimitiveStep and executes them sequentially,
        computing IK for each step and sending joint commands.

        Args:
            trajectory: List of PrimitiveStep to execute
            kinematics: Kinematics object for IK computation
            interpolate_steps: Number of interpolated points between waypoints
            time_warp_speed_bounds: If not None, (v_min, v_max) to resample trajectory with
                a random smooth speed profile for speed-diverse execution (Cartesian only).
            target_actor: Optional target actor for grasp validation
            action_name: Name of the action being executed (e.g., "pick", "place", "push", "tool_push")

        Returns:
            PrimitiveResult with execution statistics
        """
        from .interpolation import interpolate_trajectory

        self._current_actions = []
        self._current_poses = []

        agent = self.env.unwrapped.agent
        robot_name = agent.uid if hasattr(agent, "uid") else ""

        # Interpolate trajectory for smoother motion (Cartesian only).
        # Joint-space trajectories already have dense waypoints; do not run
        # interpolate_trajectory (it ignores step.joints and would corrupt them).
        all_joint_space = all(step.joints is not None for step in trajectory)
        has_grasp_phase = any(step.phase == "grasp" for step in trajectory)

        # Cache grasp context if this looks like a pick-type trajectory and we know the target.
        # This is used by check_grasp_success for pose-based grasp evaluation.
        self._last_grasp_context = None
        if has_grasp_phase and target_actor is not None:
            try:
                initial_obj_pos, initial_obj_quat = get_actor_world_pose(target_actor)
                self._last_grasp_context = {
                    "actor": target_actor,
                    "initial_obj_pos": np.asarray(initial_obj_pos, dtype=np.float32),
                }
            except Exception:
                # If pose retrieval fails, we simply fall back to width-based grasp checks.
                self._last_grasp_context = None
        if interpolate_steps is None:
            interpolate_steps = self.interpolate_steps

        if all_joint_space:
            dense_trajectory = trajectory
        elif has_grasp_phase:
            # Pick trajectory: spline would "anticipate" lift and rise during hold/grasp,
            # so close would happen after arm has moved up. Execute waypoints as-is so
            # gripper closes exactly at pregrasp (most down).
            dense_trajectory = trajectory
        elif interpolate_steps > 1 and len(trajectory) > 1:
            dense_trajectory = interpolate_trajectory(
                trajectory, num_interp=interpolate_steps
            )
        else:
            dense_trajectory = trajectory

        # Optional time-warp resampling for speed-diverse execution (Cartesian).
        # Safe-guards for grasp/pregrasp holds are handled inside time_warp (dwell runs preserved).
        if (
            time_warp_speed_bounds is not None
            and not all_joint_space
            and len(dense_trajectory) >= 2
        ):
            from .time_warp import resample_trajectory_with_speed_profile

            dense_trajectory_warped = resample_trajectory_with_speed_profile(
                dense_trajectory, speed_bounds=time_warp_speed_bounds
            )
            logger.info(
                f"Time-warped trajectory from {len(dense_trajectory)} -> {len(dense_trajectory_warped)} steps"
            )
            dense_trajectory = dense_trajectory_warped

        if progress is not None and task_id is not None:
            progress.update(task_id, total=len(dense_trajectory), completed=0, visible=True)

        # Get joint limits for normalization (if using normalized control)
        action_dim = agent.action_space.shape[0]
        active_joint_indices = self.env.agent.controller.controllers[
            "arm"
        ].active_joint_indices

        # Track last joint solution for IK continuity
        for step_i, step in enumerate(dense_trajectory):
            action = np.zeros(action_dim, dtype=np.float32)

            if step.joints is not None:
                # Joint space control
                # Assuming action space is [joint_pos, gripper] or similar
                # If using pd_joint_pos, action is just target joint positions
                # We need to ensure action_dim match.
                # Usually step.joints should be len(qpos).
                # For Panda/XArm, action includes gripper.
                if len(action) >= len(active_joint_indices):
                    action[: len(active_joint_indices)] = step.joints[active_joint_indices]
                else:
                    action[:] = step.joints[active_joint_indices][: len(action)]

                # Handle gripper for joint step
                if self.config.gripper_idx >= 0:
                    action[self.config.gripper_idx] = self.config.get_gripper_action(
                        step.gripper
                    )

            else:
                # Cartesian Control (IK)
                result = kinematics.compute_inverse_kinematics(
                    step.position,
                    step.quaternion,
                    check_reachability=False,  # We trust the trajectory was validated
                )
                if hasattr(result, "cpu"):
                    result = result.flatten().cpu().numpy()
                else:
                    # Assume numpy
                    result = result.flatten()

                # Build action
                if len(result) <= len(action):
                    action[: len(result)] = result
                else:
                    action = result[: len(action)]

                # Handle gripper
                if self.config.gripper_idx >= 0:
                    action[self.config.gripper_idx] = self.config.get_gripper_action(
                        step.gripper
                    )

            self._current_actions.append(action.copy())
            self._current_poses.append(self.get_tcp_pose())

            action_batch = action.reshape(1, -1)
            self.env.step(action_batch)

            if self.render_callback:
                self.render_callback()

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

            # --- Deviation Check ---
            # Check deviation every step or every N steps
            # Since we are using dense steps, maybe every step is fine if cheap
            if self.monitor.enabled and step.joints is None:
                # Only check during interaction (push/grasp) phases?
                # Or always? Always is safer.
                # But during approach, we might be far off initially if IK was approximate?
                # IK should be accurate.

                tcp_pos, tcp_quat = self.get_tcp_pose()
                if not self.monitor.check_deviation(tcp_pos, tcp_quat, step):
                    logger.error(f"Execution Deviation Detected at phase {step.phase}!")
                    self.last_execution_result = "DEVIATION"

                    # Trigger recovery (retreat)
                    logger.info("Triggering automatic retreat...")
                    self.retreat(kinematics)

                    return PrimitiveResult(
                        success=False,
                        action_name=action_name,
                        steps_taken=len(self._current_actions),
                        message=f"Deviation detected during {step.phase}",
                        actions=self._current_actions,
                        poses=self._current_poses,
                        trajectory_steps=trajectory,  # Return original steps
                    )

        # --- Grasp Success Check (for pick-like trajectories) ---
        # At this point, the trajectory has completed without deviation.
        # For trajectories that contain a grasp phase and use a gripper, we additionally
        # check whether the gripper ended up "holding" something based on both the
        # object pose and, when reliable, gripper width.
        if has_grasp_phase and self.config.gripper_idx >= 0:
            if not self.check_grasp_success(
                actor=target_actor if target_actor is not None else None
            ):
                self.last_execution_result = "GRASP_FAILED"
                # Clear grasp context after evaluation
                self._last_grasp_context = None
                return PrimitiveResult(
                    success=False,
                    action_name=action_name,
                    steps_taken=len(dense_trajectory),
                    message="Execution completed but grasp failed (empty gripper or object not attached).",
                    actions=self._current_actions,
                    poses=self._current_poses,
                    trajectory_steps=trajectory,  # Include original steps with metadata
                )

        # Clear grasp context on normal completion as well
        self._last_grasp_context = None
        self.last_execution_result = "SUCCESS"
        return PrimitiveResult(
            success=True,
            action_name=action_name,
            steps_taken=len(dense_trajectory),
            message=f"Executed {len(dense_trajectory)} steps (interpolated from {len(trajectory)})",
            actions=self._current_actions,
            poses=self._current_poses,
            trajectory_steps=trajectory,  # Include original steps with metadata
        )

    def sample_place_parameters(self, bounds: tuple = None) -> dict:
        """
        Sample parameters for a place action without executing.

        Args:
            bounds: (x_min, x_max, y_min, y_max) for placement area.
                    If None, uses the robot's configured xy_bounds.

        Returns:
            dict with keys: target_pos, place_quat
        """
        if bounds is None:
            bounds = self.config.xy_bounds
        x_min, x_max, y_min, y_max = bounds
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(0.02, 0.08)  # Place height
        target_pos = np.array([x, y, z], dtype=np.float32)

        place_quat = self._get_down_with_random_yaw(angle_range=np.pi / 8)

        return {
            "target_pos": target_pos,
            "place_quat": place_quat,
        }

    def sample_push_parameters(self, actor) -> dict:
        """
        Sample parameters for a push action without executing.

        Samples 1–4 consecutive push steps. When other objects lie within
        PUSH_TARGET_RADIUS, direction is chosen toward a randomly selected one;
        otherwise direction is random. Distance stays in [0.05, 0.15] per step.

        Args:
            actor: Object to push

        Returns:
            dict with keys: obj_pos, direction, distance, push_quat, steps (list of
            {direction, distance [, push_type, curve_control]} for 1–4 steps)
        """
        obj_pos, _ = get_actor_world_pose(actor)
        radius = self.PUSH_TARGET_RADIUS

        # Other scene objects (exclude current actor) and their positions
        pushable = self._get_pushable_actors()
        other_positions = [
            get_actor_world_pose(a)[0] for a in pushable if a is not actor
        ]

        num_steps = np.random.randint(1, 5)
        steps = []
        cur_obj = obj_pos.copy()

        for _ in range(num_steps):
            cur_xy = cur_obj[:2]
            # Objects within radius (XY), strictly positive distance to get a valid direction
            candidates = [
                pos
                for pos in other_positions
                if 0 < np.linalg.norm(pos[:2] - cur_xy) <= radius
            ]

            if candidates:
                target_pos = random.choice(candidates)
                vec_xy = target_pos[:2] - cur_xy
                direction_xy = vec_xy / np.linalg.norm(vec_xy)
                direction = np.array(
                    [direction_xy[0], direction_xy[1], 0.0], dtype=np.float32
                )
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array(
                    [np.cos(angle), np.sin(angle), 0.0],
                    dtype=np.float32,
                )

            distance = float(np.random.uniform(0.1, 0.25))
            step = {
                "direction": direction,
                "distance": distance,
            }
            if num_steps == 1:
                step["push_type"] = random.choice(list(PushType))
                if step["push_type"] == PushType.CURVE:
                    perp = np.array([-direction[1], direction[0], 0])
                    offset = np.random.uniform(-0.1, 0.1) * perp
                    mid_point = cur_obj + direction * (distance / 2) + offset
                    step["curve_control"] = mid_point.astype(np.float32)
            else:
                step["push_type"] = PushType.STRAIGHT
            steps.append(step)
            cur_obj = cur_obj + direction * distance
            cur_obj = self._clip_pos(cur_obj)

        push_z = self.config.push_height_offset
        push_quat = self._get_down_with_random_yaw(angle_range=np.pi / 12)

        params = {
            "obj_pos": obj_pos,
            "direction": steps[0]["direction"],
            "distance": steps[0]["distance"],
            "push_quat": push_quat.astype(np.float32),
            "push_z": push_z,
            "push_type": steps[0]["push_type"],
            "steps": steps,
        }
        if steps[0].get("curve_control") is not None:
            params["curve_control"] = steps[0]["curve_control"]
        return params

    def sample_home_parameters(self) -> dict:
        """
        Return home parameters: the initial robot pose (recorded at primitives creation).

        Returns:
            dict with keys: home_pos, home_quat (initial TCP pose), and home_qpos when
            config.home_qpos is set (for joint-space home).
        """
        home_pos = self._initial_tcp_pose[0].copy()
        home_quat = self._initial_tcp_pose[1].copy()
        out = {"home_pos": home_pos, "home_quat": home_quat}
        if self.config.home_qpos is not None:
            out["home_qpos"] = self.config.home_qpos.copy()
        return out

    def sample_pick_parameters(self, actor) -> dict:
        """
        Sample parameters for a pick action without executing.

        Uses 6DoF grasp poses from AntipodalSampler (antipodal or cardinal).

        Args:
            actor: ManiSkill/SAPIEN actor to pick

        Returns:
            dict with keys: approach_pos, approach_quat, pregrasp_pos, pregrasp_quat, obj_pos
            Returns None if no valid grasps found.
        """
        obj_pos, obj_quat = get_actor_world_pose(actor)
        actor_name = actor.name if hasattr(actor, "name") else ""

        # 0. Check object size and get height for small object adjustment
        # Getting exact height from collision shapes
        actor_height = 0.05  # Default if fail
        is_small_object = False

        # Try to get collision bounds
        try:
            # Access underlying sapien actor (handle wrapped/batched)
            sapien_actor = actor._objs[0] if hasattr(actor, "_objs") else actor
            if hasattr(sapien_actor, "entity"):
                sapien_actor = sapien_actor.entity

            max_h = 0.0
            for comp in sapien_actor.components:
                if hasattr(comp, "collision_shapes"):
                    for shape in comp.collision_shapes:
                        # Box
                        if hasattr(shape, "half_size"):
                            max_h = max(max_h, shape.half_size[2] * 2)
                        # Capsule/Cylinder
                        elif hasattr(shape, "half_length"):
                            max_h = max(max_h, shape.half_length * 2)
                            if hasattr(
                                shape, "radius"
                            ):  # Vertical cylinder? depends on pose.
                                # We assume standard orientation where length is Z-ish or max dim
                                pass
                        # Sphere
                        elif hasattr(shape, "radius"):
                            max_h = max(max_h, shape.radius * 2)

            if max_h > 0:
                actor_height = max_h
                # Check actual world Z height?
                # Better to use loose threshold on local size
                if actor_height < 0.06:  # Objects shorter than 6cm
                    is_small_object = True
        except Exception:
            pass  # Fallback to default behavior

        # 1. Retrieve pre-generated grasps from Env cache
        grasps = getattr(self.env.unwrapped, "distractor_grasps", {}).get(actor_name)

        if not grasps:
            return None

        # Transform local grasps to world frame
        obj_matrix = pose_to_matrix(obj_pos, obj_quat)
        valid_grasps = []

        for width, local_matrix in grasps:
            # Consider original orientation and ±90 deg rotations about Z (approach axis)
            for angle in [np.pi / 2, -np.pi / 2]:
                aug_local = local_matrix.copy()
                if angle != 0:
                    rotation = R.from_euler("z", angle).as_matrix()
                    aug_local[:3, :3] = aug_local[:3, :3] @ rotation

                # World Matrix = Actor Matrix * Local Matrix
                world_matrix = obj_matrix @ aug_local
                w_pos, w_quat = matrix_to_pose(world_matrix)

                # Check orientation in world frame (bias towards top-down)
                if GeometryLib.check_grasp_orientation(w_quat, threshold_degrees=30.0):
                    valid_grasps.append((w_pos, w_quat))

        if not valid_grasps:
            # Fallback to just the first grasp if none are strictly top-down in world frame
            # (Though they should be if objects are upright)
            w_matrix = obj_matrix @ grasps[0][1]
            p, q = matrix_to_pose(w_matrix)
            valid_grasps = [(p, q)]

        if not valid_grasps:
            return None

        # 2. Filter grasps to those within robot XY workspace (reachability)
        reachable = []
        # for g_pos, g_quat in valid_grasps:
        #     # Check approach position validity too
        #     approach_pos = (
        #         g_pos - R.from_quat(g_quat).as_matrix()[:, 2] * self.approach_height
        #     )
        #     if self._within_xy_bounds(g_pos) and self._within_xy_bounds(approach_pos):
        #         reachable.append((g_pos, g_quat))
        if not reachable:
            reachable = valid_grasps  # Fallback: use all, trajectory will clip

        # 3. Select a grasp
        # Grasps are sorted by width (ascending) in geometry.py.
        # Prefer the tightest grasps (top of list) but keep some variety.
        # Pick from top 3 (or fewer if not enough)
        top_k = min(len(reachable), 3)
        grasp_idx = np.random.randint(top_k)
        pregrasp_pos, pregrasp_quat = reachable[grasp_idx]

        # 4. Derive approach pose (retreat along grasp Z-axis)
        # BUG FIX: Ensure pregrasp_pos is NOT confirmed until we calculate approach from it
        # The previous code overwrote pregrasp_pos with the approach offset!

        rotation = R.from_quat(pregrasp_quat)
        z_axis = rotation.as_matrix()[:, 2]  # Extract Z column (approach direction)

        # Move BACK from the grasp position by approach_height
        # If Z points INTO object, we want -Z direction.
        # If approach_height is negative (e.g. -0.06), then - (-0.06) * Z = +0.06 * Z (into object further?)
        # Let's check:
        # PANDA config: approach_height = -0.06.
        # If Z is INTO object.
        # approach_pos = grasp_pos - (-0.06) * Z = grasp_pos + 0.06 * Z.
        # This moves it 6cm FURTHER into the table/object. This is WRONG.
        # Unless Panda's Z is pointing OUT? No, standard grasp frame is Z execution.

        # Let's assume standard behavior: we want to be ABOVE/BACK from the object.
        # We need to move against the Z axis.
        # If approach_height is meant to be a distance, it should be positive.
        # If existing code used negative, maybe it compensated for a different frame?
        # But 'approach_pos = pregrasp_pos - z_axis * self.approach_height'
        # with height=-0.06 -> pregrasp + 0.06*Z.

        # To fix logically: approach_pos should be "pregrasp - distance * Z".
        # We should use abs(self.approach_height) to ensure we move BACK.

        dist = abs(self.approach_height)
        # Make sure it's at least something reasonable (e.g. 10cm) if config is weird
        if dist < 0.01:
            dist = 0.12

        approach_pos = pregrasp_pos - z_axis * dist
        approach_quat = pregrasp_quat.copy()

        pregrasp_pos = pregrasp_pos - z_axis * self.config.approach_height

        # [Small Object Adjustment]
        # For small objects (height < 6cm), we want to ensure the grasp is low enough to catch it.
        # But we must avoid crashing into the table (Z=0).
        if is_small_object:
            # Lower grasp by 1.5cm to ensure fingertips reach object center/bottom
            pregrasp_pos -= z_axis * 0.015
            # Clamp to safe Z (0.2cm above table)
            pregrasp_pos[2] = max(pregrasp_pos[2], 0.002)

        # Pregrasp is the actual contact pose
        # We do NOT subtract anything from it. It IS the grasp pose.
        # (The variable name 'pregrasp_pos' in this function actually means 'final_grasp_pos'
        #  based on how it's used in primitives.generate_pick_trajectory logic:
        #  phase 4 moves to pregrasp -> phase 5 closes gripper.
        #  Wait, generate_pick_trajectory has:
        #  Phase 3: Approach (approach_pos)
        #  Phase 4: Pregrasp (pregrasp_pos)
        #  Phase 5: Grasp (close gripper in place)
        #  So 'pregrasp_pos' IS the position where we close the gripper.
        #  This should be the contact point on the object surface.

        return {
            "obj_pos": obj_pos,
            "approach_pos": approach_pos.astype(np.float32),
            "approach_quat": approach_quat.astype(np.float32),
            "pregrasp_pos": pregrasp_pos.astype(
                np.float32
            ),  # This is the actual contact/grasp point
            "pregrasp_quat": pregrasp_quat.astype(np.float32),
        }
