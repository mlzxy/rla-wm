"""
Kinematics helper for validating waypoint reachability.

Uses ManiSkill's Kinematics class for robust IK-based validation.
Requires environment with robot for IK computation.
"""

from pytorch_kinematics import Transform3d
import numpy as np
import torch
from typing import Optional, List, Tuple
from rich import print

from mani_skill import format_path
from .mani_kinematics_util import Kinematics
from mani_skill.utils.structs.pose import Pose
from .utils import matrix_to_pose, sapien_pose_to_numpy, numpy_to_sapien_pose


class KinematicsHelper:
    """
    Helper for validating robot pose reachability using IK.

    Uses ManiSkill's Kinematics class for accurate validation.
    Requires environment with properly configured robot.
    """

    def __init__(
        self,
        env,
        ik_threshold: float = 0.01,
    ):
        """
        Initialize the kinematics helper.

        Args:
            env: ManiSkill environment (required for IK)

        Raises:
            ValueError: If env is None or IK setup fails
        """
        if env is None:
            raise ValueError(
                "KinematicsHelper requires an environment for IK computation"
            )

        self.env = env
        self.kinematics = None
        self.ik_threshold = ik_threshold
        self._setup_from_env(env)

    def _setup_from_env(self, env):
        """Setup IK solver from environment."""
        agent = env.unwrapped.agent
        self.robot_uid = agent.uid

        # Cache base position
        base_pos, _ = sapien_pose_to_numpy(agent.robot.pose)
        self._base_position = base_pos.flatten()

        # Get IK parameters from agent
        urdf_path = format_path(str(agent.urdf_path))
        ee_link_name = agent.ee_link_name
        articulation = agent.robot

        # Get active joint indices from controller
        active_joint_indices = agent.controller.active_joint_indices
        self.active_joint_indices = torch.as_tensor(active_joint_indices).flatten()
        self.num_active_joints = len(active_joint_indices)
        self.kinematics = Kinematics(
            urdf_path=urdf_path,
            end_link_name=ee_link_name,
            articulation=articulation,
            active_joint_indices=self.active_joint_indices.tolist(),
        )

    @property
    def base_position(self) -> np.ndarray:
        """Get robot base position."""
        return self._base_position

    def compute_forward_kinematics(
        self, full_or_partial_qpos
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics for a given joint configuration."""
        if full_or_partial_qpos.shape[0] == self.num_active_joints:
            qpos = full_or_partial_qpos[
                :, self.kinematics.controlled_joints_idx_in_qmask
            ]
        else:
            qpos = full_or_partial_qpos
        ee_pos_fk = self.kinematics.pk_chain.forward_kinematics(qpos)[0]
        return sapien_pose_to_numpy(ee_pos_fk)

    def compute_inverse_kinematics(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        initial_qpos: Optional[torch.Tensor] = None,
        initial_eef_pose: Optional[torch.Tensor] = None,
        check_reachability: bool = False,
    ) -> Tuple[bool, Optional[torch.Tensor]] | torch.Tensor:
        """
        Check if a pose is reachable by the robot using IK.

        Args:
            position: World position [3] to check
            quaternion: Orientation [4] xyzw (optional, defaults to pointing down)
            initial_qpos: Initial joint configuration for IK solver.
                         If None, uses current robot qpos.

        Returns:
            Tuple of (is_reachable, qpos_solution)
            qpos_solution is None if not reachable.
        """
        try:
            agent = self.env.unwrapped.agent

            if initial_qpos is not None:
                q0 = initial_qpos
            else:
                q0 = agent.robot.get_qpos()

            if initial_eef_pose is not None:
                ee_pose = initial_eef_pose
            else:
                ee_pose = agent.tcp.pose

            quaternion = quaternion.reshape(-1)
            q_wxyz = quaternion[[3, 0, 1, 2]]  # xyzw to wxyz

            # transform target position to base frame
            target_pose_world = Pose.create_from_pq(
                position.reshape(1, 3),
                q_wxyz.reshape(1, -1),
            )
            target_pose_at_base = agent.robot.pose.inv() * target_pose_world
            ee_pose_at_base = agent.robot.pose.inv() * ee_pose

            # Compute IK (ManiSkill IK solvers expect poses relative to root/base)
            result = self.kinematics.compute_ik(
                target_pose_at_base,
                q0,
                current_pose=ee_pose_at_base,
                is_delta_pose=False,
                solver_config=dict(
                    type="levenberg_marquardt", solver_iterations=500, alpha=1.0
                ),
            )

            if check_reachability:
                if result is None:
                    return False, None, None

                fk_ee_p, q = self.compute_forward_kinematics(result)
                dist = torch.norm(
                    torch.as_tensor(fk_ee_p) - target_pose_at_base.p[0]
                ).item()
                is_valid = dist < self.ik_threshold
                result_eef = numpy_to_sapien_pose(fk_ee_p, q)
                return is_valid, result, agent.robot.pose * result_eef, dist
            else:
                return result  # qpos

        except Exception as e:
            import traceback

            print(
                f"[bold red][KinematicsHelper] Error in check_reachability: {e}[/bold red]"
            )
            traceback.print_exc()
            return False, None, None, None

    def validate_trajectory(
        self,
        trajectory: List,  # List of PrimitiveStep
        sample_rate: int = 1,  # Check every N steps
    ) -> Tuple[bool, Optional[int]]:
        """
        Validate that all waypoints in a trajectory are reachable.

        Args:
            trajectory: List of PrimitiveStep objects
            sample_rate: Check every N steps (1 = all, 5 = every 5th)

        Returns:
            Tuple of (is_valid, first_invalid_index or None)
        """
        summary = ""
        first_fail_idx = None
        last_q = None
        last_eef = None
        for i in range(0, len(trajectory), sample_rate):
            step = trajectory[i]
            is_reachable, result_q, result_eef, dist = self.compute_inverse_kinematics(
                step.position,
                step.quaternion,
                initial_qpos=last_q,
                initial_eef_pose=last_eef,
                check_reachability=True,
            )
            if is_reachable:
                summary += "o"
                last_q = result_q
                last_eef = result_eef
            else:
                summary += "x"
                if first_fail_idx is None:
                    first_fail_idx = i

        print(f"[yellow]{summary}[/yellow]")

        if first_fail_idx is not None:
            return False, first_fail_idx

        return True, None
