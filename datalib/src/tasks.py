import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from typing import Any

from mani_skill.utils import common, sapien_utils
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose, Actor


from .unified_workspace import UnifiedWorkspaceEnv, TableSceneBuilder
import mani_skill.envs.tasks.tabletop.push_t as push_t_module
from mani_skill.utils.geometry import rotation_conversions
from datalib.src.mani_tasks import (
    RollBallEnv,
    PullCubeEnv,
    PokeCubeEnv,
    PegInsertionSideEnv,
    PullCubeToolEnv,
    PushTEnv,
)


# --- 1. Push-T task ---
@register_env("PushT-v2", max_episode_steps=100, override=True)
class PushTEnvV2(UnifiedWorkspaceEnv, PushTEnv):
    goal_offset = torch.tensor([0.0, -0.1])

    def _load_scene(self, options: dict):
        PushTEnv._load_scene(self, options)
        # UnifiedWorkspaceEnv._load_scene(self, options, skip_table=True)


# --- 2. RollBall task ---
@register_env("RollBall-v1", max_episode_steps=80, override=True)
class RollBallEnvV1(UnifiedWorkspaceEnv, RollBallEnv):

    def _load_scene(self, options: dict):
        RollBallEnv._load_scene(self, options)
        # UnifiedWorkspaceEnv._load_scene(self, options, skip_table=True)
        self.reached_status = self.reached_status.to(self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_agent(env_idx)
        self.table_scene.initialize(env_idx)
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = torch.rand(b) * 0.3 - 0.3
            xyz[:, 1] = torch.rand(b) * 0.1 + 0.2
            xyz[:, 2] = self.ball_radius
            self.ball.set_pose(Pose.create_from_pq(p=xyz))
            xyz_goal = torch.zeros((b, 3), device=self.device)
            xyz_goal[:, 0] = torch.rand(b) * 0.3 - 0.3
            xyz_goal[:, 1] = torch.rand(b) * 0.15 + 0.0
            xyz_goal[:, 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(p=xyz_goal, q=euler2quat(0, np.pi / 2, 0))
            )
        self.reached_status[env_idx] = 0.0
        self._initialize_distractors(env_idx)


# --- 4. PokeCube task ---
@register_env("PokeCube-v2", max_episode_steps=50)
class PokeCubeEnvV2(UnifiedWorkspaceEnv, PokeCubeEnv):

    def _load_scene(self, options: dict):
        PokeCubeEnv._load_scene(self, options)
        # UnifiedWorkspaceEnv._load_scene(self, options, skip_table=True)
        
        
    def evaluate(self):
        is_cube_placed = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        peg_q = self.peg_head_pose.q
        peg_qmat = rotation_conversions.quaternion_to_matrix(peg_q)
        peg_euler = rotation_conversions.matrix_to_euler_angles(peg_qmat, "XYZ")
        cube_q = self.cube.pose.q
        cube_qmat = rotation_conversions.quaternion_to_matrix(cube_q)
        cube_euler = rotation_conversions.matrix_to_euler_angles(cube_qmat, "XYZ")
        angle_diff = torch.abs(peg_euler[:, 2] - cube_euler[:, 2])
        is_peg_cube_aligned = angle_diff < 0.05

        head_to_cube_dist = torch.linalg.norm(
            self.peg_head_pos[..., :2] - self.cube.pose.p[..., :2], axis=1
        )
        is_peg_cube_close = head_to_cube_dist <= self.cube_half_size + 0.005

        is_peg_cube_fit = torch.logical_and(is_peg_cube_aligned, is_peg_cube_close)

        try:
            is_peg_grasped = self.agent.is_grasping(self.peg)
        except NotImplementedError:
            # Fallback for non-grasping robots: use proximity
            tcp_to_peg_dist = torch.linalg.norm(
                self.agent.tcp.pose.p - self.peg.pose.p, axis=1
            )
            is_peg_grasped = tcp_to_peg_dist < 0.03

        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_cube_placed & is_robot_static,
            "is_cube_placed": is_cube_placed,
            "is_peg_cube_fit": is_peg_cube_fit,
            "is_peg_grasped": is_peg_grasped,
            "angle_diff": angle_diff,
            "head_to_cube_dist": head_to_cube_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reach peg
        tcp_pos = self.agent.tcp.pose.p
        tgt_tcp_pose = self.peg.pose
        tcp_to_peg_dist = torch.linalg.norm(tcp_pos - tgt_tcp_pose.p, axis=1)

        try:
            self.agent.is_grasping(self.peg)
            reached_thresh = 0.01
        except NotImplementedError:
            reached_thresh = 0.03

        reached = tcp_to_peg_dist < reached_thresh
        reaching_reward = 2 * (1 - torch.tanh(5.0 * tcp_to_peg_dist))
        reward = reaching_reward

        # peg to cube
        angle_diff = info["angle_diff"]
        align_reward = 1 - torch.tanh(5.0 * angle_diff)
        head_to_cube_dist = info["head_to_cube_dist"]
        close_reward = 1 - torch.tanh(5.0 * head_to_cube_dist)
        is_peg_grasped = info["is_peg_grasped"] * reached
        reward[is_peg_grasped] = (4 + close_reward + align_reward)[is_peg_grasped]

        # cube to goal
        cube_to_goal_dist = torch.linalg.norm(
            self.goal_region.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        is_peg_cube_fit = info["is_peg_cube_fit"] * is_peg_grasped
        reward[is_peg_cube_fit] = (7 + place_reward)[is_peg_cube_fit]

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward[info["is_cube_placed"]] += static_reward[info["is_cube_placed"]]

        reward[info["success"]] = 10
        return reward


@register_env("PullCubeTool-v1", max_episode_steps=100, override=True)
class PullCubeToolEnvV1(UnifiedWorkspaceEnv, PullCubeToolEnv):

    def _load_scene(self, options: dict):
        PullCubeToolEnv._load_scene(self, options)
        # UnifiedWorkspaceEnv._load_scene(self, options, skip_table=True)


# --- 6. PullCube task ---
@register_env("PullCube-v2", max_episode_steps=50)
class PullCubeEnvV2(UnifiedWorkspaceEnv, PullCubeEnv):
    def _load_scene(self, options: dict):
        PullCubeEnv._load_scene(self, options)
        # UnifiedWorkspaceEnv._load_scene(self, options, skip_table=True)


# --- 7. PegInsertionSide task ---
@register_env("PegInsertionSide-v1", max_episode_steps=100, override=True)
class PegInsertionSideEnvV1(UnifiedWorkspaceEnv, PegInsertionSideEnv):
    def _load_scene(self, options: dict):
        PegInsertionSideEnv._load_scene(self, options)
        # UnifiedWorkspaceEnv._load_scene(self, options, skip_table=True)
    
    

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            xy = randomization.uniform(
                low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            xy = randomization.uniform(
                low=torch.tensor([-0.05, 0.2]),
                high=torch.tensor([0.05, 0.4]),
                size=(b, 2),
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))
