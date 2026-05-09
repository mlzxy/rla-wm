import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
import os


BIG_TABLE_DIMMENSION = (2.418, 1.209, 0.9196429)
SMALL_TABLE_DIMMENSION = (1.6625001, 1.2090765, 0.04379038)


class TableSceneBuilder(SceneBuilder):
    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""

    use_small_table: bool = os.environ.get("USE_ORIGIN_TABLE", "0") == "0"
    table_model_scale: float = 1.75
    table_visual_rot_euler: tuple = (0, 0, np.pi / 2)
    table_global_rot_euler: tuple = (0, 0, np.pi / 2)
    table_global_offset: tuple = (-0.12, 0.0)

    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")
        TABLE_DIM = BIG_TABLE_DIMMENSION

        if self.use_small_table:
            table_model_file = str(model_dir / "table-small.glb")
            TABLE_DIM = SMALL_TABLE_DIMMENSION
        self.TABLE_DIM = TABLE_DIM

        table_pose = sapien.Pose(q=euler2quat(*self.table_visual_rot_euler))

        # NOTE: how to derive the collision box for the table (if you have a new table)
        # builder.add_nonconvex_collision_from_file(
        #     filename=table_model_file,
        #     scale=[scale] * 3,
        #     pose=table_pose,
        # )
        # table = builder.build_kinematic(name="table-workspace")
        # x = table._objs[0].components[0].compute_global_aabb_tight()
        # x[1] - x[0] -> (2.418, 1.209, 0.9196429)

        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, TABLE_DIM[2] / 2]),
            half_size=(TABLE_DIM[0] / 2, TABLE_DIM[1] / 2, TABLE_DIM[2] / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[self.table_model_scale] * 3,
            pose=table_pose,
        )

        builder.initial_pose = sapien.Pose(
            p=[self.table_global_offset[0], self.table_global_offset[1], -TABLE_DIM[2]],
            q=euler2quat(*self.table_global_rot_euler),
        )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        # value of the call above is saved below for large table
        # aabb = np.array(
        #     [
        #         [-0.7402168, -1.2148621, -0.91964257],
        #         [0.4688596, 1.2030163, 3.5762787e-07],
        #     ]
        # )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.table = table
        self.scene_objects: list[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(
                p=[
                    self.table_global_offset[0],
                    self.table_global_offset[1],
                    -self.TABLE_DIM[2],
                ],
                q=euler2quat(*self.table_global_rot_euler),
            )
        )
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
