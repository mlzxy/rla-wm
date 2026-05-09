import os
from dataclasses import dataclass
from typing import Union, Sequence, Optional, List, Dict

_ROBOT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import sapien
import torch
from gymnasium import spaces
from copy import deepcopy
from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda import Panda
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
import mani_skill.agents.controllers.utils.kinematics as kinematics_utils
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.types import Array
import platform

# Monkey patch Kinematics to use pytorch-kinematics on Mac CPU as Pinocchio is often missing
if platform.system() == "Darwin":
    original_setup_cpu = kinematics_utils.Kinematics._setup_cpu

    def patched_setup_cpu(self):
        self._setup_gpu()
        self.use_gpu_ik = True

    kinematics_utils.Kinematics._setup_cpu = patched_setup_cpu


@register_agent(override=True)
class PandaV2(Panda):
    uid = "panda"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
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
            ),
            pose=sapien.Pose(),
        ),
        rest_high=Keyframe(
            qpos=np.array(
                [-0.011018545, 0.2489883, -0.015446769, -1.8937913, -0.026205156, 2.116052, 0.8045866, 0.039999723, 0.039999723]
            ),
            pose=sapien.Pose(),
        ),
    )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_or(lflag, rflag)
    

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        for k in ['pd_joint_pos', 'pd_ee_pose']:
            configs[k]['gripper'] = deepcopy(configs[k]['gripper'])
            configs[k]['gripper'].normalize_action = False
        return configs


def remove_gripper(configs):
    for mode in configs:
        for k in list(configs[mode].keys()):
            if "gripper" in k:
                configs[mode].pop(k)
    return configs




@register_agent()
class PandaClosed(PandaV2):
    uid = "panda_closed"
    # gripper_force_limit = 500
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.00890558,
                    0.49010447,
                    0.013186726,
                    -1.9629638,
                    -0.041327026,
                    2.5393598,
                    0.75421184,
                    8e-44,
                    8e-44,
                ]
            ),
            pose=sapien.Pose(),
        ),
        rest_high=Keyframe(
            qpos=np.array(
                [-0.015683716, 0.25824597, 0.0005123088, -1.8930161, -0.020406188, 2.210182, 0.7178196, 0.0, 0.0]
            ),
            pose=sapien.Pose(),
        ),
    )

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        return remove_gripper(configs)

    def _after_init(self):
        super()._after_init()
        # Manually set gripper joints to closed position (over-closing)
        for joint in self.robot.get_active_joints():
            if joint.name in ["panda_finger_joint1", "panda_finger_joint2"]:
                joint.set_drive_properties(
                    stiffness=self.gripper_stiffness, damping=self.gripper_damping
                )
                for obj in joint._objs:
                    obj.drive_target = np.array([-0.1])


@register_agent(override=True)
class XArm6RobotiqV2(XArm6Robotiq):
    """XArm6 with Robotiq gripper - using upstream ManiSkill implementation."""
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    uid = "xarm6_robotiq"
    urdf_path = os.path.join(_ROBOT_DIR, "robots/xarm6/xarm6_robotiq.urdf")
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.11007055,
                    0.44809502,
                    -1.4307088,
                    0.061216623,
                    0.95184845,
                    0.045703474,
                    4.1095404e-08,
                    3.8954175e-07,
                    6.346737e-11,
                    3.526919e-07,
                    -7.2738095e-07,
                    -2.7744545e-06,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        rest_high=Keyframe(
            qpos=np.array(
                [0.1010212, 0.34049365, -1.6935898, 0.044812873, 1.3688554, 0.07900385, -1.0035441e-06, -4.080726e-09, -1.0970805e-06, -4.1200647e-11, -1.0684801e-09, -2.8531963e-06]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )
    fix_root_link = True

    gripper_force_limit = 100 # 0.1
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    
    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_or(lflag, rflag)


@register_agent()
class XArm6RobotiqClosed(XArm6RobotiqV2):
    uid = "xarm6_robotiq_closed"
    gripper_force_limit = 500
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.13424626,
                    0.46841288,
                    -1.4039639,
                    0.08751424,
                    0.87772185,
                    0.02053429,
                    3.2795136e-10,
                    2.1616957e-07,
                    7.590833e-11,
                    8.151683e-07,
                    -3.8325954e-08,
                    -2.189691e-06,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        rest_high=Keyframe(
           qpos=np.array(
                [0.1010212, 0.34049365, -1.6935898, 0.044812873, 1.3688554, 0.07900385, -1.0035441e-06, -4.080726e-09, -1.0970805e-06, -4.1200647e-11, -1.0684801e-09, -2.8531963e-06]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        return remove_gripper(configs)

    def _after_init(self):
        super()._after_init()
        # Manually set gripper joints to closed position (over-closing)
        for joint in self.robot.get_active_joints():
            if joint.name in ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]:
                joint.set_drive_properties(
                    stiffness=self.gripper_stiffness, damping=self.gripper_damping
                )
                for obj in joint._objs:
                    obj.drive_target = np.array([1.0])


class PaddedPDJointPosController(PDJointPosController):
    """Custom PDJointPosController that pads the action space with a fixed value for the last joint."""
    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        # Only take the first N-1 joints for the action space
        low, high = joint_limits[:-1, 0], joint_limits[:-1, 1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_action(self, action: Array):
        # 1. Preprocess the 5D action
        action = self._preprocess_action(action)
        
        # 2. Pad action with 0.0 for the last joint (wrist_3_joint)
        # This keeps the action space at 5D while the robot has 6 arm joints.
        padding = torch.zeros((action.shape[0], 1), device=self.device)
        padded_action = torch.cat([action, padding], dim=-1)
        
        # 3. Manually implement set_action logic for the 6 joints
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + padded_action
            else:
                self._target_qpos = self._start_qpos + padded_action
        else:
            self._target_qpos = torch.broadcast_to(
                padded_action, self._start_qpos.shape
            ).clone()
            
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

@dataclass
class PaddedPDJointPosControllerConfig(PDJointPosControllerConfig):
    controller_cls = PaddedPDJointPosController

@register_agent()
class UR10eStickV2(BaseAgent):
    """UR10e arm with a stick."""

    uid = "ur10e_stick"
    fix_root_link = True
    urdf_path = os.path.join(_ROBOT_DIR, "robots/ur10e_stick/ur10e_stick.urdf")

    # TCP at stick tip link
    ee_link_name = "tcp"

    # Arm joint names (standard UR10e)
    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.09706659, -1.1033098, 1.94184, -2.394296, -1.5409073, 0.0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.zeros(6),
            pose=sapien.Pose([0, 0, 0]),
        ),
        rest_high=Keyframe(
            qpos=np.array([0.10832267, -1.3466122, 1.7542601, -1.992004, -1.5525175, -0.0038689054]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    # Stability Tuning (matches ur10e_allegro)
    arm_friction = 0.1
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    @property
    def _controller_configs(self):
        # Joint Controllers (Padded to 5D action space)
        arm_pd_joint_pos = PaddedPDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PaddedPDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        arm_pd_joint_pos_6d = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_6d = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        
        # EE Controllers (Still use all 6 joints for stability, action space is 6D)
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=True,
            use_target=True,
            frame="root_translation:root_aligned_body_rotation",
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            pos_lower=-1.0,
            pos_upper=1.0,
            rot_lower=-np.pi,
            rot_upper=np.pi,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            frame="root_translation:root_aligned_body_rotation",
            normalize_action=False
        )

        return dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
            pd_joint_pos_6d=dict(arm=arm_pd_joint_pos_6d),
            pd_joint_delta_pos_6d=dict(arm=arm_pd_joint_delta_pos_6d),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
            pd_ee_pose=dict(arm=arm_pd_ee_pose),
        )

    def _after_init(self):
        self.tcp = self.robot.links_map.get("tcp", self.robot.links[-1])

    def is_static(self, threshold: float = 0.2):
        return torch.max(torch.abs(self.robot.get_qvel()), 1)[0] <= threshold


def get_robot_uids(actual=False):
    if actual:
        return [
            "panda",
            "xarm6_robotiq",
            "ur10e_stick",
        ]
    else:
        return [
            "panda",
            "xarm6_robotiq",
            "ur10e_stick",
            "panda_closed",
            "xarm6_robotiq_closed",
        ]
