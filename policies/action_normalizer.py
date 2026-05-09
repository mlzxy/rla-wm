from __future__ import annotations
import torch.nn as nn

import argparse
import os
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from rich import print

import gymnasium as gym
import imageio
import numpy as np
import torch
import third_party.pytorch_kinematics as pk
from jaxtyping import Float
from mani_skill.agents.registration import REGISTERED_AGENTS

from datalib.dataset import ManiSkillTrajectoryDataset
from src.datasets.trajectory_dataset import TrajectoryBatch
import datalib.src.robots as robots  # noqa: F401  # ensure robot classes are registered
import datalib.src.tasks  # noqa: F401  # ensure env ids are registered


ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))



@dataclass
class Action:
    """Normalized action representation consumed by policy code."""

    gripper: Float[torch.Tensor, "batch horizon 1"]  # unified close signal in [-1, 1], -1=open, 1=closed
    arm_joints: Float[torch.Tensor, "batch horizon joints"]
    eef: Optional[Float[torch.Tensor, "batch horizon 6"]] = None  # end-effector pose (xyz + euler, radians)



@dataclass
class Observation:
    """Minimal observation bundle used by downstream policy components."""

    state: Action
    rgbs: torch.Tensor  # (B, C, 3, H, W) camera images
    actions: Optional[Action] = None


@dataclass
class RobotSpec:
    """Static robot metadata used for action/state shape conversion."""

    full_qpos_dim: int
    arm_dim: int
    default_hidden_qpos: Float[np.ndarray, "..."]
    has_controllable_gripper: bool
    # Unified close signal is mapped to [-1, 1] where -1=open and 1=closed.
    # Robot-native conversion uses these two endpoints, which may be increasing
    # or decreasing depending on robot/controller convention.
    gripper_open_value: float = 0.0
    gripper_close_value: float = 1.0


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert input to tensor and cast dtype if already a tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _ensure_bhd(t: Any, name: str) -> Float[torch.Tensor, "batch horizon dim"]:
    """Ensure tensor has shape ``(B, H, D)`` by injecting singleton horizon if needed."""
    x = _to_tensor(t)
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected {name} to have shape (B,D) or (B,H,D), got {tuple(x.shape)}")


def _safe_affine_normalize(
    values: Float[torch.Tensor, "... dim"],
    low: Float[torch.Tensor, "... dim"],
    high: Float[torch.Tensor, "... dim"],
) -> Float[torch.Tensor, "... dim"]:
    """Affine-map values from ``[low, high]`` into ``[-1, 1]`` with NaN/Inf guards."""
    low = torch.nan_to_num(low, nan=-1.0, posinf=1.0, neginf=-1.0)
    high = torch.nan_to_num(high, nan=1.0, posinf=1.0, neginf=-1.0)
    denom = torch.clamp(high - low, min=1e-6)
    out = 2.0 * ((values - low) / denom) - 1.0
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_affine_denormalize(
    values: Float[torch.Tensor, "... dim"],
    low: Float[torch.Tensor, "... dim"],
    high: Float[torch.Tensor, "... dim"],
) -> Float[torch.Tensor, "... dim"]:
    """Inverse of `_safe_affine_normalize` with matching numerical safeguards."""
    low = torch.nan_to_num(low, nan=-1.0, posinf=1.0, neginf=-1.0)
    high = torch.nan_to_num(high, nan=1.0, posinf=1.0, neginf=-1.0)
    out = low + 0.5 * (values + 1.0) * (high - low)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)





class ActionNormalizer:
    """Normalize robot action/state tensors to a unified policy-friendly space."""

    def __init__(
        self,
        robot_uid: str,
        control_mode: Literal["pd_joint_pos", "pd_ee_pose"],
        state_source: Literal["qpos", "target_qpos"] = "target_qpos",
        device: Optional[torch.device] = None,
        debug: bool = False,
    ):
        if robot_uid not in robots.get_robot_uids():
            raise ValueError(f"Unknown robot_uid={robot_uid}. Valid: {robots.get_robot_uids()}")
        if control_mode not in ("pd_joint_pos", "pd_ee_pose"):
            raise ValueError(f"Unsupported control_mode={control_mode}")
        if state_source not in ("qpos", "target_qpos"):
            raise ValueError(f"Unsupported state_source={state_source}; expected 'qpos' or 'target_qpos'")
        self.device = device or torch.device("cpu")
        self.robot_uid = robot_uid
        self.control_mode = control_mode
        self.state_source = state_source
        self.debug = debug

        self._robot_cls = REGISTERED_AGENTS[robot_uid].agent_cls
        self._spec = self._build_robot_spec(robot_uid)
        self._ee_link_name = getattr(self._robot_cls, "ee_link_name", "tcp")

        self._fk_chain: Optional[pk.chain.Chain] = None
        self._init_fk_chain()
        self._fk_chain = self._fk_chain.to(device=self.device)

        self._action_low: Optional[torch.Tensor] = None # these actions dim are always for the pd_joint_pos
        self._action_high: Optional[torch.Tensor] = None
        self._action_dim: Optional[int] = None
        self._bootstrap_action_bounds()
        if self.debug:
            low_dbg = self._action_low.detach().cpu().tolist() if self._action_low is not None else None
            high_dbg = self._action_high.detach().cpu().tolist() if self._action_high is not None else None
            print(
                "[ActionNormalizer][init] "
                f"robot={self.robot_uid}, mode={self.control_mode}, state_source={self.state_source}, action_dim={self._action_dim}"
            )
            print(f"[ActionNormalizer][init] action_low={low_dbg}")
            print(f"[ActionNormalizer][init] action_high={high_dbg}")

    def _build_robot_spec(self, robot_uid: str) -> RobotSpec:
        """Construct static robot metadata required by conversion helpers."""
        default_qpos = None
        keyframes = getattr(self._robot_cls, "keyframes", None)
        default_qpos = np.array(keyframes["rest"].qpos, dtype=np.float32)

        if robot_uid.startswith("panda"):
            full_dim = 9
            arm_dim = 7
            has_gripper = not robot_uid.endswith("_closed")
            return RobotSpec(
                full_qpos_dim=full_dim,
                arm_dim=arm_dim,
                default_hidden_qpos=default_qpos,
                has_controllable_gripper=has_gripper,
                gripper_open_value=0.04,
                gripper_close_value=0 #-0.01,
            )

        if robot_uid.startswith("xarm6_robotiq"):
            full_dim = 12
            arm_dim = 6
            has_gripper = not robot_uid.endswith("_closed")
            return RobotSpec(
                full_qpos_dim=full_dim,
                arm_dim=arm_dim,
                default_hidden_qpos=default_qpos,
                has_controllable_gripper=has_gripper,
                gripper_open_value=0,
                gripper_close_value=0.81,
            )

        if robot_uid == "ur10e_stick":
            full_dim = 6
            arm_dim = 5 if self.control_mode == "pd_joint_pos" else 6
            return RobotSpec(
                full_qpos_dim=full_dim,
                arm_dim=arm_dim,
                default_hidden_qpos=default_qpos,
                has_controllable_gripper=False,
                gripper_open_value=0.0,
                gripper_close_value=1.0,
            )

        raise ValueError(f"Unsupported robot_uid={robot_uid}")

    def _init_fk_chain(self) -> None:
        """Initialize FK chain from URDF when available; otherwise leave FK disabled."""
        urdf_path = getattr(self._robot_cls, "urdf_path", None)
        if not urdf_path:
            if self.debug:
                print(f"[ActionNormalizer] No urdf_path on class for {self.robot_uid}; FK disabled")
            return
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.abspath(urdf_path)
        if not os.path.exists(urdf_path):
            if self.debug:
                print(f"[ActionNormalizer] URDF path does not exist: {urdf_path}; FK disabled")
            return
        with open(urdf_path, "r") as f:
            urdf_content = f.read()
        self._fk_chain = pk.build_chain_from_urdf(urdf_content)

    def _bootstrap_action_bounds(self) -> None:
        """Bootstrap action bounds from a lightweight env instance for normalization."""
        try:
            env = gym.make(
                "TableOnly-v2",
                obs_mode="state",
                control_mode="pd_joint_pos",
                robot_uids=self.robot_uid,
                render_mode="rgb_array",
                sim_backend="physx_cpu",
                include_all_cameras=False,
                max_episode_steps=10,
                # render_backend="cpu"
            )
            self.joint_names = [joint.name for joint in env.agent.robot.get_active_joints()]
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
            self._action_low = torch.from_numpy(low)
            self._action_high = torch.from_numpy(high)
            self._action_dim = int(low.shape[0])
            if self.debug:
                print(
                    "[ActionNormalizer][bounds] "
                    f"action_dim={self._action_dim}, "
                    f"low[:6]={low[:6].tolist()}, high[:6]={high[:6].tolist()}"
                )
            env.close()
        except Exception as e:
            print(f"[red][ActionNormalizer] Could not bootstrap action bounds from env: {e}[/red]")
            raise e

    def _get_bounds_for_dim(self, dim: int) -> Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
        """Return action bounds with optional padding when requested dim exceeds env dim."""
        low = self._action_low
        high = self._action_high
        if low.shape[0] >= dim:
            return low[:dim], high[:dim]

        pad = dim - low.shape[0]
        low = torch.cat([low, -torch.ones((pad,), dtype=low.dtype)], dim=0)
        high = torch.cat([high, torch.ones((pad,), dtype=high.dtype)], dim=0)
        return low, high


    def _extract_state_qpos(self, data: TrajectoryBatch) -> Float[torch.Tensor, "batch horizon joints"]:
        """Read and normalize qpos-like state from `TrajectoryBatch` to ``(B, H, J)``."""
        qpos = data.get(self.state_source)
        if qpos is None:
            raise ValueError(f"Expected '{self.state_source}' in batch")

        if isinstance(qpos, list):
            seqs = []
            for x in qpos:
                t = _to_tensor(x)
                if t.ndim == 3 and t.shape[1] == 1:
                    t = t[:, 0, :]
                seqs.append(t)
            return torch.stack(seqs, dim=0)

        t = _to_tensor(qpos)
        if t.ndim == 2:
            return t.unsqueeze(0)
        if t.ndim == 4 and t.shape[2] == 1:
            return t[:, :, 0, :]
        if t.ndim == 3:
            return t
        raise ValueError(f"Unexpected qpos shape: {tuple(t.shape)}")


    def _split_target_qpos(
        self, target_qpos: Float[torch.Tensor, "batch horizon joints"]
    ) -> Tuple[Float[torch.Tensor, "batch horizon arm_joints"], Float[torch.Tensor, "batch horizon gripper_joints"]]:
        """Split target qpos into arm and gripper channels based on robot spec."""
        arm_dim = min(self._spec.arm_dim, target_qpos.shape[-1])
        arm = target_qpos[..., :arm_dim]
        gripper = target_qpos[..., arm_dim:]
        if not self._spec.has_controllable_gripper:
            gripper = target_qpos.new_zeros((*target_qpos.shape[:-1], 0))
        return arm, gripper

    def _close_signal_from_gripper_raw(
        self, gripper_raw: Float[torch.Tensor, "batch horizon gripper_joints"], state_source=None
    ) -> Float[torch.Tensor, "batch horizon 1"]:
        """Map robot-native gripper values to unified close signal in ``[-1, 1]``."""
        if gripper_raw.shape[-1] == 0: # No controllable gripper: expose fully-open signal by default.
            out = -torch.ones((*gripper_raw.shape[:-1], 1), dtype=gripper_raw.dtype, device=gripper_raw.device)
            return out
        
        state_source = state_source or self.state_source

        open_v = self._spec.gripper_open_value
        close_v = self._spec.gripper_close_value
        denom = close_v - open_v
        if abs(denom) < 1e-6:
            progress = torch.zeros_like(gripper_raw)
        else:
            # 0=open, 1=closed regardless of endpoint ordering.
            progress = (gripper_raw - open_v) / denom

        progress = progress.clamp(0.0, 1.0)
        unified = 2.0 * progress - 1.0
        unified = unified.clamp(-1.0, 1.0)

        if 'panda' in self.robot_uid:
            if state_source == 'qpos':
                assert unified.shape[-1] == 2
            else:
                assert unified.shape[-1] == 1
            out = unified[..., [0]]
        elif 'xarm' in self.robot_uid:
            if state_source == 'qpos':
                assert unified.shape[-1] == 6
                out = unified[..., [2]]
            else:
                assert unified.shape[-1] == 1
                out = unified[..., [0]]
        return out

    def _gripper_raw_1d_from_close_signal(
        self, close_signal: Float[torch.Tensor, "batch horizon 1"]
    ) -> Float[torch.Tensor, "batch horizon 1"]:
        """Map unified close signal from ``[-1, 1]`` back to robot-native gripper value."""
        # Unified signal maps linearly from [-1, 1] to [open, close].
        c = close_signal.clamp(-1.0, 1.0)
        progress = 0.5 * (c + 1.0)
        open_v = self._spec.gripper_open_value
        close_v = self._spec.gripper_close_value
        value = open_v + progress * (close_v - open_v)
        return value

    def _target_to_full_qpos(
        self, target_qpos: Float[torch.Tensor, "batch horizon joints"]
    ) -> Float[torch.Tensor, "batch horizon full_qpos_dim"]:
        """Expand reduced target qpos into full robot qpos including hidden/mimic joints."""
        B, H, J = target_qpos.shape
        full_dim = self._spec.full_qpos_dim

        defaults = torch.from_numpy(self._spec.default_hidden_qpos).to(target_qpos).view(1, 1, full_dim).repeat(B, H, 1)
        full = defaults.clone()

        if self.robot_uid == "ur10e_stick" and J == 5:
            full[..., :5] = target_qpos # hidden wrist_3 joint keeps default value from rest keyframe

        if self.robot_uid.startswith("panda"):
            full[..., :7] = target_qpos[..., :7]
            if J == 8:
                g = target_qpos[..., 7:8].squeeze(-1)
                full[..., 7] = g # mimic joint
                full[..., 8] = g
            elif J == 9:
                g = target_qpos[..., 7:9]
                full[..., 7:9] = g # full qpos input
            else:
                g = self._spec.gripper_close_value # default value (panda_closed)
                full[..., 7] = g
                full[..., 8] = g

        if self.robot_uid.startswith("xarm6_robotiq"):
            full[..., :6] = target_qpos[..., :6]
            if J == 12:
                full[..., 6:12] = target_qpos[..., 6:12] # full qpos input
            else:
                if J == 7:
                    g = target_qpos[..., 6:].squeeze(-1)
                elif J == 6:
                    g = torch.full_like(target_qpos[..., 0], self._spec.gripper_close_value)
                full[..., 6] = g # mimic joints
                full[..., 8] = g # mimic joints
                 # emulate the passive joints of four-link fingers
                full[..., [7, 9]] = (0.88 / 0.81) * g[..., None].clamp(0, 0.81)
                full[..., [10, 11]] = - (0.88 / 0.81) * g[..., None].clamp(0, 0.81)

        return full

    def _compute_eef_from_fk(
        self, target_qpos: Float[torch.Tensor, "batch horizon joints"], *args: Any
    ) -> Float[torch.Tensor, "batch horizon 6"]:
        """Compute EEF pose ``(xyz + euler XYZ radians)`` from qpos via FK when available."""
        B, H, _ = target_qpos.shape
        if self._fk_chain is None:
            return target_qpos.new_zeros((B, H, 6))

        full_q = self._target_to_full_qpos(target_qpos)
        q = full_q.reshape(B * H, -1)

        ret = self._fk_chain.forward_kinematics(q)
        local = ret[self._ee_link_name]
        world = local
        eef_mat = world.get_matrix()
        p = eef_mat[:, :3, 3]
        rpy = pk.transforms.matrix_to_euler_angles(eef_mat[:, :3, :3], "XYZ") # in radians!
        eef = torch.cat([p, rpy], dim=-1)
        return eef.reshape(B, H, 6)

    def normalize(self, data: TrajectoryBatch) -> Action:
        """Convert configured state source to normalized arm joints + unified gripper signal."""
        qpos = self._extract_state_qpos(data)
        B, H, J = qpos.shape
        # root_poses = self._extract_root_poses(data, B, H)

        arm_raw, gripper_raw = self._split_target_qpos(qpos)
        arm_dim = arm_raw.shape[-1]

        low, high = self._get_bounds_for_dim(arm_dim)
        arm_low = low[:arm_dim].view(1, 1, -1).to(qpos)
        arm_high = high[:arm_dim].view(1, 1, -1).to(qpos)
        arm_norm = _safe_affine_normalize(arm_raw, arm_low, arm_high)

        close_signal = self._close_signal_from_gripper_raw(gripper_raw)
        eef = self._compute_eef_from_fk(qpos)

        return Action(
            gripper=close_signal,
            eef=eef,  # EEF Pose is not normalized
            arm_joints=arm_norm,
        )

    def denormalize(
        self,
        action: Action,
        return_full_qpos: bool = False,
        return_eef_action: bool = None,
    ) -> Float[torch.Tensor, "batch horizon out_dim"]:
        """Map normalized action back to robot-space qpos-like or EEF action tensors.

        - `return_full_qpos=True`: reconstruct hidden/fixed joints for full robot qpos.
        - `return_eef_action=True`: return ``[eef(6), denormalized_gripper(1)]`` per step.
        """
        arm_norm = _ensure_bhd(action.arm_joints, name="action.arm_joints")
        close_signal = _ensure_bhd(action.gripper, name="action.gripper")

        B, H, arm_dim = arm_norm.shape

        low, high = self._get_bounds_for_dim(arm_dim)
        arm_low = low[:arm_dim].view(1, 1, -1).to(arm_norm)
        arm_high = high[:arm_dim].view(1, 1, -1).to(arm_norm)
        arm_raw = _safe_affine_denormalize(arm_norm, arm_low, arm_high)

        gripper_raw = self._gripper_raw_1d_from_close_signal(close_signal) # scalar

        if (return_eef_action is None and self.control_mode == 'pd_ee_pose') or return_eef_action:
            if action.eef is None:
                eef = self._compute_eef_from_fk(arm_raw)
            else:
                eef = _ensure_bhd(action.eef, name="action.eef")
            if self._spec.has_controllable_gripper:
                eef_action = torch.cat([eef, gripper_raw], dim=-1)
            else:
                eef_action = eef
            return eef_action
        
        if self._spec.has_controllable_gripper:
            qpos = torch.cat([arm_raw, gripper_raw], dim=-1)
        else:
            qpos = arm_raw
        out = self._target_to_full_qpos(qpos) if return_full_qpos else qpos
        return out


# --- Below are utilities for smoke testing the ActionNormalizer on real trajectories and env rollouts. --- # 


def _load_one_batch_from_dataset(robot_uid: str) -> Optional[TrajectoryBatch]:
    """Load one trajectory as a single-item `TrajectoryBatch` for smoke testing."""
    root = f"data/better/ppo/{robot_uid}"
    task = os.listdir(root)[0]
    root = os.path.join(root, task, 'success')
    ds = ManiSkillTrajectoryDataset(root)
    traj_ids = ds.list_trajectories()
    if len(traj_ids) == 0:
        return None

    traj = ds.read_trajectory(
        traj_ids[0],
        video_keys=["base_camera_rgb", "front_camera_rgb"],
        metadata_keys=["qpos", "target_qpos", "root_poses", "eef_pose"],
    )
    robot_infos = ds.get_robot_infos() or []
    batch: TrajectoryBatch = {
        "qpos": [_to_tensor(traj.metadata["qpos"])],
        "target_qpos": [_to_tensor(traj.metadata["target_qpos"])],
        "root_poses": [_to_tensor(traj.metadata["root_poses"])],
        "robot_infos": [robot_infos],
    }  # type: ignore[typeddict-item]
    return batch


def _playback_and_record(
    robot_uid: str,
    control_mode: str,
    actions: Float[np.ndarray, "... action_dim"],
    recover_qposes: Float[np.ndarray, "... qpos_dim"],
    out_video_path: str,
    max_steps: int,
    env_id: str,
    joint_names: List[str],
) -> Dict[str, Any]:
    """Replay actions in env, save rollout video, and optionally export mesh RRD."""
    def _record_mesh_rrd_from_recovered_qpos() -> Optional[str]:
        try:
            import rerun as rr
            from datalib.robot_geometry import DifferentiableRobotGeometry, to_o3d_mesh, mesh_to_arrays
        except Exception as e:
            print(f"[yellow][ActionNormalizer] Skip mesh RRD export (missing deps): {e}[/yellow]")
            return None

        qseq = np.asarray(recover_qposes, dtype=np.float32)
        if qseq.ndim == 3:
            qseq = qseq.reshape(-1, qseq.shape[-1])
        elif qseq.ndim != 2:
            print(f"[yellow][ActionNormalizer] Skip mesh RRD export (unexpected recover_qposes shape): {qseq.shape}[/yellow]")
            return None
        if qseq.shape[0] == 0:
            return None

        mesh_robot_uid = robot_uid[:-7] if robot_uid.endswith("_closed") else robot_uid
        robot_cls = REGISTERED_AGENTS[mesh_robot_uid].agent_cls
        urdf_path = getattr(robot_cls, "urdf_path", None)
        if urdf_path is None:
            print(f"[yellow][ActionNormalizer] Skip mesh RRD export (no urdf_path for {mesh_robot_uid})[/yellow]")
            return None
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(ROOT_DIR, urdf_path)
        if not os.path.exists(urdf_path):
            urdf_path = os.path.abspath(urdf_path)
        if not os.path.exists(urdf_path):
            print(f"[yellow][ActionNormalizer] Skip mesh RRD export (URDF not found): {urdf_path}[/yellow]")
            return None

        base_dir = os.path.dirname(urdf_path)
        if osp.exists(urdf_path.replace(".urdf", ".stl.urdf")):
            urdf_path = urdf_path.replace(".urdf", ".stl.urdf")
        robot_geom = DifferentiableRobotGeometry(urdf_path=urdf_path, base_dir=base_dir, joint_names=joint_names)

        mesh_rrd_path = f"{os.path.splitext(out_video_path)[0]}.rrd"
        os.makedirs(os.path.dirname(mesh_rrd_path), exist_ok=True)
        rr.init(f"action_normalizer_mesh_{mesh_robot_uid}", spawn=False)
        rr.save(mesh_rrd_path)

        for frame_idx in range(qseq.shape[0]):
            rr.set_time("frame", sequence=frame_idx)
            q = torch.from_numpy(qseq[frame_idx : frame_idx + 1]).float()
            robot_geom.set_pose(q)
            mesh = to_o3d_mesh(robot_geom.sdf)
            vertices, faces, vertex_colors = mesh_to_arrays(mesh)
            rr.log(
                f"robots/{mesh_robot_uid}/mesh",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_colors=vertex_colors,
                ),
            )

        print(f"[ActionNormalizer] saved mesh rrd to {mesh_rrd_path}")
        return mesh_rrd_path

    env = gym.make(
        env_id,
        obs_mode="state+rgb+segmentation",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        robot_uids=robot_uid,
        control_mode=control_mode,
        include_all_cameras=True,
        camera_width=512,
        camera_height=512,
        max_episode_steps=max_steps,
    )
    obs, _ = env.reset(seed=0)
    del obs

    A = actions.reshape(-1, actions.shape[-1])
    T = min(max_steps, A.shape[0])

    frames: List[np.ndarray] = []
    total_reward = 0.0
    success = False
    used = 0
    for t in range(T):
        a = np.asarray(A[t], dtype=np.float32).reshape(-1)
        _, reward, term, trunc, info = env.step(a)
        frame = env.render()
        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame)
        total_reward += float(np.asarray(reward).reshape(-1)[0])
        success = success or bool(info.get("success", False))
        used += 1
        if np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0]:
            break

    env.close()

    mesh_rrd_path = _record_mesh_rrd_from_recovered_qpos()

    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    if frames:
        print(f'saving to {out_video_path}')
        writer = imageio.get_writer(out_video_path, fps=20)
        for fr in frames:
            writer.append_data(fr)
        writer.close()

    return {
        "steps": used,
        "reward": total_reward,
        "success": success,
        "video": out_video_path if len(frames) > 0 else None,
        "mesh_rrd": mesh_rrd_path,
    }


def _run_smoke_test_for_config(
    robot_uid: str,
    control_mode: str,
    state_source: Literal["qpos", "target_qpos"],
    env_id: str,
    max_steps: int,
    out_dir: str,
    debug: bool,
) -> Dict[str, Any]:
    """Run end-to-end normalize/denormalize smoke test for one robot/mode pair."""
    batch = _load_one_batch_from_dataset(robot_uid[:-7] if robot_uid.endswith('_closed') else robot_uid)

    if batch is None:
        return {
            "robot": robot_uid,
            "mode": control_mode,
            "status": "skip",
            "reason": "dataset found but no readable trajectories",
        }

    normalizer = ActionNormalizer(
        robot_uid=robot_uid,
        control_mode=control_mode,
        state_source=state_source,
        debug=debug,
    )
    normalized = normalizer.normalize(batch)
    denorm_actions = normalizer.denormalize(normalized)
    denorm_target = normalizer.denormalize(normalized, return_full_qpos=False, return_eef_action=False)
    denorm_full = normalizer.denormalize(normalized, return_full_qpos=True, return_eef_action=False)

    if not torch.isfinite(denorm_target).all():
        denorm_target = torch.nan_to_num(denorm_target, nan=0.0, posinf=0.0, neginf=0.0)

    orig_qpos = normalizer._extract_state_qpos(batch)
    min_dim = min(orig_qpos.shape[-1], denorm_target.shape[-1])
    arm_eval_dim = min(normalizer._spec.arm_dim, min_dim)
    arm_diff = torch.abs(orig_qpos[..., :arm_eval_dim] - denorm_target[..., :arm_eval_dim])
    arm_finite = torch.isfinite(arm_diff)
    arm_rt_err = float(arm_diff[arm_finite].mean().item()) if arm_finite.any() else float("nan")
    
    if state_source == "target_qpos":
        per_dim_mae: List[float] = []
        arm_labels: List[str] = []
        for d in range(arm_eval_dim):
            dim_diff = arm_diff[..., d]
            dim_finite = torch.isfinite(dim_diff)
            if dim_finite.any():
                per_dim_mae.append(float(dim_diff[dim_finite].mean().item()))
            else:
                per_dim_mae.append(float("nan"))
            arm_labels.append(f"arm[joint_{d}]")

        orig_arm, orig_gripper = normalizer._split_target_qpos(orig_qpos)
        denorm_arm, denorm_gripper = normalizer._split_target_qpos(denorm_target)
        del orig_arm, denorm_arm

        orig_close = normalizer._close_signal_from_gripper_raw(orig_gripper)
        denorm_close = normalizer._close_signal_from_gripper_raw(denorm_gripper)
        close_diff = torch.abs(orig_close - denorm_close)
        close_finite = torch.isfinite(close_diff)
        gripper_signal_mae = float(close_diff[close_finite].mean().item()) if close_finite.any() else float("nan")

        if orig_gripper.shape[-1] > 0 and denorm_gripper.shape[-1] > 0:
            approx_dim = min(orig_gripper.shape[-1], denorm_gripper.shape[-1])
            raw_diff = torch.abs(orig_gripper[..., :approx_dim] - denorm_gripper[..., :approx_dim])
            raw_finite = torch.isfinite(raw_diff)
            gripper_raw_mae_approx = float(raw_diff[raw_finite].mean().item()) if raw_finite.any() else float("nan")
        else:
            gripper_raw_mae_approx = float("nan")

        denorm_target_np = denorm_target.detach().cpu().numpy()
        denorm_full_np = denorm_full.detach().cpu().numpy()

    video_path = os.path.join(out_dir, f"{robot_uid}__{control_mode}.mp4")
    _playback_and_record(
        robot_uid=robot_uid,
        control_mode=control_mode,
        actions=denorm_actions.detach().cpu().numpy(),
        recover_qposes=denorm_full.detach().cpu().numpy(),
        out_video_path=video_path,
        max_steps=max_steps,
        env_id=env_id,
        joint_names=normalizer.joint_names,
    )


def _main() -> None:
    """CLI entry point for running smoke tests across all robots and control modes."""
    parser = argparse.ArgumentParser(description="ActionNormalizer smoke test loop")
    parser.add_argument("--env_id", type=str, default="TableOnly-v2")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="runs/action_normalizer_videos_target")
    parser.add_argument("--state_source", type=str, default="target_qpos", choices=["qpos", "target_qpos"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    all_modes = ["pd_joint_pos", "pd_ee_pose"]
    all_robots = robots.get_robot_uids()

    print("[ActionNormalizer] Running all configuration smoke tests...")
    print(f"[ActionNormalizer] robots={all_robots}")
    print(f"[ActionNormalizer] modes={all_modes}")

    results = []
    for robot_uid in all_robots:
        for mode in all_modes:
            print(f"\n[ActionNormalizer] Testing robot={robot_uid}, mode={mode}")
            try:
                res = _run_smoke_test_for_config(
                    robot_uid=robot_uid,
                    control_mode=mode,
                    state_source=args.state_source,
                    env_id=args.env_id,
                    max_steps=args.max_steps,
                    out_dir=args.out_dir,
                    debug=args.debug,
                )
            except Exception as e:
                raise e
            results.append(res)
            print(robot_uid, mode)
            # _pretty_print_result(res)

    # print("\n[ActionNormalizer] Summary")
    # for r in results:
    #     _pretty_print_result(r)


if __name__ == "__main__":
    _main()

