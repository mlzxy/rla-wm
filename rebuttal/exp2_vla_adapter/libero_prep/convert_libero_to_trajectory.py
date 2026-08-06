#!/usr/bin/env python
"""
Convert the LIBERO RLDS datasets into this repo's ManiSkill trajectory layout.

Source (downloaded by rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh):

    data/libero/libero_{spatial,object,goal,10}_no_noops/1.0.0/*.tfrecord-*

Each RLDS episode holds `observation.image` (front / agentview), `observation.wrist_image`,
`observation.state` (EEF xyz + axis-angle + 2 gripper joints), `observation.joint_state`
(7 arm joints), `action` (7-D OSC delta) and `language_instruction`.

Destination (same shape as data/iws_converted, which TrajectoryDataset reads):

    data/libero_converted/<suite>/
        index.json  metadata.json
        traj_000000/
            agentview_camera_rgb.mp4    agentview_camera_foreground_mask.mp4
            wrist_camera_rgb.mp4        wrist_camera_foreground_mask.mp4
            metadata.h5

Camera intrinsics and extrinsics are not in the RLDS export. They come from
`rebuttal/exp2_vla_adapter/libero_prep/extract_libero_camera_params.py`, which queries a live LIBERO env once per task; run it
before converting. Extrinsics are written as OpenCV-convention camera-to-world `[T, 1, 3, 4]`,
matching what TrajectoryDataset reads back (it stores them under `w2c` but never inverts them),
and they align with the RLDS image orientation directly.

RGB streams are written **visually lossless** (VideoEncoding.H264_LOSSLESS / yuv444p qp=0): full
chroma resolution and no lossy quantization, so the `*_rgb.mp4` frames match the RLDS pixels to
within RGB↔YUV rounding (|diff| <= ~2/255, no blocking) and play correctly in every viewer. The
old default (lossy H.264 crf23 / yuv420p) produced visible chroma-subsampling colour blocks, worst
on the close-up wrist camera. To re-encode already-converted data in place without touching
metadata, use `rebuttal/exp2_vla_adapter/libero_prep/reencode_libero_rgb_lossless.py`.

One genuine placeholder remains: **foreground masks are constant 255.** LIBERO ships no
segmentation, so RlaAutoencoderTrainer's `rgbs * (foreground_masks > 0)` becomes the identity
and the model sees full frames. Real masks would need re-rendering the demos in the simulator.

Usage:

    export PYTHONPATH=.:./third_party/diffusion_policy
    .venv/bin/python rebuttal/exp2_vla_adapter/libero_prep/extract_libero_camera_params.py
    .venv/bin/python rebuttal/exp2_vla_adapter/libero_prep/convert_libero_to_trajectory.py --workers 8            # all four suites
"""

import argparse
import json
import os
import sys

import numpy as np

# TensorFlow must stay off the GPU: it is only used to decode TFRecords here, and eager context
# init aborts on hosts with a faulty GPU (upstream VLA-Adapter pins TF to CPU for the same reason).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root, three levels up from rebuttal/exp2_vla_adapter/libero_prep/:
# gives datalib/, src/, utils/ and the rebuttal/ package.
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from datalib.dataset import (
    ManiSkillTrajectoryDataset,
    RobotInfo,
    TrajectoryData,
    VideoEncoding,
)

# RGB is stored visually-lossless: H.264 with full-resolution chroma (yuv444p) and qp=0. This
# removes the chroma-subsampling colour blocks the historical default (lossy crf23 / yuv420p) left
# on the close-up wrist camera, is imperceptibly close to the source (|diff| <= ~2/255, no
# blocking), and plays correctly in every viewer. (Bit-exact libx264rgb was rejected: its gbrp
# stream renders pink in standard players.)
RGB_ENCODING = VideoEncoding.H264_LOSSLESS


# RLDS directory name -> output dataset name.
SUITES = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

# Camera names must contain "cam": TrajectoryDataset._build_single_trajectory_info filters stream
# keys with `if "cam" in k` when counting frames, the same way `front_lower_camera` (maniskill) and
# `camera_0` (iws) do.
FRONT_CAM = "agentview_camera"
WRIST_CAM = "wrist_camera"

# Camera geometry comes from rebuttal/exp2_vla_adapter/libero_prep/extract_libero_camera_params.py, which queries a live LIBERO
# env per task. Do NOT try to read it out of the scene XMLs: they all declare
# `agentview pos="0.5 0 1.35"`, but LIBERO repositions the camera when the arena loads, and the
# runtime pose is scene dependent (four distinct poses across the 40 tasks). Likewise the
# `eye_in_hand` camera hangs off the hand body, ~9.7 cm behind the grip site that
# `observation.state` reports, not at the 5 cm offset the robot XML suggests.
CAMERA_PARAMS_DEFAULT = "data/libero_converted/camera_params.json"

PANDA_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
PANDA_GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def panda_robot_info() -> RobotInfo:
    """RobotInfo for the LIBERO Panda. Action space is the 7-D OSC delta LIBERO actually uses."""
    joint_names = PANDA_ARM_JOINTS + PANDA_GRIPPER_JOINTS
    return RobotInfo(
        uid="libero_panda",
        urdf_path="",
        urdf_config={},
        joint_names=joint_names,
        action_space=([-1.0] * 7, [1.0] * 7, list(range(7))),
        action_mapping={name: (i, i + 1) for i, name in enumerate(joint_names[:7])},
    )


def load_camera_params(path: str = CAMERA_PARAMS_DEFAULT) -> dict:
    """Load the simulator-extracted camera parameters."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it first:\n"
            "  .venv/bin/python rebuttal/exp2_vla_adapter/libero_prep/extract_libero_camera_params.py"
        )
    with open(path) as f:
        return json.load(f)


def episode_camera_arrays(
    task_description: str,
    eef_pos: np.ndarray,
    eef_rot: np.ndarray,
    params: dict,
    height: int,
    width: int,
) -> dict:
    """Per-frame camera intrinsics/extrinsics for one episode.

    Args:
        task_description: language instruction, the key into the params table.
        eef_pos: ``[T, 3]`` end-effector positions from ``observation.state``.
        eef_rot: ``[T, 3, 3]`` end-effector rotations.

    Returns dict of ``[T, 1, 3, 3]`` intrinsics and ``[T, 1, 3, 4]`` camera-to-world extrinsics,
    keyed by stream name. The singleton camera axis is the layout
    TrajectoryDataset.__getitem__ assigns into (`intrinsics[:, [cam_idx]] = ...`).
    """
    entry = params["tasks"].get(task_description)
    if entry is None:
        raise KeyError(
            f"No camera parameters for task {task_description!r}. Re-run "
            "rebuttal/exp2_vla_adapter/libero_prep/extract_libero_camera_params.py so every suite is covered."
        )

    num_frames = len(eef_pos)
    # The sim was queried at one resolution; rescale if the stored frames differ.
    scale = height / float(params["resolution"])
    front_k = np.array(entry["agentview_K"], dtype=np.float64)
    wrist_k = np.array(entry["wrist_K"], dtype=np.float64)
    if not np.isclose(scale, 1.0):
        front_k[:2] *= scale
        wrist_k[:2] *= scale

    # The RLDS frames are a HORIZONTAL MIRROR of the standard camera frame, so a ray built from
    # unmodified intrinsics points at the wrong side of the scene.
    #
    # Why: robosuite renders with IMAGE_CONVENTION="opengl" (robosuite/macros.py), so
    # `obs['<cam>_image']` is the raw bottom-up render, while `get_camera_extrinsic_matrix`
    # applies the OpenCV axis correction -- i.e. the matrices describe `obs[::-1]`. The RLDS
    # export stores `obs[::-1, ::-1]` (the 180-degree rotation that
    # `libero_utils.get_libero_image` undoes at eval), and
    #     obs[::-1, ::-1] == (obs[::-1])[:, ::-1]
    # leaves exactly one horizontal mirror between the matrices and the stored pixels.
    #
    # A mirror is improper, so it cannot be folded into the extrinsics without making them
    # non-rigid. Negating fx is the standard way to express it and keeps the extrinsics a proper
    # rigid transform; inv(K) then yields correctly mirrored ray directions.
    #
    # This is invisible at the robot's home pose (the gripper sits at u ~ W/2, where a horizontal
    # mirror is a no-op), which is why frame-0 reprojection checks cannot detect it. The check
    # that does is cross-view photometric consistency in rebuttal/exp2_vla_adapter/libero_prep/verify_libero_plucker.py.
    front_k[0, 0] *= -1.0
    wrist_k[0, 0] *= -1.0

    front_e = np.array(entry["agentview_c2w"], dtype=np.float64)  # [3, 4], world-fixed

    # The wrist camera rides the hand: T_world_cam = T_world_eef @ T_eef_cam.
    eef_to_cam = np.array(params["eef_to_wrist_cam"], dtype=np.float64)
    world_eef = np.tile(np.eye(4), (num_frames, 1, 1))
    world_eef[:, :3, :3] = eef_rot
    world_eef[:, :3, 3] = eef_pos
    wrist_e = (world_eef @ eef_to_cam)[:, :3, :]

    return {
        f"{FRONT_CAM}_intrinsics": np.tile(front_k, (num_frames, 1, 1, 1)).astype(np.float32),
        f"{FRONT_CAM}_extrinsics": np.tile(front_e, (num_frames, 1, 1, 1)).astype(np.float32),
        f"{WRIST_CAM}_intrinsics": np.tile(wrist_k, (num_frames, 1, 1, 1)).astype(np.float32),
        f"{WRIST_CAM}_extrinsics": wrist_e[:, None].astype(np.float32),
    }


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from a MuJoCo-convention (w, x, y, z) quaternion."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation matrices from [T, 3] axis-angle vectors (Rodrigues' formula)."""
    angle = np.linalg.norm(axis_angle, axis=-1)
    safe_angle = np.where(angle < 1e-8, 1.0, angle)
    axis = axis_angle / safe_angle[:, None]
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(kx)
    skew = np.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1
    ).reshape(-1, 3, 3)
    eye = np.broadcast_to(np.eye(3), (len(axis_angle), 3, 3))
    sin = np.sin(angle)[:, None, None]
    cos = np.cos(angle)[:, None, None]
    rot = eye + sin * skew + (1.0 - cos) * (skew @ skew)
    rot[angle < 1e-8] = np.eye(3)
    return rot


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert [T, 3] axis-angle rotations to [T, 4] wxyz quaternions."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    # Guard the angle==0 case: the axis is arbitrary there, so any unit axis gives identity.
    safe_angle = np.where(angle < 1e-8, 1.0, angle)
    axis = axis_angle / safe_angle
    half = 0.5 * angle
    quat = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
    quat[angle[:, 0] < 1e-8] = np.array([1.0, 0.0, 0.0, 0.0])
    return quat.astype(np.float32)


def episode_to_trajectory_data(episode, camera_params: dict) -> TrajectoryData:
    """Turn one decoded RLDS episode into a TrajectoryData ready for write_trajectory."""
    front, wrist, states, joints, actions = [], [], [], [], []
    instruction = ""
    for step in episode["steps"]:
        obs = step["observation"]
        front.append(obs["image"].numpy())
        wrist.append(obs["wrist_image"].numpy())
        states.append(obs["state"].numpy())
        joints.append(obs["joint_state"].numpy())
        actions.append(step["action"].numpy())
        if not instruction:
            instruction = step["language_instruction"].numpy().decode("utf-8")

    front = np.stack(front)  # [T, H, W, 3] uint8
    wrist = np.stack(wrist)
    states = np.stack(states).astype(np.float32)  # [T, 8]
    joints = np.stack(joints).astype(np.float32)  # [T, 7]
    actions = np.stack(actions).astype(np.float32)  # [T, 7]

    num_frames, height, width = front.shape[:3]

    # Full-scene masks: LIBERO has no segmentation, see module docstring.
    ones_mask = np.full((num_frames, height, width), 255, dtype=np.uint8)

    # Panda qpos is 7 arm joints + 2 gripper finger joints (the tail of `state`).
    qpos = np.concatenate([joints, states[:, 6:8]], axis=1)
    # LIBERO's action is an OSC delta, not a joint target, so use the achieved next qpos instead
    # and keep the raw action under its own key.
    target_qpos = np.concatenate([qpos[1:], qpos[-1:]], axis=0)

    eef_pose = np.concatenate([states[:, :3], axis_angle_to_quat(states[:, 3:6])], axis=1)

    root_poses = np.zeros((num_frames, 7), dtype=np.float32)
    root_poses[:, 3] = 1.0  # identity quaternion (wxyz), Panda base at the origin

    cameras = episode_camera_arrays(
        task_description=instruction,
        eef_pos=states[:, :3].astype(np.float64),
        eef_rot=axis_angle_to_matrix(states[:, 3:6].astype(np.float64)),
        params=camera_params,
        height=height,
        width=width,
    )

    return TrajectoryData(
        success=True,
        video_streams={
            f"{FRONT_CAM}_rgb": front,
            f"{FRONT_CAM}_foreground_mask": ones_mask,
            f"{WRIST_CAM}_rgb": wrist,
            f"{WRIST_CAM}_foreground_mask": ones_mask,
        },
        metadata={
            "qpos": qpos,
            "target_qpos": target_qpos,
            "action": actions,
            "eef_pose": eef_pose,
            "root_poses": root_poses,
            **cameras,
            "task_description": instruction,
        },
    )


def convert_shard(args) -> int:
    """Convert episodes [start, end) of one suite. Runs in a worker process."""
    rlds_dir, out_dir, start, end, skip_existing, camera_params_path = args

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    dataset = ManiSkillTrajectoryDataset(out_dir, rgb_encoding=RGB_ENCODING)
    builder = tfds.builder_from_directory(rlds_dir)
    camera_params = load_camera_params(camera_params_path)

    num_written = 0
    for offset, episode in enumerate(builder.as_dataset(split=f"train[{start}:{end}]")):
        traj_id = f"{start + offset:06d}"
        if skip_existing and os.path.exists(
            os.path.join(out_dir, f"traj_{traj_id}", "metadata.h5")
        ):
            continue
        dataset.write_trajectory(
            traj_id, episode_to_trajectory_data(episode, camera_params)
        )
        num_written += 1
        print(f"  [{os.path.basename(out_dir)}] wrote traj_{traj_id}", flush=True)
    return num_written


def convert_suite(
    suite: str,
    rlds_root: str,
    out_root: str,
    max_episodes: int,
    workers: int,
    skip_existing: bool,
    camera_params_path: str,
) -> None:
    rlds_dir = os.path.join(rlds_root, SUITES[suite], "1.0.0")
    if not os.path.isdir(rlds_dir):
        print(f"[skip] {suite}: {rlds_dir} does not exist")
        return

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    out_dir = os.path.join(out_root, suite)
    num_episodes = tfds.builder_from_directory(rlds_dir).info.splits["train"].num_examples
    if max_episodes > 0:
        num_episodes = min(num_episodes, max_episodes)
    print(f"[{suite}] {num_episodes} episodes -> {out_dir}")

    # Written before the shards run so every worker's ManiSkillTrajectoryDataset sees it.
    ManiSkillTrajectoryDataset(out_dir).save_robot_infos([panda_robot_info()])

    bounds = np.linspace(0, num_episodes, min(workers, num_episodes) + 1).astype(int)
    shards = [
        (rlds_dir, out_dir, int(a), int(b), skip_existing, camera_params_path)
        for a, b in zip(bounds[:-1], bounds[1:])
        if b > a
    ]

    if len(shards) == 1:
        num_written = convert_shard(shards[0])
    else:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(len(shards)) as pool:
            num_written = sum(pool.map(convert_shard, shards))

    # Rebuild after writing so index.json lists every trajectory and key.
    ManiSkillTrajectoryDataset(out_dir).build_index(force=True, verbose=False)
    print(f"[{suite}] wrote {num_written} new trajectories ({num_episodes} total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds-root", default="data/libero", help="Root of the RLDS datasets")
    parser.add_argument(
        "--out", default="data/libero_converted", help="Output root for converted trajectories"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["all"],
        choices=["all"] + list(SUITES),
        help="Suites to convert (default: all)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=-1, help="Convert only the first N episodes per suite"
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes per suite")
    parser.add_argument(
        "--camera-params",
        default=CAMERA_PARAMS_DEFAULT,
        help="JSON from rebuttal/exp2_vla_adapter/libero_prep/extract_libero_camera_params.py",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite trajectories that already exist (default: skip them)",
    )
    args = parser.parse_args()

    suites = list(SUITES) if "all" in args.suites else args.suites
    for suite in suites:
        convert_suite(
            suite=suite,
            rlds_root=args.rlds_root,
            out_root=args.out,
            max_episodes=args.max_episodes,
            workers=max(1, args.workers),
            skip_existing=not args.overwrite,
            camera_params_path=args.camera_params,
        )


if __name__ == "__main__":
    main()
