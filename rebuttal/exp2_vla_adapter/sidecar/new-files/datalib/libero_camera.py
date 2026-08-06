"""LIBERO camera geometry: simulator-extracted parameters and per-episode K / camera-to-world.

Moved verbatim out of `scripts/convert_libero_to_trajectory.py` so consumers that need the geometry
but *not* the converter can import it. The converter sets `CUDA_VISIBLE_DEVICES=""` at module scope
(it only decodes TFRecords and must keep TensorFlow off the GPU), which blinds torch for anything
that imports it -- `scripts/extract_rla_latents.py` needs both TF-on-CPU and torch-on-GPU in one
process, so it imports from here instead. The converter re-exports these names, so existing
importers (`scripts/backfill_libero_camera_params.py`,
`scripts/test_libero_camera_geometry.py`) keep working unchanged.

Everything here is pure numpy: no TensorFlow, no torch, no environment side effects.
"""

import json
import os

import numpy as np

# Camera names must contain "cam": TrajectoryDataset._build_single_trajectory_info filters stream
# keys with `if "cam" in k` when counting frames, the same way `front_lower_camera` (maniskill) and
# `camera_0` (iws) do.
FRONT_CAM = "agentview_camera"
WRIST_CAM = "wrist_camera"

# Camera geometry comes from scripts/extract_libero_camera_params.py, which queries a live LIBERO
# env per task. Do NOT try to read it out of the scene XMLs: they all declare
# `agentview pos="0.5 0 1.35"`, but LIBERO repositions the camera when the arena loads, and the
# runtime pose is scene dependent (four distinct poses across the 40 tasks). Likewise the
# `eye_in_hand` camera hangs off the hand body, ~9.7 cm behind the grip site that
# `observation.state` reports, not at the 5 cm offset the robot XML suggests.
CAMERA_PARAMS_DEFAULT = "data/libero_converted/camera_params.json"


def load_camera_params(path: str = CAMERA_PARAMS_DEFAULT) -> dict:
    """Load the simulator-extracted camera parameters."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it first:\n"
            "  source scripts/vla_adapter_env.sh\n"
            "  .venv/bin/python scripts/extract_libero_camera_params.py"
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
            "scripts/extract_libero_camera_params.py so every suite is covered."
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
    # that does is cross-view photometric consistency in scripts/verify_libero_plucker.py.
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
