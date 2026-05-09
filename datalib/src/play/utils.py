from numpy import isin
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien
from pytorch_kinematics import Transform3d


def sapien_pose_to_numpy(
    pose: sapien.Pose, return_6d: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a SAPIEN Pose to numpy arrays.

    Handles batched poses from ManiSkill environments by squeezing singleton
    batch dimensions (e.g., shape (1, 3) -> (3,)).

    IMPORTANT: SAPIEN uses wxyz quaternion format, but we convert to xyzw
    for compatibility with scipy.spatial.transform.Rotation.

    Args:
        pose: SAPIEN Pose object

    Returns:
        Tuple of (position [3], quaternion [4] in xyzw format for scipy)
    """
    if isinstance(pose, Transform3d):
        mat = pose.get_matrix()
        if mat.ndim == 3:
            mat = mat[0]
        p, q = matrix_to_pose(mat.cpu().numpy())
        return p, q

    import torch

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)

    p = to_np(pose.p)
    q = to_np(pose.q)  # SAPIEN: wxyz

    # Handle batched poses (e.g., from ManiSkill num_envs=1)
    # Squeeze singleton batch dimensions: (1, 3) -> (3,), (1, 4) -> (4,)
    if p.ndim == 2 and p.shape[0] == 1:
        p = p.squeeze(axis=0)
    if q.ndim == 2 and q.shape[0] == 1:
        q = q.squeeze(axis=0)

    # Convert wxyz (SAPIEN) to xyzw (scipy)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])

    if return_6d:
        # Return position and euler angles (xyz)
        euler = R.from_quat(q_xyzw).as_euler("xyz")
        return np.concatenate([p, euler])

    return p, q_xyzw


def numpy_to_sapien_pose(position: np.ndarray, quaternion: np.ndarray) -> sapien.Pose:
    """
    Convert numpy arrays to a SAPIEN Pose.

    IMPORTANT: Input quaternion is in xyzw format (scipy convention),
    but SAPIEN expects wxyz, so we convert.

    Args:
        position: Position array [3]
        quaternion: Quaternion array [4] in xyzw format (scipy convention)

    Returns:
        SAPIEN Pose object
    """
    # Convert xyzw (scipy) to wxyz (SAPIEN)
    q_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    return sapien.Pose(p=position, q=q_wxyz)


def pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Convert position + quaternion to a 4x4 homogeneous transformation matrix.

    Args:
        position: Position array [3]
        quaternion: Quaternion array [4] in xyzw format (SAPIEN convention)

    Returns:
        4x4 transformation matrix
    """
    # Convert xyzw to scipy's expected format (xyzw is already scipy's format)
    rotation = R.from_quat(quaternion)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position

    return matrix


def matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 homogeneous transformation matrix to position + quaternion.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        Tuple of (position [3], quaternion [4] in xyzw format)
    """
    position = matrix[:3, 3].copy()
    rotation = R.from_matrix(matrix[:3, :3])
    quaternion = rotation.as_quat()  # Returns xyzw format

    return position, quaternion


def get_actor_world_pose(actor) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the world pose of an actor (handles both SAPIEN and ManiSkill actors).

    Args:
        actor: SAPIEN Actor or ManiSkill Actor wrapper

    Returns:
        (position, quaternion) in world frame
    """
    # Handle both SAPIEN actors and ManiSkill wrapped actors
    if hasattr(actor, "pose"):
        pose = actor.pose
    else:
        # Fallback for raw SAPIEN actors
        pose = actor.get_pose()

    return sapien_pose_to_numpy(pose)


def interpolate_pose(
    pose1: tuple[np.ndarray, np.ndarray], pose2: tuple[np.ndarray, np.ndarray], t: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate between two poses.

    Args:
        pose1: Start pose (position, quaternion)
        pose2: End pose (position, quaternion)
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated pose (position, quaternion)
    """
    # Linear interpolation for position
    position = (1 - t) * pose1[0] + t * pose2[0]

    # SLERP for rotation
    r1 = R.from_quat(pose1[1])
    r2 = R.from_quat(pose2[1])

    # Use scipy's SLERP
    from scipy.spatial.transform import Slerp

    slerp = Slerp([0, 1], R.concatenate([r1, r2]))
    quaternion = slerp(t).as_quat()

    return position, quaternion


def just_run_env(env):
    env.reset()
    try:
        while True:
            # Step physics directly to avoid RL/reward overhead
            if env.agent.control_mode == "pd_ee_pose":
                action = np.zeros_like(env.action_space.sample())
                tmp = sapien_pose_to_numpy(
                    env.agent.robot.pose.inv() * env.agent.tcp_pose, return_6d=True
                )
                action[: len(tmp)] = tmp
            elif "delta" in env.agent.control_mode:
                action = np.zeros_like(env.action_space.sample())
            else:
                action = env.agent.robot.qpos
            env.step(action)
            env.render()

            # Check for keyboard input
            viewer = env.unwrapped.viewer
            if viewer is None or viewer.window is None:
                break

            if viewer.window.should_close:
                break

            # env.step()

            if viewer.window.key_press("q"):
                print("Quitting...")
                break
            if viewer.window.key_press("r"):
                print("Resetting environment...")
                obs, info = env.reset()
                print(f"New distractors: {len(env.unwrapped.distractors)}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Done!")
