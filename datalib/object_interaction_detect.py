import sys
import os
import os.path as osp

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
import h5py
import json
import argparse
import imageio
import cv2
from tqdm import tqdm

import third_party.pytorch_kinematics as pk
from mani_skill.agents.registration import REGISTERED_AGENTS
import datalib.src.robots  # Ensure custom agents are registered


@torch.inference_mode()
def calculate_eef_positions(
    traj_path: str, override_cache: bool = False, log: bool = False
):
    """Calculate EEF positions for a given trajectory."""
    h5_path = os.path.join(traj_path, "metadata.h5")

    if not override_cache:
        try:
            with h5py.File(h5_path, "r") as f:
                if "eef_pose" in f:
                    if log:
                        print(f"Using cached EEF positions for {traj_path}")
                    return f["eef_pose"][:], f["object_poses"][:]
        except Exception:
            pass

    meta_json_path = os.path.join(os.path.dirname(traj_path), "metadata.json")
    if not os.path.exists(meta_json_path):
        meta_json_path = os.path.join(os.path.dirname(traj_path), "../metadata.json")

    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    robot_info = meta["robot_infos"][0]
    robot_type = robot_info.get("uid", robot_info.get("robot_type"))
    urdf_path_abs = os.path.join(str(ROOT_DIR), robot_info["urdf_path"])
    if osp.exists(urdf_path_abs.replace(".urdf", ".stl.urdf")):
        urdf_path_abs = urdf_path_abs.replace(".urdf", ".stl.urdf")

    assert robot_type in REGISTERED_AGENTS
    agent_cls = REGISTERED_AGENTS[robot_type].agent_cls
    eef_link_name = agent_cls.ee_link_name

    with open(urdf_path_abs, "r") as f:
        urdf_content = f.read()
    chain = pk.build_chain_from_urdf(urdf_content)

    with h5py.File(h5_path, "r") as f:
        qpos = f["qpos"][:]
        root_poses = f["root_poses"][:]
        object_poses = f["object_poses"][:]

    qpos_tensor = torch.tensor(qpos[:, 0, :], dtype=torch.float32)

    root_pos = root_poses[:, 0, :3]
    root_rot = root_poses[:, 0, 3:]  # qw, qx, qy, qz

    root_transform = pk.Transform3d(
        pos=torch.tensor(root_pos, dtype=torch.float32),
        rot=torch.tensor(root_rot, dtype=torch.float32),
    )

    ret = chain.forward_kinematics(qpos_tensor)
    eef_local_transform = ret[eef_link_name]

    eef_world_transform = root_transform.compose(eef_local_transform)
    eef_mat = eef_world_transform.get_matrix()
    p_torch = eef_mat[:, :3, 3]
    q_torch = pk.transforms.matrix_to_quaternion(eef_mat[:, :3, :3])
    eef_pose = torch.cat([p_torch, q_torch], dim=-1).numpy()

    try:
        with h5py.File(h5_path, "r+") as f:
            if "eef_pose" in f:
                del f["eef_pose"]
            f.create_dataset("eef_pose", data=eef_pose)
    except Exception as e:
        print(f"Warning: Could not save eef_pose to {h5_path}: {e}")

    return eef_pose, object_poses


def detect_interactions(
    traj_path: str,
    dist_threshold: float = 0.25,
    movement_threshold: float = 0.002,
    buffer_window: int = 5,
    output_vis_path: str = None,
    cam_name: str = "base_camera",
    override_cache: bool = False,
):
    eef_pose, object_poses = calculate_eef_positions(
        traj_path, override_cache=override_cache, log=(output_vis_path is not None)
    )
    eef_pos = eef_pose[..., :3]
    T = eef_pos.shape[0]
    num_objects = object_poses.shape[1]

    obj_pos = object_poses[..., :3]

    eef_pos_expanded = eef_pos[:, None, :]
    distances = np.linalg.norm(obj_pos - eef_pos_expanded, axis=-1)

    close_to_eef = distances < dist_threshold

    obj_movement = np.zeros((T, num_objects))
    obj_movement[1:] = np.linalg.norm(obj_pos[1:] - obj_pos[:-1], axis=-1)

    moving_objects = obj_movement > movement_threshold

    # object interaction happens only when object is moving AND object is close to eef
    raw_interacting = np.any(close_to_eef & moving_objects, axis=-1)

    buffered_frames = np.copy(raw_interacting)
    for i in range(T):
        if raw_interacting[i]:
            start = max(0, i - buffer_window)
            end = min(T, i + buffer_window + 1)
            buffered_frames[start:end] = True

    if output_vis_path is not None:
        video_path = os.path.join(traj_path, f"{cam_name}_rgb.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found for visualization.")
            # fallback to the first available mp4
            import glob

            mp4s = glob.glob(os.path.join(traj_path, "*_rgb.mp4"))
            if mp4s:
                video_path = mp4s[0]
                print(f"Using fallback video: {video_path}")

        if os.path.exists(video_path):
            print(f"Visualizing onto {video_path}, saving to {output_vis_path}")
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data().get("fps", 20)
            writer = imageio.get_writer(output_vis_path, fps=fps)

            for t, frame in enumerate(tqdm(reader, total=T, desc="Rendering video")):
                if t >= T:
                    break

                img = frame.copy()

                if raw_interacting[t]:
                    label = "INTERACTING"
                    color = (255, 0, 0)  # Red in RGB
                elif buffered_frames[t]:
                    label = "BUFFER"
                    color = (255, 165, 0)  # Orange
                else:
                    label = "IDLE"
                    color = (128, 128, 128)  # Gray

                # cv2 draws in BGR if we use imwrite, but imageio works with RGB. The color tuple here is applied directly to the array.
                cv2.putText(
                    img,
                    f"Frame: {t} | Status: {label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                cv2.putText(
                    img,
                    f"Min Dist: {np.min(distances[t]):.3f}m | Max Move: {np.max(obj_movement[t]):.4f}m",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                # Draw a colored border
                cv2.rectangle(
                    img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 8
                )

                writer.append_data(img)

            writer.close()
            reader.close()

    return (
        buffered_frames,
        raw_interacting,
        {
            "distances": distances,
            "movement": obj_movement,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="Process a single trajectory")
    single_parser.add_argument("--traj", type=str, required=True)
    single_parser.add_argument("--vis-out", type=str, default=None)
    single_parser.add_argument("--dist-thresh", type=float, default=0.25)
    single_parser.add_argument("--move-thresh", type=float, default=0.002)
    single_parser.add_argument("--buffer", type=int, default=5)
    single_parser.add_argument("--cam", type=str, default="base_camera")
    single_parser.add_argument("--override-cache", action="store_true")

    batch_parser = subparsers.add_parser("batch", help="Batch process from config")
    batch_parser.add_argument("--config", type=str, required=True)
    batch_parser.add_argument("--override-cache", action="store_true")
    batch_parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="Number of parallel workers (default: cpu count / 2)",
    )

    args = parser.parse_args()

    if args.command == "single":
        buffered, raw, stats = detect_interactions(
            args.traj,
            output_vis_path=args.vis_out,
            dist_threshold=args.dist_thresh,
            movement_threshold=args.move_thresh,
            buffer_window=args.buffer,
            cam_name=args.cam,
            override_cache=args.override_cache,
        )

        print(f"Total frames: {len(buffered)}")
        print(f"Raw interacting frames: {np.sum(raw)}")
        print(f"Buffered interacting frames: {np.sum(buffered)}")
        print("Mask:", buffered.astype(int).tolist())

    elif args.command == "batch":
        import yaml
        import glob
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        dataset_config = config.get("dataset", {}).get("args", {})
        root = dataset_config.get("root", "")
        configs = dataset_config.get("configs", {})

        expanded_paths = []
        for config_name, data_config in configs.items():
            paths = data_config.get("paths", [])
            for path in paths:
                if "*" in path or "?" in path:
                    if os.path.isabs(path):
                        expanded = sorted(glob.glob(path))
                    else:
                        expanded = sorted(glob.glob(os.path.join(root, path)))
                    expanded_paths.extend(expanded)
                else:
                    if os.path.isabs(path):
                        expanded_paths.append(path)
                    else:
                        expanded_paths.append(os.path.join(root, path))

        all_trajectories = []
        for dataset_path in expanded_paths:
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path does not exist: {dataset_path}")
                continue

            trajs = glob.glob(os.path.join(dataset_path, "traj_*"))
            all_trajectories.extend(trajs)

        print("--- Found Datasets ---")
        for dp in expanded_paths:
            print(f"  {dp}")

        print(
            f"\nFound {len(all_trajectories)} trajectories across {len(expanded_paths)} dataset folders."
        )

        if all_trajectories:
            print("\nSample trajectories (first 5):")
            for t in all_trajectories[:5]:
                print(f"  {t}")

        reply = input("\nProceed with processing? [y/N]: ")
        if reply.lower() != 'y':
            print("Aborting operation.")
            sys.exit(0)

        def process_traj(traj_path):
            try:
                calculate_eef_positions(
                    traj_path, override_cache=args.override_cache, log=False
                )
                return True
            except Exception as e:
                return str(e)

        if args.num_workers > 1:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(process_traj, p): p for p in all_trajectories
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Caching EEF positions",
                ):
                    res = future.result()
                    if res is not True:
                        p = futures[future]
                        print(f"Error processing {p}: {res}")
        else:
            for traj_path in tqdm(all_trajectories, desc="Caching EEF positions"):
                res = process_traj(traj_path)
                if res is not True:
                    print(f"Error processing {traj_path}: {res}")
        print("Done batch caching.")
