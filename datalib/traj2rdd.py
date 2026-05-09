import argparse
from tqdm import tqdm
import numpy as np
import torch
import os
from typing import Optional, Tuple, List
import rerun as rr
from datalib.dataset import ManiSkillTrajectoryDataset
from datalib.robot_geometry import (
    DifferentiableRobotGeometry,
    to_o3d_mesh,
    mesh_to_arrays,
)
from mani_skill import PACKAGE_ASSET_DIR
from utils.mesh import to_mesh
from utils.voxel import VoxelizationLayer
import os.path as osp

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: Optional[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    max_depth: float = 2.0,
    foreground_mask: Optional[np.ndarray] = None,
    attrs: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Convert depth image to point cloud using camera intrinsics.

    Args:
        depth: Depth image (H, W) in meters
        rgb: RGB image (H, W, 3) or None
        intrinsics: Camera intrinsics matrix (3, 3) in OpenCV format
        extrinsics: Camera extrinsics matrix (4, 4) or None (world to camera transform)
        max_depth: Maximum depth in meters to include (default: 2.0). Points with depth greater than this are ignored.
        foreground_mask: Optional boolean mask (H, W) where True indicates foreground pixels to include.
                        Points outside the foreground mask are ignored.
        attrs: Optional list of attributes (H, W) or (H, W, C) to sample at valid points.

    Returns:
        Tuple of (points, colors, out_attrs):
        - points: (N, 3) point cloud in world coordinates (or camera coordinates if extrinsics is None)
        - colors: (N, 3) RGB colors or None
        - out_attrs: List of (N,) or (N, C) attribute arrays, or None if attrs is None
    """
    H, W = depth.shape

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Get valid depth pixels (non-zero, non-inf, non-nan, and within max_depth)
    # Points beyond max_depth are ignored (not included in the point cloud)
    valid_mask = (depth > 0) & (depth <= max_depth) & np.isfinite(depth)

    # Apply foreground mask if provided
    if foreground_mask is not None:
        if foreground_mask.shape != (H, W):
            raise ValueError(
                f"foreground_mask shape {foreground_mask.shape} does not match depth shape {(H, W)}"
            )
        # Convert to boolean if needed
        if foreground_mask.dtype != bool:
            foreground_mask = foreground_mask.astype(bool)
        valid_mask = valid_mask & foreground_mask

    # Filter valid pixels
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]

    # Convert to camera coordinates using intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Camera coordinates: x = (u - cx) * z / fx, y = (v - cy) * z / fy, z = depth
    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = (v_valid - cy) * depth_valid / fy
    z_cam = depth_valid

    # Stack into (N, 3) array
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Transform to world coordinates if extrinsics provided
    if extrinsics is not None:
        # Extrinsics is world-to-camera, so we need camera-to-world
        T_cam_to_world = np.linalg.inv(extrinsics)

        # Homogeneous coordinates
        points_cam_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1))], axis=-1
        )

        # Transform: points_world = T_cam_to_world @ points_cam
        points_world_h = (T_cam_to_world @ points_cam_h.T).T
        points = points_world_h[:, :3]
    else:
        points = points_cam

    # Extract colors if RGB provided
    colors = None
    if rgb is not None:
        colors = rgb[valid_mask]
        # Normalize to [0, 1] if needed
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0

    # Extract attrs if provided
    out_attrs = None
    if attrs is not None:
        out_attrs = []
        for attr in attrs:
            # attr should be (H, W) or (H, W, C)
            # if (H, W), attr[valid_mask] -> (N,)
            # if (H, W, C), attr[valid_mask] -> (N, C)
            out_attrs.append(attr[valid_mask])

    return points, colors, out_attrs


def visualize_trajectory(
    dataset: ManiSkillTrajectoryDataset,
    traj_id: str,
    output_path: str,
    use_target_qpos: bool = False,
    max_frames: Optional[int] = None,
    img_size: Optional[int] = None,
    voxelize: bool = False,
    voxel_resolution: Optional[Tuple[int, int, int]] = None,
    voxel_min_cell_size: Optional[Tuple[float, float, float]] = None,
    voxel_center_offset_ratio: float = 0.1,
    voxel_bound_ratio: float = 0.05,
    vis_masks: bool = False,
):
    """
    Visualize a trajectory and save to RRD file.

    Args:
        dataset: ManiSkillTrajectoryDataset instance
        traj_id: Trajectory ID to visualize
        output_path: Path to save RRD file
        use_target_qpos: If True, use target_qpos instead of qpos for robot visualization
        max_frames: Maximum number of frames to visualize (None = all)
        img_size: If specified, resize all images to img_size x img_size
        voxelize: If True, voxelize point clouds before visualization
        voxel_resolution: Resolution of the voxel grid (D, H, W). Default: (64, 64, 64)
        voxel_min_cell_size: Physical size of one voxel in meters (x, y, z). Default: (0.05, 0.05, 0.05)
        voxel_center_offset_ratio: Augmentation parameter for center offset. Default: 0.1
        voxel_bound_ratio: Augmentation parameter for bound scaling. Default: 0.05
    """
    # Read trajectory data (with optional resizing)
    traj_id = traj_id.zfill(6)
    print(f"Reading trajectory {traj_id}...")
    traj_data = dataset.read_trajectory(
        traj_id, max_frames=max_frames, img_size=img_size
    )

    # Get robot infos
    robot_infos = dataset.get_robot_infos()
    if robot_infos is None or len(robot_infos) == 0:
        print("Warning: No robot infos found in dataset")
        robot_infos = []

    # Initialize rerun recording
    print(f"Initializing rerun recording to {output_path}...")
    rr.init("trajectory_visualization", spawn=False)
    rr.save(output_path)

    # Get number of frames
    first_key = next(iter(traj_data.video_streams))
    first_frames = traj_data.video_streams[first_key]
    num_frames = first_frames.shape[0]

    print(f"Visualizing {num_frames} frames...")

    # Initialize robot geometry loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize voxelization layer if enabled
    voxel_layer = None
    if voxelize:
        resolution = (
            voxel_resolution if voxel_resolution is not None else (192, 192, 192)
        )
        min_cell_size = (
            voxel_min_cell_size
            if voxel_min_cell_size is not None
            else (0.01, 0.01, 0.01)
        )
        voxel_layer = VoxelizationLayer(
            resolution=resolution,
            min_cell_size=min_cell_size,
        ).to(device)
        print(
            f"Voxelization enabled: resolution={resolution}, min_cell_size={min_cell_size}"
        )

    robot_geometries: dict[str, DifferentiableRobotGeometry] = {}
    for robot_info in robot_infos:
        urdf_path_abs = os.path.join(str(ROOT_DIR), robot_info.urdf_path)
        urdf_dir = os.path.dirname(urdf_path_abs)
        
        # We need to compute joint names by temporarily instantiating the env 
        # based on the robot_uid to keep identical behavior with other scripts
        import gymnasium as gym
        import datalib.src.tasks 
        env_tmp = gym.make(
            "TableOnly-v2",
            obs_mode="state",
            control_mode="pd_joint_pos",
            robot_uids=robot_info.uid,
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            include_all_cameras=False,
            max_episode_steps=10,
        )
        joint_names = [joint.name for joint in env_tmp.agent.robot.get_active_joints()]
        env_tmp.close()
        
        robot_geom = DifferentiableRobotGeometry(
            urdf_path=urdf_path_abs,
            base_dir=urdf_dir,
            joint_names=joint_names,
        )
        robot_geometries[robot_info.uid] = robot_geom
        print(f"Loaded robot geometry for {robot_info.uid} with joints {joint_names}")

    # Extract camera names from video stream keys
    camera_names = set()
    for key in traj_data.video_streams.keys():
        if key.endswith("_rgb"):
            cam_name = key[:-4]  # Remove '_rgb' suffix
            camera_names.add(cam_name)

    # Process each frame
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        rr.set_time("frame", sequence=frame_idx)

        # Log RGB images and masks for each camera
        for cam_name in sorted(camera_names):
            rgb_key = f"{cam_name}_rgb"
            depth_key = f"{cam_name}_depth"
            robot_mask_key = f"{cam_name}_robot_mask"
            foreground_mask_key = f"{cam_name}_foreground_mask"
            static_mask_key = f"{cam_name}_static_mask"

            # Get frames (already resized if img_size was specified in read_trajectory)
            rgb_frame = traj_data.video_streams[rgb_key][frame_idx]
            depth_frame = (
                traj_data.video_streams[depth_key][frame_idx].astype(float) / 1000
            )
            robot_mask_frame = traj_data.video_streams[robot_mask_key][frame_idx]
            foreground_mask_frame = traj_data.video_streams[foreground_mask_key][
                frame_idx
            ]
            static_mask_frame = None
            if static_mask_key in traj_data.video_streams:
                static_mask_frame = traj_data.video_streams[static_mask_key][frame_idx]

            # Get image dimensions
            H, W = rgb_frame.shape[:2]

            # Get camera intrinsics and extrinsics (already adjusted if resized)
            intrinsics_key = f"{cam_name}_intrinsics"
            extrinsics_key = f"{cam_name}_extrinsics"

            intrinsics = traj_data.metadata[intrinsics_key][frame_idx].squeeze()
            extrinsics = traj_data.metadata[extrinsics_key][frame_idx].squeeze()

            # Convert extrinsics (3x4) to (4x4)
            if extrinsics.shape == (3, 4):
                extrinsics = np.vstack([extrinsics, np.array([0, 0, 0, 1])])

            # Log RGB image
            rr.log(f"cameras/{cam_name}/rgb", rr.Image(rgb_frame))

            # Log depth image
            # rr.log(f"cameras/{cam_name}/depth", rr.Image(depth_frame))

            # Log robot mask
            if vis_masks:
                rr.log(f"cameras/{cam_name}/robot_mask", rr.Image(robot_mask_frame))

            # Log foreground mask
            if vis_masks:
                rr.log(
                    f"cameras/{cam_name}/foreground_mask",
                    rr.Image(foreground_mask_frame),
                )

            # Log static mask if available
            if static_mask_frame is not None and vis_masks:
                rr.log(f"cameras/{cam_name}/static_mask", rr.Image(static_mask_frame))

            # Log camera transform (extrinsics is world-to-camera)
            # For visualization, we need camera-to-world transform
            # Rerun Transform3D with mat3x3 and translation expects transform from parent to child
            # So we invert extrinsics to get camera-to-world (positioning camera in world space)
            camera_to_world = np.linalg.inv(extrinsics)
            rr.log(
                f"cameras/{cam_name}",
                rr.Transform3D(
                    mat3x3=camera_to_world[:3, :3], translation=camera_to_world[:3, 3]
                ),
            )

            # Log camera intrinsics (Pinhole camera model)
            # This should be logged as a child of the camera transform
            # Rerun expects image_from_camera matrix (intrinsics)
            rr.log(
                f"cameras/{cam_name}/pinhole",
                rr.Pinhole(image_from_camera=intrinsics, width=W, height=H),
            )

            attrs = [robot_mask_frame]
            if static_mask_frame is not None and vis_masks:
                attrs.append(static_mask_frame)

            points, colors, out_attrs = depth_to_point_cloud(
                depth_frame,
                rgb_frame,
                intrinsics,
                extrinsics,
                max_depth=2.0,
                foreground_mask=foreground_mask_frame.astype(bool),
                attrs=attrs,
            )

            robot_mask_pts = out_attrs[0]
            static_mask_pts = out_attrs[1] if len(out_attrs) > 1 else None

            # Apply voxelization if enabled
            if voxel_layer is not None and len(points) > 0:
                # Convert to torch tensors
                pts_tensor = torch.tensor(points, dtype=torch.float32, device=device)

                # Estimate voxel parameters and voxelize
                norm_params = voxel_layer.estimate_voxel_parameters(
                    pts_tensor, batch=None
                )
                (voxel_coords, voxel_batch), pool_indices = voxel_layer.voxelize(
                    pts_tensor, norm_params, batch=None
                )

                # Devoxelize to get voxel center positions
                voxel_points = voxel_layer.devoxelize(
                    voxel_coords, voxel_batch, norm_params
                )

                # Pool colors if available
                if colors is not None:
                    colors_tensor = torch.tensor(
                        colors, dtype=torch.float32, device=device
                    )
                    voxel_colors = voxel_layer.feature_voxel_pool(
                        pool_indices, colors_tensor
                    )
                    colors = voxel_colors.cpu().numpy()

                # Pool robot mask (reduce="min")
                if robot_mask_pts is not None and vis_masks:
                    rm_tensor = torch.tensor(
                        robot_mask_pts, dtype=torch.float32, device=device
                    ).unsqueeze(-1)
                    # Use min reduction for masks
                    voxel_rm = voxel_layer.feature_voxel_pool(
                        pool_indices, rm_tensor, reduce="min"
                    )
                    robot_mask_pts = voxel_rm.squeeze(-1).cpu().numpy()

                # Pool static mask (reduce="min")
                if static_mask_pts is not None and vis_masks:
                    sm_tensor = torch.tensor(
                        static_mask_pts, dtype=torch.float32, device=device
                    ).unsqueeze(-1)
                    voxel_sm = voxel_layer.feature_voxel_pool(
                        pool_indices, sm_tensor, reduce="min"
                    )
                    static_mask_pts = voxel_sm.squeeze(-1).cpu().numpy()

                points = voxel_points.cpu().numpy()

            # Log point cloud
            if colors is not None:
                rr.log(f"point_clouds/{cam_name}", rr.Points3D(points, colors=colors))
            else:
                rr.log(f"point_clouds/{cam_name}", rr.Points3D(points))

            # Log robot mask points
            if robot_mask_pts is not None and vis_masks:
                # Color based on mask: Red for active, Grey for inactive
                mask_colors = np.full((len(points), 3), 128, dtype=np.uint8)  # Grey
                mask_indices = robot_mask_pts > 0.5
                if np.any(mask_indices):
                    mask_colors[mask_indices] = [255, 0, 0]  # Red

                rr.log(
                    f"point_clouds/{cam_name}_robot_mask",
                    rr.Points3D(points, colors=mask_colors),
                )

            # Log static mask points
            if static_mask_pts is not None and vis_masks:
                # Color based on mask: Green for active, Grey for inactive
                mask_colors = np.full((len(points), 3), 128, dtype=np.uint8)  # Grey
                mask_indices = static_mask_pts > 0.5
                if np.any(mask_indices):
                    mask_colors[mask_indices] = [0, 255, 0]  # Green

                rr.log(
                    f"point_clouds/{cam_name}_static_mask",
                    rr.Points3D(points, colors=mask_colors),
                )

        # Visualize robot geometry
        qpos_key = "target_qpos" if use_target_qpos else "qpos"

        root_poses_data = traj_data.metadata["root_poses"]
        assert len(root_poses_data) == num_frames
        root_poses_frame = root_poses_data[frame_idx]

        qpos_data = traj_data.metadata[qpos_key]
        assert len(qpos_data) == num_frames
        qpos_frame = qpos_data[frame_idx]

        robot_uids = list(robot_geometries.keys())

        assert len(robot_infos) == len(robot_geometries)
        for robot_idx, robot_uid in enumerate(robot_uids):
            robot_geom = robot_geometries[robot_uid]
            robot_info = robot_infos[robot_idx]

            qpos_robot = qpos_frame[robot_idx]
            root_pose_robot = root_poses_frame[robot_idx]

            qpos_tensor = torch.tensor(
                qpos_robot,
                dtype=torch.float32,
            ).unsqueeze(0)  # Add batch dimension

            root_pose_tensor = torch.tensor(
                root_pose_robot,
                dtype=torch.float32,
            ).unsqueeze(0)  # Add batch dimension

            robot_geom.set_pose(qpos_tensor, root_pose_tensor)
            robot_mesh = to_o3d_mesh(robot_geom.sdf)
            vertices, faces, vertex_colors = mesh_to_arrays(robot_mesh)
            # Log mesh
            rr.log(
                f"robots/{robot_uid}/mesh",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_colors=vertex_colors,
                ),
            )

    print(f"Visualization complete. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory data to Rerun RRD file for visualization"
    )
    parser.add_argument("dataset_dir", type=str, help="Root directory of the dataset")
    parser.add_argument("traj_id", type=str, help="Trajectory ID to visualize")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output RRD file path (default: <dataset_dir>/<traj_id>.rrd)",
    )
    parser.add_argument(
        "--use-qpos",
        action="store_true",
        help="Use qpos instead of target_qpos for robot visualization",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to visualize. If not set, uses --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Limit the number of frames visualized (default: 40)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Resize all images to img_size x img_size (default: no resizing)",
    )

    # Voxelization arguments
    parser.add_argument(
        "--voxelize", action="store_true", help="Enable voxelization of point clouds"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Voxel grid resolution. Single value (e.g., 64) or three values (e.g., 64 64 64). Default: 64 64 64",
    )
    parser.add_argument(
        "--min-cell-size",
        type=float,
        nargs="+",
        default=None,
        help="Physical size of one voxel in meters. Single value (e.g., 0.05) or three values (e.g., 0.05 0.05 0.05). Default: 0.05 0.05 0.05",
    )
    parser.add_argument(
        "--center-offset-ratio",
        type=float,
        default=0.1,
        help="Voxelization center offset ratio for augmentation. Default: 0.1",
    )
    parser.add_argument(
        "--bound-ratio",
        type=float,
        default=0.05,
        help="Voxelization bound ratio for augmentation. Default: 0.05",
    )

    parser.add_argument(
        "--vis-masks",
        default=False,
        action="store_true",
        help="Visualize the robot mask and static mask (default: False)",
    )

    args = parser.parse_args()

    # Parse voxelization parameters
    def parse_tuple_arg(value: Optional[List], n: int = 3):
        """Parse a list argument into a tuple, duplicating single values."""
        if value is None:
            return None
        if len(value) == 1:
            return tuple([value[0]] * n)
        elif len(value) == n:
            return tuple(value)
        else:
            raise ValueError(f"Expected 1 or {n} values, got {len(value)}")

    voxel_resolution = parse_tuple_arg(args.resolution, 3)
    voxel_min_cell_size = parse_tuple_arg(args.min_cell_size, 3)

    # Determine output path
    if args.output is None:
        output_path = os.path.join(args.dataset_dir, "rerun", f"{args.traj_id}.rrd")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_path = args.output

    if not output_path.endswith(".rrd"):
        output_path = output_path + ".rrd"

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize dataset
    dataset = ManiSkillTrajectoryDataset(args.dataset_dir)

    # Visualize trajectory
    visualize_trajectory(
        dataset=dataset,
        traj_id=args.traj_id,
        output_path=output_path,
        use_target_qpos=not args.use_qpos,
        max_frames=args.max_frames if args.max_frames is not None else args.limit,
        img_size=args.img_size,
        voxelize=args.voxelize,
        voxel_resolution=voxel_resolution,
        voxel_min_cell_size=voxel_min_cell_size,
        voxel_center_offset_ratio=args.center_offset_ratio,
        voxel_bound_ratio=args.bound_ratio,
        vis_masks=args.vis_masks,
    )


if __name__ == "__main__":
    main()
