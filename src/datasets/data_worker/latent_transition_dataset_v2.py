import os.path as osp
import time
import mani_skill
from functools import reduce
from rich import print
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import torch
import numpy as np
from dataclasses import dataclass
from easydict import EasyDict as edict

import third_party.pytorch_kinematics as pk
from utils.pv_k3d import render_pcds
from utils.vis import to_pil, sdf_to_colors, sdf_grad_to_colors
from src import models, datasets, trainers
from jaxtyping import Float32, Bool, Int32
from torch import Tensor
from src.modules.sparse import SparseTensor
from src.datasets.trajectory_dataset import RobotInfo
from third_party.pytorch_volumetric.robot_anchors import RobotAnchorSDF
from transformers import AutoTokenizer, CLIPTextModel
from .base import DataWorker


# ================================================
# Interface
# ================================================


class TransitionSample(TypedDict):
    """
    Represents a single physics transition sample processed from a trajectory.

    A transition sample contains structured (sparse voxels) and unstructured (latent vectors)
    representations of the scene across multiple frames, along with auxiliary information
    for SDF flow computation, rendering, and text conditioning.

    Attributes:
        structure: A list of length K (e.g., 2 for t and T), where each element is a tuple
            of (coords, feats). Coords has shape (P, 3) and feats has shape (P, zdim).
        unstructure: A list of length T (full trajectory), where each element is a latent
            vector of shape (z_len, 1) or similar, flattened from (D, D, D, C).
        sdf_t_T: A tuple (coords, flow_data) or None. coords has shape (P, 3).
            flow_data has shape (P, horizon, 5), where 5 dims are (dist, grad_x, grad_y, grad_z, mask).
        anchor_points: Tensor of shape (SH, N, 3) representing robot anchor points across the horizon,
            where SH is the effective horizon and N is the number of anchors.
        rgbs_T: Multi-view RGB images for the target (future) frame T. Shape: (CAM, 3, H, W).
        rgbs_t: Multi-view RGB images for the source (current) frame t. Shape: (CAM, 3, H, W).
        masks_fg_T: Combined foreground masks (non-background) for frame T. Shape: (CAM, H, W).
        robot_masks_T: Segmentation masks for the robot at frame T. Shape: (CAM, H, W).
        static_masks_T: Segmentation masks for static environment objects at frame T. Shape: (CAM, H, W).
        voxel_params: Normalization parameters for the voxel grid (e.g., scale/offset). Shape: (3, 3).
        aug_params: Augmentation parameters (e.g., rotation/offset) applied to the point cloud. Shape: (3).
        w2cs: World-to-camera transformation matrices for each camera. Shape: (CAM, 4, 4).
        ints: Intrinsic camera matrices for each camera. Shape: (CAM, 3, 3).
        frame_id: Sequence of frame indices within the original trajectory. Shape: (T).
        num_frames: Total number of frames in the trajectory (or max frames). Shape: (T).
        text_embedding: CLIP text embedding for the task description. Shape: (text_len, text_dim).
    """

    structure: list[Tuple[Int32[Tensor, "P 3"], Float32[Tensor, "P zdim"]]]
    unstructure: list[Float32[Tensor, "z_len 1"]]

    sdf_t_T: Tuple[Int32[Tensor, "P 3"], Float32[Tensor, "P horizon 5"]] | None
    anchor_points: Float32[Tensor, "SH N 3"] | None

    rgbs_T: Float32[Tensor, "CAM 3 H W"]
    rgbs_t: Float32[Tensor, "CAM 3 H W"]
    masks_fg_T: Bool[Tensor, "CAM H W"]
    robot_masks_T: Bool[Tensor, "CAM H W"]
    static_masks_T: Bool[Tensor, "CAM H W"]

    voxel_params: Float32[Tensor, "3 3"]
    aug_params: Float32[Tensor, "3"]

    w2cs: Float32[Tensor, "CAM 4 4"]
    ints: Float32[Tensor, "CAM 3 3"]

    frame_id: Int32[Tensor, "T"]
    num_frames: Int32[Tensor, "T"]
    text_embedding: Float32[Tensor, "text_len text_dim"]

    # GT Gaussian parameters for target frame T (used in ControlNet dynamics trainer)
    z_T_mean: Optional[Float32[Tensor, "z_len z_dim"]]
    z_T_logvar: Optional[Float32[Tensor, "z_len z_dim"]]
    z_T_h_normed: Optional[Float32[Tensor, "C D H W"]]

    sample_removal_mask: Bool[Tensor, ""]
    task_ind: Tensor | None
    traj_id: Tensor | None
    valid_masks_T: Bool[Tensor, "1 D H W"] | None
    valid_masks_t: Bool[Tensor, "1 D H W"] | None


class TransitionBatch(TypedDict):
    """
    Represents a batched collection of physics transition samples.

    All fields are batched versions of `TransitionSample` attributes, with a leading
    batch dimension `B`. Structured components (SparseTensors) handle batching internally
    via coordinate offsets.

    Attributes:
        structure: List of SparseTensors (batch size B), one for each kept frame.
        unstructure: List of batched latent vectors. Shape: (B, z_len, z_dim).
        sdf_t_T: Batched SDF flow data stored as a SparseTensor if available.
        rgbs_T: Batched target RGB images. Shape: (B, CAM, 3, H, W).
        rgbs_t: Batched source RGB images. Shape: (B, CAM, 3, H, W).
        masks_fg_T: Batched foreground masks. Shape: (B, CAM, H, W).
        robot_masks_T: Batched robot masks. Shape: (B, CAM, H, W).
        static_masks_T: Batched static masks. Shape: (B, CAM, H, W).
        voxel_params: Batched normalization parameters. Shape: (B, 3, 3).
        aug_params: Batched augmentation parameters. Shape: (B, 3).
        w2cs: Batched world-to-camera matrices. Shape: (B, CAM, 4, 4).
        ints: Batched intrinsic matrices. Shape: (B, CAM, 3, 3).
        frame_id: Batched frame indices. Shape: (B, T).
        num_frames: Batched total frame counts. Shape: (B, T).
        text_embedding: Batched CLIP embeddings. Shape: (B, text_len, text_dim).
    """

    structure: list[SparseTensor]
    unstructure: list[Float32[Tensor, "B z_len z_dim"]]

    sdf_t_T: SparseTensor | None

    rgbs_T: Float32[Tensor, "B CAM 3 H W"]
    rgbs_t: Float32[Tensor, "B CAM 3 H W"]
    masks_fg_T: Bool[Tensor, "B CAM H W"]
    robot_masks_T: Bool[Tensor, "B CAM H W"]
    static_masks_T: Bool[Tensor, "B CAM H W"]

    voxel_params: Float32[Tensor, "B 3 3"]
    aug_params: Float32[Tensor, "B 3"]

    w2cs: Float32[Tensor, "B CAM 4 4"]
    ints: Float32[Tensor, "B CAM 3 3"]

    frame_id: Int32[Tensor, "B T"]
    num_frames: Int32[Tensor, "B T"]
    text_embedding: Float32[Tensor, "B text_len text_dim"]

    # GT Gaussian parameters for target frame T (used in ControlNet dynamics trainer)
    z_T_mean: Optional[Float32[Tensor, "B z_len z_dim"]]
    z_T_logvar: Optional[Float32[Tensor, "B z_len z_dim"]]
    z_T_h_normed: Optional[Float32[Tensor, "B C D H W"]]

    sample_removal_mask: Bool[Tensor, "B"]
    task_ind: Int32[Tensor, "B"]
    traj_id: Int32[Tensor, "B"]


@dataclass
class DataWorkerConfig:
    """
    Configuration for the LatentTransitionWorker.

    Attributes:
        nop_action_prob: Probability of generating a 'No-Operation' (NOP) action when
            horizon is at minimum. Defaults to 0.35.
        anchor_num: Number of anchor points to sample on the robot surface for SDF flow.
        flow_on_robot: Whether to compute correspondence-based flow for points on the robot.
        min_points_per_link: Minimum number of anchor points per robot link.
        compute_sdf: Whether to compute SDF flow features.
        text_cond_model: HuggingFace model path for text conditioning (e.g., CLIP).
        structure_only_keep_last_k: Number of frames to keep in the 'structure' output list.
            Usually 2 (source 't' and target 'T').
    """

    nop_action_prob: float = 0.35
    anchor_num: int = 10000
    flow_on_robot: bool = True
    min_points_per_link: int = 100
    compute_sdf: bool = True
    text_cond_model: str = "openai/clip-vit-large-patch14"
    structure_only_keep_last_k: int = 2
    use_mean_for_unstructure: bool = False

    n_cache_anchor_state: int = 10
    update_anchor_state_every: int = 100
    encode_text: bool = False
    label_only_T: bool = False  # If True, export z_T_mean/z_T_logvar as GT labels
    skip_t_inference: bool = (
        False  # If True, skip encoder on frame t (use voxelize_only)
    )


# ================================================
# Worker
# ================================================


class RobotSDFWorkerMixin:
    """
    Mixin class to provide RobotSDF loading and caching capabilities.
    """

    def load_robot_sdf(self, robot_infos: List[RobotInfo]) -> RobotAnchorSDF:
        """
        Load or retrieve cached RobotSDF for the given robot information.

        Args:
            robot_infos: List of RobotInfo objects containing UID and URDF path for each robot.

        Returns:
            A RobotAnchorSDF object initialized with the robot's kinematic chains.
        """
        # _asset_path = osp.join(mani_skill.__path__[0], "assets")
        _asset_path = osp.abspath(osp.join(osp.dirname(__file__), "../../../"))
        if not hasattr(self, "_robot_sdfs"):
            self._robot_sdfs = {}
        names = tuple([robot_info.uid for robot_info in robot_infos])
        if names in self._robot_sdfs:
            return self._robot_sdfs[names]

        chains = []
        for robot_info in robot_infos:
            urdf_path = osp.join(_asset_path, robot_info.urdf_path)
            if osp.exists(urdf_path.replace(".urdf", ".stl.urdf")):
                urdf_path = urdf_path.replace(".urdf", ".stl.urdf")
            chain = pk.build_chain_from_urdf(open(urdf_path).read())
            chain = chain.to(device=self.device)
            chains.append(chain)

        s = RobotAnchorSDF(
            chains, path_prefix=osp.dirname(urdf_path), use_collision_mesh=False
        )
        self._robot_sdfs[names] = s
        return s


class LatentTransitionWorker(DataWorker, RobotSDFWorkerMixin):
    """
    Worker implementation for physics-based transitions.
    This worker processes trajectory data using a VAE model and an SDF engine to generate
    SDF values and gradients as features.
    """

    def __init__(
        self,
        cfg: edict,  # old
        data_override: edict,  # new
        device: Union[str, torch.device],
        batch_size: int = 1,
        num_workers: int = 0,
        debug: bool = False,
        dataset_is_none: bool = False,
    ):
        """
        Initialize the LatentTransitionWorker.

        Loads the VAE model, trajectory dataset, and initializes trainer and robot SDF cache.

        Args:
            cfg: Global configuration object (EasyDict).
            data_override: Configuration overrides for dataset and worker-specific settings.
            device: Computing device (str or torch.device).
            batch_size: Batch size (currently only 1 is supported).
            num_workers: Number of subprocesses for data loading.
            debug: Whether to enable debug mode for logging and visualization.
        """
        super().__init__(
            cfg,
            data_override,
            device,
            batch_size,
            num_workers,
            debug,
        )
        # assert batch_size == 1
        self.cfg = cfg
        self.data_override = data_override
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug

        # Merge configs
        if hasattr(data_override.dataset, "full_args") and not dataset_is_none:
            cfg.dataset = data_override.dataset.full_args

        # Initialize DataWorkerConfig
        worker_kwargs = data_override.dataset.args.get("kwargs", {})
        self.config_dw = DataWorkerConfig(**worker_kwargs)

        if self.config_dw.compute_sdf:
            self.min_horizon, self.horizon = data_override.dataset.args.get(
                "horizon",
                data_override.dataset.get(
                    "full_args", edict(args=edict(horizon=[1, 1]))
                ).args.horizon,
            )
            print(
                f"[yellow][DataWorkerConfig] Setting horizon to {self.min_horizon}, {self.horizon}[/yellow]"
            )
        else:
            self.min_horizon, self.horizon = 1, 1

        if hasattr(data_override.dataset, "trainer_args"):
            for k, v in data_override.dataset.trainer_args.items():
                print(f"[red][Trainer Config] Setting {k} to {v}[/red]")
                setattr(cfg.trainer.args, k, v)

        print("Initializing models...")
        self.model_dict = {
            name: getattr(models, model.name)(**model.args).to(device)
            for name, model in cfg.models.items()
            if (name != "decoder") or debug
        }

        if dataset_is_none:
            self.dataset = None
        else:
            # usually `TrajectoryDataset`
            self.dataset = getattr(datasets, cfg.dataset.name)("", **cfg.dataset.args)

        # Use output_dir from config if available
        load_dir = cfg.get("output_dir", ".")
        print(f"Initializing trainer (loading from {load_dir})...")
        cfg.trainer.args.debug = debug
        if "load_dir" in cfg.trainer.args:
            cfg.trainer.args.pop("load_dir")
        self.trainer = getattr(trainers, cfg.trainer.name)(
            self.model_dict,
            self.dataset,
            **cfg.trainer.args,
            output_dir="/tmp",
            load_dir=None,
            step=None,
            wandb_run=None,
            inference_only=True,
        )
        self.trainer.load(
            load_dir=load_dir, step=data_override.trainer.args.vae_step_suffix
        )
        self.image_keys = data_override.dataset.args.image_keys
        print(f"[yellow]Image keys: {self.image_keys}[/yellow]")
        self.prefix = "cpt_" if "V4" in cfg.trainer.name else ""
        self.robot_sdfs = {}

        # Text conditioning initialization (lazy)
        self.text_model = None
        self.tokenizer = None

    def encode_text(self, text: List[str]) -> Float32[Tensor, "B text_len text_dim"]:
        """
        Encode a list of text descriptions into CLIP embeddings.

        Args:
            text: List of strings (task descriptions).

        Returns:
            A tensor of shape (B, text_len, text_dim) containing the last hidden
            states from the CLIP text model.
        """
        if self.text_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config_dw.text_cond_model
            )
            self.text_model = (
                CLIPTextModel.from_pretrained(self.config_dw.text_cond_model)
                .to("cpu")
                .eval()
            )

        if not hasattr(self, "_text_cache"):
            self._text_cache = {}

        uncached = [t for t in text if t not in self._text_cache]
        if uncached:
            encoding = self.tokenizer(
                uncached,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = encoding["input_ids"].to("cpu")
            with torch.no_grad():
                embs = self.text_model(input_ids=tokens).last_hidden_state
            for t, emb in zip(uncached, embs):
                self._text_cache[t] = emb.to("cpu")

        return torch.stack([self._text_cache[t] for t in text]).to(self.device)

    def visualize(self, transitions: List[TransitionSample], count: int = 0) -> None:
        """
        Visualize the generated transition samples.

        Decodes the source latent (z_t) to produce multi-view RGB images and renders
        the SDF flow (values and gradients) on the point cloud if available.

        Args:
            transitions: List of TransitionSample objects to visualize.
            count: Number of transitions to visualize (typically only the first one).
        """

        for i, sample in enumerate(transitions):
            if i > count:
                break
            # Use the second to last frame as 't' for visualization
            # unstructure is now a list, so we take the second to last element
            unstructure_t = sample["unstructure"][-2]
            D = int(round(unstructure_t.shape[0] ** (1 / 3)))
            inp = {
                self.prefix + "z": unstructure_t.reshape(D, D, D, -1).permute(
                    3, 0, 1, 2
                )[None]
            }
            out = self.trainer.decode(
                **inp,
                w2c=sample["w2cs"],
                intrinsics=sample["ints"],
            )
            to_pil(out[self.prefix + "color"]).save(
                "runs/debug/latent_transition_worker.png"
            )

            # Check if SDF data is present before visualizing
            if sample["sdf_t_T"] is not None and sample["anchor_points"] is not None:
                # sdf_t_T is a tuple (coords, feats), feats has shape (P, horizon, 5)
                # We need the data for the second to last frame (t_idx)
                # The 5th dimension (index 4) indicates valid actions.
                # We need to find the maximum index where this is 1.0 across all points
                # to determine the actual number of actions (act_dim)
                # The original code used [0, :, 4].sum().item() which implies it was looking at the first point's horizon.
                # Let's assume act_dim refers to the effective_horizon used during sdf_t_T creation.
                # The sdf_t_T[1] has shape (P, self.horizon, 5).
                # The valid actions are up to `effective_horizon`.
                # The 5th dimension (index 4) is 1.0 for valid actions and 0.0 for padded.
                # So, act_dim should be the number of non-zero entries in the 5th column for any point.
                # We can find the max index where it's 1.0.
                valid_horizon_mask = sample["sdf_t_T"][1][0, :, 4] == 1.0
                act_dim = (
                    valid_horizon_mask.sum().item()
                )  # This is the effective_horizon

                # devoxelized uses structure_xt, which is sample["structure"][-2]
                # The original code used sample["sdf_t_T"][0] for coords, which is correct for structure_xt coords
                devoxelized = self.trainer.voxelization.devoxelize(
                    sample["structure"][-2][0],  # Use coords from structure_xt
                    batch=None,
                    norm_params=sample["voxel_params"][None],
                    aug_params=sample["aug_params"][None]
                    if sample["aug_params"] is not None
                    else None,
                )

                render_pcds(
                    reduce(
                        lambda a, b: a | b,
                        [
                            {
                                f"sdf_val_{ai}": devoxelized,
                                f"sdf_grad_{ai}": devoxelized,
                                f"anchor_{ai}": sample["anchor_points"][ai],
                            }
                            for ai in range(act_dim)
                        ],
                    ),
                    reduce(
                        lambda a, b: a | b,
                        [
                            {
                                f"sdf_val_{ai}": sdf_to_colors(
                                    sample["sdf_t_T"][1][:, ai, 0]
                                ),
                                f"sdf_grad_{ai}": sdf_grad_to_colors(
                                    sample["sdf_t_T"][1][:, ai, 1:4]
                                ),
                            }
                            for ai in range(act_dim)
                        ],
                    ),
                    node_radius=0.02,
                    save_to="runs/debug/sdf.latent_trainsition_worker.html",
                )

    def _compute_sdf_flow(
        self,
        pts_t0: Float32[Tensor, "P 3"],
        robot_mask_t0: Bool[Tensor, "P"],
        anchors_T: Float32[Tensor, "SH N 3"],
    ) -> Float32[Tensor, "P SH 5"]:
        """
        Compute the Signed Distance Field (SDF) and flow gradients for a set of points.

        The computation follows two distinct paths:
        1.  **Robot Points**: Points belonging to the robot at time t=0 are assigned a
            motion based on the closest robot surface anchor at t=0. Their future
            'targets' are the same anchors at future time T.
        2.  **Environment Points**: Points not belonging to the robot (or if flow_on_robot is False)
            are assigned to the closest anchor at *each* future time T' independently.

        Args:
            pts_t0: Point cloud coordinates at source time t=0. Shape: (P, 3).
            robot_mask_t0: Boolean mask identifying robot points in `pts_t0`. Shape: (P).
            anchors_T: Robot surface anchor positions over the horizon. Shape: (SH, N, 3),
                where SH is the effective horizon and N is the number of anchors.

        Returns:
            SDF flow tensor of shape (P, SH, 5). The 5 dimensions are:
                - index 0: Distance from point to its assigned target (in voxel units).
                - index 1-3: Gradient vector (dx, dy, dz) normalized by distance.
                - index 4: Validity mask (always 1.0 here, used for padding in outputs).
        """
        num_T = anchors_T.shape[0]
        num_pts = pts_t0.shape[0]
        # sdf_results shape: (num_pts, num_T, 5)
        # 5th dimension (index 4) is 1.0 for valid actions
        sdf_results = torch.zeros(
            (num_pts, num_T, 5), device=self.device, dtype=pts_t0.dtype
        )
        sdf_results[:, :, 4] = 1.0

        # 1. Handle Robot Points (Correspondence-based Flow)
        if self.config_dw.flow_on_robot and robot_mask_t0.any():
            robot_pts = pts_t0[robot_mask_t0]
            # Find closest anchor at t=0
            dists_0 = torch.cdist(
                robot_pts, anchors_T[0:1]
            )  # (P_robot, 1, num_anchors)
            anchor_indices = dists_0.argmin(dim=-1).squeeze()  # (P_robot,)

            # For each future time T', the target is anchors_T[T', anchor_indices]
            assigned_anchors = anchors_T[:, anchor_indices]  # (num_T, P_robot, 3)
            assigned_anchors = assigned_anchors.transpose(0, 1)  # (P_robot, num_T, 3)

            diff = assigned_anchors - robot_pts.unsqueeze(1)  # (P_robot, num_T, 3)
            dist = torch.linalg.norm(diff, dim=-1, keepdim=True)  # (P_robot, num_T, 1)
            grad = diff / (dist + 1e-8)

            sdf_results[robot_mask_t0, :, 0:1] = dist
            sdf_results[robot_mask_t0, :, 1:4] = grad

        # 2. Handle Non-Robot Points (Strictly Closest Assignment)
        # If not flow_on_robot, treat robot points as environment points
        env_mask = (
            ~robot_mask_t0
            if self.config_dw.flow_on_robot
            else torch.ones_like(robot_mask_t0)
        )
        if env_mask.any():
            env_pts = pts_t0[env_mask]  # (P_env, 3)

            if self.debug:
                for ti in range(num_T):
                    dists_ti = torch.cdist(
                        env_pts, anchors_T[ti : ti + 1]
                    )  # (P_env, 1, num_anchors)
                    min_dists, min_indices = dists_ti.min(dim=-1)  # (P_env, 1)

                    closest_anchors = anchors_T[
                        ti, min_indices.squeeze(1)
                    ]  # (P_env, 3)
                    diff = closest_anchors - env_pts  # (P_env, 3)
                    dist = torch.linalg.norm(diff, dim=-1, keepdim=True)  # (P_env, 1)
                    grad = diff / (dist + 1e-8)

                    sdf_results[env_mask, ti, 0:1] = dist
                    sdf_results[env_mask, ti, 1:4] = grad

                original_dist = sdf_results[env_mask, :, 0:1].clone()
                original_grad = sdf_results[env_mask, :, 1:4].clone()

            # Vectorized version
            env_pts_exp = env_pts.unsqueeze(0).expand(
                num_T, -1, -1
            )  # (num_T, P_env, 3)
            dists_vec = torch.cdist(
                env_pts_exp, anchors_T
            )  # (num_T, P_env, num_anchors)
            min_indices_vec = dists_vec.argmin(dim=-1)  # (num_T, P_env)

            batch_indices = torch.arange(num_T, device=self.device).unsqueeze(
                1
            )  # (num_T, 1)
            closest_anchors_vec = anchors_T[
                batch_indices, min_indices_vec
            ]  # (num_T, P_env, 3)

            diff_vec = closest_anchors_vec - env_pts_exp  # (num_T, P_env, 3)
            dist_vec = torch.linalg.norm(
                diff_vec, dim=-1, keepdim=True
            )  # (num_T, P_env, 1)
            grad_vec = diff_vec / (dist_vec + 1e-8)  # (num_T, P_env, 3)

            # Transpose to match (P_env, num_T, dim)
            dist_vec = dist_vec.transpose(0, 1)  # (P_env, num_T, 1)
            grad_vec = grad_vec.transpose(0, 1)  # (P_env, num_T, 3)

            if self.debug:
                assert torch.allclose(original_dist, dist_vec, atol=1e-5), (
                    "Vectorized dist does not match original"
                )
                assert torch.allclose(original_grad, grad_vec, atol=1e-5), (
                    "Vectorized grad does not match original"
                )

            sdf_results[env_mask, :, 0:1] = dist_vec
            sdf_results[env_mask, :, 1:4] = grad_vec

        # 3. Convert Distance to Voxel Space
        sdf_results[:, :, 0] /= self.trainer.granularity
        return sdf_results

    @torch.no_grad()
    def process_batch(self, batch: Dict[str, Any], is_eval=False) -> List[TransitionSample]:
        """
        Process a batch of trajectory data into physics transition samples.

        This is the main entry point for the worker. It performs several steps:
        1.  **Inference**: Runs the VAE model to extract features (structure) and latents (z).
        2.  **Text Encoding**: Generates CLIP embeddings for the task description.
        3.  **NOP Detection**: Randomly decides if this transition should be a 'No-Op'
            (target frame = source frame) based on configuration and horizon.
        4.  **SDF Flow Computation**: If enabled, computes the SDF values and gradients
            of source points relative to robot motions over the sampled horizon.
        5.  **Sample Construction**: Packagesthe processed data into `TransitionSample` objects.

        Args:
            batch: Dictionary containing trajectory data:
                - "rgbs": (B, T, CAM, 3, H, W)
                - "qpos": (B, horizon, Q) - Robot joint positions.
                - "root_poses": (B, horizon, 7) - Robot root poses.
                - "task_description": List[str] - Natural language task.
                - etc.

        Returns:
            A list of `TransitionSample` objects ready for training or evaluation.
        """
        if self.debug:
            _t_total_start = time.time()
            _t_start = time.time()

        if self.config_dw.skip_t_inference:
            # Split inference: voxelize_only for frame t, full inference_batch for frame T
            # Slice batch to get per-frame sub-batches
            def _slice_batch_only_tensors(batch, frame_indices):
                """Create a sub-batch containing only the given frame indices."""
                sub = {}
                for k, v in batch.items():
                    if (
                        isinstance(v, torch.Tensor)
                        and v.dim() >= 2
                        and v.shape[1] == T_orig
                    ):
                        # Tensors with time dimension: (B, T, ...)
                        sub[k] = v[:, frame_indices]
                    else:
                        # usually not needed for encoding process
                        pass
                return sub

            B_orig, T_orig, CAM = batch["rgbs"].shape[:3]
            t_idx_frame = T_orig - 2  # second to last
            T_idx_frame = T_orig - 1  # last

            # Frame t: lightweight voxelization only (no encoder)
            batch_t = _slice_batch_only_tensors(batch, [t_idx_frame])
            results_t = self.trainer.voxelize_only(batch_t, augment_voxel_grid=False, disable_voxel_subsample=is_eval)

            # Frame T: full inference (with encoder, for z_T)
            batch_T = _slice_batch_only_tensors(batch, [T_idx_frame])
            results_T = self.trainer.inference_batch(
                self.model_dict,
                batch_T,
                augment_voxel_grid=False,
                skip_decoding=True,
                voxel_params=results_t["voxel_params"],
                aug_params=results_t.get("aug_params"),
                disable_voxel_subsample=is_eval
            )

            assert torch.allclose(
                results_t["voxel_params"], results_T["voxel_params"]
            ), "Voxelization params do not match"

            # Merge results into a unified format as if T=2 inference was done
            # Concatenate input_feats (with adjusted batch coords)
            feats_t = results_t["input_feats"]
            feats_T = results_T["input_feats"]
            coords_t = feats_t.coords.clone()
            coords_T = feats_T.coords.clone()
            coords = []
            feats = []
            voxels_robot = []

            for _bi in range(B_orig):
                index_t = coords_t[:, 0] == _bi
                index_T = coords_T[:, 0] == _bi

                new_coords_T = torch.cat(
                    [coords_T[index_T, :1] * 2 + 1, coords_T[index_T, 1:]], dim=1
                )
                new_coords_t = torch.cat(
                    [coords_t[index_t, :1] * 2, coords_t[index_t, 1:]], dim=1
                )

                coords.append(new_coords_t)
                coords.append(new_coords_T)
                feats.append(feats_t.feats[index_t])
                feats.append(feats_T.feats[index_T])

                voxels_robot.append(results_t["voxels_robot"][index_t])
                voxels_robot.append(results_T["voxels_robot"][index_T])

            merged_input_feats = SparseTensor(
                coords=torch.cat(coords, dim=0),
                feats=torch.cat(feats, dim=0),
            )

            # Stack voxels_robot: shape is (P, ...) so direct concat is fine
            merged_voxels_robot = torch.cat(voxels_robot, dim=0)

            if results_t["sample_removal_mask"] is not None:
                merged_sample_removal_mask = torch.cat(
                    [
                        results_t["sample_removal_mask"].reshape(B_orig, 1),
                        results_T["sample_removal_mask"].reshape(B_orig, 1),
                    ],
                    dim=1,
                ).flatten()
            else:
                merged_sample_removal_mask = torch.zeros(
                    B_orig * 2, dtype=torch.bool, device=feats_t.device
                )

            # Build merged results dict
            results = {
                "input_feats": merged_input_feats,
                "voxel_params": results_t["voxel_params"]
                .unsqueeze(1)
                .repeat(1, 2, 1, 1)
                .flatten(0, 1),  # t and T shall have the same voxelization params
                "aug_params": None,
                "voxels_robot": merged_voxels_robot,
                "w2c": torch.cat(
                    [
                        results_t["w2c"].reshape(B_orig, CAM, 4, 4).unsqueeze(1),
                        results_T["w2c"].reshape(B_orig, CAM, 4, 4).unsqueeze(1),
                    ],
                    dim=1,
                ).flatten(0, 2),
                "intrinsics": torch.cat(
                    [
                        results_t["intrinsics"].reshape(B_orig, CAM, 3, 3).unsqueeze(1),
                        results_T["intrinsics"].reshape(B_orig, CAM, 3, 3).unsqueeze(1),
                    ],
                    dim=1,
                ).flatten(0, 2),
                "sample_removal_mask": merged_sample_removal_mask,
            }

            if results_T["valid_mask"] is not None:
                results["valid_mask"] = torch.cat(
                    [
                        results_t["valid_mask"].unsqueeze(1),
                        results_T["valid_mask"].unsqueeze(1),
                    ],
                    dim=1,
                ).flatten(0, 1)

            # z/mean/logvar: only available for frame T; create zeros for frame t
            z_T = results_T[self.prefix + "z"]
            mean_T = results_T[self.prefix + "mean"]
            logvar_T = results_T[self.prefix + "logvar"]

            z_dummy = torch.zeros_like(z_T)
            results[self.prefix + "z"] = torch.cat(
                [z_dummy.unsqueeze(1), z_T.unsqueeze(1)], dim=1
            ).flatten(0, 1)
            results[self.prefix + "mean"] = torch.cat(
                [torch.zeros_like(mean_T).unsqueeze(1), mean_T.unsqueeze(1)], dim=1
            ).flatten(0, 1)
            results[self.prefix + "logvar"] = torch.cat(
                [torch.zeros_like(logvar_T).unsqueeze(1), logvar_T.unsqueeze(1)], dim=1
            ).flatten(0, 1)

            # Propagate intermediate_feats from frame T (contains h_normed etc.)
            results["intermediate_feats"] = results_T.get("intermediate_feats", {})
        else:
            results = self.trainer.inference_batch(
                self.model_dict, batch, augment_voxel_grid=False, skip_decoding=True
            )  # BUG: note that this will voxelize t and T differently, but shall be very very closed though

        if self.debug:
            torch.cuda.synchronize()
            _time_inference = time.time() - _t_start
        B, T, CAM, _, H, W = batch["rgbs"].shape
        assert T >= 2, f"num frames must be at least 2, got {T}"

        # Generate text embeddings for the whole batch
        if self.debug:
            _t_start = time.time()

        if self.config_dw.encode_text:
            text_embeddings = self.encode_text(batch["task_description"])  # (B, 77, D)
        else:
            text_embeddings = None

        if self.debug:
            torch.cuda.synchronize()
            _time_text_encode = time.time() - _t_start

        def chwd_to_lc(chwd):
            return chwd.flatten(1).permute(1, 0)

        if self.debug:
            _t_start = time.time()

        input_feats = results["input_feats"]
        transitions = []
        aug_params = results["aug_params"]
        assert (
            input_feats.coords[:, 0].unique().numel() == len(results["voxel_params"])
            # == len(aug_params)
        )
        devoxelized = self.trainer.voxelization.devoxelize(
            input_feats.coords,
            batch=None,
            norm_params=results["voxel_params"],
            aug_params=aug_params,
        )
        voxels_robot = results["voxels_robot"]

        if self.config_dw.skip_t_inference:
            # When skip_t_inference: z is dummy for frame t, real for frame T
            z = results[self.prefix + "z"]
            z_mean_all = results[self.prefix + "mean"]
            z_logvar_all = results[self.prefix + "logvar"]
        else:
            if self.config_dw.use_mean_for_unstructure:
                z = results[self.prefix + "mean"]
            else:
                z = results[self.prefix + "z"]

            # Extract mean/logvar for GT labels when label_only_T is enabled
            z_mean_all = (
                results[self.prefix + "mean"] if self.config_dw.label_only_T else None
            )
            z_logvar_all = (
                results[self.prefix + "logvar"] if self.config_dw.label_only_T else None
            )

        # Extract h_normed from intermediate_feats if available
        intermediate_feats = results.get("intermediate_feats", {})
        h_normed_T = (
            intermediate_feats.get("h_normed", None)
            if self.config_dw.label_only_T
            else None
        )

        if self.debug:
            torch.cuda.synchronize()
            _time_devoxelize = time.time() - _t_start
            _time_struct_unstruct = 0.0
            _time_robot_sdf_anchors = 0.0
            _time_sdf_flow = 0.0
            _time_sample_construct = 0.0

        for global_bi in range(B):
            time_bi_start = global_bi * T

            qpos = torch.from_numpy(batch["qpos"][global_bi]).to(
                self.device
            )  # NOTE: we are using qpos, not target_qpos
            sampled_horizon = qpos.shape[0]

            # NOP Action logic: triggered when horizon is minimum and prob hits
            is_nop = (sampled_horizon == self.min_horizon) and (
                np.random.random() < self.config_dw.nop_action_prob
            )

            # Build list of structure and unstructure for all T frames
            if self.debug:
                _t_start = time.time()

            structure = []
            unstructure = []
            for t in range(T):
                t_mask = input_feats.coords[:, 0] == (time_bi_start + t)
                structure.append(
                    (input_feats.coords[t_mask, 1:], input_feats.feats[t_mask])
                )
                unstructure.append(chwd_to_lc(z[time_bi_start + t]))

            if self.debug:
                torch.cuda.synchronize()
                _time_struct_unstruct += time.time() - _t_start

            # t is second to last, T is last
            t_idx = T - 2
            T_idx = T - 1

            t_mask = input_feats.coords[:, 0] == (time_bi_start + t_idx)
            pts_t0 = devoxelized[t_mask]  # (P, 3)
            robot_mask_t0 = voxels_robot[t_mask].bool().view(-1)  # (P,)

            # Load robot SDF and compute anchors if requested
            if self.debug:
                _t_start = time.time()

            if self.config_dw.compute_sdf:
                robot_infos = batch["robot_infos"][global_bi]
                robot_sdf = self.load_robot_sdf(robot_infos)
                root_poses = torch.from_numpy(batch["root_poses"][global_bi]).to(
                    self.device
                )

                if self.config_dw.n_cache_anchor_state <= 0:
                    anchor_state = robot_sdf.initialize_anchors(
                        self.config_dw.anchor_num,
                        seed=-1,
                        min_points_per_link=self.config_dw.min_points_per_link,
                    )
                else:
                    key = robot_infos[0].urdf_path
                    if not hasattr(self, "_anchor_state_cache"):
                        self._anchor_state_cache = {}
                        self._anchor_state_counter = {}

                    if key not in self._anchor_state_cache:
                        self._anchor_state_cache[key] = []
                        self._anchor_state_counter[key] = 0

                    self._anchor_state_counter[key] += 1
                    cache_list = self._anchor_state_cache[key]

                    if (
                        len(cache_list) < self.config_dw.n_cache_anchor_state
                        or self._anchor_state_counter[key]
                        % self.config_dw.update_anchor_state_every
                        == 0
                    ):
                        new_state = robot_sdf.initialize_anchors(
                            self.config_dw.anchor_num,
                            seed=-1,
                            min_points_per_link=self.config_dw.min_points_per_link,
                        )
                        if len(cache_list) >= self.config_dw.n_cache_anchor_state:
                            replace_idx = np.random.randint(len(cache_list))
                            cache_list[replace_idx] = new_state
                        else:
                            cache_list.append(new_state)

                    anchor_state = cache_list[np.random.randint(len(cache_list))]

                tracked_points = robot_sdf.forward_kinematic_anchors(
                    anchor_state,
                    qpos.reshape(1, sampled_horizon, 1, -1),
                    root_poses.reshape(1, sampled_horizon, 1, -1),
                )  # (1, SH, N, 3)
                anchors_sh = tracked_points.squeeze(0)  # (SH, N, 3)
            else:
                anchors_sh = None

            if self.debug:
                torch.cuda.synchronize()
                _time_robot_sdf_anchors += time.time() - _t_start

            if is_nop:
                # NOP: Use second to last frame data for last frame
                rgbs_T = batch["rgbs"][global_bi, t_idx]
                masks_fg_T = batch["foreground_masks"][global_bi, t_idx]
                robot_masks_T = batch["robot_masks"][global_bi, t_idx]
                static_masks_T = batch["static_masks"][global_bi, t_idx]

                w2cs = results["w2c"].reshape(B, T, CAM, 4, 4)[global_bi, t_idx]
                ints = results["intrinsics"].reshape(B, T, CAM, 3, 3)[global_bi, t_idx]

                anchors_T = anchors_sh[:1] if anchors_sh is not None else None
                effective_horizon = 1
            else:
                # Normal processing: use last frame
                rgbs_T = batch["rgbs"][global_bi, T_idx]
                masks_fg_T = batch["foreground_masks"][global_bi, T_idx]
                robot_masks_T = batch["robot_masks"][global_bi, T_idx]
                static_masks_T = batch["static_masks"][global_bi, T_idx]
     
                w2cs = results["w2c"].reshape(B, T, CAM, 4, 4)[global_bi, T_idx]
                ints = results["intrinsics"].reshape(B, T, CAM, 3, 3)[global_bi, T_idx]

                anchors_T = anchors_sh
                effective_horizon = sampled_horizon

            robot_masks_t = batch["robot_masks"][global_bi, t_idx]
            static_masks_t = batch["static_masks"][global_bi, t_idx]

            # Compute SDF flow if requested
            if self.debug:
                _t_start = time.time()

            sdf_t_T = None
            if self.config_dw.compute_sdf:
                sdf_sh = self._compute_sdf_flow(pts_t0, robot_mask_t0, anchors_T)
                # Pad to self.horizon (max)
                num_pts = pts_t0.shape[0]
                sdf_results = torch.zeros(
                    (num_pts, self.horizon, 5), device=self.device, dtype=pts_t0.dtype
                )
                sdf_results[:, :effective_horizon] = sdf_sh
                sdf_t_T = (input_feats.coords[t_mask, 1:], sdf_results)

            if self.debug:
                torch.cuda.synchronize()
                _time_sdf_flow += time.time() - _t_start

            if self.debug:
                _t_start = time.time()

            # Determine sample removal mask natively from voxelization
            batch_srm = results["sample_removal_mask"]
            sm = (
                batch_srm[time_bi_start + t_idx]
                if batch_srm is not None
                else torch.tensor(False, device=self.device)
            )

            # Extract valid_mask if available
            valid_masks_t = None
            valid_masks_T = None
            if "valid_mask" in results:
                valid_mask_batch = results["valid_mask"]
                valid_masks_t = valid_mask_batch[time_bi_start + t_idx]
                valid_masks_T = valid_mask_batch[time_bi_start + T_idx]

            sample = TransitionSample(
                structure=structure[
                    -self.config_dw.structure_only_keep_last_k :
                ]  # 2 -> t and T
                if self.config_dw.structure_only_keep_last_k > 0
                else structure,
                unstructure=unstructure,
                sdf_t_T=sdf_t_T,
                rgbs_T=rgbs_T * (masks_fg_T > 0)[:, None],
                rgbs_t=batch["rgbs"][global_bi, t_idx]
                * (batch["foreground_masks"][global_bi, t_idx] > 0)[:, None],

                masks_fg_t=batch["foreground_masks"][global_bi, t_idx],
                masks_fg_T=masks_fg_T,
                robot_masks_T=robot_masks_T,
                static_masks_T=static_masks_T,
                robot_masks_t=robot_masks_t,
                static_masks_t=static_masks_t,

                voxel_params=results["voxel_params"][time_bi_start + t_idx],
                aug_params=None,  # results["aug_params"][time_bi_start + t_idx],
                w2cs=w2cs,
                ints=ints,
                anchor_points=anchors_T if self.config_dw.compute_sdf else None,
                frame_id=batch["frame_id"][global_bi],
                num_frames=batch["max_frames"][global_bi]
                if "max_frames" in batch
                else batch["num_frames"][global_bi],
                text_embedding=text_embeddings[global_bi]
                if text_embeddings is not None
                else None,
                z_T_mean=chwd_to_lc(z_mean_all[time_bi_start + T_idx])
                if z_mean_all is not None
                else None,
                z_T_logvar=chwd_to_lc(z_logvar_all[time_bi_start + T_idx])
                if z_logvar_all is not None
                else None,
                z_T_h_normed=h_normed_T[global_bi] if h_normed_T is not None else None,
                horizon=batch["horizon"][global_bi],
                sample_removal_mask=sm,
                task_ind=batch["task_ind"][global_bi] if "task_ind" in batch else torch.tensor(0, dtype=torch.int32, device=self.device),
                traj_id=batch["traj_id"][global_bi] if "traj_id" in batch else torch.tensor(0, dtype=torch.int32, device=self.device),
                valid_masks_T=valid_masks_T,
                valid_masks_t=valid_masks_t,
            )
            transitions.append(sample)

            if self.debug:
                _time_sample_construct += time.time() - _t_start

        if self.debug:
            _time_total = time.time() - _t_total_start
            print(
                f"[cyan][process_batch timing (B={B})]\n"
                f"  inference_batch : {_time_inference:.4f}s\n"
                f"  text_encode     : {_time_text_encode:.4f}s\n"
                f"  devoxelize+z    : {_time_devoxelize:.4f}s\n"
                f"  struct/unstruct : {_time_struct_unstruct:.4f}s\n"
                f"  robot_sdf+anchor: {_time_robot_sdf_anchors:.4f}s\n"
                f"  sdf_flow        : {_time_sdf_flow:.4f}s\n"
                f"  sample_construct: {_time_sample_construct:.4f}s\n"
                f"  total           : {_time_total:.4f}s[/cyan]"
            )

        return transitions
