import os.path as osp
import mani_skill
from functools import reduce
from rich import print
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
import pickle
from torch.utils.data import DataLoader, Dataset

from utils.misc import move_to_device
import third_party.pytorch_kinematics as pk
import third_party.pytorch_volumetric as pv
import open3d as o3d
from utils.pv_k3d import render_pcds
from utils.vis import to_pil, sdf_to_colors, sdf_grad_to_colors
from src import models, datasets, trainers
from jaxtyping import Float32, Bool, Int32
from torch import Tensor
from src.modules.sparse import SparseTensor
from src.datasets.trajectory_dataset import TrajectoryDataset, RobotInfo
from datalib.remote_dataset import RemoteQueueDataset
from third_party.pytorch_volumetric.robot_anchors import RobotAnchorSDF
from .base import DataWorker


# ================================================
# Interface
# ================================================


class TransitionSample(TypedDict):
    """
    Represents a single physics transition sample.
    """

    structure_xt: Tuple[Int32[Tensor, "P 3"], Float32[Tensor, "P zdim"]]
    structure_xT: Tuple[Int32[Tensor, "Q 3"], Float32[Tensor, "Q zdim"]]

    unstructure_xt: Float32[Tensor, "a b"]
    unstructure_xT: Float32[Tensor, "a b"]

    sdf_t_T: Tuple[Int32[Tensor, "P 3"], Float32[Tensor, "P horizon 5"]]

    rgbs_T: Float32[Tensor, "CAM 3 H W"]
    rgbs_t: Float32[Tensor, "CAM 3 H W"]
    masks_fg_T: Bool[Tensor, "CAM H W"]
    robot_masks_T: Bool[Tensor, "CAM H W"]
    static_masks_T: Bool[Tensor, "CAM H W"]

    voxel_params: Float32[Tensor, "3 3"]
    aug_params: Float32[Tensor, "3"]

    w2cs: Float32[Tensor, "CAM 4 4"]
    ints: Float32[Tensor, "CAM 3 3"]
    anchor_points: Float32[Tensor, "SH N 3"]


class TransitionBatch(TypedDict):
    """
    Represents a batched collection of physics transition samples.
    """

    structure_xt: SparseTensor
    structure_xT: SparseTensor

    unstructure_xt: Float32[Tensor, "B a b"]
    unstructure_xT: Float32[Tensor, "B a b"]

    sdf_t_T: SparseTensor

    rgbs_T: Float32[Tensor, "B CAM 3 H W"]
    rgbs_t: Float32[Tensor, "B CAM 3 H W"]
    masks_fg_T: Bool[Tensor, "B CAM H W"]
    robot_masks_T: Bool[Tensor, "B CAM H W"]
    static_masks_T: Bool[Tensor, "B CAM H W"]

    voxel_params: Float32[Tensor, "B 3 3"]
    aug_params: Float32[Tensor, "B 3"]

    w2cs: Float32[Tensor, "B CAM 4 4"]
    ints: Float32[Tensor, "B CAM 3 3"]


# ================================================
# Worker
# ================================================


class RobotSDFWorkerMixin:
    def load_robot_sdf(self, robot_infos: List[RobotInfo]) -> RobotAnchorSDF:
        """
        Load or retrieve cached RobotSDF for the given robot information.
        """
        _asset_path = osp.join(mani_skill.__path__[0], "assets")
        if not hasattr(self, "_robot_sdfs"):
            self._robot_sdfs = {}
        names = tuple([robot_info.uid for robot_info in robot_infos])
        if names in self._robot_sdfs:
            return self._robot_sdfs[names]

        chains = []
        for robot_info in robot_infos:
            urdf_path = osp.join(_asset_path, robot_info.urdf_path)
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
        cfg: edict,
        data_override: edict,
        device: Union[str, torch.device],
        batch_size: int,
        num_workers: int,
        debug: bool = False,
    ):
        super().__init__(
            cfg,
            data_override,
            device,
            batch_size,
            num_workers,
            debug,
        )
        assert batch_size == 1
        self.cfg = cfg
        self.data_override = data_override
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.min_horizon, self.horizon = data_override.dataset.full_args.args.horizon

        # Merge configs
        if hasattr(data_override.dataset, "full_args"):
            cfg.dataset = data_override.dataset.full_args

        self.kwargs = data_override.dataset.args.kwargs

        if hasattr(data_override.dataset, "trainer_args"):
            for k, v in data_override.dataset.trainer_args.items():
                print(f"[yellow][Trainer Config] Setting {k} to {v}[/yellow]")
                setattr(cfg.trainer.args, k, v)

        print("Initializing models...")
        self.model_dict = {
            name: getattr(models, model.name)(**model.args).to(device)
            for name, model in cfg.models.items()
        }

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
        )
        self.trainer.load(
            load_dir=load_dir, step=data_override.trainer.args.vae_step_suffix
        )
        self.image_keys = data_override.dataset.args.image_keys
        self.prefix = "cpt_" if "V4" in cfg.trainer.name else ""
        self.robot_sdfs = {}
        self.nop_action_prob = self.kwargs.get("nop_action_prob", 0.35)
        self.anchor_num = self.kwargs.get("anchor_num", 10000)
        self.flow_on_robot = self.kwargs.get("flow_on_robot", True)
        self.min_points_per_link = self.kwargs.get("min_points_per_link", 100)

    def visualize(self, transitions: List[TransitionSample], count: int) -> None:
        """
        Entry point for visualization.
        """

        for i, sample in enumerate(transitions):
            if i > 0:
                break
            D = int(round(sample["unstructure_xt"].shape[0] ** (1 / 3)))
            inp = {
                self.prefix + "z": sample["unstructure_xt"]
                .reshape(D, D, D, -1)
                .permute(3, 0, 1, 2)[None]
            }
            out = self.trainer.decode(
                **inp,
                w2c=sample["w2cs"],
                intrinsics=sample["ints"],
            )
            to_pil(out[self.prefix + "color"]).save(
                "runs/debug/latent_transition_worker.png"
            )
            act_dim = int(sample["sdf_t_T"][1][0, :, 4].sum().item())
            devoxelized = self.trainer.voxelization.devoxelize(
                sample["sdf_t_T"][0],
                batch=None,
                norm_params=sample["voxel_params"][None],
                aug_params=sample["aug_params"][None],
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
        pts_t0: torch.Tensor,
        robot_mask_t0: torch.Tensor,
        anchors_T: torch.Tensor,
    ) -> torch.Tensor:
        num_T = anchors_T.shape[0]
        num_pts = pts_t0.shape[0]
        # sdf_results shape: (num_pts, num_T, 5)
        # 5th dimension (index 4) is 1.0 for valid actions
        sdf_results = torch.zeros(
            (num_pts, num_T, 5), device=self.device, dtype=pts_t0.dtype
        )
        sdf_results[:, :, 4] = 1.0

        # 1. Handle Robot Points (Correspondence-based Flow)
        if self.flow_on_robot and robot_mask_t0.any():
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
            ~robot_mask_t0 if self.flow_on_robot else torch.ones_like(robot_mask_t0)
        )
        if env_mask.any():
            env_pts = pts_t0[env_mask]  # (P_env, 3)

            for ti in range(num_T):
                dists_ti = torch.cdist(
                    env_pts, anchors_T[ti : ti + 1]
                )  # (P_env, 1, num_anchors)
                min_dists, min_indices = dists_ti.min(dim=-1)  # (P_env, 1)

                closest_anchors = anchors_T[ti, min_indices.squeeze(1)]  # (P_env, 3)
                diff = closest_anchors - env_pts  # (P_env, 3)
                dist = torch.linalg.norm(diff, dim=-1, keepdim=True)  # (P_env, 1)
                grad = diff / (dist + 1e-8)

                sdf_results[env_mask, ti, 0:1] = dist
                sdf_results[env_mask, ti, 1:4] = grad

        # 3. Convert Distance to Voxel Space
        sdf_results[:, :, 0] /= self.trainer.granularity
        return sdf_results

    @torch.no_grad()
    def process_batch(self, batch: Dict[str, Any]) -> List[TransitionSample]:
        """
        Runs inference and generates physics transitions.
        """
        results = self.trainer.inference_batch(
            self.model_dict, batch, augment_voxel_grid=True, skip_decoding=True
        )
        B, T, CAM, _, H, W = batch["rgbs"].shape
        assert T == 2, "num frames must be 2"

        def chwd_to_lc(chwd):
            return chwd.flatten(1).permute(1, 0)

        input_feats = results["input_feats"]
        transitions = []
        aug_params = results["aug_params"]
        assert (
            input_feats.coords[:, 0].unique().numel()
            == len(aug_params)
            == len(results["voxel_params"])
        )
        devoxelized = self.trainer.voxelization.devoxelize(
            input_feats.coords,
            batch=None,
            norm_params=results["voxel_params"],
            aug_params=aug_params,
        )
        voxels_robot = results["voxels_robot"]
        z = results[self.prefix + "z"]

        for global_bi in range(B):
            time_bi = global_bi * T
            qpos = torch.from_numpy(batch["qpos"][global_bi]).to(self.device)
            sampled_horizon = qpos.shape[0]

            # NOP Action logic: triggered when horizon is minimum and prob hits
            is_nop = (sampled_horizon == self.min_horizon) and (
                np.random.random() < self.nop_action_prob
            )

            xt_mask = input_feats.coords[:, 0] == time_bi
            pts_t0 = devoxelized[xt_mask]  # (P, 3)
            robot_mask_t0 = voxels_robot[xt_mask].bool().view(-1)  # (P,)

            # Load robot SDF and compute anchors for all actions
            robot_infos = batch["robot_infos"][global_bi]
            robot_sdf = self.load_robot_sdf(robot_infos)
            root_poses = torch.from_numpy(batch["root_poses"][global_bi]).to(
                self.device
            )
            anchor_state = robot_sdf.initialize_anchors(
                self.anchor_num,
                seed=-1,
                min_points_per_link=self.min_points_per_link,
            )

            tracked_points = robot_sdf.forward_kinematic_anchors(
                anchor_state,
                qpos.reshape(1, sampled_horizon, 1, -1),
                root_poses.reshape(1, sampled_horizon, 1, -1),
            )  # (1, SH, N, 3)
            anchors_sh = tracked_points.squeeze(0)  # (SH, N, 3)

            if is_nop:
                # NOP: x_T = x_t, z_T = z_t
                structure_xT = (
                    input_feats.coords[xt_mask, 1:],
                    input_feats.feats[xt_mask],
                )
                unstructure_xT = z[time_bi]

                # Interface uses index 0 instead of 1 for NOP
                rgbs_T = batch["rgbs"][global_bi, 0]
                masks_fg_T = batch["foreground_masks"][global_bi, 0]
                robot_masks_T = batch["robot_masks"][global_bi, 0]
                static_masks_T = batch["static_masks"][global_bi, 0]
                w2cs = results["w2c"].reshape(B, T, CAM, 4, 4)[global_bi, 0]
                ints = results["intrinsics"].reshape(B, T, CAM, 3, 3)[global_bi, 0]

                # Use only the first frame of anchors for NOP
                anchors_T = anchors_sh[:1]
                effective_horizon = 1
            else:
                # Normal processing: x_T = x_{t+1}
                xT_mask = input_feats.coords[:, 0] == (time_bi + 1)
                structure_xT = (
                    input_feats.coords[xT_mask, 1:],
                    input_feats.feats[xT_mask],
                )
                unstructure_xT = z[time_bi + 1]

                rgbs_T = batch["rgbs"][global_bi, 1]
                masks_fg_T = batch["foreground_masks"][global_bi, 1]
                robot_masks_T = batch["robot_masks"][global_bi, 1]
                static_masks_T = batch["static_masks"][global_bi, 1]
                w2cs = results["w2c"].reshape(B, T, CAM, 4, 4)[global_bi, 1]
                ints = results["intrinsics"].reshape(B, T, CAM, 3, 3)[global_bi, 1]

                anchors_T = anchors_sh
                effective_horizon = sampled_horizon

            # Compute SDF flow (P, effective_horizon, 5)
            # with index 4 set to 1.0 for all entries in anchors_T
            sdf_sh = self._compute_sdf_flow(pts_t0, robot_mask_t0, anchors_T)

            # Pad to self.horizon (max)
            num_pts = pts_t0.shape[0]
            sdf_results = torch.zeros(
                (num_pts, self.horizon, 5), device=self.device, dtype=pts_t0.dtype
            )
            sdf_results[:, :effective_horizon] = sdf_sh
            # The 5th dimension is 1.0 for valid frames [:effective_horizon]
            # and 0.0 for padded frames [effective_horizon:] due to zeros initialization.

            sample = TransitionSample(
                structure_xt=(
                    input_feats.coords[xt_mask, 1:],
                    input_feats.feats[xt_mask],
                ),
                structure_xT=structure_xT,
                unstructure_xt=chwd_to_lc(z[time_bi]),
                unstructure_xT=chwd_to_lc(unstructure_xT),
                sdf_t_T=(input_feats.coords[xt_mask, 1:], sdf_results),
                rgbs_T=rgbs_T,
                rgbs_t=batch["rgbs"][global_bi, 0],
                masks_fg_T=masks_fg_T,
                robot_masks_T=robot_masks_T,
                static_masks_T=static_masks_T,
                voxel_params=results["voxel_params"][time_bi],
                aug_params=results["aug_params"][time_bi],
                w2cs=w2cs,
                ints=ints,
                anchor_points=anchors_T,
            )
            transitions.append(sample)

        return transitions
