import os.path as osp
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
from utils.pv import o3d_mesh_to_pv, render_pcds, Plotter
from utils.vis import sdf_to_colors, sdf_grad_to_colors
import third_party.pytorch_volumetric.model_to_sdf_cuda as pv_cuda
from src import models, datasets
from jaxtyping import Float32, Bool, Int32
from torch import Tensor
from src.trainers.vae.structured_latent_vae_gaussian_vcw import (
    VcwSLatVaeGaussianTrainer,
    InferenceBatchOutput,
    TrajectoryBatch,
)
from src.modules.sparse import SparseTensor
from src.datasets.trajectory_dataset import TrajectoryDataset, RobotInfo
from datalib.remote_dataset import RemoteQueueDataset
from .base import DataWorker


class TransitionSample(TypedDict):
    """
    Represents a single physics transition sample.
    """

    xt: Tuple[Int32[Tensor, "P 3"], Float32[Tensor, "P zdim"]]
    xt_plus_T: Tuple[Int32[Tensor, "Q 3"], Float32[Tensor, "Q zdim"]]
    v_at_to_T: Tuple[Int32[Tensor, "P 3"], Float32[Tensor, "P adim"]]
    text: str

    voxel_params: Float32[Tensor, "3 3"]
    mask_robot_t_plus_T: Bool[Tensor, " Q "]
    rgbs_t_plus_T: Float32[Tensor, "CAM 3 H W"]
    mask_fg_t_plus_T: Bool[Tensor, "CAM H W"]
    w2cs: Float32[Tensor, "CAM 4 4"]
    ints: Float32[Tensor, "CAM 3 3"]


class TransitionBatch(TypedDict):
    """
    Represents a batched collection of physics transition samples.
    """

    xt: SparseTensor
    xt_plus_T: SparseTensor
    v_at_to_T: SparseTensor
    text: List[str]

    voxel_params: Float32[Tensor, "B 3 3"]

    mask_robot_t: Bool[Tensor, " concat_P "]
    mask_static_t: Bool[Tensor, " concat_P "]
    mask_robot_t_plus_T: Bool[Tensor, " concat_Q "]
    mask_static_t_plus_T: Bool[Tensor, " concat_Q "]

    rgbs_t_plus_T: Float32[Tensor, "B CAM 3 H W"]
    mask_fg_t_plus_T: Bool[Tensor, "B CAM H W"]
    w2cs: Float32[Tensor, "B CAM 4 4"]
    ints: Float32[Tensor, "B CAM 3 3"]


class PhysicsTransitionDatasetV0(RemoteQueueDataset):
    """
    Dataset class for remote queue transitions.
    """

    value_range = (0.0, 1.0)

    @staticmethod
    def collate_fn(samples: List[TransitionSample]) -> TransitionBatch:
        """
        Custom collate function to handle SparseTensors in a batch.
        """

        def _collate_sparse(
            list_of_tuples: List[Tuple[Tensor, Tensor]],
        ) -> SparseTensor:
            # List of (coords, feats)
            batched_coords = []
            batched_feats = []
            for i, (coords, feats) in enumerate(list_of_tuples):
                # coords: (P, 3) -> (P, 4) with batch index at 0
                batch_idx = torch.full(
                    (coords.shape[0], 1), i, dtype=coords.dtype, device=coords.device
                )
                batched_coords.append(torch.cat([batch_idx, coords], dim=1))
                batched_feats.append(feats)

            return SparseTensor(
                coords=torch.cat(batched_coords, dim=0),
                feats=torch.cat(batched_feats, dim=0),
            )

        xt = _collate_sparse([s["xt"] for s in samples])
        xt_plus_T = _collate_sparse([s["xt_plus_T"] for s in samples])
        v_at_to_T = _collate_sparse([s["v_at_to_T"] for s in samples])

        return {
            "xt": xt,
            "xt_plus_T": xt_plus_T,
            "v_at_to_T": v_at_to_T,
            "voxel_params": torch.stack([s["voxel_params"] for s in samples]),
            "mask_robot_t_plus_T": torch.cat(
                [s["mask_robot_t_plus_T"] for s in samples]
            ),
            "mask_robot_t": torch.cat([s["mask_robot_t"] for s in samples]),
            "mask_static_t": torch.cat([s["mask_static_t"] for s in samples]),
            "mask_static_t_plus_T": torch.cat(
                [s["mask_static_t_plus_T"] for s in samples]
            ),
            "rgbs_t_plus_T": torch.stack([s["rgbs_t_plus_T"] for s in samples]),
            "mask_fg_t_plus_T": torch.stack([s["mask_fg_t_plus_T"] for s in samples]),
            "w2cs": torch.stack([s["w2cs"] for s in samples]),
            "ints": torch.stack([s["ints"] for s in samples]),
        }


class TransitionEngine:
    """
    Engine for generating physics transitions from trajectory data.
    """

    def __init__(
        self,
        trainer: VcwSLatVaeGaussianTrainer,
        device: Union[str, torch.device] = "cpu",
        debug: bool = False,
    ):
        import mani_skill

        self.device = device
        self.trainer = trainer
        self.debug = debug
        self._asset_path = osp.join(mani_skill.__path__[0], "assets")
        self._robot_sdfs = {}

    def load_robot_sdf(self, robot_infos: List[RobotInfo]) -> pv_cuda.RobotSDF:
        """
        Load or retrieve cached RobotSDF for the given robot information.
        """
        names = tuple([robot_info.uid for robot_info in robot_infos])
        if names in self._robot_sdfs:
            return self._robot_sdfs[names]

        chains = []
        for robot_info in robot_infos:
            urdf_path = osp.join(self._asset_path, robot_info.urdf_path)
            chain = pk.build_chain_from_urdf(open(urdf_path).read())
            chain = chain.to(device=self.device)
            chains.append(chain)

        s = pv_cuda.RobotSDF(
            chains, path_prefix=osp.dirname(urdf_path), use_collision_mesh=False
        )
        self._robot_sdfs[names] = s
        return s

    def to_transitions(
        self, batch: TrajectoryBatch, results: InferenceBatchOutput
    ) -> List[TransitionSample]:
        """
        Convert inference results and trajectory data into a list of TransitionSamples.
        """
        (B, T, CAM, H, W) = results["shape"]
        assert T == 2, "num frames must be 2"
        z = results["z"]

        transitions = []
        aug_params = results["aug_params"]
        assert (
            z.coords[:, 0].unique().numel()
            == len(aug_params)
            == len(results["voxel_params"])
        )
        devoxelized = self.trainer.voxelization.devoxelize(
            z.coords,
            batch=None,
            norm_params=results["voxel_params"],
            aug_params=aug_params,
        )

        for global_bi in range(B):
            time_bi = global_bi * T
            xt_batch_indices = z.coords[:, 0] == time_bi
            xt_plus_T_batch_indices = z.coords[:, 0] == (time_bi + 1)

            robot_infos = batch["robot_infos"][global_bi]
            robot_sdf = self.load_robot_sdf(robot_infos)
            qpos = torch.from_numpy(batch["qpos"][global_bi][1:]).to(
                self.device
            )  # actions
            root_poses = torch.from_numpy(batch["root_poses"][global_bi][1:]).to(
                self.device
            )  # actions

            action_feats = []
            devoxelized_xt = devoxelized[xt_batch_indices]

            for ti in range(len(qpos)):
                robot_sdf.set_joint_configuration(
                    [qpos[ti, ri] for ri in range(len(robot_infos))]
                )

                base_transforms = []
                for root_pose in root_poses[ti]:
                    translation = root_pose[:3]
                    rotation_quat = root_pose[3:]
                    obj_to_world = pk.Transform3d(
                        pos=translation, rot=rotation_quat, device=self.device
                    )
                    world_to_obj = obj_to_world.inverse()
                    base_transforms.append(world_to_obj)

                robot_sdf.set_base_transform(
                    base_transforms[0] if len(base_transforms) == 1 else base_transforms
                )
                sdf_vals, sdf_grads = robot_sdf(devoxelized_xt)
                if self.debug:
                    self._visualize_debug(
                        robot_sdf,
                        devoxelized_xt,
                        sdf_vals,
                        sdf_grads,
                        global_bi,
                        ti + 1,
                    )

                action_feats.append(
                    torch.cat([sdf_vals.reshape(-1, 1), sdf_grads], dim=1)
                )

            xt = (z.coords[xt_batch_indices, 1:], z.feats[xt_batch_indices])
            xt_plus_T = (
                z.coords[xt_plus_T_batch_indices, 1:],
                z.feats[xt_plus_T_batch_indices],
            )

            transition = {
                "xt": xt,
                "xt_plus_T": xt_plus_T,
                "v_at_to_T": (
                    z.coords[xt_batch_indices, 1:],
                    torch.cat(
                        action_feats, dim=1
                    ),  # horizon=10 -> size = 36 = (10-1)*4
                ),
                "text": batch["task_description"][global_bi],
                "voxel_params": results["voxel_params"][time_bi],
                "mask_robot_t_plus_T": results["attrs"]["robot"].flatten()[
                    xt_plus_T_batch_indices
                ],
                "mask_robot_t": results["attrs"]["robot"].flatten()[xt_batch_indices],
                "mask_static_t_plus_T": results["attrs"]["static"].flatten()[
                    xt_plus_T_batch_indices
                ],
                "mask_static_t": results["attrs"]["static"].flatten()[xt_batch_indices],
                "rgbs_t_plus_T": batch["rgbs"][global_bi, 1],
                "mask_fg_t_plus_T": batch["foreground_masks"][global_bi, 1],
                "w2cs": results["w2c"].reshape(B, T, CAM, 4, 4)[global_bi, 1, :],
                "ints": results["intrinsics"].reshape(B, T, CAM, 3, 3)[global_bi, 1, :],
            }
            transitions.append(transition)

        if self.debug:
            PhysicsTransitionDatasetV0.collate_fn(transitions)
        return transitions

    def _visualize_debug(
        self,
        robot_sdf: pv_cuda.RobotSDF,
        devoxelized: Tensor,
        sdf_vals: Tensor,
        sdf_grads: Tensor,
        batch_idx: int,
        t_idx: int,
    ) -> None:
        """
        Internal debug visualization of SDF and gradients.
        """
        meshes = pv.get_transformed_meshes(robot_sdf)
        combined_mesh = o3d.geometry.TriangleMesh()

        # Merge all links into the combined mesh
        for mesh in meshes:
            combined_mesh += mesh

        combined_mesh.compute_vertex_normals()
        pv_mesh = o3d_mesh_to_pv(combined_mesh)

        # Visualize SDF
        plotter = Plotter(backend=f"remote:/tmp/pvlib.state.{batch_idx}.{t_idx}.sdf")
        plotter.update_param(
            "meshes",
            [{"name": "robot", "mesh": pv_mesh, "kwargs": {"color": "gray"}}],
        )
        plotter.update_param("means", devoxelized.cpu().numpy())
        plotter.update_param("colors", sdf_to_colors(sdf_vals))
        plotter.render()

        # Visualize Gradients
        plotter = Plotter(backend=f"remote:/tmp/pvlib.state.{batch_idx}.{t_idx}.grad")
        plotter.update_param(
            "meshes",
            [{"name": "robot", "mesh": pv_mesh, "kwargs": {"color": "gray"}}],
        )
        plotter.update_param("means", devoxelized.cpu().numpy())
        plotter.update_param("colors", sdf_grad_to_colors(sdf_grads))
        plotter.render()


def visualize_transitions(transitions: List[TransitionSample], count: int) -> None:
    """
    Visualize masks and generated transitions.
    """
    for i, trans in enumerate(transitions):
        # Visualize masks
        coords = trans["xt"][0]
        mask_static = trans["mask_static_t"].bool()
        mask_robot = trans["mask_robot_t"].bool()

        pcd_dict = {}
        if mask_static.sum() > 0:
            pcd_dict["static"] = coords[mask_static]
        if mask_robot.sum() > 0:
            pcd_dict["robot"] = coords[mask_robot]

        mask_other = ~(mask_static | mask_robot)
        if mask_other.sum() > 0:
            pcd_dict["other"] = coords[mask_other]
        render_pcds(pcd_dict, backend=f"/tmp/vis_{count}_{i}_masks.pkl")


class PhysicsTransitionWorker(DataWorker):
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
        self.cfg = cfg
        self.data_override = data_override
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug

        # Merge configs
        if hasattr(data_override.dataset, "full_args"):
            cfg.dataset.args = data_override.dataset.full_args

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
        self.trainer = VcwSLatVaeGaussianTrainer(
            self.model_dict,
            self.dataset,
            **cfg.trainer.args,
            output_dir="/tmp",
            load_dir=load_dir,
            step=None,
            wandb_run=None,
        )
        self.trainer.load(load_dir=load_dir)

        self.engine = TransitionEngine(trainer=self.trainer, device=device, debug=debug)
        self.image_keys = data_override.dataset.args.image_keys

    @torch.no_grad()
    def process_batch(self, batch_cuda: Dict[str, Any]) -> List[TransitionSample]:
        """
        Runs inference and generates physics transitions.
        """
        inference_results = self.trainer.inference_batch(
            self.model_dict, batch_cuda, augment_voxel_grid=True
        )
        transitions = self.engine.to_transitions(batch_cuda, inference_results)
        return transitions

    def visualize(self, transitions: List[TransitionSample], count: int) -> None:
        """
        Entry point for visualization.
        """
        visualize_transitions(transitions, count)
