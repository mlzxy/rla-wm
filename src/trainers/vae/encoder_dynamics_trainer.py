"""
Dynamics Trainer (V6x).

Predicts latent z_T from observation at time t using the modulated SparseSDFSideCar architecture.
No latent-to-latent transformer denoiser is used.
"""

import copy
import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from rich import print
from torch import nn
from torch.utils.data import DataLoader

from src import models
from utils.misc import load_config, move_to_device
from utils.vis import to_pil, annotate_images

from ...utils.loss_utils import (
    gradient_l1_loss,
    l1_loss,
    l2_loss,
    lpips,
    ssim,
    compute_zoom_in_loss,
)
from utils.loss import dice_loss, focal_loss
from utils.voxel import hash_coords

def compute_iou(logits, labels):
    preds = (logits > 0).float()
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

from ...datasets.data_worker.latent_transition_dataset_v2 import (
    TransitionBatch,
    LatentTransitionWorker,
)
from ...datasets.data_worker.base import SimpleTransitionDataset
from ...modules.sparse import SparseTensor

from ..basic import BasicTrainer
from .structured_latent_vae_gaussian_vcw import GaussianRenderingMixin


# ═══════════════════════════════════════════════════════════════════
# Loss Utilities
# ═══════════════════════════════════════════════════════════════════


def kl_divergence_gaussians(
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
) -> torch.Tensor:
    """
    KL(p || q) where p = N(mean_p, diag(exp(logvar_p))),
                      q = N(mean_q, diag(exp(logvar_q))).

    Per-element KL divergence, averaged over all dims.

    Args:
        mean_p, logvar_p: Parameters of distribution p (predicted).
        mean_q, logvar_q: Parameters of distribution q (ground truth target).

    Returns:
        Scalar mean KL divergence.
    """
    var_p = logvar_p.exp()
    var_q = logvar_q.exp()
    kl = 0.5 * (
        logvar_q - logvar_p + var_p / var_q + (mean_p - mean_q) ** 2 / var_q - 1
    )
    return kl.mean()


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════


class EncoderDynamicsTrainer(GaussianRenderingMixin, BasicTrainer):
    """
    Modulated dynamics trainer for structured latent VAE.

    Predicts z_T distribution from raw observation at time t, conditioned on
    SDF/horizon features. No latent-to-latent transformer denoiser.

    Models (from config):
        - "encoder": Learnable encoder
        - "sdf_encoder" (optional): SDF feature encoder
        - "time_embedder" (optional): Timestep embedder (flow_t)
        - "task_embedder" (optional): Task index embedder (task_ind)
    Frozen internal models (loaded in __init__):
        - "decoder": Frozen decoder for optional render loss

    Args:
        lambda_kl_dynamics: KL loss weight (predicted vs GT distributions)
        lambda_render: Render loss weight (0 = disabled)
        lambda_ssim: SSIM weight for render loss
        lambda_lpips: LPIPS weight for render loss
        vae_model_dir: Path to pretrained VAE checkpoint
        internal_models: Dict of internal model configs
        internal_ckpt: Dict of checkpoint paths for internal models
        horizon: Max SDF horizon steps
        use_horizon_t: Encode horizon timestep as per-voxel features
        coarse_resolution: Coarse resolution for dense latent (D, H, W)
        num_samples: Max samples for snapshot
        clamp_z: Clamp z values if > 0
        normalize_z: Divide z by this value if > 0
    """

    def __init__(
        self,
        *args,
        lambda_kl_dynamics: float = 1.0,
        lambda_feat_mse: float = 0.0,
        lambda_render: float = 0.0,
        loss_type: str = "l1",
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_edge: float = 0.0,
        lambda_diff: float = 10.0,
        lambda_focus: float = 0.0,
        vae_model_dir: str = "",
        vae_step_suffix: str = "",
        internal_models: Dict = {},
        internal_ckpt: Dict = {},
        resolution: int = 128,
        coarse_resolution: int = 8,
        horizon: int = 10,
        num_samples: int = 4,
        process_data_locally: bool = False,
        process_data_locally_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lambda_kl_dynamics = lambda_kl_dynamics
        self.lambda_feat_mse = lambda_feat_mse
        self.lambda_render = lambda_render
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_edge = lambda_edge
        self.lambda_diff = lambda_diff
        self.lambda_focus = lambda_focus
        self.vae_model_dir = vae_model_dir
        self.vae_step_suffix = vae_step_suffix
        self.resolution = resolution
        self.coarse_resolution = coarse_resolution
        self.max_horizon = horizon
        self.num_samples = num_samples

        # Local data processing (for dev/debug)
        self.process_data_locally = process_data_locally or self.debug
        self.process_data_locally_eval = process_data_locally_eval
        if self.process_data_locally or self.process_data_locally_eval:
            self.local_data_cfg = edict(
                load_config(os.path.join(vae_model_dir, "config.yaml"))
            )
            self.local_data_cfg.output_dir = vae_model_dir
            self.dataset_worker = LatentTransitionWorker(
                cfg=self.local_data_cfg,
                data_override=kwargs["cfg"],
                device=self.device,
                dataset_is_none=True,
                debug=self.debug,
            )
        else:
            self.local_data_cfg = None
            self.dataset_worker = None

        # ----- Load frozen decoder for render loss / visualization -----
        if internal_models and "decoder" in internal_models:
            decoder_config = internal_models["decoder"]
            self.decoder = getattr(models, decoder_config["name"])(
                **decoder_config["args"]
            ).to(self.device)

            if internal_ckpt and "decoder" in internal_ckpt:
                ckpt_path = internal_ckpt["decoder"]
                print(f"[yellow]Loading decoder from {ckpt_path}[/yellow]")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt
                self.decoder.load_state_dict(state_dict)
                print("[green]Decoder loaded successfully[/green]")

            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            self.decoder = None

        # Load pretrained weights for the learnable encoder
        if vae_model_dir and "encoder" in self.models:
            self._load_encoder_weights(vae_model_dir, vae_step_suffix)
        
        # Sync the newly loaded weights to master_params for 'inflat_all'
        model_ckpts = {
            name: model.state_dict()
            for name, model in self.models.items()
            if name not in getattr(self, "freeze_models", [])
        }
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        print("[green]Master params synced[/green]")

    # ═══════════════════════════════════════════════════════════════
    # Weight Loading
    # ═══════════════════════════════════════════════════════════════

    def _load_encoder_weights(self, vae_model_dir: str, vae_step_suffix: str):
        """
        Load pretrained encoder weights with partial channel adaptation.
        Only loads matching parameters; for input_layer, copies only
        the first matching channels from the pretrained weights.
        """
        ckpt_dir = os.path.join(vae_model_dir, "ckpts")
        if not os.path.exists(ckpt_dir):
            print(f"[red]Checkpoint dir not found: {ckpt_dir}[/red]")
            return

        if vae_step_suffix:
            target_ckpt = f"encoder_step{vae_step_suffix}.pt"
            ckpt_path = os.path.join(ckpt_dir, target_ckpt)
            if not os.path.exists(ckpt_path):
                alt_ckpt_path = os.path.join(ckpt_dir, f"encoder_step{vae_step_suffix}")
                if os.path.exists(alt_ckpt_path):
                    ckpt_path = alt_ckpt_path
                else:
                    print(f"[red]Encoder checkpoint not found: {ckpt_path}[/red]")
                    raise FileNotFoundError(
                        f"Encoder checkpoint not found: {ckpt_path}"
                    )
        else:
            encoder_ckpts = [
                f for f in os.listdir(ckpt_dir) if f.startswith("encoder_step")
            ]
            if not encoder_ckpts:
                print(f"[red]No encoder checkpoints found in {ckpt_dir}[/red]")
                raise FileNotFoundError(f"No encoder checkpoints found in {ckpt_dir}")
            encoder_ckpts.sort(key=lambda x: int(x.split("step")[1].split(".")[0]))
            ckpt_path = os.path.join(ckpt_dir, encoder_ckpts[-1])

        print(f"[yellow]Loading encoder weights from {ckpt_path}[/yellow]")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            pretrained_state = ckpt["model_state_dict"]
        else:
            pretrained_state = ckpt

        current_state = self.models["encoder"].state_dict()
        loaded_keys = []
        skipped_keys = []
        partial_keys = []

        for key, pretrained_param in pretrained_state.items():
            if key not in current_state:
                skipped_keys.append(f"{key} (not in current model)")
                continue

            current_param = current_state[key]

            if current_param.shape == pretrained_param.shape:
                current_state[key] = pretrained_param
                loaded_keys.append(key)
            elif "input_layer" in key and current_param.dim() == 2:
                # input_layer weight: (out_channels, in_channels)
                pretrained_in = pretrained_param.shape[1]
                current_in = current_param.shape[1]
                if current_in >= pretrained_in:
                    current_state[key][:, :pretrained_in] = pretrained_param
                    partial_keys.append(
                        f"{key}: copied [{pretrained_in}] into [{current_in}]"
                    )
                else:
                    current_state[key] = pretrained_param[:, :current_in]
                    partial_keys.append(
                        f"{key}: truncated [{pretrained_in}] to [{current_in}]"
                    )
            else:
                skipped_keys.append(
                    f"{key} (shape mismatch: {pretrained_param.shape} vs {current_param.shape})"
                )

        self.models["encoder"].load_state_dict(current_state)

        print(
            f"[green]Encoder weights loaded: "
            f"{len(loaded_keys)} full, {len(partial_keys)} partial, "
            f"{len(skipped_keys)} skipped[/green]"
        )
        if partial_keys:
            print("[yellow]Partially loaded keys:[/yellow]")
            for k in partial_keys:
                print(f"  - {k}")
        if skipped_keys:
            print("[yellow]Skipped keys:[/yellow]")
            for k in skipped_keys:
                print(f"  - {k}")

    # ═══════════════════════════════════════════════════════════════
    # Flow/Horizon Time
    # ═══════════════════════════════════════════════════════════════

    def _make_flow_t(self, data: TransitionBatch) -> Optional[torch.Tensor]:
        """
        Return horizon-based time signal, or None if use_horizon_t is False.

        When enabled, computes:
            t_delta = ((T_frame_id - t_frame_id) / (num_frames - 1)) * 1000
        """
        horizon = data["horizon"]  # (B,)
        t_delta = horizon * 100  # (B,) in [0, 1000]
        return t_delta.flatten()

    # ═══════════════════════════════════════════════════════════════
    # Local Data Processing
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _preprocess_trajectory_batch(self, data: dict, is_eval=False) -> TransitionBatch:
        """
        Convert a raw trajectory batch into TransitionBatch via the
        LatentTransitionWorker pipeline.
        """
        data = move_to_device(data, self.device)
        transitions = self.dataset_worker.process_batch(data, is_eval=is_eval)
        # if self.debug:
            # self.dataset_worker.visualize(transitions)
        batch = SimpleTransitionDataset.collate_fn(transitions)
        return move_to_device(batch, self.device)

    # ═══════════════════════════════════════════════════════════════
    # Rendering Helper
    # ═══════════════════════════════════════════════════════════════

    def _render_latent(
        self,
        lat_5d: torch.Tensor,
        data: dict,
        gt_img: torch.Tensor,
        cam_idx: int = 0,
    ):
        """Decode a 5D latent and render to images for visualization."""
        if self.decoder is None:
            return None

        decode_out = self.decoder(
            lat_5d,
            gt_structure=None,
            to_rep_kwargs={"aabb": [0, 0, 0, 1.0, 1.0, 1.0]},
        )
        if decode_out.get("inference_diverged", False):
            blank_img = torch.zeros_like(gt_img)
            annotated = annotate_images(
                blank_img, ["Diverged"] * gt_img.shape[0], font_size=25
            )
            return annotated.to(self.device)
        else:
            gaussians = decode_out["reps"]
            w2c = data["w2cs"][:, cam_idx]
            intrinsics = data["ints"][:, cam_idx]
            H_img, W_img = gt_img.shape[-2:]
            return self._render_batch(
                gaussians,
                w2c,
                intrinsics,
                bg_color=(0, 0, 0),
                img_wh=(W_img, H_img),
            )["color"]
            
    # ═══════════════════════════════════════════════════════════════
    # Training Losses
    # ═══════════════════════════════════════════════════════════════

    def training_losses(self, **args) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Pipeline:
            1. Extract GT mean_T, logvar_T from data batch
            2. Prepare condition features (SDF + horizon)
            3. Run encoder(s) to predict (mean_T_pred, logvar_T_pred)
            4. Compute KL divergence loss
            5. Optionally compute render loss

        Returns:
            Tuple of (loss_dict, info_dict)
        """
        data: TransitionBatch = args

        if self.process_data_locally:
            data = self._preprocess_trajectory_batch(data)

        terms = edict()

        # 1. Extract GT distribution parameters for frame T
        z_T_mean_gt = data["z_T_mean"]  # (B, L, C)
        z_T_logvar_gt = data["z_T_logvar"]  # (B, L, C)

        if z_T_mean_gt is None or z_T_logvar_gt is None:
            raise ValueError(
                "z_T_mean and z_T_logvar must be present in data. "
                "Set label_only_T=True in DataWorkerConfig."
            )

        # 2-3. Prepare condition features & forward pass through encoder
        structure_xt: SparseTensor = data["structure"][-2]
        
        mod = None
        if "time_embedder" in self.training_models:
            flow_t = self._make_flow_t(data)
            if flow_t is None:
                # Fallback to zero if physical horizon not available
                flow_t = torch.zeros(structure_xt.shape[0], device=self.device)
            mod_time = self.training_models["time_embedder"](flow_t)
            mod = mod_time

            # Generate task embedding
            task_ind_b = data["task_ind"].long().flatten()
            assert task_ind_b.min() >= 0
            mod_task = self.training_models["task_embedder"](task_ind_b)
            # Concatenate time and task embeddings to form (B, C) mod feature
            mod = torch.cat([mod_time, mod_task], dim=-1)
            
        cond = None
        if "sdf_encoder" in self.training_models and data.get("sdf_t_T") is not None:
            sdf_t_T: SparseTensor = data["sdf_t_T"]
            # Process the SDF input features to the target format (P, C)
            P = sdf_t_T.feats.shape[0]
            sdf_feats = sdf_t_T.feats.reshape(P, self.max_horizon, 5)
            encoded_sdf = self.training_models["sdf_encoder"](sdf_feats)  # (P, sdf_out_channels)
            cond_input = sdf_t_T.replace(encoded_sdf.reshape(P, -1))
            cond = self.training_models['sdf_downsample'](cond_input)

        z_pred, mean_pred, logvar_pred, encoder_intermediate = self.training_models[
            "encoder"
        ](structure_xt, sample_posterior=False, mod=mod, cond=cond)

        # 4. Feature MSE loss on h_normed (primary loss when enabled)
        if self.lambda_feat_mse > 0 and data.get("z_T_h_normed") is not None:
            h_normed_pred = encoder_intermediate.get("h_normed")  # (B, C, D, H, W)
            h_normed_gt = data["z_T_h_normed"]  # (B, C, D, H, W)
            feat_mse = F.mse_loss(h_normed_pred.float(), h_normed_gt.float())
            terms["feat_mse"] = feat_mse
            terms["loss"] = self.lambda_feat_mse * feat_mse
        else:
            terms["loss"] = torch.tensor(0.0, device=mean_pred.device)

        # Reshape to match GT format: (B, C, D, H, W) -> (B, L, C)
        # mean_pred/logvar_pred are (B, C, D, H, W) from encoder
        B, C, D, H, W = mean_pred.shape
        mean_pred_flat = mean_pred.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        logvar_pred_flat = logvar_pred.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        # KL divergence loss (primary loss when feat_mse is not enabled)
        kl_loss = kl_divergence_gaussians(
            mean_pred_flat, logvar_pred_flat, z_T_mean_gt, z_T_logvar_gt
        )
        terms["kl_dynamics"] = kl_loss
        terms["loss"] = terms["loss"] + self.lambda_kl_dynamics * kl_loss

        # Also log L1 on means for monitoring
        terms["l1_mean"] = F.l1_loss(mean_pred_flat, z_T_mean_gt)

        # 5. Optional render loss
        if self.lambda_render > 0 and self.decoder is not None:
            # Sample z from predicted distribution
            #std_pred = torch.exp(0.5 * logvar_pred)
            z_sample = mean_pred #+ std_pred * torch.randn_like(std_pred)

            structure_T: SparseTensor = data["structure"][-1]
            gt_occ = structure_T.occupancy(self.resolution)

            # Passing valid_mask_T directly as float
            valid_mask_T = data["valid_masks_T"].float()

            decode_out = self.decoder(
                z_sample,
                # gt_structure=gt_occ,
                to_rep_kwargs={"aabb": [0, 0, 0, 1.0, 1.0, 1.0]},
                return_labels=True,
                # valid_mask=valid_mask_T,
            )

            # cpt_recon_z = decode_out.get("reconstruction")
            # if cpt_recon_z is not None:
            #     z_hashes = hash_coords(structure_T.coords, res=self.resolution)
            #     cpt_hashes = hash_coords(cpt_recon_z.coords, res=self.resolution)

            #     z_order = torch.argsort(z_hashes)
            #     cpt_order = torch.argsort(cpt_hashes)

            #     perm = torch.zeros_like(z_order)
            #     perm[z_order] = cpt_order

            #     cpt_recon_z = cpt_recon_z.replace(
            #         cpt_recon_z.feats[perm], cpt_recon_z.coords[perm]
            #     )
            #     decode_out["reconstruction"] = cpt_recon_z

            #     if self.debug:
            #         assert torch.all(cpt_recon_z.coords == structure_T.coords)

            # logit_occupancies = decode_out.get("occupancy_logits", [])
            # label_occupancies = decode_out.get("occupancy_labels", [])
            # valid_masks_list = decode_out.get("occupancy_valid_masks", [])

            # for scale_i, (logit_occupancy, label_occupancy) in enumerate(
            #     zip(logit_occupancies, label_occupancies)
            # ):
            #     if label_occupancy is None:
            #         continue
            #     logit_occupancy = logit_occupancy.float()
            #     label_occupancy = label_occupancy.float()

            #     mask = None
            #     if len(valid_masks_list) > scale_i:
            #         mask = valid_masks_list[scale_i].float()

            #     loss_weight = mask if mask is not None else None

            #     # BCE
            #     bce_per_element = F.binary_cross_entropy_with_logits(
            #         logit_occupancy, label_occupancy, reduction="none"
            #     )
            #     if loss_weight is not None:
            #         bce = (bce_per_element * loss_weight).sum() / (loss_weight.sum() + 1e-6)
            #     else:
            #         bce = bce_per_element.mean()

            #     terms[f"cpt_occ_bce_{scale_i}"] = bce

            #     # Metrics
            #     if mask is not None:
            #         valid_bool = mask > 0.5
            #         acc = (((logit_occupancy[valid_bool] > 0) == label_occupancy[valid_bool].bool()).float().mean())
            #         iou = compute_iou(logit_occupancy[valid_bool], label_occupancy[valid_bool])
            #     else:
            #         acc = ((logit_occupancy > 0) == label_occupancy.bool()).float().mean()
            #         iou = compute_iou(logit_occupancy, label_occupancy)

            #     terms[f"train_cpt_occ_acc_{scale_i}"] = acc
            #     terms[f"train_cpt_occ_iou_{scale_i}"] = iou

            #     terms["loss"] = terms["loss"] + bce

            #     # Focal Loss
            #     lambda_focal_occ = getattr(self, "lambda_focal_occ", 0.0)
            #     if lambda_focal_occ > 0:
            #         focal_per_element = focal_loss(
            #             logit_occupancy, label_occupancy, reduction="none"
            #         )
            #         if loss_weight is not None:
            #             focal = (focal_per_element * loss_weight).sum() / (
            #                 loss_weight.sum() + 1e-6
            #             )
            #         else:
            #             focal = focal_per_element.mean()

            #         terms[f"cpt_occ_focal_{scale_i}"] = focal
            #         terms["loss"] = terms["loss"] + lambda_focal_occ * focal

            #     # Dice Loss
            #     probs = torch.sigmoid(logit_occupancy)
            #     if loss_weight is not None:
            #         probs_valid = probs * loss_weight
            #         label_valid = label_occupancy * loss_weight
            #         intersection = (probs_valid * label_valid).sum()
            #         union = probs_valid.sum() + label_valid.sum()
            #     else:
            #         intersection = (probs * label_occupancy).sum()
            #         union = probs.sum() + label_occupancy.sum()

            #     dice = 1 - (2 * intersection + 1) / (union + 1)
            #     terms[f"cpt_occ_dice_{scale_i}"] = dice
            #     terms["loss"] = terms["loss"] + dice

            gaussians = decode_out["reps"]
            B_orig, CAM = data["w2cs"].shape[:2]
            w2c = data["w2cs"].reshape(B_orig * CAM, 4, 4)
            intrinsics = data["ints"].reshape(B_orig * CAM, 3, 3)
            H_img, W_img = data["rgbs_T"].shape[-2:]
            gt_img = (
                data["rgbs_T"].reshape(B_orig * CAM, 3, H_img, W_img).float()
                / 255.0
            )
            t_gt_img = (
                data["rgbs_t"].reshape(B_orig * CAM, 3, H_img, W_img).float()
                / 255.0
            )

            rendered = self._render_batch(
                gaussians,
                w2c,
                intrinsics,
                bg_color=(0, 0, 0),
                img_wh=(W_img, H_img),
            )
            pred_img = rendered["color"]

            sample_removal_mask = data["sample_removal_mask"]

            if sample_removal_mask.sum() > 0:
                robot_masks = data["robot_masks_T"]
                fg_masks = data["masks_fg_T"]
                if robot_masks is not None and fg_masks is not None:
                    robot_m = (
                        robot_masks.reshape(
                            B_orig * CAM, 1, *robot_masks.shape[-2:]
                        ).float()
                        > 0
                    ).float()
                    fg_m = (
                        fg_masks.reshape(
                            B_orig * CAM, 1, *fg_masks.shape[-2:]
                        ).float()
                        > 0
                    ).float()

                    if sample_removal_mask is not None:
                        removal_mask_cam = (
                            sample_removal_mask.view(B_orig, 1, 1, 1, 1)
                            .expand(-1, CAM, -1, -1, -1)
                            .reshape(B_orig * CAM, 1, 1, 1)
                            .float()
                        )
                        robot_invalid_mask = robot_m * removal_mask_cam
                    else:
                        robot_invalid_mask = robot_m

                    valid_masks = fg_m * (1 - robot_invalid_mask)
                    pred_img = pred_img * valid_masks + gt_img * (1 - valid_masks)

            T_object_masks = data["masks_fg_T"] * (1 - data["robot_masks_T"]) * (1 - data["static_masks_T"])
            t_object_masks = data["masks_fg_t"] * (1 - data["robot_masks_t"]) * (1 - data["static_masks_t"])
            t_T_object_masks = (T_object_masks.bool() | t_object_masks.bool()).float()
            t_T_object_masks = t_T_object_masks.flatten(0, 1)[:, None]
            render_terms = self._compute_render_loss(pred_img, gt_img, data, terms, t_gt_image=t_gt_img, 
                                                     object_masks=t_T_object_masks)
            terms["loss"] = (
                terms["loss"] + self.lambda_render * render_terms["render_loss"]
            )
            terms.update(render_terms)

        if torch.isnan(terms["loss"]):
            print("[red]NAN LOSS![/red]")
            raise Exception("NAN LOSS!")

        return terms, {}

    def _compute_render_loss(
        self,
        rec_image: torch.Tensor,
        gt_image: torch.Tensor,
        data: TransitionBatch,
        terms: dict,
        t_gt_image: torch.Tensor|None=None,
        object_masks: torch.Tensor|None=None
    ) -> torch.Tensor:
        """
        Compute render loss with l1/l2 + SSIM + LPIPS + edge + focus components.

        Mirrors the v5x trainer's render loss pattern (without weight masks).

        Args:
            rec_image: Predicted rendered image (B, 3, H, W).
            gt_image: Ground truth image (B, 3, H, W).
            data: TransitionBatch for optional masks.
            terms: Loss terms dict to update in-place.

        Returns:
            Scalar total render loss.
        """
        weight_mask = None # (BTC, 1, H, W)
        if t_gt_image is not None:
            diff = torch.abs(gt_image - t_gt_image).mean(dim=1, keepdim=True)
            weight_mask = 1.0 + diff * (self.lambda_diff - 1.0)
            if object_masks is not None:
                weight_mask = weight_mask * object_masks + (1.0 - object_masks)

        # Primary reconstruction loss
        if self.loss_type == "l1":
            primary_loss = l1_loss(rec_image, gt_image, weight_mask=weight_mask)
            terms["l1"] = primary_loss
        elif self.loss_type == "l2":
            primary_loss = mse_loss(rec_image, gt_image, weight_mask=weight_mask)
            terms["l2"] = primary_loss
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        total = primary_loss

        # SSIM loss
        if self.lambda_ssim > 0:
            ssim_loss = ssim(rec_image, gt_image, return_loss=True, weight_mask=weight_mask)
            terms["ssim"] = ssim_loss
            total = total + self.lambda_ssim * ssim_loss

        # LPIPS loss
        if self.lambda_lpips > 0:
            lpips_loss = lpips(rec_image, gt_image)
            terms["lpips"] = lpips_loss
            total = total + self.lambda_lpips * lpips_loss

        # Edge (gradient) loss
        if self.lambda_edge > 0:
            edge_loss = gradient_l1_loss(rec_image, gt_image)
            terms["render_edge"] = edge_loss
            total = total + self.lambda_edge * edge_loss

        # Focus loss on object regions
        if self.lambda_focus > 0:
            # Build object masks from available data
            B = gt_image.shape[0]
            robot_masks = data.get("robot_masks_T")
            static_masks = data.get("static_masks_T")
            fg_masks = data.get("masks_fg_T")

            if (
                robot_masks is not None
                and static_masks is not None
                and fg_masks is not None
            ):
                # Use all cameras, flatten to (B*CAM, 1, H, W)
                B_orig, CAM = robot_masks.shape[:2]
                robot_m = (
                    robot_masks.reshape(
                        B_orig * CAM, 1, *robot_masks.shape[-2:]
                    ).float()
                    > 0
                ).float()
                static_m = (
                    static_masks.reshape(
                        B_orig * CAM, 1, *static_masks.shape[-2:]
                    ).float()
                    > 0
                ).float()
                fg_m = (
                    fg_masks.reshape(B_orig * CAM, 1, *fg_masks.shape[-2:]).float() > 0
                ).float()
                objects_masks = fg_m * (1 - robot_m) * (1 - static_m)

                focus_terms = compute_zoom_in_loss(
                    rec_image,
                    gt_image,
                    objects_masks,
                    lambda_ssim=self.lambda_ssim,
                    lambda_lpips=self.lambda_lpips,
                    debug=self.debug,
                    weight_mask=weight_mask
                )
                if focus_terms:
                    terms.update(focus_terms)
                    total = total + self.lambda_focus * focus_terms["focus_loss"]

        terms["render_loss"] = total
        return terms



    # ═══════════════════════════════════════════════════════════════
    # Snapshot Visualization
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _run_snapshot(
        self,
        dataset,
        num_samples: int,
        batch_size: int,
        prefix: str,
        verbose: bool = False,
    ) -> Dict:
        """Generate samples and visualize predictions for a given dataset."""
        if dataset is None:
            return {}

        if self.num_samples > 0:
            batch_size = min(batch_size, self.num_samples)

        dataloader = DataLoader(
            copy.deepcopy(dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=(dataset.collate_fn if hasattr(dataset, "collate_fn") else None),
        )

        gt_images = []
        t_gt_images = []
        rec_images = []

        for i in range(0, num_samples, batch_size):
            try:
                data: TransitionBatch = next(iter(dataloader))
            except StopIteration:
                break

            if self.process_data_locally or (
                "val" in prefix and self.process_data_locally_eval
            ):
                data = self._preprocess_trajectory_batch(data, is_eval=True)

            data = move_to_device(data, self.device)
            # Prepare condition and run encoder
            structure_xt: SparseTensor = data["structure"][-2]
            
            mod = None
            if "time_embedder" in self.models:
                flow_t = self._make_flow_t(data)
                if flow_t is None:
                    flow_t = torch.zeros(structure_xt.shape[0], device=self.device)
                mod_time = self.models["time_embedder"](flow_t)
                mod = mod_time

                task_ind_b = data["task_ind"].long().flatten()
                assert task_ind_b.min() >= 0
                mod_task = self.models["task_embedder"](task_ind_b)
                mod = torch.cat([mod_time, mod_task], dim=-1)
            cond = None
            if "sdf_encoder" in self.models and data.get("sdf_t_T") is not None:
                sdf_t_T: SparseTensor = data["sdf_t_T"]
                P = sdf_t_T.feats.shape[0]
                sdf_feats = sdf_t_T.feats.reshape(P, self.max_horizon, 5)
                encoded_sdf = self.models["sdf_encoder"](sdf_feats)  # (P, sdf_out_channels)
                cond_input = sdf_t_T.replace(encoded_sdf.reshape(P, -1))
                cond = self.models['sdf_downsample'](cond_input)

            z_pred, mean_pred, logvar_pred, _ = self.models["encoder"](
                structure_xt, sample_posterior=True, mod=mod, cond=cond
            )

            B, C, D, H, W = z_pred.shape

            # Annotation info
            action_texts = []
            if "horizon" in data and data["horizon"] is not None:
                for b_idx in range(B):
                    num_valid = int(data["horizon"][b_idx].item())
                    action_texts.append(f"Acts: {num_valid}")
            else:
                action_texts = [""] * B

            cam_idx = 0
            gt_img_original = data["rgbs_T"][:, cam_idx].float() / 255.0
            gt_img_vis = gt_img_original

            annotated_gt = annotate_images(gt_img_vis, action_texts, font_size=25)
            gt_images.append(annotated_gt)

            t_gt_img = data["rgbs_t"][:, cam_idx].float() / 255.0
            t_gt_images.append(t_gt_img)

            if self.decoder is not None:
                rec_images.append(
                    self._render_latent(z_pred, data, gt_img_original, cam_idx)
                )

        ret_dict = {}

        if gt_images:
            ret_dict[f"{prefix}T_gt_image"] = {
                "value": torch.cat(gt_images, dim=0),
                "type": "image",
            }

        if t_gt_images:
            ret_dict[f"{prefix}t_gt_image"] = {
                "value": torch.cat(t_gt_images, dim=0),
                "type": "image",
            }

        if rec_images:
            ret_dict[f"{prefix}T_rec_image"] = {
                "value": torch.cat(rec_images, dim=0),
                "type": "image",
            }

        return ret_dict

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        **kwargs,
    ) -> Dict:
        """Generate samples and visualize for both validation and training data."""
        gc.collect()
        torch.cuda.empty_cache()

        for mod in self.models.values():
            mod.eval()

        ret_dict = {}

        val_dict = self._run_snapshot(
            dataset=self.snapshot_dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            prefix="val_",
            verbose=verbose,
        )
        ret_dict.update(val_dict)

        train_dict = self._run_snapshot(
            dataset=self.dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            prefix="train_",
            verbose=verbose,
        )
        ret_dict.update(train_dict)

        for mod in self.models.values():
            mod.train()

        gc.collect()
        torch.cuda.empty_cache()

        return ret_dict
