#!/usr/bin/env python3
"""DynamicsPredictor for RLA-WM evaluation on Maniskill.

Robot selection via os.environ["EVAL_ROBOT"] (default: "panda").
Supports horizons 5, 15 (single-step) and 30 (3-chunk autoregressive rollout).
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF
from PIL import Image

ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # eval/predictors/<this> -> repo root
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rich import print
from utils.misc import load_config
from src.utils.loss_utils import ssim as compute_ssim, lpips as compute_lpips
import src.models as models

ROBOT_CONFIGS = {
    "panda": {
        "config": "configs/rla_wm/panda.yaml",
        "workdir": "runs/weights/rla-wm/maniskill/panda/20260405_11-00-59",
    },
    "xarm": {
        "config": "configs/rla_wm/xarm.yaml",
        "workdir": "runs/weights/rla-wm/maniskill/xarm/20260404_15-49-08",
    },
    "ur10e": {
        "config": "configs/rla_wm/ur10e.yaml",
        "workdir": "runs/weights/rla-wm/maniskill/ur10e/20260404_15-49-19",
    },
}

EULER_STEPS = 30
IMAGE_DECODER_CKPT = "runs/weights/dino-to-image_unet/maniskill/20260404_22-53-36"


def _build_trainer(robot_name: str, device: torch.device):
    """Build a DinoLatentActionFlowTrainer in inference_only mode."""
    from src.trainers.dino_latent_action_flow_trainer import DinoLatentActionFlowTrainer

    rcfg = ROBOT_CONFIGS[robot_name]
    cfg = load_config(rcfg["config"])

    # Build models from config (same as train.py)
    model_dict = {
        name: getattr(models, model["name"])(**model["args"]).to(device)
        for name, model in cfg["models"].items()
    }

    # Instantiate trainer in inference_only mode — skips optimizer, EMA, dataloaders
    trainer_args = dict(cfg["trainer"]["args"])
    trainer_args.pop('load_dir', None)
    trainer_args["image_decoder_ckpt"] = IMAGE_DECODER_CKPT
    trainer = DinoLatentActionFlowTrainer(
        model_dict,
        None,  # no dataset needed
        output_dir=rcfg["workdir"],
        load_dir=rcfg["workdir"],
        step=None,
        # max_steps=0,
        inference_only=True,
        **trainer_args,
    )

    for m in trainer.models.values():
        m.eval()

    return trainer


# ---------------------------------------------------------------------------
# DynamicsPredictor
# ---------------------------------------------------------------------------


class DynamicsPredictor:
    """Latent Action Flow dynamics predictor for eval_wrapper.py."""

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,  # noqa: ARG002
        device: str | None = None,
        gpu_id: int | None = None,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            if gpu_id is None:
                gpu_id = rank % max(torch.cuda.device_count(), 1)
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        robot_name = os.environ.get("EVAL_ROBOT", "panda")
        if robot_name not in ROBOT_CONFIGS:
            raise ValueError(
                f"Unknown EVAL_ROBOT={robot_name!r}, expected one of {list(ROBOT_CONFIGS)}"
            )

        self.trainer = _build_trainer(robot_name, self.device)
        print(f"[DynamicsPredictor] Robot={robot_name}, device={self.device}")

    # ------------------------------------------------------------------
    # Helpers (delegate to trainer where possible)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_dino_tokens(
        self, images: Tensor
    ) -> Tuple[Tensor, Tuple[int, int]]:
        """Extract DINO patch tokens from (N, 3, H, W) float [0,1] images."""
        # Trainer expects (B, Cam, 3, H, W); add camera dim
        imgs_4d = images.unsqueeze(1).to(self.device)
        tokens, patch_hw = self.trainer._extract_dino_tokens(imgs_4d)
        return tokens.squeeze(1), patch_hw  # (N, Lp, C)

    @torch.no_grad()
    def _decode_to_image_tensor(
        self, dino_tokens: Tensor, patch_hw: Tuple[int, int]
    ) -> Tensor:
        """Decode DINO tokens (1, Lp, C) to RGB image (1, 3, H, W) float [0, 1]."""
        tokens_4d = dino_tokens.unsqueeze(1)  # add camera dim
        decoded = self.trainer.models["image_decoder"](
            tokens_4d.contiguous(), patch_hw=patch_hw
        )
        return decoded[:, 0].clamp(0, 1)

    @staticmethod
    def _tensor_to_pil(t: Tensor) -> Image.Image:
        if t.dim() == 4:
            t = t[0]
        return TF.to_pil_image(t.clamp(0, 1).cpu())

    @torch.no_grad()
    def _run_single_step(
        self,
        x_t_tokens: Tensor,
        target_qpos: Tensor,
        horizon: int,
        robot_id: int,
        task_ind: int,
    ) -> Tuple[Tensor, Tensor]:
        """Run one flow sampling step.

        Returns:
            pred_latent: (1, num_tokens, token_dim) denormalized
            pred_x_T_tokens: (1, Lp, C) predicted DINO tokens
        """
        robot_ids = torch.tensor([robot_id], device=self.device)
        task_inds = torch.tensor([task_ind], device=self.device)
        horizons = torch.tensor([horizon], device=self.device, dtype=torch.float)
        full_qpos_list = self.trainer._target_qpos_to_full_qpos(
            [target_qpos.to(self.device)], robot_ids, task_inds
        )

        flow_model = self.trainer.models["flow_model"]
        num_latent_tokens = self.trainer.models["encoder"].num_tokens
        token_dim = flow_model.token_dim

        noise = torch.randn(1, num_latent_tokens, token_dim, device=self.device)

        sampled_flow = self.trainer._euler_sample(
            flow_model=flow_model,
            noise=noise,
            xt_tokens=x_t_tokens,
            task_inds=task_inds,
            horizons=horizons,
            robot_ids=robot_ids,
            full_qpos_list=full_qpos_list,
            steps=EULER_STEPS,
        )

        pred_latent = self.trainer._denormalize_latent_tokens(
            sampled_flow, flow_model=flow_model
        )
        _, pred_x_T_tokens = self.trainer.models["decoder"](
            x_t_tokens, tokens=pred_latent
        )
        return pred_latent, pred_x_T_tokens

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return torch.from_numpy(x) if hasattr(x, '__array__') else torch.tensor(x)

    @torch.no_grad()
    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        rgbs = self._to_tensor(sample["rgbs"])
        masks = self._to_tensor(sample["foreground_masks"])
        target_qpos = self._to_tensor(sample["target_qpos"])
        robot_id = int(self._to_tensor(sample["robot_id"]).reshape(-1)[0].item())
        task_ind = int(self._to_tensor(sample["task_ind"]).reshape(-1)[0].item())
        horizon_val = int(self._to_tensor(sample["horizon"]).reshape(-1)[0].item())

        # Mask RGB and normalize to [0, 1]
        mask_float = (masks > 0).float().unsqueeze(2)
        masked_rgb = rgbs.float() * mask_float / 255.0

        # Extract DINO tokens for frame 0 and frame 1 (single camera)
        imgs = masked_rgb[:, 0]  # (2, 3, H, W)
        all_tokens, patch_hw = self._extract_dino_tokens(imgs)
        all_tokens = all_tokens.float()
        x_t_tokens = all_tokens[0:1]
        x_T_tokens = all_tokens[1:2]

        # GT images with background removed (masked RGB)
        gt_t_pil = self._tensor_to_pil(masked_rgb[0, 0])
        gt_T_pil = self._tensor_to_pil(masked_rgb[1, 0])

        # GT image as (1, 3, H, W) float tensor for metric computation (background removed)
        gt_T_img = (masked_rgb[1, 0:1]).to(self.device)

        if horizon_val <= 15:
            return self._apply_single_step(
                x_t_tokens, x_T_tokens, target_qpos, patch_hw,
                horizon_val, robot_id, task_ind, gt_t_pil, gt_T_pil, gt_T_img,
            )
        else:
            return self._apply_multi_step(
                x_t_tokens, x_T_tokens, target_qpos, patch_hw,
                horizon_val, robot_id, task_ind, gt_t_pil, gt_T_pil, gt_T_img,
            )

    def _apply_single_step(
        self,
        x_t_tokens: Tensor,
        x_T_tokens: Tensor,
        target_qpos: Tensor,
        patch_hw: Tuple[int, int],
        horizon_val: int,
        robot_id: int,
        task_ind: int,
        gt_t_pil: Image.Image,
        gt_T_pil: Image.Image,
        gt_T_img: Tensor,
    ) -> Dict[str, Any]:
        pred_latent, pred_x_T = self._run_single_step(
            x_t_tokens, target_qpos, horizon_val, robot_id, task_ind,
        )

        # Latent L1 (encoder latent space)
        gt_latent, _ = self.trainer.models["encoder"](
            (x_T_tokens - x_t_tokens).float()
        )
        latent_l1 = F.l1_loss(pred_latent.float(), gt_latent.float()).item()

        # DINO token L1 (pred vs GT xT tokens)
        dino_l1 = F.l1_loss(pred_x_T.float(), x_T_tokens.float()).item()

        # Decode predicted tokens to image for LPIPS/SSIM against original GT
        pred_img = self._decode_to_image_tensor(pred_x_T, patch_hw)
        lpips_val = compute_lpips(pred_img, gt_T_img).item()
        ssim_val = compute_ssim(pred_img, gt_T_img).item()

        pred_T_pil = self._tensor_to_pil(pred_img)

        return {
            "metrics": {
                "latent_l1": latent_l1,
                "dino_l1": dino_l1,
                "lpips": lpips_val,
                "ssim": ssim_val,
            },
            "images": {
                "gt_t": gt_t_pil,
                "gt_T": gt_T_pil,
                f"pred_T_dinol1_{dino_l1:.3f}": pred_T_pil,
            },
        }

    def _apply_multi_step(
        self,
        x_t_tokens: Tensor,
        x_T_tokens: Tensor,
        target_qpos: Tensor,
        patch_hw: Tuple[int, int],
        horizon_val: int,  # noqa: ARG002
        robot_id: int,
        task_ind: int,
        gt_t_pil: Image.Image,
        gt_T_pil: Image.Image,
        gt_T_img: Tensor,
    ) -> Dict[str, Any]:
        # 3-chunk autoregressive rollout: 3 × 10 target_qpos steps = 30 total
        # All chunks use drop_last=True (trainer convention)
        chunks = [
            (0, 10, 10),
            (10, 20, 10),
            (20, 30, 10),
        ]

        current_x_t = x_t_tokens
        intermediate_images: Dict[str, Image.Image] = {}

        for chunk_idx, (start, end, chunk_horizon) in enumerate(chunks):
            chunk_qpos = target_qpos[start:end]

            _, pred_x_T = self._run_single_step(
                current_x_t, chunk_qpos, chunk_horizon, robot_id, task_ind,
            )

            pred_img = self._decode_to_image_tensor(pred_x_T, patch_hw)
            intermediate_images[f"pred_step{chunk_idx + 1}"] = self._tensor_to_pil(
                pred_img
            )

            current_x_t = pred_x_T.float()

        # Final metrics — DINO token L1 (pred vs GT xT tokens)
        final_pred_tokens = current_x_t
        dino_l1 = F.l1_loss(final_pred_tokens.float(), x_T_tokens.float()).item()

        final_pred_img = self._decode_to_image_tensor(final_pred_tokens, patch_hw)
        lpips_val = compute_lpips(final_pred_img, gt_T_img).item()
        ssim_val = compute_ssim(final_pred_img, gt_T_img).item()

        pred_T_pil = self._tensor_to_pil(final_pred_img)

        images = {
            "gt_t": gt_t_pil,
            "gt_T": gt_T_pil,
            f"pred_T_dinol1_{dino_l1:.3f}": pred_T_pil,
        }
        images.update(intermediate_images)

        return {
            "metrics": {
                "dino_l1": dino_l1,
                "lpips": lpips_val,
                "ssim": ssim_val,
            },
            "images": images,
        }
