import copy
import gc
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from src.trainers.basic import BasicTrainer
from src.utils.loss_utils import l1_loss, ssim, lpips
from utils.dino import (
    DINOv3FeatureExtractor,
    get_dinov3_model_for_channels,
    visualize_dino_to_imgs,
)
from utils.misc import TimerContext, move_to_device


class DinoToImageTrainerV1(BasicTrainer):
    def __init__(
        self,
        *args,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        dino_channels: int = 1024,
        rgb_key: str = "rgbs",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.rgb_key = rgb_key
        self.dino_channels = dino_channels

        # Initialize frozen DINO feature extractor
        dino_model_name = get_dinov3_model_for_channels(dino_channels)
        self.dino_extractor = DINOv3FeatureExtractor(
            model_name=dino_model_name,
        ).to(self.device)
        self.dino_extractor.eval()
        for p in self.dino_extractor.parameters():
            p.requires_grad = False

        self.dino_patch_size = self.dino_extractor.patch_size

    @torch.no_grad()
    def _extract_dino_tokens(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract DINO tokens from images.
        images: (B, Cam, 3, H, W)
        Returns: (B, Cam, pH*pW, C), (pH, pW)
        """
        bsz, cams, _, _, _ = images.shape
        imgs = images.float()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
        
        flat = imgs.reshape(bsz * cams, *imgs.shape[2:])
        _, patch_grid = self.dino_extractor(flat, return_spatial_grid=True)
        
        _, ch, ph, pw = patch_grid.shape
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)  # (B*Cam, pH*pW, C)
        tokens = patch_tokens.reshape(bsz, cams, ph * pw, ch)
        
        return tokens, (ph, pw)

    def get_rgb_t(self, data: dict):
        rgb = data[self.rgb_key][:, 0]
        if "foreground_masks" in data:
            mask = (data["foreground_masks"][:, 0] > 0).float()
            rgb = rgb * mask.unsqueeze(2)
        return rgb

    def training_losses(self, **args) -> Tuple[Dict, Dict]:
        data = move_to_device(args, self.device)
        terms = edict()

        # Get target image (Frame t)
        rgb_t = self.get_rgb_t(data)
        
        # 1. Extract DINO parameters
        with TimerContext("Timer - DINO Extraction", self.debug):
            x_t, patch_hw = self._extract_dino_tokens(rgb_t)
        
        # 2. Decode features -> RGB
        with TimerContext("Timer - Decoder Forward", self.debug):
            # Pass patch_hw to dynamically calculate spatial reshaping
            pred_rgb_t = self.training_models["decoder"](
                x_t, patch_hw=patch_hw
            )
        
        # Target needs to be normalized to [0, 1] if max > 1.5
        gt_image = rgb_t.float()
        if gt_image.max() > 1.5:
            gt_image = gt_image / 255.0

        # Reshape to combine batch and cam dimension for standard image loss functions
        B, Cams, C, H, W = gt_image.shape
        pred_rgb_t_flat = pred_rgb_t.reshape(B * Cams, C, H, W)
        gt_image_flat = gt_image.reshape(B * Cams, C, H, W)

        # 3. Calculate Losses
        terms["loss"] = 0.0

        if self.lambda_l1 > 0:
            terms["l1"] = l1_loss(pred_rgb_t_flat, gt_image_flat)
            terms["loss"] = terms["loss"] + self.lambda_l1 * terms["l1"]

        if self.lambda_ssim > 0:
            terms["ssim"] = ssim(pred_rgb_t_flat, gt_image_flat, return_loss=True)
            terms["loss"] = terms["loss"] + self.lambda_ssim * terms["ssim"]

        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(pred_rgb_t_flat, gt_image_flat)
            terms["loss"] = terms["loss"] + self.lambda_lpips * terms["lpips"]

        return terms, {}

    def _tokens_to_vis(
        self,
        tokens: torch.Tensor,
        patch_hw: Tuple[int, int],
    ) -> torch.Tensor:
        # tokens: (B, Cam, Lp1, C)
        bsz, cams, _, ch = tokens.shape
        ph, pw = patch_hw
        vis_by_cam = []
        for ci in range(cams):
            patch_tokens = (
                tokens[:, ci, :, :].reshape(bsz, ph, pw, ch).permute(0, 3, 1, 2)
            )
            vis = visualize_dino_to_imgs(patch_tokens, patch_size=self.dino_patch_size)
            vis_by_cam.append(vis)
        return torch.cat(vis_by_cam, dim=-1)

    @torch.no_grad()
    def _run_snapshot(
        self,
        dataset,
        num_samples: int,
        batch_size: int,
        prefix: str,
    ) -> Dict:
        if dataset is None:
            return {}
        dataloader = DataLoader(
            copy.deepcopy(dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=(dataset.collate_fn if hasattr(dataset, "collate_fn") else None),
        )

        gt_imgs = []
        # dino_vis_imgs = []
        pred_imgs = []
        
        for _ in range(0, num_samples, batch_size):
            try:
                data = next(iter(dataloader))
            except StopIteration:
                break
            data = move_to_device(data, self.device)
            
            rgb_t = self.get_rgb_t(data)
            x_t, patch_hw = self._extract_dino_tokens(rgb_t)
            pred_rgb_t = self.models["decoder"](x_t, patch_hw=patch_hw)
            
            # Format GT (B, Cam, 3, H, W)
            imgs = rgb_t.float()
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
            
            bsz, cams, c, h, w = imgs.shape
            
            # Concat view horizontally to match dino_to_imgs [B, 3, H, W * Cam] format 
            # Note: _tokens_to_vis concatenates along W matching this behavior.
            gt_imgs.append(torch.cat([imgs[:, ci] for ci in range(cams)], dim=-1))
            # dino_vis_imgs.append(self._tokens_to_vis(x_t, patch_hw))
            
            # Format and concat predictions
            pred = pred_rgb_t.clamp(0, 1)
            pred_imgs.append(torch.cat([pred[:, ci] for ci in range(cams)], dim=-1))

        ret = {}
        if gt_imgs:
            ret[f"{prefix}gt_rgb"] = {
                "value": torch.cat(gt_imgs, dim=0),
                "type": "image",
            }
        # if dino_vis_imgs:
        #     ret[f"{prefix}dino_features"] = {
        #         "value": torch.cat(dino_vis_imgs, dim=0),
        #         "type": "image",
        #     }
        if pred_imgs:
            ret[f"{prefix}pred_rgb"] = {
                "value": torch.cat(pred_imgs, dim=0),
                "type": "image",
            }
        return ret

    @torch.no_grad()
    def run_snapshot(
        self, num_samples: int, batch_size: int, verbose: bool = False, **kwargs
    ) -> Dict:
        del verbose, kwargs
        gc.collect()
        torch.cuda.empty_cache()
        for mod in self.models.values():
            mod.eval()

        ret = {}
        ret.update(
            self._run_snapshot(self.snapshot_dataset, num_samples, batch_size, "val_")
        )
        ret.update(self._run_snapshot(self.dataset, num_samples, batch_size, "train_"))

        for mod in self.models.values():
            mod.train()
        gc.collect()
        torch.cuda.empty_cache()
        return ret
