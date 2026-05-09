import copy
import gc
from typing import Any, Dict, List, Optional, Tuple

from easydict import EasyDict as edict
from jaxtyping import Float32
from rich import print
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.trainers.basic import BasicTrainer
from src.utils.loss_utils import lpips
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels
from utils.misc import (
    fetch_state_dict,
    local_seed_scope,
    make_worker_seed_init_fn,
    move_to_device,
)


class RlaAutoencoderTrainer(BasicTrainer):
    def __init__(
        self,
        *args,
        lambda_l1: float = 1.0,
        lambda_mse: float = 1.0,
        lambda_vq: float = 1.0,
        dino_channels: int = 1024,
        rgb_key: str = "rgbs",
        image_decoder_ckpt: str = "",
        snapshot_eval_samples: int = 200,
        snapshot_eval_seed: int = 2026,
        revival_every: int = -1,
        inverse_input_mode: str = "sub",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.revival_every = revival_every
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_vq = lambda_vq
        self.rgb_key = rgb_key
        self.dino_channels = dino_channels
        self.snapshot_eval_samples = snapshot_eval_samples
        self.snapshot_eval_seed = snapshot_eval_seed
        self.inverse_input_mode = inverse_input_mode.lower()
        valid_modes = {"sub", "concat", "append"}
        if self.inverse_input_mode not in valid_modes:
            raise ValueError(
                f"Unsupported inverse_input_mode={inverse_input_mode!r}. "
                f"Expected one of {sorted(valid_modes)}"
            )
        self.decoder_only_mode = (
            "decoder" in self.models
            and "encoder" not in self.models
            and "vq" not in self.models
        )
        self.use_vq = "vq" in self.models

        required_models = ("decoder",)
        missing = [k for k in required_models if k not in self.models]
        if missing:
            raise ValueError(f"Missing required models: {missing}")

        if not self.decoder_only_mode and "encoder" not in self.models:
            raise ValueError(
                "Model 'encoder' is required unless decoder-only mode is used"
            )

        dino_model_name = get_dinov3_model_for_channels(dino_channels)
        self.dino_extractor = DINOv3FeatureExtractor(model_name=dino_model_name).to(self.device)
        self.dino_extractor.eval()
        for p in self.dino_extractor.parameters():
            p.requires_grad = False

        if image_decoder_ckpt:
            self.models["image_decoder"].load_state_dict(
                fetch_state_dict("decoder", image_decoder_ckpt, self.device),
                strict=True,
            )
            print("[green]Image decoder loaded successfully (strict)[/green]")

    @torch.no_grad()
    def _extract_dino_tokens(
        self,
        images: Float32[Tensor, "B Cam 3 H W"],
    ) -> Tuple[Float32[Tensor, "B Cam Lp C"], Tuple[int, int]]:
        """Extract per-frame DINO patch tokens for a multi-camera image batch."""
        bsz, cams, _, _, _ = images.shape
        imgs = images.float()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
        flat = imgs.reshape(bsz * cams, *imgs.shape[2:])
        _, patch_grid = self.dino_extractor(flat, return_spatial_grid=True)
        _, ch, ph, pw = patch_grid.shape
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)
        tokens = patch_tokens.reshape(bsz, cams, ph * pw, ch)
        return tokens, (ph, pw)

    @torch.no_grad()
    def _extract_dino_tokens_sequence(
        self,
        images: Float32[Tensor, "B K Cam 3 H W"],
    ) -> Tuple[Float32[Tensor, "B K Cam Lp C"], Tuple[int, int]]:
        """Extract DINO patch tokens for all frames in a sequence batch."""
        bsz, steps, cams = images.shape[:3]
        flat_images = images.reshape(bsz * steps, cams, *images.shape[3:])
        flat_tokens, patch_hw = self._extract_dino_tokens(flat_images)
        tokens = flat_tokens.reshape(
            bsz,
            steps,
            cams,
            flat_tokens.shape[-2],
            flat_tokens.shape[-1],
        )
        return tokens, patch_hw

    def _get_masked_rgb_sequence(
        self,
        data: Dict[str, Any],
    ) -> Float32[Tensor, "B K Cam 3 H W"]:
        """Apply foreground masks to RGB sequence frames."""
        rgb = data[self.rgb_key]
        if rgb.shape[1] < 2:
            raise ValueError(f"Expected at least 2 frames in '{self.rgb_key}', got {rgb.shape[1]}")
        mask = (data["foreground_masks"] > 0).float()
        return rgb * mask.unsqueeze(3)

    @staticmethod
    def _normalize_rgb_for_lpips(
        imgs: Float32[Tensor, "B Cam 3 H W"],
    ) -> Float32[Tensor, "B Cam 3 H W"]:
        """Normalize images to [0, 1] for LPIPS computation."""
        imgs = imgs.float()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
        return imgs.clamp(0, 1)

    @torch.no_grad()
    def _decode_tokens_to_vis(
        self,
        tokens: Float32[Tensor, "B Cam Lp C"],
        patch_hw: Tuple[int, int],
    ) -> Float32[Tensor, "B 3 H WStrip"]:
        """Decode DINO tokens and stitch camera views horizontally for visualization."""
        decoded = self.models["image_decoder"](tokens.contiguous(), patch_hw=patch_hw).clamp(0, 1)
        return torch.cat([decoded[:, ci] for ci in range(decoded.shape[1])], dim=-1)

    @staticmethod
    def _scalar_tensor(value: float, device: torch.device) -> torch.Tensor:
        return torch.tensor(float(value), device=device)

    def _compose_inverse_input(
        self,
        x_t_flat: Float32[Tensor, "B N C"],
        x_T_flat: Float32[Tensor, "B N C"],
    ) -> Float32[Tensor, "B N CIn"]:
        """Compose encoder input from current/target tokens using configured mode."""
        if self.inverse_input_mode == "sub":
            return x_T_flat - x_t_flat
        if self.inverse_input_mode == "concat":
            return torch.cat([x_T_flat, x_t_flat], dim=2)
        if self.inverse_input_mode == "append":
            return torch.cat([x_T_flat, x_t_flat], dim=1)
        raise RuntimeError(f"Unhandled inverse_input_mode={self.inverse_input_mode!r}")

    def inference_batch(
        self,
        models: Dict[str, nn.Module],
        data: Dict[str, Any],
        training: bool = False,
    ) -> Dict[str, Any]:
        """Shared inference path used by both training and snapshot."""
        rgbs_seq = self._get_masked_rgb_sequence(data)
        tokens_seq, patch_hw = self._extract_dino_tokens_sequence(rgbs_seq)

        x0 = tokens_seq[:, 0]  # [B, Cam, Lp, C]
        x1 = tokens_seq[:, 1]  # [B, Cam, Lp, C]

        bsz, cams, lp, dino_ch = x0.shape
        assert cams == 1, f"Expected single-camera input, got {cams} cameras"

        x0_flat = x0.reshape(bsz, cams * lp, dino_ch)
        x1_flat = x1.reshape(bsz, cams * lp, dino_ch)

        if self.decoder_only_mode:
            # Decoder-only mode: force decoder to map x0 -> x1 without latent tokens.
            _, pred_x1_flat = models["decoder"](x0_flat, tokens=None)
            pred_x1 = pred_x1_flat.reshape_as(x1)
            return {
                "x0": x0,
                "x1_gt": x1,
                "x1_pred": pred_x1,
                "patch_hw": patch_hw,
                "vq_loss": pred_x1.new_zeros(()),
                "vq_metrics": {},
                "vq_code_ids": None,
                "enc_tokens": None,
                "quant_tokens": None,
                "target_rgb": rgbs_seq[:, 1],
            }

        # Encoder consumes an inverse-dynamics token composition of (xT, xt).
        inv_encoder_input = self._compose_inverse_input(x_t_flat=x0_flat, x_T_flat=x1_flat)
        enc_tokens, _ = models["encoder"](inv_encoder_input)
        if enc_tokens.ndim != 3:
            raise ValueError(f"Encoder token output must be rank-3 [B, Ntok, C], got {enc_tokens.shape}")

        if self.use_vq:
            vq_model = models["vq"]
            run_revival = (
                training
                and self.revival_every > 0
                and self.step > 0
                and self.step % self.revival_every == 0
            )
            token_dim = enc_tokens.shape[-1]
            if getattr(vq_model, "expects_per_token_input", False):
                quant_tokens, vq_loss, code_ids, vq_metrics = vq_model(
                    enc_tokens,
                    run_revival=run_revival,
                )
            else:
                quantized_flat, vq_loss, code_ids, vq_metrics = vq_model(
                    enc_tokens.reshape(-1, token_dim),
                    run_revival=run_revival,
                )
                quant_tokens = quantized_flat.reshape(enc_tokens.shape)
        else:
            quant_tokens = enc_tokens
            vq_loss = enc_tokens.new_zeros(())
            code_ids = None
            vq_metrics = {}

        # from pudb.remote import set_trace; set_trace()
        # Decoder predicts x1 given x0 sequence tokens and quantized latent tokens.
        _, pred_x1_flat = models["decoder"](x0_flat, tokens=quant_tokens)
        pred_x1 = pred_x1_flat.reshape_as(x1)

        return {
            "x0": x0,
            "x1_gt": x1,
            "x1_pred": pred_x1,
            "patch_hw": patch_hw,
            "vq_loss": vq_loss,
            "vq_metrics": vq_metrics,
            "vq_code_ids": code_ids,
            "enc_tokens": enc_tokens,
            "quant_tokens": quant_tokens,
            "target_rgb": rgbs_seq[:, 1],
        }

    def training_losses(self, **args: Any) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute one-step inverse dynamics losses for simplified VQ pipeline."""
        data = move_to_device(args, self.device)
        out = self.inference_batch(self.training_models, data, training=True)

        terms = edict()
        status = edict()

        terms["l1"] = F.l1_loss(out["x1_pred"], out["x1_gt"])
        terms["mse"] = F.mse_loss(out["x1_pred"], out["x1_gt"].clone())
        terms["vq_loss"] = out["vq_loss"]
        terms["loss"] = (
            self.lambda_l1 * terms["l1"]
            + self.lambda_mse * terms["mse"]
            + self.lambda_vq * terms["vq_loss"]
        )

        for k, v in out["vq_metrics"].items():
            if isinstance(v, (float, int)):
                terms[f"vq_{k}"] = self._scalar_tensor(float(v), device=self.device)

        return terms, status

    def _make_snapshot_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        deterministic: bool,
        seed: int,
    ) -> DataLoader:
        """Create deterministic or random snapshot dataloader copy."""
        generator = None
        worker_init_fn = None
        if deterministic:
            generator = torch.Generator(device="cpu")
            rank = int(getattr(self, "rank", 0))
            base_seed = seed + rank
            generator.manual_seed(base_seed)
            worker_init_fn = make_worker_seed_init_fn(base_seed)
        return DataLoader(
            copy.deepcopy(dataset),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            collate_fn=(dataset.collate_fn if hasattr(dataset, "collate_fn") else None),
        )

    @torch.no_grad()
    def _run_snapshot(
        self,
        dataset: Any,
        num_samples: int,
        batch_size: int,
        prefix: str,
        compute_metrics: bool = False,
    ) -> Dict[str, Dict[str, Tensor | str]]:
        """Run snapshot pass for either images or scalar metrics."""
        if dataset is None:
            return {}

        dataloader = self._make_snapshot_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            deterministic=compute_metrics,
            seed=self.snapshot_eval_seed,
        )
        iterator = iter(dataloader)

        decoded_t_imgs: List[torch.Tensor] = []
        decoded_gt_last_imgs: List[torch.Tensor] = []
        decoded_pred_best_last_imgs: List[torch.Tensor] = []
        lpips_vals: List[torch.Tensor] = []
        dino_l1_vals: List[torch.Tensor] = []

        loop_iter = (
            trange(0, num_samples, batch_size)
            if compute_metrics
            else range(0, num_samples, batch_size)
        )
        rank = int(getattr(self, "rank", 0))
        local_seed = self.snapshot_eval_seed + rank

        with local_seed_scope(local_seed) if compute_metrics else torch.no_grad():
            for _ in loop_iter:
                data = next(iterator)
                data = move_to_device(data, self.device)
                out = self.inference_batch(self.models, data, training=False)

                if not compute_metrics:
                    decoded_t_imgs.append(self._decode_tokens_to_vis(out["x0"], out["patch_hw"]))
                    decoded_gt_last_imgs.append(
                        self._decode_tokens_to_vis(out["x1_gt"], out["patch_hw"])
                    )
                    decoded_pred_best_last_imgs.append(
                        self._decode_tokens_to_vis(out["x1_pred"], out["patch_hw"])
                    )
                    continue

                dino_l1_vals.append(F.l1_loss(out["x1_pred"], out["x1_gt"], reduction="mean"))
                pred_img = self.models["image_decoder"](
                    out["x1_pred"].contiguous(),
                    patch_hw=out["patch_hw"],
                ).clamp(0, 1)
                target_img = self._normalize_rgb_for_lpips(out["target_rgb"])
                pred_flat = pred_img.reshape(-1, *pred_img.shape[2:])
                target_flat = target_img.reshape(-1, *target_img.shape[2:])
                lpips_vals.append(lpips(pred_flat, target_flat))

        ret: Dict[str, Dict[str, Tensor | str]] = {}
        if not compute_metrics:
            if decoded_t_imgs:
                ret[f"{prefix}decoded_t"] = {
                    "value": torch.cat(decoded_t_imgs, dim=0),
                    "type": "image",
                }
            if decoded_gt_last_imgs:
                ret[f"{prefix}decoded_gt_last"] = {
                    "value": torch.cat(decoded_gt_last_imgs, dim=0),
                    "type": "image",
                }
            if decoded_pred_best_last_imgs:
                ret[f"{prefix}decoded_pred_best_last"] = {
                    "value": torch.cat(decoded_pred_best_last_imgs, dim=0),
                    "type": "image",
                }
            return ret

        if dino_l1_vals:
            ret[f"{prefix}dino_l1"] = {
                "value": torch.stack(dino_l1_vals).mean(),
                "type": "scalar",
            }
        if lpips_vals:
            ret[f"{prefix}lpips"] = {
                "value": torch.stack(lpips_vals).mean(),
                "type": "scalar",
            }
        return ret

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int = 4,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Tensor | str]]:
        """Run snapshot image/metric logging for train and validation datasets."""
        del verbose, kwargs
        gc.collect()
        torch.cuda.empty_cache()
        for mod in self.models.values():
            mod.eval()

        ret = {}
        ret.update(
            self._run_snapshot(
                self.snapshot_dataset,
                num_samples,
                batch_size,
                "val_",
                compute_metrics=False,
            )
        )
        ret.update(
            self._run_snapshot(
                self.dataset,
                num_samples,
                batch_size,
                "train_",
                compute_metrics=False,
            )
        )
        ret.update(
            self._run_snapshot(
                self.snapshot_dataset,
                self.snapshot_eval_samples,
                batch_size,
                "val_",
                compute_metrics=True,
            )
        )

        for mod in self.models.values():
            mod.train()
        gc.collect()
        torch.cuda.empty_cache()
        return ret
