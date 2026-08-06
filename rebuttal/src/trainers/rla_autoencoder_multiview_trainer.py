from typing import Any, Dict, List, Tuple

from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rebuttal.src.models.plucker_embedding import compute_plucker_rays
from src.trainers.rla_autoencoder_trainer import RlaAutoencoderTrainer
from utils.misc import move_to_device


class RlaAutoencoderMultiViewTrainer(RlaAutoencoderTrainer):
    """RlaAutoencoderTrainer for datasets with more than one camera (e.g. LIBERO).

    LIBERO gives a front (`agentview`) and a `wrist` view per frame, while the single-view trainer
    asserts `cams == 1`. Everything else in the parent — the DINO extraction, VQ path, snapshot
    rendering (which already stitches views horizontally) and LPIPS metrics — is camera-generic, so
    this subclass only replaces `inference_batch` and adds a per-view L1 breakdown to
    `training_losses`.

    The per-view token blocks are concatenated along the sequence dimension, exactly as the parent
    does for one camera. `num_views` and, when the model asks for them, Plücker rays are forwarded
    to encoder/decoder modules that advertise `expects_num_views` / `expects_plucker`
    (`MultiViewTokenTransformer`), so a plain `SimpleTokenTransformer` still works unchanged.
    """

    def __init__(self, *args, num_views: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        self.num_views = num_views
        if self.inverse_input_mode == "append":
            # "append" concatenates (x_T, x_t) along the sequence, so the sequence is no longer
            # V equally-sized per-view blocks and the view embedding would be misaligned.
            raise ValueError(
                "inverse_input_mode='append' is not supported with multiple views; "
                "use 'sub' or 'concat'"
            )
        self._needs_plucker = any(
            getattr(self.models.get(name), "expects_plucker", False)
            for name in ("encoder", "decoder")
        )

    @staticmethod
    def _forward_views(
        model: nn.Module,
        x: Tensor,
        tokens: Tensor | None,
        num_views: int,
        plucker: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Call a token transformer, passing num_views/plucker only if its forward accepts them.

        The capability flags are probed on the inner module: under DDP `model` is the wrapper,
        which does not proxy plain attribute lookups, so probing it directly would silently drop
        num_views/plucker on every multi-GPU run. The call itself still goes through `model` so
        gradients are synchronised.
        """
        inner = getattr(model, "module", model)
        if not getattr(inner, "expects_num_views", False):
            return model(x, tokens=tokens)
        if getattr(inner, "expects_plucker", False):
            return model(x, tokens=tokens, num_views=num_views, plucker=plucker)
        return model(x, tokens=tokens, num_views=num_views)

    def _plucker_rays(
        self, data: Dict[str, Any], patch_hw: Tuple[int, int], image_hw: Tuple[int, int]
    ) -> Tensor | None:
        """Plücker rays for the frame the encoder/decoder condition on (frame 0).

        The wrist camera moves between frame 0 and frame 1, so there is no single ray set for a
        transition. Frame 0 is the observation both the encoder input and the decoder are anchored
        on, so its geometry is what gets injected.
        """
        if not self._needs_plucker:
            return None
        missing = [k for k in ("intrinsics", "w2c") if k not in data]
        if missing:
            raise KeyError(
                f"pos_embed_mode='plucker' needs {missing} in the batch; the dataset must provide "
                "camera intrinsics and extrinsics"
            )
        rays = compute_plucker_rays(
            intrinsics=data["intrinsics"][:, 0].float(),
            cam2world=data["w2c"][:, 0].float(),
            patch_hw=patch_hw,
            image_hw=image_hw,
        )  # [B, Cam, Lp, 6]
        return rays.reshape(rays.shape[0], -1, 6)

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
        if cams != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} cameras (num_views), got {cams}. "
                "Check the dataset's `cameras`/`max_num_cameras` config."
            )

        x0_flat = x0.reshape(bsz, cams * lp, dino_ch)
        x1_flat = x1.reshape(bsz, cams * lp, dino_ch)

        plucker = self._plucker_rays(
            data, patch_hw=patch_hw, image_hw=rgbs_seq.shape[-2:]
        )

        if self.decoder_only_mode:
            # Decoder-only mode: force decoder to map x0 -> x1 without latent tokens.
            _, pred_x1_flat = self._forward_views(
                models["decoder"], x0_flat, None, cams, plucker=plucker
            )
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
        enc_tokens, _ = self._forward_views(
            models["encoder"], inv_encoder_input, None, cams, plucker=plucker
        )
        if enc_tokens.ndim != 3:
            raise ValueError(
                f"Encoder token output must be rank-3 [B, Ntok, C], got {enc_tokens.shape}"
            )

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

        # Decoder predicts x1 given x0 sequence tokens and quantized latent tokens.
        _, pred_x1_flat = self._forward_views(
            models["decoder"], x0_flat, quant_tokens, cams, plucker=plucker
        )
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
        """Same losses as the parent, plus a per-view L1 breakdown in `status` (logged only)."""
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

        # Which view the model is struggling on (wrist views are usually harder).
        with torch.no_grad():
            for view_idx in range(out["x1_pred"].shape[1]):
                status[f"l1_view{view_idx}"] = F.l1_loss(
                    out["x1_pred"][:, view_idx], out["x1_gt"][:, view_idx]
                )

        return terms, status

    @torch.no_grad()
    def _stitch_cameras(self, imgs: Tensor) -> Tensor:
        """[B, Cam, 3, H, W] in [0, 1] -> [B, 3, H, Cam*W], views laid out left to right."""
        return torch.cat([imgs[:, ci] for ci in range(imgs.shape[1])], dim=-1)

    @torch.no_grad()
    def _gt_vs_pred_panels(
        self,
        dataset: Any,
        num_samples: int,
        batch_size: int,
        prefix: str,
    ) -> Dict[str, Dict[str, Tensor | str]]:
        """One image per sample stacking the original target frame over its RLA reconstruction.

        Layout per sample, both cameras side by side within each row::

            [ original agentview | original wrist ]      <- ground truth frame t+1 (raw pixels)
            [ recon    agentview | recon    wrist ]      <- decoder(x1_pred), the RLA prediction

        The top row is raw pixels and is always meaningful. The bottom row is only meaningful once
        the image decoder is trained (an untrained DinoToImageDecoderV1 is zero-initialised and
        renders black); train rebuttal/exp2_vla_adapter/configs/unet/dino_to_image_v1_libero.yaml and set image_decoder_ckpt.
        """
        if dataset is None:
            return {}

        dataloader = self._make_snapshot_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            deterministic=False,
            seed=self.snapshot_eval_seed,
        )
        iterator = iter(dataloader)

        panels: List[Tensor] = []
        for _ in range(0, num_samples, batch_size):
            data = move_to_device(next(iterator), self.device)
            out = self.inference_batch(self.models, data, training=False)
            original = self._stitch_cameras(self._normalize_rgb_for_lpips(out["target_rgb"]))
            recon = self._decode_tokens_to_vis(out["x1_pred"], out["patch_hw"])
            panels.append(torch.cat([original, recon], dim=-2))  # stack rows -> [B, 3, 2H, Cam*W]

        if not panels:
            return {}
        return {f"{prefix}gt_vs_pred": {"value": torch.cat(panels, dim=0), "type": "image"}}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int = 4,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Tensor | str]]:
        """Parent snapshot plus a combined original-vs-reconstruction panel per dataset."""
        ret = super().run_snapshot(num_samples, batch_size, verbose=verbose, **kwargs)
        for mod in self.models.values():
            mod.eval()
        ret.update(self._gt_vs_pred_panels(self.snapshot_dataset, num_samples, batch_size, "val_"))
        ret.update(self._gt_vs_pred_panels(self.dataset, num_samples, batch_size, "train_"))
        for mod in self.models.values():
            mod.train()
        return ret
