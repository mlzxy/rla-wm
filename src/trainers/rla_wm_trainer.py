"""Trainer for flow-matching on action latent codes.

Trains a ``LatentActionFlowModel`` to predict velocity fields over action latent
tokens produced by a frozen inverse-dynamics encoder. Supports either centroid
per-dimension normalization or scalar normalization for latent flow space.
"""

from __future__ import annotations

import copy
import gc
from typing import Any, Dict, List, Optional, Tuple

from easydict import EasyDict as edict
from jaxtyping import Float32
import numpy as np
from rich import print

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import trange

from policies.action_normalizer import ActionNormalizer
from src.trainers.basic import BasicTrainer
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels
from src.datasets.trajectory_dataset import TASKS
from utils.misc import (
    fetch_state_dict,
    local_seed_scope,
    make_worker_seed_init_fn,
    move_to_device,
    unwrap,
)


# ── robot-id → robot-uid mapping (matches dataset convention) ──────────────
_DEFAULT_ROBOT_UID_MAP: Dict[int, str] = {
    0: "panda",
    1: "xarm6_robotiq",
    2: "ur10e_stick",
}

_CENTROID_STD_EPS = 1e-6


class RLAWMTrainer(BasicTrainer):
    """Train a flow-matching model over latent action tokens."""

    def __init__(
        self,
        *args,
        # dino
        dino_channels: int = 1024,
        rgb_key: str = "rgbs",
        # frozen model ckpts
        encoder_ckpt: str = "",
        image_decoder_ckpt: str = "",
        # flow matching
        sigma_min: float = 1e-5,
        t_schedule: Optional[Dict[str, Any]] = None,
        use_centroid_per_dim_normalization: bool = False,
        latent_scalar_normalization: float = 10.0,
        encoder_latent_mode: str = "auto",
        vae_sampling: bool = True,
        # losses
        lambda_mse: float = 1.0,
        lambda_l1: float = 0.0,
        # robot / action
        robot_uid_map: Optional[Dict[int, str]] = None,
        control_mode: str = "pd_joint_pos",
        state_source: str = "target_qpos",
        # snapshot
        snapshot_eval_samples: int = 200,
        snapshot_eval_seed: int = 2026,
        directly_use_target_qpos: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.dino_channels = dino_channels
        self.rgb_key = rgb_key
        self.sigma_min = sigma_min
        self.t_schedule = t_schedule or {
            "name": "logitNormal",
            "args": {"mean": 0.0, "std": 1.0},
        }
        if latent_scalar_normalization <= 0:
            raise ValueError(
                f"latent_scalar_normalization must be > 0, got {latent_scalar_normalization}"
            )
        if encoder_latent_mode not in {"auto", "deterministic", "vae"}:
            raise ValueError(
                "encoder_latent_mode must be one of {'auto', 'deterministic', 'vae'}, "
                f"got {encoder_latent_mode}"
            )
        self.use_centroid_per_dim_normalization = use_centroid_per_dim_normalization
        self.latent_scalar_normalization = float(latent_scalar_normalization)
        self.encoder_latent_mode = encoder_latent_mode
        self.vae_sampling = vae_sampling
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.snapshot_eval_samples = snapshot_eval_samples
        self.snapshot_eval_seed = snapshot_eval_seed
        self.directly_use_target_qpos = directly_use_target_qpos
        self.task_names = TASKS

        # ---- Validate required models ----
        if "flow_model" not in self.models:
            raise ValueError(
                "Model 'flow_model' is required for DinoLatentActionFlowTrainer"
            )

        # ---- Frozen DINO feature extractor ----
        dino_model_name = get_dinov3_model_for_channels(dino_channels)
        self.dino_extractor = DINOv3FeatureExtractor(model_name=dino_model_name).to(
            self.device
        )
        self.dino_extractor.eval()
        for p in self.dino_extractor.parameters():
            p.requires_grad = False

        # ---- Load frozen encoder ----
        if "encoder" in self.models and encoder_ckpt:
            self.models["encoder"].load_state_dict(
                fetch_state_dict("encoder", encoder_ckpt, self.device),
                strict=True,
            )
            print("[green]Encoder loaded successfully (strict)[/green]")

        # ---- Load frozen decoder ----
        if "decoder" in self.models and encoder_ckpt:
            self.models["decoder"].load_state_dict(
                fetch_state_dict("decoder", encoder_ckpt, self.device),
                strict=True,
            )
            print("[green]Decoder loaded successfully (strict)[/green]")

        # ---- Load frozen image decoder ----
        if "image_decoder" in self.models and image_decoder_ckpt:
            self.models["image_decoder"].load_state_dict(
                fetch_state_dict("decoder", image_decoder_ckpt, self.device),
                strict=True,
            )
            print("[green]Image decoder loaded successfully (strict)[/green]")

        # ---- Action normalizers (lazy-init per robot) ----
        self._robot_uid_map = robot_uid_map or _DEFAULT_ROBOT_UID_MAP
        self._control_mode = control_mode
        self._state_source = state_source
        self._normalizers: Dict[str, ActionNormalizer] = {}

    # ------------------------------------------------------------------
    # Action normalizer helpers
    # ------------------------------------------------------------------

    def _get_normalizer(self, robot_id: int, task_ind: int) -> ActionNormalizer:
        uid = self._robot_uid_map.get(robot_id)
        if uid is None:
            raise ValueError(f"Unknown robot_id={robot_id}, not in robot_uid_map")
        task_name = self.task_names[task_ind]
        if ("PushT" in task_name or "RollBall" in task_name) and "ur10" not in uid:
            uid += "_closed"
        if uid not in self._normalizers:
            self._normalizers[uid] = ActionNormalizer(
                robot_uid=uid,
                control_mode=self._control_mode,
                state_source=self._state_source,
                device=self.device,
            )
        return self._normalizers[uid]

    @torch.no_grad()
    def _target_qpos_to_full_qpos(
        self,
        target_qpos_list: List[Tensor],  # list of (horizon_i, J_i)
        robot_ids: Tensor,  # (B,)
        task_inds: Tensor,  # (B,)
    ) -> List[Optional[Tensor]]:
        """Prepare per-sample qpos conditioning tensors.

        When ``directly_use_target_qpos`` is True, this bypasses ActionNormalizer and
        returns ``target_qpos`` directly (moved to device). Otherwise, converts each
        sample to full_qpos via ActionNormalizer. Returns None when robot_id == -1.
        """
        results: List[Optional[Tensor]] = []
        for i, tq in enumerate(target_qpos_list):
            rid = int(robot_ids[i].item())
            if rid < 0:
                results.append(None)
                continue

            if self.directly_use_target_qpos:
                results.append(tq.squeeze().to(self.device))
                continue

            normalizer = self._get_normalizer(rid, int(task_inds[i].item()))
            # tq shape: (horizon_i, J_i) — add batch dim
            tq_3d = tq.squeeze().to(self.device)  # (1, H, J)
            # Build a minimal batch dict for normalize
            batch = {self._state_source: [tq_3d]}
            action = normalizer.normalize(batch)
            full_qpos = normalizer.denormalize(
                action, return_full_qpos=True
            )  # (1, H, full_qpos_dim)
            results.append(full_qpos.squeeze(0).detach()[:-1])  # (H-1, full_qpos_dim)
        return results

    # ------------------------------------------------------------------
    # DINO token extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_dino_tokens(
        self,
        images: Float32[Tensor, "B Cam 3 H W"],
    ) -> Tuple[Float32[Tensor, "B Cam Lp C"], Tuple[int, int]]:
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
        rgb = data[self.rgb_key]
        if rgb.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 frames in '{self.rgb_key}', got {rgb.shape[1]}"
            )
        mask = (data["foreground_masks"] > 0).float()
        return rgb * mask.unsqueeze(3)

    # ------------------------------------------------------------------
    # Flow matching helpers
    # ------------------------------------------------------------------

    def _sample_t(self, batch_size: int) -> Tensor:
        name = self.t_schedule["name"]
        if name == "uniform":
            return torch.rand(batch_size)
        elif name == "logitNormal":
            mean = self.t_schedule["args"]["mean"]
            std = self.t_schedule["args"]["std"]
            return torch.sigmoid(torch.randn(batch_size) * std + mean)
        raise ValueError(f"Unknown t_schedule: {name}")

    def _diffuse(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        t_v = t.view(-1, *([1] * (x_0.ndim - 1)))
        return (1 - t_v) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t_v) * noise

    def _get_velocity(self, x_0: Tensor, noise: Tensor) -> Tensor:
        return (1 - self.sigma_min) * noise - x_0

    def _extract_encoder_latent_tokens(
        self,
        dino_delta_tokens: Tensor,
        flow_model: nn.Module,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Extract latent action tokens from encoder supporting deterministic and VAE outputs."""
        with torch.no_grad():
            encoder_tokens, _ = self.models["encoder"](dino_delta_tokens)

        if encoder_tokens.ndim != 3:
            raise ValueError(
                "Encoder token output must be rank-3 [B, Ntok, C], got "
                f"{tuple(encoder_tokens.shape)}"
            )

        model = unwrap(flow_model)
        token_dim = int(model.token_dim)
        out_dim = int(encoder_tokens.shape[-1])

        if self.encoder_latent_mode == "deterministic":
            if out_dim != token_dim:
                raise ValueError(
                    f"Expected deterministic encoder output dim={token_dim}, got {out_dim}"
                )
            return encoder_tokens, None, None

        if self.encoder_latent_mode == "vae":
            if out_dim != 2 * token_dim:
                raise ValueError(
                    f"Expected VAE encoder output dim={2 * token_dim}, got {out_dim}"
                )
            mean, logvar = encoder_tokens.chunk(2, dim=-1)
            if self.vae_sampling:
                std = torch.exp(0.5 * logvar)
                latent = mean + std * torch.randn_like(std)
            else:
                latent = mean
            return latent, mean, logvar

        # auto mode
        if out_dim == token_dim:
            return encoder_tokens, None, None
        if out_dim == 2 * token_dim:
            mean, logvar = encoder_tokens.chunk(2, dim=-1)
            if self.vae_sampling:
                std = torch.exp(0.5 * logvar)
                latent = mean + std * torch.randn_like(std)
            else:
                latent = mean
            return latent, mean, logvar
        raise ValueError(
            "Encoder output dim does not match deterministic or VAE expectations: "
            f"got {out_dim}, expected {token_dim} or {2 * token_dim}"
        )

    def _normalize_latent_tokens(self, tokens: Tensor, flow_model: nn.Module) -> Tensor:
        """Map latent tokens to flow space via centroid or scalar normalization."""
        if self.use_centroid_per_dim_normalization:
            model = unwrap(flow_model)
            centroid = model.centroid.to(device=tokens.device, dtype=tokens.dtype)
            centroid_std = model.centroid_std.to(
                device=tokens.device,
                dtype=tokens.dtype,
            ).clamp_min(_CENTROID_STD_EPS)
            return (tokens - centroid.unsqueeze(0)) / centroid_std.unsqueeze(0)
        return tokens / self.latent_scalar_normalization

    def _denormalize_latent_tokens(self, tokens: Tensor, flow_model: nn.Module) -> Tensor:
        """Map flow-space latent tokens back to latent token space."""
        if self.use_centroid_per_dim_normalization:
            model = unwrap(flow_model)
            centroid = model.centroid.to(device=tokens.device, dtype=tokens.dtype)
            centroid_std = model.centroid_std.to(
                device=tokens.device,
                dtype=tokens.dtype,
            ).clamp_min(_CENTROID_STD_EPS)
            return tokens * centroid_std.unsqueeze(0) + centroid.unsqueeze(0)
        return tokens * self.latent_scalar_normalization

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_rgb_for_lpips(imgs):
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
        decoded = self.models["image_decoder"](
            tokens.contiguous(), patch_hw=patch_hw
        ).clamp(0, 1)
        return torch.cat([decoded[:, ci] for ci in range(decoded.shape[1])], dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses(
        self, **args: Any
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        data = move_to_device(args, self.device)
        terms = edict()
        status = edict()

        # 1. Extract DINO tokens
        rgbs_seq = self._get_masked_rgb_sequence(data)
        tokens_seq, patch_hw = self._extract_dino_tokens_sequence(rgbs_seq)

        x_t = tokens_seq[:, 0]  # (B, Cam, Lp, C)
        x_T = tokens_seq[:, 1]  # (B, Cam, Lp, C)

        bsz, cams, lp, dino_ch = x_t.shape
        assert cams == 1, f"Expected single camera, got {cams}"

        x_t_flat = x_t.reshape(bsz, cams * lp, dino_ch)
        x_T_flat = x_T.reshape(bsz, cams * lp, dino_ch)

        flow_model = self.training_models["flow_model"]

        # 2. Compute GT latent tokens via frozen encoder.
        gt_tokens, _, _ = self._extract_encoder_latent_tokens(
            x_T_flat - x_t_flat,
            flow_model=flow_model,
        )

        # 3. Prepare qpos
        robot_ids = data["robot_id"].long().flatten()  # (B,)
        task_inds = data["task_ind"].long().flatten()  # (B,)
        horizons = data["horizon"].float().flatten()  # (B,)
        target_qpos_list = data["target_qpos"]  # list of (horizon_i, J_i)

        full_qpos_list = self._target_qpos_to_full_qpos(
            target_qpos_list, robot_ids, task_inds
        )

        # 4. Flow matching
        x_0 = self._normalize_latent_tokens(gt_tokens, flow_model=flow_model)

        noise = torch.randn_like(x_0)
        t = self._sample_t(bsz).to(self.device).float()
        x_t_flow = self._diffuse(x_0, t, noise=noise)
        v_target = self._get_velocity(x_0, noise)

        # 5. Forward
        v_pred = flow_model(
            xt_tokens=x_t_flat,
            noisy_latent=x_t_flow,
            flow_t=t,
            task_inds=task_inds,
            horizons=horizons,
            robot_ids=robot_ids,
            full_qpos_list=full_qpos_list,
        )

        # 6. Losses
        terms["mse"] = F.mse_loss(v_pred, v_target)
        terms["l1"] = F.l1_loss(v_pred, v_target)
        terms["loss"] = self.lambda_mse * terms["mse"] + self.lambda_l1 * terms["l1"]

        # Log per-timestep-bin losses (10 bins over [0, 1]).
        with torch.no_grad():
            l1_per_instance = (v_pred - v_target).abs().mean(dim=(1, 2))
            time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
            status["l1"] = {}
            for i in range(10):
                mask_np = time_bin == i
                if not mask_np.any():  continue
                t_val = (i + 1) / 10.0
                mask = torch.from_numpy(mask_np).to(device=t.device)
                status["l1"][f"t={t_val:.1f}"] = float(l1_per_instance[mask].mean().item())

        terms["loss"] = terms["loss"] + 0.0 * sum(
            p.sum() for p in flow_model.parameters() if p.requires_grad
        )

        return terms, status

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _make_snapshot_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        deterministic: bool,
        seed: int,
    ) -> DataLoader:
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
    def _euler_sample(
        self,
        flow_model: nn.Module,
        noise: Tensor,  # (B, num_latent_tokens, token_dim)
        xt_tokens: Tensor,  # (B, Lp, dino_channels)
        task_inds: Tensor,
        horizons: Tensor,
        robot_ids: Tensor,
        full_qpos_list: List[Optional[Tensor]],
        steps: int = 50,
    ) -> Tensor:
        """Run Euler ODE sampling to produce latent tokens.

        Computes conditioning once, then iterates only the flow stage.
        """
        model = unwrap(flow_model)

        # One-time conditioning
        cond_queries = model.forward_cond(
            xt_tokens=xt_tokens,
            task_inds=task_inds,
            horizons=horizons,
            robot_ids=robot_ids,
            full_qpos_list=full_qpos_list,
        )  # (B, num_latent_tokens, model_channels)

        # Euler integration (only flow stage per step)
        dt = 1.0 / steps
        x = noise.clone()
        for i in range(steps):
            # Integrate backward in time: start from t=1 (noise) and move to t=0.
            t_val = 1.0 - i / steps
            t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
            v = model.forward_flow(cond_queries, x, t)
            x = x - v * dt
        return x

    @torch.no_grad()
    def _run_snapshot(
        self,
        dataset: Any,
        num_samples: int,
        batch_size: int,
        prefix: str,
        compute_metrics: bool = False,
    ) -> Dict[str, Dict[str, Tensor | str]]:
        if dataset is None:
            return {}

        dataloader = self._make_snapshot_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            deterministic=compute_metrics,
            seed=self.snapshot_eval_seed,
        )
        iterator = iter(dataloader)

        decoded_t_imgs: List[Tensor] = []
        decoded_gt_last_imgs: List[Tensor] = []
        decoded_pred_last_imgs: List[Tensor] = []
        dino_l1_vals: List[Tensor] = []

        loop_iter = (
            trange(0, num_samples, batch_size)
            if compute_metrics
            else range(0, num_samples, batch_size)
        )
        rank = int(getattr(self, "rank", 0))
        local_seed = self.snapshot_eval_seed + rank

        flow_model = self.models["flow_model"]
        has_decoder = "decoder" in self.models
        has_image_decoder = "image_decoder" in self.models

        with local_seed_scope(local_seed) if compute_metrics else torch.no_grad():
            for _ in loop_iter:
                data = next(iterator)
                data = move_to_device(data, self.device)

                rgbs_seq = self._get_masked_rgb_sequence(data)
                tokens_seq, patch_hw = self._extract_dino_tokens_sequence(rgbs_seq)

                x_t = tokens_seq[:, 0]  # (B, 1, Lp, C)
                x_T = tokens_seq[:, 1]
                bsz, cams, lp_dim, dino_ch = x_t.shape
                x_t_flat = x_t.reshape(bsz, cams * lp_dim, dino_ch)
                x_T_flat = x_T.reshape(bsz, cams * lp_dim, dino_ch)

                # GT latent
                gt_tokens, _, _ = self._extract_encoder_latent_tokens(
                    x_T_flat - x_t_flat,
                    flow_model=flow_model,
                )

                # Prepare qpos
                robot_ids = data["robot_id"].long().flatten()
                task_inds = data["task_ind"].long().flatten()
                horizons = data["horizon"].float().flatten()
                full_qpos_list = self._target_qpos_to_full_qpos(
                    data["target_qpos"], robot_ids, task_inds
                )

                # Sample via Euler
                noise = torch.randn_like(gt_tokens)
                sampled_flow = self._euler_sample(
                    flow_model=flow_model,
                    noise=noise,
                    xt_tokens=x_t_flat,
                    task_inds=task_inds,
                    horizons=horizons,
                    robot_ids=robot_ids,
                    full_qpos_list=full_qpos_list,
                    steps=50,
                )
                pred_tokens = self._denormalize_latent_tokens(
                    sampled_flow,
                    flow_model=flow_model,
                )

                if compute_metrics:
                    dino_l1_vals.append(
                        F.l1_loss(pred_tokens, gt_tokens, reduction="mean")
                    )

                    if has_decoder and has_image_decoder:
                        # Use decoder to reconstruct predicted x_T from x_t + predicted latent
                        _, pred_x_T_flat = self.models["decoder"](
                            x_t_flat, tokens=pred_tokens
                        )
                        pred_x_T = pred_x_T_flat.reshape_as(x_T)
                        # No LPIPS here to keep simple; just DINO L1
                    continue

                if has_decoder and has_image_decoder:
                    # Decode x_t for visualization
                    decoded_t_imgs.append(self._decode_tokens_to_vis(x_t, patch_hw))

                    # Decode GT x_T via decoder from GT latent tokens
                    _, gt_x_T_flat = self.models["decoder"](x_t_flat, tokens=gt_tokens)
                    gt_x_T = gt_x_T_flat.reshape_as(x_T)
                    decoded_gt_last_imgs.append(
                        self._decode_tokens_to_vis(gt_x_T, patch_hw)
                    )

                    # Decode predicted x_T via decoder
                    _, pred_x_T_flat = self.models["decoder"](
                        x_t_flat, tokens=pred_tokens
                    )
                    pred_x_T = pred_x_T_flat.reshape_as(x_T)
                    decoded_pred_last_imgs.append(
                        self._decode_tokens_to_vis(pred_x_T, patch_hw)
                    )

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
            if decoded_pred_last_imgs:
                ret[f"{prefix}decoded_pred_last"] = {
                    "value": torch.cat(decoded_pred_last_imgs, dim=0),
                    "type": "image",
                }
        else:
            if dino_l1_vals:
                ret[f"{prefix}dino_l1"] = {
                    "value": torch.stack(dino_l1_vals).mean(),
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
        # ret.update(
        #     self._run_snapshot(
        #         self.snapshot_dataset,
        #         self.snapshot_eval_samples,
        #         batch_size,
        #         "val_",
        #         compute_metrics=True,
        #     )
        # )

        for mod in self.models.values():
            mod.train()
        gc.collect()
        torch.cuda.empty_cache()
        return ret
