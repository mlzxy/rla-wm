"""Training workspace for multi-task BC / Diffusion policy on v4world ManiSkill tasks.

Handles multiple task/robot combos with balanced sampling, per-robot
normalizers, and per-task evaluation.
"""

import copy
import json
import math
import os
import random
from typing import Any, cast

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from rich import print
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from policies.dataset.multitask_sequence_dataset import MultiTaskSequenceDataset
from policies.train_loop import create_eval_env
from policies.workspace.checkpoint_util import (
    resume_training,
    save_checkpoint_with_epoch,
)
from policies.workspace.eval_utils import evaluate_policy_in_sim_env
from utils.misc import print_slurm_environment_summary

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _EvalPolicyWrapper:
    """Wraps a multi-task policy for single-task evaluation by injecting robot_ind."""

    def __init__(self, policy, robot_ind: int, device: torch.device):
        self.policy = policy
        self.robot_ind = robot_ind
        self.device = device
        self.n_obs_steps = policy.n_obs_steps
        self.n_action_steps = policy.n_action_steps

    def eval(self):
        self.policy.eval()
        return self

    def train(self, mode=True):
        self.policy.train(mode)
        return self

    @property
    def training(self):
        return self.policy.training

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @torch.no_grad()
    def predict_action(self, obs_dict):
        obs_dict = dict(obs_dict)
        B = obs_dict["image"].shape[0]
        obs_dict["robot_ind"] = torch.full((B,), self.robot_ind, device=self.device, dtype=torch.long)
        return self.policy.predict_action(obs_dict)


class TrainMultiTaskVLABCWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # --- Policy ---
        policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
        self.model = hydra.utils.instantiate(cfg.policy)

        # --- EMA ---
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # --- Optimizer ---
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=[p for p in self.model.parameters() if p.requires_grad],
        )

        self.global_step = 0
        self.epoch = 0
        self._wandb_enabled = False

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # ---- Resume ----
        resume_training(self, cfg)

        # ---- Dataset ----
        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        if not isinstance(dataset_cfg, dict):
            raise TypeError("cfg.dataset must resolve to a mapping")
        dataset = MultiTaskSequenceDataset(**cast(dict[str, Any], dataset_cfg))
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
            drop_last=True,
        )

        # ---- Validation ----
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        # ---- Per-robot normalizers ----
        normalizers = dataset.get_normalizer()
        self.model.set_normalizer(normalizers)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizers)

        # ---- LR Scheduler ----
        total_steps = (len(dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        warmup_steps = cfg.training.lr_warmup_steps

        def _cosine_with_warmup(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_cosine_with_warmup,
            last_epoch=self.global_step - 1 if self.global_step > 0 else -1,
        )

        # ---- EMA ----
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # ---- Logging ----
        logging_cfg_raw = OmegaConf.to_container(cfg.logging, resolve=True)
        logging_cfg: dict[str, object] = dict(logging_cfg_raw) if isinstance(logging_cfg_raw, dict) else {}
        self._wandb_enabled = bool(logging_cfg.pop("enable", True))

        wandb_run = None
        if self._wandb_enabled:
            wandb_config = cast(Any, OmegaConf.to_container(cfg, resolve=True))
            wandb_init_kwargs = cast(dict[str, Any], logging_cfg)
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=wandb_config,
                **wandb_init_kwargs,
            )
            wandb.config.update({"output_dir": self.output_dir})
        else:
            print("[TrainMultiTaskWorkspace] WandB disabled.")

        print_slurm_environment_summary(prefix="[TrainMultiTaskWorkspace]")

        # ---- Checkpoint manager ----
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        # ---- Device ----
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # ---- Debug mode ----
        if cfg.training.debug:
            cfg.training.max_train_steps = 3
            cfg.training.rollout_every = 3
            cfg.training.checkpoint_every = 3
            cfg.training.val_every = 1
            cfg.eval.num_episodes = 1
            cfg.eval.max_episode_steps = 10

        # ---- Eval seeds ----
        eval_seeds = [cfg.training.seed + i for i in range(cfg.eval.num_episodes)]

        # ---- Console logging ----
        console_log_every = int(cfg.training.get("console_log_every", 25))
        console_log_enabled = bool(cfg.training.get("console_log_enable", True))
        train_tqdm_disable = bool(cfg.training.get("train_tqdm_disable", True))

        train_sampling_batch = None

        # ---- Training Loop ----
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                # ================== Train ==================
                if cfg.training.get("freeze_encoder", False):
                    self.model.img_encoder.eval()
                    self.model.img_encoder.requires_grad_(False)

                self.model.train()
                if cfg.training.use_ema:
                    self.ema_model.train()

                train_losses = []
                with tqdm.tqdm(
                    dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                    disable=train_tqdm_disable,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        loss_result = self.model.compute_loss(batch)
                        if isinstance(loss_result, dict):
                            raw_loss = loss_result["loss"]
                            loss_components = {k: v.item() for k, v in loss_result.items()}
                        else:
                            raw_loss = loss_result
                            loss_components = {}
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        for comp_k, comp_v in loss_components.items():
                            if comp_k != "loss":
                                step_log[f"train_{comp_k}"] = comp_v

                        is_last_batch = (batch_idx == (len(dataloader) - 1))
                        if not is_last_batch:
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)

                            if console_log_enabled and (self.global_step % max(1, console_log_every) == 0):
                                msg = (
                                    f"[train] epoch={self.epoch} step={self.global_step} "
                                    f"total_loss={raw_loss_cpu:.6f} lr={step_log['lr']:.6e}"
                                )
                                if loss_components:
                                    comp_parts = [
                                        f"{k}={v:.6f}" for k, v in sorted(loss_components.items()) if k != "loss"
                                    ]
                                    if comp_parts:
                                        msg += " " + " ".join(comp_parts)
                                print(msg)

                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # ============ end of epoch ============
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # ======== Rollout (multi-task eval) ========
                if (self.epoch % cfg.training.rollout_every) == 0:
                    eval_metrics = self._evaluate_all_tasks(policy, cfg, eval_seeds)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    step_log.update({f"eval/{k}": v for k, v in eval_metrics.items()})

                # ======== Validation ========
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataset) > 0:
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                vloss = self.model.compute_loss(batch)
                                val_losses.append(
                                    vloss.get("action_loss", vloss["loss"]).item()
                                    if isinstance(vloss, dict) else vloss.item()
                                )
                                if cfg.training.get("max_val_steps") is not None and batch_idx >= cfg.training.max_val_steps - 1:
                                    break
                        if val_losses:
                            step_log["val_loss"] = np.mean(val_losses)

                # ======== Train sampling diagnostic ========
                if (self.epoch % cfg.training.get("sample_every", 5)) == 0:
                    with torch.no_grad():
                        if train_sampling_batch is not None:
                            batch_ = dict_apply(cast(Any, train_sampling_batch), lambda x: x.to(device, non_blocking=True))
                            batch_d = cast(dict[str, Any], batch_)
                            rid_int = batch_d["robot_ind"][0].item()
                            real_adim = self.model.robot_dim_map[rid_int]["action_dim"]
                            obs_for_pred = {
                                "image": batch_d["obs"]["image"],
                                "state": batch_d["obs"]["state"],
                                "robot_ind": batch_d["robot_ind"],
                            }
                            result = policy.predict_action(obs_for_pred)
                            pred_action = result["action_pred"]
                            gt_action = batch_d["action"][:, :pred_action.shape[1], :real_adim]
                            pred_action_real = pred_action[:, :, :real_adim]
                            mse = F.mse_loss(pred_action_real, gt_action)
                            step_log["train_action_mse_error"] = mse.item()

                # ======== Checkpoint ========
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        save_checkpoint_with_epoch(self, epoch=self.epoch)
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()

                # ======== Memory usage ========
                import psutil
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                step_log["memory/rss_gb"] = mem_info.rss / (1024 ** 3)
                step_log["memory/vms_gb"] = mem_info.vms / (1024 ** 3)

                # End of epoch log
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                if console_log_enabled:
                    summary_parts = [
                        f"[epoch] {self.epoch}",
                        f"step={self.global_step}",
                        f"train_loss={float(step_log.get('train_loss', float('nan'))):.6f}",
                    ]
                    if "val_loss" in step_log:
                        summary_parts.append(f"val_loss={float(step_log['val_loss']):.6f}")
                    if "eval/mean_success_rate" in step_log:
                        summary_parts.append(f"mean_success={float(step_log['eval/mean_success_rate']):.3f}")
                        for k, v in step_log.items():
                            if k.startswith("eval/") and k.endswith("/success_rate"):
                                tag = k[len("eval/"):-len("/success_rate")]
                                summary_parts.append(f"{tag}={float(v):.3f}")
                    if "memory/rss_gb" in step_log:
                        summary_parts.append(f"rss={float(step_log['memory/rss_gb']):.2f}GB")
                    print(" ".join(summary_parts))
                self.global_step += 1
                self.epoch += 1

        if wandb_run is not None:
            wandb_run.finish()

    # ------------------------------------------------------------------ #
    # Multi-task evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_all_tasks(self, policy, cfg, eval_seeds: list[int]) -> dict:
        """Evaluate policy on every task/robot combo, return aggregated metrics."""
        device = torch.device(cfg.training.device)
        img_size = cfg.img_size
        cameras = list(cfg.cameras) if not isinstance(cfg.cameras, list) else cfg.cameras
        save_video = cfg.eval.get("save_video", True)
        n_vis = cfg.eval.get("n_vis", min(3, cfg.eval.num_episodes))
        sim_freq = cfg.eval.get("sim_freq", 1)
        video_dir = os.path.join(self.output_dir, cfg.eval.get("video_dir", "videos"))

        all_metrics = {}
        success_rates = []

        for setting in cfg.settings:
            task = setting["task"]
            robot = setting["robot"]
            robot_ind = int(setting["robot_ind"])
            control_mode = setting.get("control_mode", cfg.control_mode)

            # Build a per-task config for create_eval_env
            task_cfg = copy.deepcopy(cfg)
            OmegaConf.set_struct(task_cfg, False)
            task_cfg.task = task
            task_cfg.robot = robot
            task_cfg.control_mode = control_mode
            OmegaConf.set_struct(task_cfg, True)

            wrapped_policy = _EvalPolicyWrapper(policy, robot_ind, device)

            print(f"[eval] Evaluating {task} / {robot} (robot_ind={robot_ind}) epoch {self.epoch}...")
            env = create_eval_env(task_cfg)
            try:
                metrics, all_ep_frames = evaluate_policy_in_sim_env(
                    env,
                    wrapped_policy,
                    eval_seeds=eval_seeds,
                    max_episode_steps=cfg.eval.max_episode_steps,
                    sim_freq=sim_freq,
                    cameras=cameras,
                    device=device,
                    camera_height=img_size,
                    camera_width=img_size,
                    n_vis=n_vis if save_video else 0,
                    progress_desc=f"Eval {task}/{robot} epoch {self.epoch}",
                )
            finally:
                env.close()

            tag = f"{task}_{robot}"
            all_metrics[f"{tag}/success_rate"] = metrics["success_rate"]
            all_metrics[f"{tag}/avg_reward"] = metrics["avg_reward"]
            success_rates.append(metrics["success_rate"])

            # Save videos
            if save_video:
                task_video_dir = os.path.join(video_dir, tag)
                os.makedirs(task_video_dir, exist_ok=True)
                for ep_idx, frames in enumerate(all_ep_frames):
                    if frames is not None and len(frames) > 0:
                        vpath = os.path.join(
                            task_video_dir,
                            f"epoch{self.epoch:04d}_ep{ep_idx:02d}.mp4",
                        )
                        imageio.mimsave(vpath, frames, fps=30)
                        if self._wandb_enabled:
                            wandb.log({
                                f"eval_video/{tag}/ep{ep_idx}": wandb.Video(vpath, fps=30),
                            }, step=self.global_step)

        all_metrics["mean_success_rate"] = float(np.mean(success_rates)) if success_rates else 0.0
        all_metrics["success_rate"] = all_metrics["mean_success_rate"]
        return all_metrics
