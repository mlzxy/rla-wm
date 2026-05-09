
"""Training workspace for BC / Diffusion policy on v4world ManiSkill tasks.

Uses diffusion-policy-style dataset (ManiSkillSequenceDataset) and
LinearNormalizer — no ActionNormalizer.  Supports both BC and diffusion policies.
"""

import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
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

from policies.dataset.maniskill_sequence_dataset import ManiSkillSequenceDataset
from policies.dataset.few_shot_mixed_dataset import FewShotMixedDataset
from policies.train_loop import create_eval_env
from policies.workspace.checkpoint_util import (
    resume_training,
    save_checkpoint_with_epoch,
)
from policies.workspace.eval_utils import evaluate_policy_in_sim_env
from utils.misc import print_slurm_environment_summary

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainVLABCWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # --- Task bookkeeping ---
        self.task = cfg.task
        self.robot = cfg.robot
        self.control_mode = cfg.control_mode

        # --- Policy (action_dim / state_dim set in config) ---
        self.model = hydra.utils.instantiate(cfg.policy)

        # --- EMA ---
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # --- Optimizer ---
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=[param for param in self.model.parameters() if param.requires_grad])

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
        dataset_cfg_dict = cast(dict[str, Any], dataset_cfg)
        if "robot_dataset_cfg" in dataset_cfg_dict:
            # Few-shot mixed dataset mode
            dataset = FewShotMixedDataset(**dataset_cfg_dict)
        else:
            dataset = ManiSkillSequenceDataset(**dataset_cfg_dict)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
            drop_last=True,
        )

        # ---- Validation dataset ----
        skip_validation = cfg.training.get("skip_validation", False)
        if skip_validation:
            val_dataset = []
            val_dataloader = None
        else:
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.dataloader.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )

        # ---- Normalizer (fit from dataset, pass to policy) ----
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

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
        logging_cfg: dict[str, object]
        if isinstance(logging_cfg_raw, dict):
            logging_cfg = dict(logging_cfg_raw)
        else:
            logging_cfg = {}
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
            print("[TrainVLABCWorkspace] WandB disabled (cfg.logging.enable=false).")

        print_slurm_environment_summary(prefix="[TrainVLABCWorkspace]")

        # ---- Checkpoint manager ----
        topk_manager = None
        if cfg.checkpoint.get("enable_topk", True):
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
            # cfg.training.num_epochs = 2
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

        # ---- Save a batch for sampling diagnostic ----
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
                        # batch: {"obs": {"image": ..., "state": ...}, "action": ...}
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        loss_result = self.model.compute_loss(batch)
                        # Support both scalar and dict loss returns
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
                            # log of last step is combined with validation and rollout
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)

                            if console_log_enabled and (self.global_step % max(1, console_log_every) == 0):
                                msg = (
                                    f"[train] epoch={self.epoch} step={self.global_step} "
                                    f"total_loss={raw_loss_cpu:.6f} lr={step_log['lr']:.6e}"
                                )
                                if loss_components:
                                    comp_parts = []
                                    for comp_k in sorted(loss_components.keys()):
                                        if comp_k == "loss":
                                            continue
                                        comp_parts.append(f"{comp_k}={loss_components[comp_k]:.6f}")
                                    if comp_parts:
                                        msg += " " + " ".join(comp_parts)
                                if "val_loss" in step_log:
                                    msg += f" val_loss={float(step_log['val_loss']):.6f}"
                                print(msg)

                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # ============ end of epoch ============
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # ======== Rollout ========
                if (self.epoch + 1) % cfg.training.rollout_every == 0:
                    eval_metrics = self._evaluate(policy, cfg, eval_seeds)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    step_log.update({f"eval/{k}": v for k, v in eval_metrics.items()})

                                # ======== Validation ========
                if not skip_validation and (self.epoch + 1) % cfg.training.val_every == 0 and len(val_dataset) > 0:
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                vloss = self.model.compute_loss(batch)
                                val_losses.append(vloss.get('action_loss', vloss['loss']).item() if isinstance(vloss, dict) else vloss.item() )
                                if cfg.training.get("max_val_steps") is not None and batch_idx >= cfg.training.max_val_steps - 1:
                                    break
                        if val_losses:
                            step_log['val_loss'] = np.mean(val_losses)

                # ======== Train sampling diagnostic ========
                if (self.epoch + 1) % cfg.training.get("sample_every", 5) == 0:
                    with torch.no_grad():
                        if train_sampling_batch is None:
                            raise RuntimeError("train_sampling_batch was not initialized")
                        batch = dict_apply(cast(Any, train_sampling_batch), lambda x: x.to(device, non_blocking=True))
                        B = batch['action'].shape[0]
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = F.mse_loss(pred_action, gt_action[:, :pred_action.shape[1]])
                        step_log['train_action_mse_error'] = mse.item()

                # ======== Checkpoint ========
                if (self.epoch + 1) % cfg.training.checkpoint_every == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        save_checkpoint_with_epoch(self, epoch=self.epoch)
                    if topk_manager is not None:
                        metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                # back to train mode
                policy.train()

                # ======== Memory usage ========
                import psutil
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                step_log["memory/rss_gb"] = mem_info.rss / (1024 ** 3)
                step_log["memory/vms_gb"] = mem_info.vms / (1024 ** 3)

                # end of epoch — log the combined step
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
                    val_component_keys = sorted(
                        k for k in step_log.keys() if k.startswith("val_") and k != "val_loss"
                    )
                    for k in val_component_keys:
                        summary_parts.append(f"{k}={float(step_log[k]):.6f}")
                    if "eval/success_rate" in step_log:
                        summary_parts.append(f"success={float(step_log['eval/success_rate']):.3f}")
                    if "eval/avg_reward" in step_log:
                        summary_parts.append(f"reward={float(step_log['eval/avg_reward']):.3f}")
                    if "memory/rss_gb" in step_log:
                        summary_parts.append(f"rss={float(step_log['memory/rss_gb']):.2f}GB")
                    print(" ".join(summary_parts))
                self.global_step += 1
                self.epoch += 1

        if wandb_run is not None:
            wandb_run.finish()

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _evaluate(self, policy, cfg, eval_seeds: list[int]) -> dict:
        """Evaluate policy in-process using the shared rollout utilities."""
        num_episodes = cfg.eval.num_episodes
        save_video = cfg.eval.get("save_video", True)
        n_vis = cfg.eval.get("n_vis", min(3, num_episodes))
        sim_freq = cfg.eval.get("sim_freq", 1)
        video_dir = os.path.join(self.output_dir, cfg.eval.get("video_dir", "videos"))
        cameras = list(cfg.cameras) if not isinstance(cfg.cameras, list) else cfg.cameras
        device = torch.device(cfg.training.device)
        img_size = cfg.img_size

        print(f"[eval] Running in-process evaluation for epoch {self.epoch}...")
        env = create_eval_env(cfg)
        try:
            metrics, all_ep_frames = evaluate_policy_in_sim_env(
                env,
                policy,
                eval_seeds=eval_seeds,
                max_episode_steps=cfg.eval.max_episode_steps,
                sim_freq=sim_freq,
                cameras=cameras,
                device=device,
                camera_height=img_size,
                camera_width=img_size,
                n_vis=n_vis if save_video else 0,
                progress_desc=f"Eval epoch {self.epoch}",
            )
        finally:
            env.close()

        metrics: dict[str, Any] = {
            "success_rate": metrics["success_rate"],
            "avg_reward": metrics["avg_reward"],
        }

        video_paths: dict[str, str] = {}
        if save_video:
            os.makedirs(video_dir, exist_ok=True)
            for ep in range(num_episodes):
                saved_frames = all_ep_frames[ep]
                if saved_frames is None:
                    continue
                try:
                    video_path = os.path.join(
                        video_dir,
                        f"eval_epoch{self.epoch:05d}_ep{ep}_seed{eval_seeds[ep]}.mp4",
                    )
                    writer = imageio.get_writer(video_path, fps=20)
                    for frame in saved_frames:
                        writer.append_data(frame)
                    writer.close()
                    video_paths[str(ep)] = video_path
                except Exception as e:
                    print(f"  Could not save video ep{ep}: {e}")

        if save_video and self._wandb_enabled:
            for ep_str, video_path in video_paths.items():
                if os.path.exists(video_path):
                    metrics[f"sim_video_ep{ep_str}"] = wandb.Video(
                        video_path, fps=20, format="mp4"
                    )

        return metrics

    def _evaluate_off_process(self, policy, cfg, eval_seeds: list[int]) -> dict:
        """Evaluate policy in a subprocess to avoid ManiSkill memory leaks.

        Saves a temporary checkpoint, spawns eval_subprocess.py which creates
        its own env, runs rollouts, saves videos and writes results to JSON.
        The main process reads the JSON and handles wandb reporting.
        """
        num_episodes = cfg.eval.num_episodes
        save_video = cfg.eval.get("save_video", True)
        n_vis = cfg.eval.get("n_vis", min(3, num_episodes))
        sim_freq = cfg.eval.get("sim_freq", 1)
        video_dir = os.path.join(self.output_dir, cfg.eval.get("video_dir", "videos"))
        cameras = list(cfg.cameras) if not isinstance(cfg.cameras, list) else cfg.cameras
        device_str = str(cfg.training.device)
        img_size = cfg.img_size
        use_ema = cfg.training.use_ema

        # ---- Save a dedicated eval checkpoint (synchronous) ----
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        eval_ckpt_path = os.path.join(checkpoint_dir, "eval_tmp.ckpt")
        # Wait for any in-flight async save to finish
        saving_thread = getattr(self, "_saving_thread", None)
        if saving_thread is not None and saving_thread.is_alive():
            saving_thread.join()
        # save_checkpoint_with_epoch renames path to {stem}_epoch{N}.ckpt
        eval_ckpt_path = save_checkpoint_with_epoch(self, path=eval_ckpt_path, epoch=self.epoch, use_thread=False)

        # ---- Write eval config JSON ----
        results_json_path = os.path.join(self.output_dir, f"eval_results_epoch{self.epoch}.json")
        eval_config = {
            "eval_seeds": list(eval_seeds),
            "max_episode_steps": cfg.eval.max_episode_steps,
            "sim_freq": sim_freq,
            "cameras": cameras,
            "device": device_str,
            "img_size": img_size,
            "n_vis": n_vis,
            "save_video": save_video,
            "video_dir": video_dir,
            "epoch": self.epoch,
            "use_ema": use_ema,
            "results_json_path": results_json_path,
        }
        eval_config_path = os.path.join(self.output_dir, f"eval_config_epoch{self.epoch}.json")
        with open(eval_config_path, "w") as f:
            json.dump(eval_config, f)

        # ---- Spawn subprocess ----
        cmd = [
            sys.executable, "-m", "policies.workspace.eval_subprocess",
            "--checkpoint", eval_ckpt_path,
            "--eval-config", eval_config_path,
        ]
        eval_timeout = cfg.eval.get("subprocess_timeout", 1800)  # 30 min default
        print(f"[eval] Spawning subprocess for epoch {self.epoch} evaluation...")
        try:
            result = subprocess.run(
                cmd,
                timeout=eval_timeout,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print(f"[eval] Subprocess failed (exit code {result.returncode})")
                if result.stderr:
                    print(result.stderr)
                return {"success_rate": 0.0, "avg_reward": 0.0}
        except subprocess.TimeoutExpired:
            print(f"[eval] Subprocess timed out after {eval_timeout}s, skipping eval.")
            return {"success_rate": 0.0, "avg_reward": 0.0}

        # ---- Read results ----
        if not os.path.exists(results_json_path):
            print("[eval] Results JSON not found, subprocess may have crashed.")
            return {"success_rate": 0.0, "avg_reward": 0.0}

        with open(results_json_path, "r") as f:
            results = json.load(f)

        metrics: dict[str, Any] = {
            "success_rate": results["success_rate"],
            "avg_reward": results["avg_reward"],
        }

        # ---- Attach wandb video artifacts from saved files ----
        if save_video and self._wandb_enabled:
            for ep_str, video_path in results.get("video_paths", {}).items():
                if os.path.exists(video_path):
                    metrics[f"sim_video_ep{ep_str}"] = wandb.Video(
                        video_path, fps=20, format="mp4"
                    )

        # ---- Cleanup temp files ----
        for tmp_path in [eval_ckpt_path, eval_config_path, results_json_path]:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return metrics
