"""
Training workspace for VLA Diffusion Policy on v4world ManiSkill tasks.

Follows the atomic_policy workspace pattern (BaseWorkspace, hydra config,
EMA, wandb, TopK checkpoints) but uses:
  - v4world's TrajectoryDataset for data loading
  - v4world's ActionNormalizer for state / action normalization
  - v4world's ManiSkill evaluation loop (no env_runner)
"""

import gc
import copy
import os
import random
import time

import math

import hydra
import imageio
import numpy as np
import torch
import tqdm
import wandb
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datalib.env_utils import get_background_ids, extract_rgbs_from_obs
from policies.action_normalizer import ActionNormalizer, Action, Observation
from policies.train_loop import (
    PUSH_TASKS,
    VALID_CONFIGS,
    create_eval_env,
    prepare_rgbs,
)
from policies.workspace.checkpoint_util import (
    resume_training,
    save_checkpoint_with_epoch,
)
from src.datasets.trajectory_dataset import TrajectoryDataset, TrajectoryBatch
from utils.misc import move_to_device

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainVLADiffusionWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # --- Robot / task bookkeeping ---
        self.task = cfg.task
        self.robot = cfg.robot
        self.control_mode = cfg.control_mode

        # Auto-adjust robot UID for push tasks (append _closed)
        robot_uid = self.robot
        if self.task in PUSH_TASKS and not robot_uid.endswith("_closed") and robot_uid != "ur10e_stick":
            robot_uid = robot_uid + "_closed"
        self.robot_uid = robot_uid

        # --- Normalizers ---
        self.state_normalizer = ActionNormalizer(robot_uid, self.control_mode, state_source="qpos")
        self.action_normalizer = ActionNormalizer(robot_uid, self.control_mode, state_source="target_qpos")

        # Compute action / state dims for the policy
        arm_dim = self.action_normalizer._spec.arm_dim
        action_dim = arm_dim + 1  # arm_joints + gripper close signal
        state_dim = action_dim    # same: arm + gripper

        # Inject dims into the policy config so hydra can instantiate it
        if "policy" in cfg:
            cfg.policy.action_dim = action_dim
            cfg.policy.state_dim = state_dim

        # --- Policy ---
        self.model = hydra.utils.instantiate(cfg.policy)

        # --- EMA ---
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # --- Optimizer ---
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # ---- Resume ----
        resume_training(self, cfg)

        # ---- Dataset ----
        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        dataset = TrajectoryDataset(**dataset_cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers,
            drop_last=True,
            collate_fn=TrajectoryDataset.collate_fn,
        )

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
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

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

        # Move FK chains in normalizers to training device
        for norm in (self.state_normalizer, self.action_normalizer):
            if norm._fk_chain is not None:
                norm._fk_chain = norm._fk_chain.to(device=device)

        # ---- Debug mode ----
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.rollout_every = 3
            cfg.training.checkpoint_every = 3
            cfg.training.val_every = 1
            cfg.eval.num_episodes = 1
            cfg.eval.max_episode_steps = 10

        # ---- Eval seeds ----
        eval_seeds = [cfg.training.seed + i for i in range(cfg.eval.num_episodes)]

        # ---- Eval env (lazy-created on first eval, reused afterwards) ----
        self._eval_env = None

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
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch: TrajectoryBatch = move_to_device(batch, device)

                        # Prepare observations
                        rgbs = prepare_rgbs(batch)  # (B, CAM, 3, H, W)
                        state_action = self.state_normalizer.normalize(batch)
                        target_action = self.action_normalizer.normalize(batch)

                        # Take first step for state
                        for k in ["gripper", "eef", "arm_joints"]:
                            setattr(state_action, k, getattr(state_action, k)[:, :1])

                        B = rgbs.shape[0]
                        task_descs = [cfg.task_description] * B

                        obs = Observation(state=state_action, rgbs=rgbs, actions=target_action)

                        # Forward — compute diffusion loss
                        raw_loss = self.model.compute_loss(obs, prompt=task_descs)
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

                        # ======== Eval (step-based) ========
                        if self.global_step > 0 and (self.global_step % cfg.training.rollout_every) == 0:
                            policy = self.model
                            if cfg.training.use_ema:
                                policy = self.ema_model
                            policy.eval()
                            eval_metrics = self._evaluate(policy, cfg, eval_seeds)
                            step_log.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                            self.model.train()
                            if cfg.training.use_ema:
                                self.ema_model.train()

                        # ======== Checkpoint (step-based) ========
                        if self.global_step > 0 and (self.global_step % cfg.training.checkpoint_every) == 0:
                            if cfg.checkpoint.save_last_ckpt:
                                save_checkpoint_with_epoch(self, epoch=self.epoch)
                            metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path)

                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # ============ end epoch ============
                self.epoch += 1

    # ------------------------------------------------------------------ #
    # Evaluation (v4world-style per-episode ManiSkill rollout)
    # ------------------------------------------------------------------ #

    def _obs_to_frame(self, obs, env, cameras, device, img_size) -> np.ndarray:
        """Extract a single obs-view video frame (background-filtered RGB, uint8).

        Returns (H, W*C, 3) uint8 numpy array with cameras tiled horizontally.
        """
        env_device = next(iter(obs['sensor_data'].values()))['rgb'].device
        bg_ids = get_background_ids(env).to(env_device)
        rgbs = extract_rgbs_from_obs(obs, bg_ids, cameras).to(device)

        if rgbs.shape[-1] != img_size or rgbs.shape[-2] != img_size:
            C_cam = rgbs.shape[1]
            rgbs = rgbs.reshape(-1, 3, rgbs.shape[-2], rgbs.shape[-1])
            rgbs = torch.nn.functional.interpolate(
                rgbs, size=(img_size, img_size), mode='bilinear', align_corners=False)
            rgbs = rgbs.reshape(1, C_cam, 3, img_size, img_size)

        cam_imgs = []
        for c in range(rgbs.shape[1]):
            img_np = rgbs[0, c].cpu().numpy().transpose(1, 2, 0)
            cam_imgs.append(img_np)
        frame = np.concatenate(cam_imgs, axis=1)
        return np.clip(frame * 255, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def _evaluate(self, policy, cfg, eval_seeds: list[int]) -> dict:
        """Run evaluation rollouts in ManiSkill env. Returns metrics dict."""
        num_episodes = cfg.eval.num_episodes
        save_video = cfg.eval.get("save_video", True)
        n_vis = cfg.eval.get("n_vis", min(3, num_episodes))
        sim_freq = cfg.eval.get("sim_freq", 1)
        video_dir = os.path.join(self.output_dir, cfg.eval.get("video_dir", "videos"))

        if self._eval_env is None:
            torch.cuda.empty_cache()
            self._eval_env = create_eval_env(cfg)
        env = self._eval_env
        cameras = cfg.cameras
        device = torch.device(cfg.training.device)
        img_size = cfg.img_size

        successes = []
        episode_rewards = []

        all_ep_frames: list[list[np.ndarray] | None] = [None] * num_episodes

        max_steps = cfg.eval.max_episode_steps
        total_steps = num_episodes * max_steps
        pbar = tqdm.tqdm(total=total_steps, desc=f"Eval epoch {self.epoch}", leave=False)

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=eval_seeds[ep])
            env_device = next(iter(obs['sensor_data'].values()))['rgb'].device
            bg_ids = get_background_ids(env).to(env_device)

            policy.reset()
            ep_reward = 0.0
            record_this_ep = save_video and ep < n_vis
            ep_frames: list[np.ndarray] = []

            step = 0
            while step < max_steps:
                rgbs = extract_rgbs_from_obs(obs, bg_ids, cameras).to(device)

                if record_this_ep:
                    frame = self._obs_to_frame(obs, env, cameras, device, img_size)
                    ep_frames.append(frame)

                qpos = env.agent.robot.qpos.to(device)
                root_pose = env.agent.robot.pose.raw_pose.to(device)
                state_batch = {
                    self.state_normalizer.state_source: qpos.unsqueeze(1),
                    "root_poses": root_pose.unsqueeze(1),
                }
                state_action = self.state_normalizer.normalize(state_batch)

                policy_obs = Observation(state=state_action, rgbs=rgbs)

                task_desc = [cfg.get("task_description", self.task)]
                chunk_action = policy.predict_action(policy_obs, prompt=task_desc)

                chunk_action_raw = self.action_normalizer.denormalize(chunk_action)
                chunk_action_env = chunk_action_raw.to(env_device)
                n_action_steps = policy.n_action_steps
                terminated = False

                for t in range(n_action_steps):
                    if step >= max_steps or terminated:
                        break
                    for _sim_i in range(sim_freq):
                        done = _sim_i == (sim_freq - 1)
                        obs, reward, terminated, _, info = (
                            env.step if done else env.step_wo_obs
                        )(chunk_action_env[:, t])
                        terminated = terminated.any()
                        if reward is not None:
                            ep_reward += reward.sum().item()
                        if terminated:
                            break
                    step += 1
                    pbar.update(1)

                    if record_this_ep and not terminated and step < max_steps and t < n_action_steps - 1:
                        frame = self._obs_to_frame(obs, env, cameras, device, img_size)
                        ep_frames.append(frame)

                if terminated:
                    break

            remaining = max_steps - step
            if remaining > 0:
                pbar.update(remaining)
            succ = info["success"].item()
            successes.append(succ)
            episode_rewards.append(ep_reward)
            pbar.set_postfix(ep=ep, reward=f"{ep_reward:.2f}", succ=succ)
            if record_this_ep and ep_frames:
                all_ep_frames[ep] = ep_frames

        pbar.close()

        # ---- Save per-episode videos and build wandb log ----
        metrics: dict = {}
        if save_video:
            os.makedirs(video_dir, exist_ok=True)
            for ep in range(num_episodes):
                if all_ep_frames[ep] is None:
                    continue
                try:
                    video_path = os.path.join(
                        video_dir,
                        f"eval_epoch{self.epoch:05d}_ep{ep}_seed{eval_seeds[ep]}.mp4",
                    )
                    writer = imageio.get_writer(video_path, fps=20)
                    for frame in all_ep_frames[ep]:
                        writer.append_data(frame)
                    writer.close()
                    metrics[f"sim_video_ep{ep}"] = wandb.Video(
                        video_path, fps=20, format="mp4"
                    )
                except Exception as e:
                    print(f"  Could not save video ep{ep}: {e}")

        avg_success = float(np.mean(successes)) if successes else 0.0
        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        metrics["success_rate"] = avg_success
        metrics["avg_reward"] = avg_reward
        return metrics
