"""
Imitation Learning Policy Training & Evaluation Loop (Simplified)

Uses ActionNormalizer for state/action normalization, easydict for config,
and supports chunked action execution with a control frequency multiplier.

Usage:
    .venv/bin/python -m policies.train_loop --config policies/example.yaml --output_dir runs/my_run
    .venv/bin/python -m policies.train_loop --config policies/example.yaml --eval_only
    .venv/bin/python -m policies.train_loop --list-configs
"""
import os
import sys
import json
import argparse
import importlib
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from rich import print
from rich.table import Table
from rich.console import Console
from easydict import EasyDict as edict
from datalib.env_utils import get_background_ids, extract_rgbs_from_obs

try:
    import wandb
except ImportError:
    wandb = None

from utils.vis import to_pil
from src.datasets.trajectory_dataset import TrajectoryDataset, TrajectoryBatch
from utils.misc import load_config, move_to_device, apply_cli_overrides, pretty_print_config, edict_to_dict
from policies.action_normalizer import ActionNormalizer, Action, Observation
from policies.base_policy import BasePolicy


# ---------------------------------------------------------------------------
# Task + Robot registry
# ---------------------------------------------------------------------------


VALID_CONFIGS = [
    ("PokeCube-v2", "xarm6_robotiq"),
    ("PullCube-v2", "xarm6_robotiq"),
    ("PullCubeTool-v1", "xarm6_robotiq"),
    ("PushT-v2", "xarm6_robotiq"),
    ("RollBall-v1", "xarm6_robotiq"),
    ("PegInsertionSide-v1", "xarm6_robotiq"),

    ("PullCube-v2", "panda"),
    ("RollBall-v1", "panda"),
    ("PegInsertionSide-v1", "panda"),
    ("PushT-v2", "panda"),
    ("PokeCube-v2", "panda"),
    ("PullCubeTool-v1", "panda"),

    ("PullCube-v2", "ur10e_stick"),
    ("RollBall-v1", "ur10e_stick"),
    ("PushT-v2", "ur10e_stick"),
] 
PUSH_TASKS = ['PushT-v2', 'RollBall-v1']



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_policy(cfg, device="cuda") -> BasePolicy:
    """Dynamically import and instantiate a policy class from cfg.policy."""
    module_path, class_name = cfg.policy.module.split(":")
    args = cfg.policy.get("args", {})
    checkpoint = cfg.policy.get("checkpoint", None)

    print(f"Loading policy: {module_path}.{class_name}")
    mod = importlib.import_module(module_path)
    policy_cls = getattr(mod, class_name)
    policy = policy_cls(**args).to(device)

    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        policy.load_state_dict(state_dict)

    return policy


def prepare_rgbs(batch: TrajectoryBatch) -> torch.Tensor:
    """Apply foreground masking and normalization to batch RGBs.
    
    Input batch['rgbs'] shape: (B, T, CAM, 3, H, W) uint8
    Input batch['foreground_masks'] shape: (B, T, CAM, H, W)
    Returns: (B, CAM, 3, H, W) float, first frame only, masked and /255.
    """
    rgbs = batch['rgbs'].float()  # (B, T, CAM, 3, H, W)
    masks = batch['foreground_masks']  # (B, T, CAM, H, W)
    rgbs = rgbs * masks[:, :, :, None]  # broadcast mask over channel dim
    rgbs = rgbs[:, 0] / 255.0  # take first frame, normalize to [0,1] -> (B, CAM, 3, H, W)
    return rgbs


def create_eval_env(cfg):
    """Create a ManiSkill environment for evaluation."""
    import gymnasium as gym
    import datalib.src.tasks  # noqa: F401
    import datalib.src.robots  # noqa: F401
    assert cfg.env.get("num_envs", 1) == 1, "Evaluation environment currently only supports num_envs=1"

    eval_on_cpu = cfg.eval.get("cpu", False)
    env_kwargs = dict(
        obs_mode="state+rgb+segmentation",
        control_mode=cfg.control_mode,
        robot_uids=cfg.robot,
        num_envs=cfg.env.get("num_envs", 1),
        shader_dir=cfg.env.get("shader_dir", "rt-clean"),
        camera_width=cfg.env.get("camera_width", 512),
        camera_height=cfg.env.get("camera_height", 512),
        render_mode="rgb_array",
        include_all_cameras=True,
        max_episode_steps=1e6, # we gonna manage the episode length ourself
    )
    if eval_on_cpu:
        env_kwargs["sim_backend"] = "physx_cpu"
        # env_kwargs["render_backend"] = "cpu", render_backend shall always be GPU
    cameras = cfg.env.get("cameras", None)
    if cameras:
        env_kwargs["camera_names"] = cameras
    return gym.make(cfg.task, **env_kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    policy: BasePolicy,
    cfg,
    state_normalizer: ActionNormalizer,
    action_normalizer: ActionNormalizer,
    iteration: int,
    output_dir: str,
    eval_seeds: list[int]
):
    """Run evaluation rollouts. Returns dict with metrics."""
    num_episodes = cfg.eval.num_episodes
    save_video = cfg.eval.get("save_video", True)
    sim_freq = cfg.eval.get("sim_freq", 1)
    video_dir = os.path.join(output_dir, cfg.eval.get("video_dir", "videos"))

    env = create_eval_env(cfg)
    obs, _ = env.reset()

    cameras = cfg.dataset.cameras

    successes = []
    episode_rewards = []
    video_frames = []
    device = None
    assert len(eval_seeds) == num_episodes, "Length of eval_seeds must match num_episodes"

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=eval_seeds[ep])
        # pause here to check the env seed effect

        # Build background mask from segmentation
        device = obs['sensor_data'][cameras[0]]['segmentation'].device
        background_ids = get_background_ids(env).to(device)

        policy.reset()
        ep_reward = 0.0
        ep_frames = []
        max_steps = cfg.eval.max_episode_steps

        step = 0
        while step < max_steps:
            # Build observation for policy
            # what this does is to mask out the background in the RGB images using the segmentation masks, then normalize to [0,1]
            rgbs = extract_rgbs_from_obs(obs, background_ids, cameras) # (1, cams, 3, H, W)

            # Build a state batch dict for normalization
            qpos = env.agent.robot.qpos  # (num_envs, J)
            root_pose = env.agent.robot.pose.raw_pose  # (num_envs, 7)
            state_batch = {
                state_normalizer.state_source: qpos.unsqueeze(1),  # (B, 1, J)
                "root_poses": root_pose.unsqueeze(1),  # (B, 1, 7)
            }
            state_action = state_normalizer.normalize(state_batch)

            policy_obs = Observation(state=state_action, rgbs=rgbs)
            chunk_action: Action = policy.forward(policy_obs, return_loss=False)
            if chunk_action is None:
                print(f"  [red]Policy returned None action, assuming this is a test run! [/red]")
                chunk_action = Action(eef=None, gripper=torch.zeros((1, 5)).to(device),
                                    arm_joints=torch.rand(1, 5, action_normalizer._action_dim - int(action_normalizer._spec.has_controllable_gripper)).to(device)  - 0.5)
        
            chunk_action = action_normalizer.denormalize(chunk_action)  # (B, H, D)

            # chunk_action has shape (B, H, D) — iterate over H (chunk steps)
            chunk_size = chunk_action.shape[1]
            terminated = False
            for t in range(chunk_size):
                if step >= max_steps or terminated:
                    break

                # Control frequency: repeat same action ctrl_freq times
                for _sim_i in range(sim_freq):
                    done = _sim_i == (sim_freq - 1)  # only get obs and reward on last sim step
                    obs, reward, terminated, _, info = (env.step if done
                                else env.step_wo_obs)(chunk_action[:, t])
                        
                    terminated = terminated.any()
                    ep_reward += reward.sum().item()

                    if save_video and ep == 0:
                        frame = env.render()
                        if frame is not None:
                            ep_frames.append(frame)

                    if terminated: break
                step += 1
            successes.append(info["success"].item())
        episode_rewards.append(ep_reward)
        if ep_frames:
            video_frames.extend(ep_frames)

    # Save video
    if save_video and video_frames:
        os.makedirs(video_dir, exist_ok=True)
        try:
            import imageio
            video_path = os.path.join(video_dir, f"eval_iter{iteration:07d}.mp4")
            writer = imageio.get_writer(video_path, fps=20)
            for frame in video_frames:
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4:
                    frame = frame[0]
                writer.append_data(frame)
            writer.close()
            print(f"  Saved video: {video_path}")
        except Exception as e:
            print(f"  [red]Could not save video: {e}[/red]")

    env.close()

    avg_success = np.mean(successes) if successes else 0.0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

    return {"success_rate": float(avg_success), "avg_reward": float(avg_reward)}


def print_eval_table(eval_metrics: dict, iteration: int):
    """Print evaluation results as a Rich table."""
    console = Console()
    table = Table(title=f"Evaluation Results (iter {iteration})", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in eval_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


def save_eval_json(eval_metrics: dict, iteration: int, output_dir: str):
    """Append evaluation metrics to a JSON results file."""
    results_path = os.path.join(output_dir, "eval_results.json")
    results = []
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    results.append({"iteration": iteration, **eval_metrics})
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved eval results: {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Imitation Learning Training Loop")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    args, unknown = parser.parse_known_args()

    # ---- Load config as easydict ----
    raw_cfg = load_config(args.config)
    if args.debug:
        raw_cfg["debug"] = True
    if args.eval_only:
        raw_cfg["eval_only"] = True
    if unknown:  apply_cli_overrides(raw_cfg, unknown)

    cfg = edict(raw_cfg)

    if cfg.debug:
        cfg.eval.max_episode_steps = 10
        cfg.eval.num_episodes = 1
        cfg.eval.interval = 4
        cfg.training.num_workers = 0
        cfg.wandb.enabled = False

    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    
    output_dir = cfg.output_dir or args.output_dir or "runs/policy_training"
    output_dir = os.path.join(output_dir, f"{cfg.task}_{cfg.robot}", timestamp)
    cfg.output_dir = output_dir
    pretty_print_config(edict_to_dict(cfg))
        
    # ---- Validate task + robot ----
    assert (cfg.task, cfg.robot) in VALID_CONFIGS, f"Error: ({cfg.task}, {cfg.robot}) is not a valid configuration."
    if cfg.task in PUSH_TASKS and cfg.robot != 'ur10e_stick':
        cfg.robot += '_closed'
        print(f"  [yellow]Auto-adjusted robot to {cfg.robot} for push task[/yellow]")
    
    # ---- Output directory ----
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ---- Optional Weights & Biases logging ----
    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "policy-training"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", f"{cfg.task}_{cfg.robot}_{timestamp}"),
            group=wandb_cfg.get("group", None),
            tags=wandb_cfg.get("tags", None),
            dir=output_dir,
            config=raw_cfg,
        )

    # ---- Seed & Device ----
    setup_seed(cfg.training.seed)
    all_eval_seeds = [42 + i for i in range(cfg.eval.num_episodes)]
    device = cfg.training.device

    # ---- ActionNormalizers ----
    state_normalizer = ActionNormalizer(cfg.robot, cfg.control_mode, state_source="qpos")
    action_normalizer = ActionNormalizer(cfg.robot, cfg.control_mode, state_source="target_qpos")
    # When EEF, xyz + rpy + gripper 1 dim -> 7 dim
    # When joints, number of arm joints + gripper 1 dim -> action_dim
    cfg.policy.args.action_dim = action_normalizer.action_dim if cfg.control_mode == 'pd_joints_pos' else 7

    # ---- Policy ----
    policy = load_policy(cfg, device=device)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # ---- Eval-only mode ----
    if cfg.eval_only:
        print("[bold yellow]Running in eval-only mode[/bold yellow]")
        policy.eval()
        eval_metrics = evaluate(policy, cfg, state_normalizer, action_normalizer, iteration=0, output_dir=output_dir, eval_seeds=all_eval_seeds)
        print_eval_table(eval_metrics, 0)
        save_eval_json(eval_metrics, 0, output_dir)
        if wandb_run is not None:
            wandb.log({f"eval/{k}": float(v) for k, v in eval_metrics.items()}, step=0)
            wandb.finish()
        return

    # ---- Dataset ----
    print("Initializing dataset...")
    dataset = TrajectoryDataset(**cfg.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=TrajectoryDataset.collate_fn,
    )

    # ---- Optimizer ----
    opt_result = policy.configure_optimizers()
    if isinstance(opt_result, tuple):
        optimizer, scheduler = opt_result
    else:
        optimizer = opt_result
        scheduler = None

    # ---- Training settings ----
    max_iterations = cfg.training.max_iterations
    eval_interval = cfg.eval.get("interval", 5000)
    log_interval = cfg.training.get("log_interval", 50)

    # ---- Training loop ----
    policy.train()
    data_iter = iter(dataloader)
    running_log = {}
    best_success = 0.0
    start_time = time.time()

    for iteration in range(1, max_iterations + 1):
        batch = next(data_iter)
        batch: TrajectoryBatch = move_to_device(batch, device)

        # Prepare RGBs: mask background and normalize
        rgbs = prepare_rgbs(batch)  # (B, CAM, 3, H, W)

        # Normalize state and target actions
        state_action = state_normalizer.normalize(batch)
        target_action = action_normalizer.normalize(batch)

        # Build Observation
        for k in ['gripper', 'eef', 'arm_joints']:
            setattr(state_action, k, getattr(state_action, k)[:, :1]) # take first step only for state
        obs = Observation(state=state_action, rgbs=rgbs, actions=target_action)

        # Forward pass -> loss
        loss_dict = policy.forward(obs, return_loss=True)
        loss = loss_dict['total']

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accumulate log metrics
        for k, v in loss_dict.items():
            if k not in running_log:
                running_log[k] = []
            val = v.item() if isinstance(v, torch.Tensor) else v
            running_log[k].append(val)

        # Log
        if iteration % log_interval == 0:
            elapsed = time.time() - start_time
            eta_seconds = elapsed / iteration * (max_iterations - iteration)
            eta_str = f"{int(eta_seconds // 3600)}h{int((eta_seconds % 3600) // 60)}m"

            log_parts = [f"[bold]Iter {iteration}/{max_iterations}[/bold]"]
            log_parts.append(f"ETA: {eta_str}")
            train_metrics = {}
            for k, vals in running_log.items():
                avg = np.mean(vals)
                train_metrics[f"train/{k}"] = float(avg)
                log_parts.append(f"{k}: {avg:.6f}")
            print(" | ".join(log_parts))
            if wandb_run is not None:
                train_metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
                wandb.log(train_metrics, step=iteration)
            running_log = {}

        # Evaluate
        if iteration % eval_interval == 0:
            policy.eval()
            eval_metrics = evaluate(
                policy, cfg, state_normalizer, action_normalizer, iteration, output_dir, eval_seeds=all_eval_seeds
            )
            print_eval_table(eval_metrics, iteration)
            save_eval_json(eval_metrics, iteration, output_dir)
            if wandb_run is not None:
                wandb.log({f"eval/{k}": float(v) for k, v in eval_metrics.items()}, step=iteration)
            policy.train()

            # Save checkpoint
            ckpt_dir = os.path.join(output_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"policy_iter{iteration:07d}.pt")
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_metrics": eval_metrics,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

            # Save best
            if eval_metrics.get("success_rate", 0) > best_success:
                best_success = eval_metrics["success_rate"]
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": policy.state_dict(),
                        "eval_metrics": eval_metrics,
                    },
                    best_path,
                )
                print(f"  [bold green]New best! success={best_success:.3f}[/bold green]")
                if wandb_run is not None:
                    wandb.log({"eval/best_success_rate": float(best_success)}, step=iteration)

    # Final save
    final_path = os.path.join(output_dir, "ckpts", "final.pt")
    torch.save(
        {
            "iteration": max_iterations,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_path,
    )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
