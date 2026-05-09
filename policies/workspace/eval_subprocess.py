"""Standalone subprocess script for out-of-process policy evaluation.

Launched by TrainVLABCWorkspace to evaluate a checkpoint in a separate process,
avoiding ManiSkill/GPU memory leaks in the training process.

Usage:
    python -m policies.workspace.eval_subprocess \
        --checkpoint /path/to/eval_tmp.ckpt \
        --eval-config /path/to/eval_config.json
"""

import argparse
import json
import os
import sys

import imageio
import numpy as np
import torch
import dill
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser(description="Out-of-process policy evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--eval-config", type=str, required=True, help="Path to eval config JSON")
    args = parser.parse_args()

    # Load eval config
    with open(args.eval_config, "r") as f:
        eval_cfg = json.load(f)

    eval_seeds = eval_cfg["eval_seeds"]
    max_episode_steps = eval_cfg["max_episode_steps"]
    sim_freq = eval_cfg["sim_freq"]
    cameras = eval_cfg["cameras"]
    device_str = eval_cfg["device"]
    img_size = eval_cfg["img_size"]
    n_vis = eval_cfg["n_vis"]
    save_video = eval_cfg["save_video"]
    video_dir = eval_cfg["video_dir"]
    epoch = eval_cfg["epoch"]
    use_ema = eval_cfg["use_ema"]
    results_json_path = eval_cfg["results_json_path"]

    device = torch.device(device_str)

    # Load checkpoint and reconstruct policy
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]

    # Instantiate the policy from config
    import hydra
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["model"])
    policy.to(device)

    if use_ema and "ema_model" in payload["state_dicts"]:
        import copy
        ema_policy = copy.deepcopy(policy)
        ema_policy.load_state_dict(payload["state_dicts"]["ema_model"])
        ema_policy.to(device)
        policy = ema_policy

    policy.eval()

    # Create eval environment (this is the memory-heavy object we isolate)
    from policies.train_loop import create_eval_env
    env = create_eval_env(cfg)

    # Run evaluation
    from policies.workspace.eval_utils import evaluate_policy_in_sim_env

    num_episodes = len(eval_seeds)
    metrics, all_ep_frames = evaluate_policy_in_sim_env(
        env,
        policy,
        eval_seeds=eval_seeds,
        max_episode_steps=max_episode_steps,
        sim_freq=sim_freq,
        cameras=cameras,
        device=device,
        camera_height=img_size,
        camera_width=img_size,
        n_vis=n_vis if save_video else 0,
        progress_desc=f"Eval epoch {epoch}",
    )

    # Save videos
    video_paths = {}
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        for ep in range(num_episodes):
            saved_frames = all_ep_frames[ep]
            if saved_frames is None:
                continue
            try:
                video_path = os.path.join(
                    video_dir,
                    f"eval_epoch{epoch:05d}_ep{ep}_seed{eval_seeds[ep]}.mp4",
                )
                writer = imageio.get_writer(video_path, fps=20)
                for frame in saved_frames:
                    writer.append_data(frame)
                writer.close()
                video_paths[str(ep)] = video_path
            except Exception as e:
                print(f"  Could not save video ep{ep}: {e}", file=sys.stderr)

    # Write results JSON
    results = {
        "success_rate": metrics["success_rate"],
        "avg_reward": metrics["avg_reward"],
        "video_paths": video_paths,
    }
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w") as f:
        json.dump(results, f)

    # Cleanup: close env
    env.close()


if __name__ == "__main__":
    main()
