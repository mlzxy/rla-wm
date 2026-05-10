from __future__ import annotations

from collections import deque
from typing import Any, Callable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from datalib.env_utils import extract_rgbs_from_obs, get_background_ids


def obs_to_eval_frame(
    obs: dict[str, Any],
    env: Any,
    cameras: Sequence[str],
    device: torch.device,
    camera_height: int,
    camera_width: int,
) -> np.ndarray:
    env_device = next(iter(obs["sensor_data"].values()))["rgb"].device
    bg_ids = get_background_ids(env).to(env_device)
    rgbs = extract_rgbs_from_obs(obs, bg_ids, list(cameras)).to(device)

    if rgbs.shape[-2] != camera_height or rgbs.shape[-1] != camera_width:
        num_cams = rgbs.shape[1]
        rgbs = rgbs.reshape(-1, 3, rgbs.shape[-2], rgbs.shape[-1])
        rgbs = torch.nn.functional.interpolate(
            rgbs,
            size=(camera_height, camera_width),
            mode="bilinear",
            align_corners=False,
        )
        rgbs = rgbs.reshape(1, num_cams, 3, camera_height, camera_width)

    cam_imgs = []
    for cam_idx in range(rgbs.shape[1]):
        img_np = rgbs[0, cam_idx].detach().cpu().numpy().transpose(1, 2, 0)
        cam_imgs.append(img_np)
    frame = np.concatenate(cam_imgs, axis=1)
    return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)


def build_eval_obs(
    obs: dict[str, Any],
    env: Any,
    cameras: Sequence[str],
    device: torch.device,
    camera_height: int,
    camera_width: int,
    obs_history: deque[tuple[torch.Tensor, torch.Tensor]],
    n_obs_steps: int,
) -> dict[str, torch.Tensor]:
    env_device = next(iter(obs["sensor_data"].values()))["rgb"].device
    bg_ids = get_background_ids(env).to(env_device)
    rgbs = extract_rgbs_from_obs(obs, bg_ids, list(cameras)).to(device)

    if rgbs.shape[-2] != camera_height or rgbs.shape[-1] != camera_width:
        num_cams = rgbs.shape[1]
        rgbs = rgbs.reshape(-1, 3, rgbs.shape[-2], rgbs.shape[-1])
        rgbs = torch.nn.functional.interpolate(
            rgbs,
            size=(camera_height, camera_width),
            mode="bilinear",
            align_corners=False,
        )
        rgbs = rgbs.reshape(1, num_cams, 3, camera_height, camera_width)

    qpos = env.agent.robot.qpos.to(device)
    obs_history.append((rgbs, qpos))
    while len(obs_history) < n_obs_steps:
        obs_history.appendleft(obs_history[0])

    imgs = torch.stack([item[0] for item in list(obs_history)[-n_obs_steps:]], dim=1)
    states = torch.stack([item[1] for item in list(obs_history)[-n_obs_steps:]], dim=1)
    return {"image": imgs, "state": states}


def _to_bool_done(done_value: Any) -> bool:
    if isinstance(done_value, torch.Tensor):
        if done_value.numel() == 1:
            return bool(done_value.item())
        return bool(done_value.any().item())
    return bool(done_value)


def _to_float_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return float(value.float().mean().item())
    return float(value)


@torch.no_grad()
def evaluate_policy_in_sim_env(
    env: Any,
    policy: Any,
    *,
    eval_seeds: Sequence[int],
    max_episode_steps: int,
    sim_freq: int,
    cameras: Sequence[str],
    device: torch.device,
    camera_height: int,
    camera_width: int,
    n_vis: int = 0,
    video_max_steps: int = 0,
    progress_desc: str = "Eval",
    reset_policy_fn: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], list[list[np.ndarray] | None]]:
    if len(eval_seeds) == 0:
        raise ValueError("eval_seeds must be non-empty")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be > 0")

    episode_rewards: list[float] = []
    successes: list[float] = []
    max_video_steps = video_max_steps if video_max_steps > 0 else max_episode_steps
    all_ep_frames: list[list[np.ndarray] | None] = [None] * len(eval_seeds)
    was_training = bool(getattr(policy, "training", False))
    if hasattr(policy, "eval"):
        policy.eval()

    total_steps = len(eval_seeds) * max_episode_steps
    pbar = tqdm(total=total_steps, desc=progress_desc, leave=False)
    try:
        for ep, seed in enumerate(eval_seeds):
            obs, _ = env.reset(seed=int(seed))
            if reset_policy_fn is not None:
                reset_policy_fn()
            else:
                reset_fn = getattr(policy, "reset", None)
                if callable(reset_fn):
                    reset_fn()

            ep_reward = 0.0
            record_this_ep = ep < n_vis
            ep_frames: list[np.ndarray] = []
            obs_history: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
                maxlen=int(policy.n_obs_steps)
            )
            step = 0
            terminated = False
            info: dict[str, Any] | None = None

            while step < max_episode_steps and not terminated:
                if record_this_ep and len(ep_frames) < max_video_steps:
                    ep_frames.append(
                        obs_to_eval_frame(
                            obs,
                            env,
                            cameras,
                            device,
                            camera_height,
                            camera_width,
                        )
                    )

                obs_dict = build_eval_obs(
                    obs,
                    env,
                    cameras,
                    device,
                    camera_height,
                    camera_width,
                    obs_history,
                    int(policy.n_obs_steps),
                )
                result = policy.predict_action(obs_dict)
                action_chunk = result["action"].to(device=env.agent.robot.qpos.device)
                chunk_size = action_chunk.shape[1]

                for action_idx in range(chunk_size):
                    if step >= max_episode_steps or terminated:
                        break
                    for sim_idx in range(sim_freq):
                        with_obs = sim_idx == (sim_freq - 1)
                        obs, reward, terminated_value, _truncated, info = (
                            env.step if with_obs else env.step_wo_obs
                        )(action_chunk[:, action_idx])
                        terminated = _to_bool_done(terminated_value)
                        if reward is not None:
                            ep_reward += _to_float_scalar(reward)
                        if terminated:
                            break
                    step += 1
                    pbar.update(1)

                    if (
                        record_this_ep
                        and not terminated
                        and step < max_episode_steps
                        and action_idx < chunk_size - 1
                        and len(ep_frames) < max_video_steps
                    ):
                        ep_frames.append(
                            obs_to_eval_frame(
                                obs,
                                env,
                                cameras,
                                device,
                                camera_height,
                                camera_width,
                            )
                        )

                if terminated:
                    break

            remaining = max_episode_steps - step
            if remaining > 0:
                pbar.update(remaining)

            success = _to_float_scalar(info.get("success") if info is not None else None)
            successes.append(success)
            episode_rewards.append(ep_reward)
            pbar.set_postfix(ep=ep, reward=f"{ep_reward:.2f}", succ=success)

            if record_this_ep and ep_frames:
                all_ep_frames[ep] = ep_frames
    finally:
        pbar.close()
        if was_training and hasattr(policy, "train"):
            policy.train()

    metrics = {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        # "per_episode_success": [float(s) for s in successes],
        # "per_episode_reward": [float(r) for r in episode_rewards],
        # "seeds": [int(s) for s in eval_seeds],
    }
    return metrics, all_ep_frames