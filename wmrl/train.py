"""BC + PPO training loop for imagined rollouts inside FlowWorldModelVecEnv."""
import os.path as osp

import os
import random
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import trange, tqdm
from utils.vis import to_pil
from utils.misc import pretty_print_config
from policies.workspace.eval_utils import evaluate_policy_in_sim_env
from wmrl.logger import Logger
from wmrl.rl_utils import import_cls, parse_dataclass_with_optional_yaml, save_video
from wmrl.rl_types import RolloutBatch, StepResult
from wmrl.world_model_env import RunningRewardNormalizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Args:
    run_id: int = 0
    tag: str = ''
    env_cls: str = "wmrl.world_model_env.FlowWorldModelVecEnv"
    agent_cls: str = "wmrl.agent_critic.WMRLAgentWithCritic"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)

    seed: int = 1
    resume_ckpt: Optional[str] = None
    task: Optional[str] = None
    robot: Optional[str] = None
    control_mode: Optional[str] = None

    num_envs: int = 4
    env_batch_num: int = 1

    total_iterations: int = 50
    total_timesteps: int = 1_000_000
    num_steps: int = 6
    learning_rate: float = 1e-4
    gamma: float = 0.95
    mini_batch_size: int = 64
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.05
    norm_adv: bool = True
    reward_scale: float = 1.0
    reward_shaping: str = "none"
    reward_norm_ema_decay: float = 0.99
    action_chunk_size: int = 8
    n_obs_steps: int = 1
    camera_width: int = 126
    camera_height: int = 126

    bc_dataset_cfg: Dict[str, Any] = field(default_factory=dict)
    bc_batch_size: int = 64
    bc_num_workers: int = 4
    bc_loss_weight: float = 1.0
    bc_minibatches_per_update: int = 4

    policy_cls: str = "policies.policy.vla_bc_policy.VLABCPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    pretrained_ckpt: Optional[str] = None

    eval_freq: int = 25
    eval_num_episodes: int = 16
    eval_max_episode_steps: int = 100
    eval_sim_freq: int = 1
    eval_cpu: bool = True
    save_eval_video: bool = False
    eval_video_fps: int = 10
    eval_video_max_steps: int = 100
    save_freq: int = 25
    disp_freq: int = 10
    run_dir: str = "runs/wmrl"
    no_eval: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_tb: bool = False
    use_wandb: bool = False
    wandb_project: str = "world-model-rl"
    wandb_entity: Optional[str] = None

    # --- Critic baseline ---
    use_critic: bool = False
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    value_lr: float = 1e-4
    value_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])

    # --- DINO-WM baseline ---
    env_type: str = "flow"  # "flow" or "dino_wm"

    config_file: str = ""
    debug: bool = False
    run_initial_eval: bool = False



# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_env(args: Args, device: torch.device):
    # Auto-select multi-process env when multiple visible GPUs are available.
    env_cls_str = args.env_cls
    visible_gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    multi_gpu = visible_gpu_count > 1


    if multi_gpu and env_cls_str == "wmrl.world_model_env.FlowWorldModelVecEnv":
        env_cls_str = "wmrl.multi_process_env.MultiProcessWorldModelVecEnv"
        print(f"[red]Using multi-process env with {visible_gpu_count} visible GPUs[/red]")
    
    args.env_cls = env_cls_str  # Update for logging

    EnvCls = import_cls(env_cls_str)
    kw = dict(args.env_kwargs)
    kw.setdefault("num_envs", args.num_envs)
    kw.setdefault("device", device)
    kw.setdefault("chunk_size", args.action_chunk_size)
    kw.setdefault("n_obs_steps", args.n_obs_steps)
    kw.setdefault("reward_scale", args.reward_scale)
    kw.setdefault("camera_width", args.camera_width)
    kw.setdefault("camera_height", args.camera_height)
    kw.setdefault("env_batch_num", args.env_batch_num)
    kw.setdefault("seed", args.seed)
    kw.setdefault("flow_seed", args.seed)
    return EnvCls(**kw)


def _make_agent(args: Args, device: torch.device):
    if args.use_critic:
        from wmrl.agent_critic import WMRLAgentWithCritic
        AgentCls = WMRLAgentWithCritic
        args.agent_cls = "wmrl.agent_critic.WMRLAgentWithCritic"
    else:
        AgentCls = import_cls(args.agent_cls)
    action_dim = int(args.policy_kwargs.get('action_dim', 0))
    if action_dim <= 0:
        raise ValueError("policy_kwargs must contain a positive 'action_dim'")
    camera_h = args.env_kwargs.get('camera_height', args.camera_height)
    camera_w = args.env_kwargs.get('camera_width', args.camera_width)
    # Fall back to img_size when camera dims are not explicitly set.
    img_size = int(args.env_kwargs.get('img_size', 256))
    camera_h = int(camera_h) if camera_h else img_size
    camera_w = int(camera_w) if camera_w else img_size
    obs_shape = (args.n_obs_steps, 3, camera_h, camera_w)
    kw: Dict[str, Any] = dict(
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=device,
        lr=args.learning_rate,
        chunk_size=args.action_chunk_size,
        phase="joint",
        policy_cls=args.policy_cls,
        policy_kwargs=args.policy_kwargs,
        pretrained_ckpt=args.pretrained_ckpt,
        bc_dataset_cfg=args.bc_dataset_cfg,
        bc_batch_size=args.bc_batch_size,
        bc_num_workers=args.bc_num_workers,
        bc_loss_weight=args.bc_loss_weight,
        bc_minibatches_per_update=args.bc_minibatches_per_update,
        seed=args.seed,
    )
    kw.update(args.agent_kwargs)
    if args.use_critic:
        kw["value_hidden_dims"] = tuple(args.value_hidden_dims)
        kw["value_lr"] = args.value_lr
        kw["vf_coef"] = args.vf_coef
        kw["gae_lambda"] = args.gae_lambda
    return AgentCls(**kw)


def _make_eval_env(args: Args):
    from policies.train_loop import create_eval_env

    if args.task is None:
        raise ValueError("train_wmrl requires args.task for simulator evaluation")
    robot = args.robot or args.env_kwargs.get("robot_uid")
    if robot is None:
        raise ValueError("train_wmrl requires task/robot info for simulator evaluation")
    cameras = [str(c) for c in args.env_kwargs.get("cameras", ["front_lower_camera"])]
    cfg = SimpleNamespace(
        task=str(args.task),
        robot=str(robot),
        control_mode=str(args.control_mode or args.env_kwargs.get("control_mode", "pd_joint_pos")),
        env={"num_envs": 1, "shader_dir": args.env_kwargs.get("shader_dir", "rt-clean"),
             "camera_width": args.camera_width, "camera_height": args.camera_height,
             "cameras": cameras},
        eval={"cpu": args.eval_cpu, "num_episodes": args.eval_num_episodes,
              "max_episode_steps": args.eval_max_episode_steps,
              "save_video": args.save_eval_video,
              "n_vis": 1 if args.save_eval_video else 0,
              "video_dir": "eval_videos", "sim_freq": args.eval_sim_freq},
    )
    return create_eval_env(cfg), cameras


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def collect_rollout_with_state(
    env,
    agent,
    obs: torch.Tensor,
    state_obs: torch.Tensor,
    num_steps: int,
    deterministic: bool = False,
    reward_normalizer: Optional[RunningRewardNormalizer] = None,
    use_critic: bool = False,
) -> tuple[RolloutBatch, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    device = env.device
    num_envs = env.num_envs
    obs_shape = obs.shape[1:]
    state_shape = state_obs.shape[1:]
    action_dim = env.action_dim
    chunk_size = agent.chunk_size

    all_obs = torch.zeros(num_steps, num_envs, *obs_shape, device=device)
    all_state_obs = torch.zeros(num_steps, num_envs, *state_shape, device=device)
    all_actions = torch.zeros(num_steps, num_envs, chunk_size, action_dim, device=device)
    all_logprobs = torch.zeros(num_steps, num_envs, device=device)
    all_rewards = torch.zeros(num_steps, num_envs, device=device)
    all_dones = torch.zeros(num_steps, num_envs, device=device)
    all_terminal_obs = torch.zeros(num_steps, num_envs, *obs_shape, device=device)
    all_values = torch.zeros(num_steps, num_envs, device=device) if use_critic else None

    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_metrics: dict = {"successes": []}

    for t in range(num_steps):
        all_obs[t] = obs
        all_state_obs[t] = state_obs
        all_dones[t] = done.float()

        if use_critic:
            action_chunk, logprob, value = agent.get_action_value_and_logprob(
                obs, state_obs=state_obs,
                deterministic=deterministic,
            )
            assert all_values is not None
            all_values[t] = value
        else:
            action_chunk, logprob = agent.get_action_and_logprob(
                obs, state_obs=state_obs,
                deterministic=deterministic,
            )
        all_actions[t] = action_chunk
        all_logprobs[t] = logprob

        result: StepResult = env.step_chunked(agent.unnormalize_action(action_chunk))
        obs, state_obs, done = result.obs, result.info["state_history"], result.done

        if reward_normalizer is not None:
            all_rewards[t] = reward_normalizer.normalize(-result.reward) 
        else:
            all_rewards[t] = result.reward

        # Store pre-reset terminal obs for visualization.
        if "pre_reset_obs" in result.info:
            pre_reset_mask = result.info["pre_reset_done_mask"]
            all_terminal_obs[t, pre_reset_mask] = result.info["pre_reset_obs"]

        done_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            ep_metrics["successes"].extend(
                result.success[done_idx].float().detach().cpu().tolist()
            )

    # Bootstrap value for GAE (critic only).
    next_value = None
    if use_critic:
        next_value = agent.get_bootstrap_value(obs, state_obs=state_obs)

    batch = RolloutBatch(
        obs=all_obs, actions=all_actions, logprobs=all_logprobs,
        rewards=all_rewards, dones=all_dones,
        next_obs=obs, next_done=done.float(),
        state_obs=all_state_obs, next_state_obs=state_obs,
        terminal_obs=all_terminal_obs,
        values=all_values,
        next_value=next_value,
    )
    return batch, obs, state_obs, done, ep_metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    env, agent, args: Args,
    eval_seeds: list[int],
    cameras: list[str],
    video_path: str | None = None,
    deterministic: bool = True,
) -> dict:
    is_train = agent.policy.training
    agent.policy.eval()
    # Evaluation uses real sim images — disable DINO skip.
    _prev_skip = getattr(agent.policy, 'skip_dino_preprocess', False)
    if _prev_skip:
        agent.policy.skip_dino_preprocess = False
    # Store deterministic flag so predict_action can read it.
    _prev_deterministic = getattr(agent.policy, '_deterministic_eval', None)
    agent.policy._deterministic_eval = deterministic
    metrics, all_ep_frames = evaluate_policy_in_sim_env(
        env,
        agent.policy,
        eval_seeds=eval_seeds,
        max_episode_steps=args.eval_max_episode_steps,
        sim_freq=args.eval_sim_freq,
        cameras=cameras,
        device=agent.device,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        n_vis=1 if video_path else 0,
        video_max_steps=args.eval_video_max_steps,
        progress_desc="Eval",
        reset_policy_fn=agent.reset,
    )
    agent.policy.train(is_train)
    if _prev_skip:
        agent.policy.skip_dino_preprocess = True
    agent.policy._deterministic_eval = _prev_deterministic
    if video_path is not None:
        for frames in all_ep_frames:
            if frames is not None and len(frames) > 0:
                save_video(frames, output_path=video_path, fps=args.eval_video_fps)
                metrics["video_frames"] = float(len(frames))
                break
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: Args) -> list[float]:
    """Run BC+RL training. Returns the list of eval success rates (initial first, then per-eval-iter)."""
    assert args.config_file, "train_wmrl requires --config-file <path> argument"

    eval_success_rates: list[float] = []

    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    exp_name = f"{osp.basename(args.config_file).split('.')[0]}"

    if args.tag:
        exp_name += f"-{args.tag}"
    
    exp_name += f"-run{args.run_id}" 
    run_name = f"{exp_name}/{int(time.time())}"
    if args.debug:
        run_name = f"DEBUG/{run_name}"
        args.num_envs = 8
        args.num_steps = min(args.num_steps, 4)
        args.eval_num_episodes = 5
        args.eval_freq = 1
        args.bc_dataset_cfg['end_traj_id'] = 10
        args.env_kwargs['end_traj_id'] = 10
        args.save_freq = 1
        args.total_timesteps = min(args.total_timesteps, 1024)
        args.mini_batch_size = min(args.mini_batch_size, 4)
        args.bc_minibatches_per_update = 1
        args.use_wandb = False
        args.use_tb = False

    run_path = os.path.join(args.run_dir, run_name)
    os.makedirs(run_path, exist_ok=True)

    # Persist and display the effective run configuration.
    cfg_yaml = OmegaConf.to_yaml(OmegaConf.create(vars(args)))
    cfg_path = os.path.join(run_path, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_yaml)

    pretty_print_config(vars(args))

    logger = Logger(
        log_dir=run_path, use_tb=args.use_tb, use_wandb=args.use_wandb,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_config={**vars(args), "name": run_name} if args.use_wandb else None,
    )

    def _video_path(tag: str) -> str | None:
        if not args.save_eval_video:
            return None
        return os.path.join(run_path, "eval_videos", f"{tag}.mp4")

    def _debug_rollout_path(iteration: int) -> str:
        return os.path.join(run_path, "debug_rollouts", f"iter_{iteration:06d}.html")

    eval_seeds = [42 + i for i in range(args.eval_num_episodes)] # DO NOT CHANGE THIS LINE, I WANT TO FIX EVAL SEEDS TO A KNOWN SET FOR CONSISTENT EVALUATION ACROSS RUNS

    eval_env, eval_cameras = (_make_eval_env(args) if not args.no_eval else (None, []))
    agent = _make_agent(args, device)

    if args.resume_ckpt is not None:
        agent.load_state_dict(torch.load(args.resume_ckpt, map_location=device))
        print("Resumed from checkpoint: " + args.resume_ckpt)

    # --- Initial evaluation ---
    if eval_env is not None and args.run_initial_eval:
        print("[blue]Running initial evaluation...[/blue]")
        init_eval = evaluate(eval_env, agent, args, eval_seeds, eval_cameras,
                             video_path=_video_path("init_eval"),
                             deterministic=True)
        logger.scalars(init_eval, step=0)
        logger.log(f"initial eval: {init_eval}")
        eval_success_rates.append(float(init_eval.get("success_rate", float("nan"))))

    env = _make_env(args, device)

    env_steps_per_iter = args.num_envs * args.num_steps
    num_iterations = min(int(args.total_timesteps // env_steps_per_iter), args.total_iterations)
    if num_iterations <= 0:
        raise ValueError("num_iterations must be > 0; adjust total_timesteps")

    global_step = 0
    start_time = time.time()
    obs = env.reset()
    state_obs = env.get_state_history()

    reward_normalizer: Optional[RunningRewardNormalizer] = None
    if args.reward_shaping == "running_norm":
        reward_normalizer = RunningRewardNormalizer(ema_decay=args.reward_norm_ema_decay)

    logger.log("=== wmrl ===")
    logger.log(f"env_cls={args.env_cls}  agent_cls={args.agent_cls}")
    logger.log(
        f"num_envs={args.num_envs} seed={args.seed}  "
    )
    logger.log(f"env_steps/iter={env_steps_per_iter}  num_iterations={num_iterations}  run_path={run_path}")

    # --- Training loop ---
    for iteration in trange(0, num_iterations):
        iter_start = time.perf_counter()

        batch, obs, state_obs, _done, ep_metrics = collect_rollout_with_state(
            env, agent, obs, state_obs, args.num_steps,
            reward_normalizer=reward_normalizer,
            use_critic=args.use_critic,
        )
        # if args.debug or iteration == 0:
        #     html_render_path = _debug_rollout_path(iteration)
        #     print(f"Saving debug rollout visualization to {html_render_path}")
        #     batch.render(html_render_path, img_size=128, max_envs=8)

        global_step += env_steps_per_iter

        metrics = agent.update(
            batch, gamma=args.gamma,
            clip_coef=args.clip_coef, ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm, mini_batch_size=args.mini_batch_size,
            update_epochs=args.update_epochs, norm_adv=args.norm_adv,
            target_kl=args.target_kl,
            update_step=global_step,
            gae_lambda=args.gae_lambda if args.use_critic else 0.0,
            vf_coef=args.vf_coef if args.use_critic else 0.0,
        )

        sps = int(global_step / max(time.time() - start_time, 1e-6))
        log_dict = {
            "charts/SPS": sps,
            "charts/mean_return": float(batch.rewards.mean().item()),
            "charts/mean_success": float(np.mean(ep_metrics["successes"])) if ep_metrics["successes"] else 0.0,
            "losses/policy_loss": metrics.policy_loss,
            "losses/entropy": metrics.entropy,
            "losses/approx_kl": metrics.approx_kl,
            "losses/clipfrac": metrics.clipfrac,
            "losses/bc_loss": metrics.bc_loss,
        }
        if args.use_critic:
            log_dict["losses/value_loss"] = metrics.value_loss
        vf_log = f" vf={metrics.value_loss:.4f}" if args.use_critic else ""
        logger.scalars(log_dict, step=iteration)
        logger.log(
            f"[{iteration}/{num_iterations}] step={global_step} phase={agent.phase} SPS={sps} "
            f"pg={metrics.policy_loss:.4f}{vf_log} bc={metrics.bc_loss:.4f} "
            f"kl={metrics.approx_kl:.4f} t={time.perf_counter() - iter_start:.1f}s", console = tqdm.write if iteration % args.disp_freq == 0 else None
        )

        if eval_env is not None and (iteration % args.eval_freq == 0 or iteration == num_iterations - 1) and agent.phase == 'joint' and iteration > 0:
            print(f"[blue]Running evaluation at iteration {iteration}...[/blue]")
            eval_metrics = evaluate(eval_env, agent, args, eval_seeds, eval_cameras,
                                    video_path=_video_path(f"iter_{iteration:06d}"))
            logger.scalars(eval_metrics, step=iteration)
            logger.log(f"iter={iteration}  eval: {eval_metrics}", console=tqdm.write)
            eval_success_rates.append(float(eval_metrics.get("success_rate", float("nan"))))

        if args.save_freq > 0 and iteration % args.save_freq == 0 and iteration > 0:
            ckpt_path = os.path.join(run_path, f"ckpt_{iteration}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            logger.log(f"saved {ckpt_path}")

    final_path = os.path.join(run_path, "final.pt")
    torch.save(agent.state_dict(), final_path)
    logger.log(f"Saved final checkpoint to {final_path}")

    env.close()
    if eval_env is not None:
        eval_env.close()
    logger.close()
    return eval_success_rates


if __name__ == "__main__":
    train(parse_dataclass_with_optional_yaml(Args))