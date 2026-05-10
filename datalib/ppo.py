import os


from collections import defaultdict
import sys
import json
import random
import time
from dataclasses import dataclass
from typing import Optional, Union, get_type_hints
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig
from gymnasium import vector as gym_vector

# Datalib specific imports
from .src import tasks, robots


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill (state-rl)"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    eval_num_episodes: int = 32
    """number of episodes to run in evaluation environments"""
    eval_num_runs: int = 16
    """number of evaluation runs to perform for statistical significance (only used when evaluate=True)"""
    no_eval: bool = False
    """if toggled, skip deliberate evaluation loop and rely on training metrics"""
    checkpoint: Optional[str] = None
    hparams: Optional[str] = None
    """path to a hyperparameters file to load (key: value format)"""
    """path to a pretrained checkpoint file to start evaluation/training from"""
    robot_uid: str = "panda"
    """the id of the robot"""
    base_run_dir: str = "runs/PPO"
    """the base directory to save runs"""

    # Algorithm specific arguments
    env_id: str = "PushT-v2"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    force_cpu_sim: bool = False
    """if toggled, force physx_cpu simulation backend (single env) even when CUDA is available"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 100
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = None
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # Torch optimizations
    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""
    no_progress: bool = False
    """whether to disable the rich progress bar."""
    metadata_dir: Optional[str] = None
    """directory to store JSON metadata for search coordination."""

    tensordict: bool = False
    """whether to use tensordict for storage and agent synchronization."""
    eval_output_dir: Optional[str] = None
    """directory to save evaluation videos"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    mask_obs: bool = False
    """whether to mask the 6th and 12th element of the observation (0-indexed 5 and 11)"""


class SpecificObservationMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.is_vector_env = getattr(env, "is_vector_env", False)

        # Calculate new shape
        if isinstance(self.observation_space, gym.spaces.Box):
            old_shape = self.observation_space.shape
            # Assuming 1D observation (plus batch dim if vector env)
            # If vector env, shape is (num_envs, obs_dim) or just (obs_dim, ) in single env??
            # Usually single env box shape is (N,). Vector env box shape is (M, N).
            # We assume the last dimension is the feature dimension.

            self.feature_dim = old_shape[-1]
            new_shape = list(old_shape)
            new_shape[-1] = self.feature_dim - 2

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(new_shape),
                dtype=self.observation_space.dtype,
            )
        elif isinstance(self.observation_space, gym.spaces.Dict):
            # Handle Dict space (e.g. for collect_all.py which uses obs_mode="state_dict" or similar)
            if "state" not in self.observation_space.spaces:
                raise ValueError(
                    "SpecificObservationMaskWrapper with Dict space requires 'state' key"
                )

            old_box = self.observation_space.spaces["state"]
            old_shape = old_box.shape
            self.feature_dim = old_shape[-1]
            new_shape = list(old_shape)
            new_shape[-1] = self.feature_dim - 2

            new_box = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=tuple(new_shape), dtype=old_box.dtype
            )
            self.observation_space.spaces["state"] = new_box
        else:
            raise NotImplementedError(
                f"Unsupported observation space: {type(self.observation_space)}"
            )

        # Update single_observation_space if it exists (e.g. for VectorEnv)
        if hasattr(self.env, "single_observation_space"):
            self.single_observation_space = self.env.single_observation_space
            if isinstance(self.single_observation_space, gym.spaces.Box):
                old_shape = self.single_observation_space.shape
                # Assume last dim is feature dim
                new_shape = list(old_shape)
                new_shape[-1] = old_shape[-1] - 2
                self.single_observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=tuple(new_shape),
                    dtype=self.single_observation_space.dtype,
                )
            elif isinstance(self.single_observation_space, gym.spaces.Dict):
                if "state" in self.single_observation_space.spaces:
                    old_box = self.single_observation_space.spaces["state"]
                    old_shape = old_box.shape
                    new_shape = list(old_shape)
                    new_shape[-1] = old_shape[-1] - 2
                    new_box = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=tuple(new_shape),
                        dtype=old_box.dtype,
                    )
                    # We need to copy and update the dict space
                    new_spaces = dict(self.single_observation_space.spaces)
                    new_spaces["state"] = new_box
                    self.single_observation_space = gym.spaces.Dict(new_spaces)

        all_indices = np.arange(self.feature_dim)
        self.keep_indices = np.delete(all_indices, [5, 11])

        # For torch slicing, we might need a tensor if we want to use efficient indexing
        # But simple slicing concatenation might be faster or clearer?
        # A boolean mask is also good.

    def observation(self, observation):
        # observation can be numpy or torch
        # It can be a Dict or a Box (array/tensor)

        if isinstance(observation, dict):
            state = observation["state"]
            observation["state"] = self._mask_item(state)
            return observation
        else:
            return self._mask_item(observation)

    def _mask_item(self, item):
        # item is vector or batch of vectors. Last dim is feature dim.
        if isinstance(item, torch.Tensor):
            return item[..., self.keep_indices]
        elif isinstance(item, np.ndarray):
            return item[..., self.keep_indices]
        else:
            return item


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, np.prod(envs.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class Logger:
    def __init__(
        self,
        log_wandb=False,
        tensorboard: SummaryWriter = None,
        log_file_path: str = None,
        text_log_path: str = None,
        console: Console = None,
    ) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
        self.log_file_path = log_file_path
        self.text_log_path = text_log_path
        self.console = console if console is not None else Console()
        if self.log_file_path is not None:
            dirname = os.path.dirname(self.log_file_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.log_file_path, "w") as f:
                f.write("step,tag,value\n")
        if self.text_log_path is not None:
            dirname = os.path.dirname(self.text_log_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

    def log(self, msg: str):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_msg = f"{timestamp} {msg}"
        self.console.print(msg)
        if self.text_log_path is not None:
            with open(self.text_log_path, "a") as f:
                f.write(formatted_msg + "\n")

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)
        if self.log_file_path is not None:
            if (
                tag.startswith("train/")
                or tag.startswith("eval/")
                or tag.startswith("final_eval/")
                or tag.startswith("final/")
            ) and tag.split("/")[-1] in [
                "success",
                "reward",
                "success_at_end",
                "success_rate",
            ]:
                with open(self.log_file_path, "a") as f:
                    f.write(f"{step},{tag},{scalar_value}\n")

    def close(self):
        if self.writer is not None:
            self.writer.close()


class RecordEpisodeSubset(RecordEpisode):
    def __init__(self, *args, max_recorded_envs: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_recorded_envs = min(max_recorded_envs, self.num_envs)
        self.video_nrows = int(np.sqrt(self.max_recorded_envs))

    def capture_image(self, infos=None):
        # We only want to capture images from a subset of environments
        # RecordEpisode.render() returns a tiled image if num_envs > 1
        # But RecordEpisode.capture_image calls self.env.render()
        # For ManiSkillVectorEnv, render() returns a batch of images [N, H, W, 3]

        # Bypass RecordEpisode.capture_image logic and implement our own subset logic
        # to ensure we only transfer the first max_recorded_envs to CPU.
        scene = self.env.unwrapped.scene
        scene.update_render(update_sensors=False, update_human_render_cameras=True)

        if not scene.human_render_cameras:
            return None

        # Pick the first camera (usually "render_camera")
        camera = list(scene.human_render_cameras.values())[0]

        # Fetch data as torch tensors (views of GPU buffers)
        camera.capture()
        pic_tensors = camera.get_obs(rgb=True)["rgb"]
        if pic_tensors is None:
            return None

        # pic_tensors is already a torch tensor of shape [N, H, W, 3] on GPU
        # SLICE BEFORE CPU COPY
        img_tensor = pic_tensors[: self.max_recorded_envs]

        # Now convert to numpy (this triggers the copy)
        img = img_tensor.cpu().numpy()

        if infos is not None:
            from mani_skill.utils.visualization.misc import put_info_on_image

            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)

        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=self.video_nrows)
        return img


def compute_gae(
    rewards,
    values,
    dones,
    next_value,
    next_done,
    final_values,
    gamma,
    gae_lambda,
    finite_horizon_gae,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    num_steps = rewards.shape[0]

    if finite_horizon_gae:
        lam_coef_sum = torch.zeros(rewards.shape[1], device=rewards.device)
        reward_term_sum = torch.zeros(rewards.shape[1], device=rewards.device)
        value_term_sum = torch.zeros(rewards.shape[1], device=rewards.device)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_not_done = 1.0 - next_done
                nextvalues = next_value
            else:
                next_not_done = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            real_next_values = next_not_done * nextvalues + final_values[t]

            lam_coef_sum = lam_coef_sum * next_not_done
            reward_term_sum = reward_term_sum * next_not_done
            value_term_sum = value_term_sum * next_not_done

            lam_coef_sum = 1 + gae_lambda * lam_coef_sum
            reward_term_sum = (
                gae_lambda * gamma * reward_term_sum + lam_coef_sum * rewards[t]
            )
            value_term_sum = (
                gae_lambda * gamma * value_term_sum + gamma * real_next_values
            )
            advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[
                t
            ]
    else:
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_not_done = 1.0 - next_done
                nextvalues = next_value
            else:
                next_not_done = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            real_next_values = next_not_done * nextvalues + final_values[t]
            delta = rewards[t] + gamma * real_next_values - values[t]
            advantages[t] = lastgaelam = (
                delta + gamma * gae_lambda * next_not_done * lastgaelam
            )

    return advantages, advantages + values


def update_step(
    agent,
    optimizer,
    b_obs,
    b_actions,
    b_logprobs,
    b_advantages,
    b_returns,
    b_values,
    clip_coef,
    norm_adv,
    ent_coef,
    vf_coef,
    max_grad_norm,
    clip_vloss,
):
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
    logratio = newlogprob - b_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    mb_advantages = b_advantages
    if norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if clip_vloss:
        v_loss_unclipped = (newvalue - b_returns) ** 2
        v_clipped = b_values + torch.clamp(
            newvalue - b_values,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()

    return (
        approx_kl,
        v_loss,
        pg_loss,
        entropy_loss,
        old_approx_kl,
        clipfrac,
        grad_norm,
    )


def check_agent_nans(agent, logger):
    for name, param in agent.named_parameters():
        if torch.isnan(param).any():
            logger.log(f"[CRITICAL] NaNs detected in agent parameter: {name}")
            return True
    return False


def train(args=None, callback=None, progress: Progress = None, task_id: TaskID = None):
    if args is None:
        args = tyro.cli(Args)

    if args.hparams is not None:
        print(f"Loading hyperparameters from {args.hparams}")
        with open(args.hparams, "r") as f:
            lines = f.readlines()

        type_hints = get_type_hints(Args)
        cli_args = [arg.split("=")[0] for arg in sys.argv[1:]]
        for line in lines:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if hasattr(args, key) and key not in ["hparams", "checkpoint", "evaluate"]:
                # Check if this key was provided on CLI
                cli_key = "--" + key.replace("_", "-")
                if cli_key in cli_args:
                    print(f"Skipping {key} from file (overridden by CLI)")
                    continue

                target_type = type_hints.get(key)
                # Handle basic types
                try:
                    if target_type is bool:
                        setattr(args, key, val.lower() == "true")
                    elif target_type is int:
                        setattr(args, key, int(val))
                    elif target_type is float:
                        setattr(args, key, float(val))
                    elif (
                        target_type is str
                        or target_type is Optional[str]
                        or target_type is Union[str, None]
                    ):
                        setattr(args, key, None if val == "None" else val)
                    elif (
                        hasattr(target_type, "__origin__")
                        and target_type.__origin__ is Union
                    ):
                        # Simple Union/Optional handling for int/float
                        inner_types = target_type.__args__
                        if int in inner_types:
                            setattr(args, key, None if val == "None" else int(val))
                        elif float in inner_types:
                            setattr(args, key, None if val == "None" else float(val))
                        else:
                            setattr(args, key, None if val == "None" else val)
                except Exception as e:
                    print(
                        f"Warning: Could not parse {key}: {val} as {target_type}: {e}"
                    )

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.robot_uid}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Configuration validation
    NO_GRIPPER_TASKS = ["StackCube-v1", "PegInsertionSide-v1"]
    if args.robot_uid == "ur10e_stick" and args.env_id in NO_GRIPPER_TASKS:
        raise ValueError(
            f"Robot 'ur10e_stick' cannot perform task '{args.env_id}' because it requires a gripper."
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.force_cpu_sim:
        device = torch.device("cpu")

    console = Console(force_terminal=not args.no_progress, quiet=args.no_progress)
    if args.evaluate:
        console.quiet = False

    # logger setup
    writer = None
    if not args.evaluate:
        writer = SummaryWriter(f"{args.base_run_dir}/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    logger = Logger(
        log_wandb=args.track and not args.evaluate,
        tensorboard=writer,
        log_file_path=f"{args.base_run_dir}/{run_name}/metrics.csv"
        if not args.evaluate
        else None,
        text_log_path=f"{args.base_run_dir}/{run_name}/log.txt",
        console=console,
    )

    if not args.evaluate:
        hps_path = f"{args.base_run_dir}/{run_name}/hparams.txt"
        os.makedirs(os.path.dirname(hps_path), exist_ok=True)
        with open(hps_path, "w") as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
        logger.log(f"Hyperparameters saved to {hps_path}")

    if args.track and not args.evaluate:
        import wandb

        config = vars(args)
        # env_cfg/eval_env_cfg will be filled after env creation if needed
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="PPO",
            tags=["ppo", "walltime_efficient"],
        )

    eval_success = 0.0
    # env setup
    env_kwargs = dict(
        obs_mode="state",
        # render_mode=None, #"rgb_array",
        sim_backend="physx_cpu" if args.force_cpu_sim else ("physx_cuda" if torch.cuda.is_available() else "physx_cpu"),
        robot_uids=args.robot_uid,
    )
    if env_kwargs["sim_backend"] == "physx_cpu":
        args.num_envs = 1
        args.eval_num_runs = args.num_eval_envs
        args.num_eval_envs = 1
        logger.log(f"Using CPU simulation backend with num_envs={args.num_envs} and num_eval_envs={args.num_eval_envs}")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    # Evaluation runs only on CPU sim (Sync/AsyncVectorEnv of physx_cpu + CPUGymWrapper).
    eval_env_kwargs = {**env_kwargs, "sim_backend": "physx_cpu" }
    use_cpu_eval = True
    use_async_eval = args.num_eval_envs > 1

    def cpu_make_env(eval_env: bool = True, render_mode=None):
        """Return a thunk that creates a single CPU env with CPUGymWrapper."""
        def thunk():
            reconfig = args.eval_reconfiguration_freq if eval_env else args.reconfiguration_freq
            ignore_term = args.eval_partial_reset if eval_env else not args.partial_reset
            kw = dict(eval_env_kwargs)
            kw["num_envs"] = 1
            if render_mode:
                kw["render_mode"] = render_mode
                kw["camera_width"] = 128
                kw["camera_height"] = 128
            env = gym.make(args.env_id, reconfiguration_freq=reconfig, **kw)
            env = CPUGymWrapper(env, ignore_terminations=ignore_term, record_metrics=True)
            return env
        return thunk

    vector_cls_eval = (
        gym_vector.SyncVectorEnv
        if args.num_eval_envs == 1
        else lambda x: gym_vector.AsyncVectorEnv(x, context="forkserver")
    )

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        render_mode=None,
        **env_kwargs,
    )
    logger.log(f"envs.single_action_space.shape {envs.single_action_space.shape}")
    logger.log(f"envs.observation_space.shape {envs.observation_space.shape}")

    eval_envs = None
    if not args.no_eval:
        eval_render = "rgb_array" if args.capture_video else None
        logger.log(
            f"Using {'Sync' if args.num_eval_envs == 1 else 'Async'}VectorEnv for eval with num_eval_envs={args.num_eval_envs} (physx_cpu, CPUGymWrapper); eval video {'enabled' if args.capture_video else 'disabled'}"
        )
        eval_fns = [
            cpu_make_env(eval_env=True, render_mode=eval_render)
            for _ in range(args.num_eval_envs)
        ]
        eval_envs = vector_cls_eval(eval_fns)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)  # two handle multiple robot
        if eval_envs is not None:
            eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"{args.base_run_dir}/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        if args.eval_output_dir:
            eval_output_dir = args.eval_output_dir
        logger.log(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (
                (x // args.num_steps) % args.save_train_video_freq == 0
            )
            envs = RecordEpisode(
                envs,
                output_dir=f"{args.base_run_dir}/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        if eval_envs is not None:
            eval_envs = RecordEpisodeSubset(
                eval_envs,
                output_dir=eval_output_dir,
                save_trajectory=args.evaluate,
                trajectory_name="trajectory",
                max_steps_per_video=args.num_eval_steps,
                video_fps=30,
                max_recorded_envs=8,
            )
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )
    if args.mask_obs:
        envs = SpecificObservationMaskWrapper(envs)

    if eval_envs is not None and args.mask_obs:
        eval_envs = SpecificObservationMaskWrapper(eval_envs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    if not args.evaluate:
        logger.log("Running training")
        if args.track:
            # Update wandb config with env details
            wandb.config.update(
                dict(
                    env_cfg=dict(
                        **env_kwargs,
                        num_envs=args.num_envs,
                        env_id=args.env_id,
                        reward_mode="normalized_dense",
                        env_horizon=max_episode_steps,
                        partial_reset=args.partial_reset,
                    ),
                    eval_env_cfg=dict(
                        **env_kwargs,
                        num_envs=args.num_eval_envs,
                        env_id=args.env_id,
                        reward_mode="normalized_dense",
                        env_horizon=max_episode_steps,
                        partial_reset=False,
                    ),
                )
            )
    else:
        logger.log("Running evaluation")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    if args.tensordict:
        import tensordict

        storage = tensordict.TensorDict(
            {
                "obs": torch.zeros(
                    (args.num_steps, args.num_envs)
                    + envs.single_observation_space.shape
                ),
                "actions": torch.zeros(
                    (args.num_steps, args.num_envs) + envs.single_action_space.shape
                ),
                "logprobs": torch.zeros((args.num_steps, args.num_envs)),
                "rewards": torch.zeros((args.num_steps, args.num_envs)),
                "dones": torch.zeros((args.num_steps, args.num_envs)),
                "values": torch.zeros((args.num_steps, args.num_envs)),
            },
            batch_size=[args.num_steps, args.num_envs],
        ).to(device)
    else:
        storage = {
            "obs": torch.zeros(
                (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                device=device,
            ),
            "actions": torch.zeros(
                (args.num_steps, args.num_envs) + envs.single_action_space.shape,
                device=device,
            ),
            "logprobs": torch.zeros((args.num_steps, args.num_envs), device=device),
            "rewards": torch.zeros((args.num_steps, args.num_envs), device=device),
            "dones": torch.zeros((args.num_steps, args.num_envs), device=device),
            "values": torch.zeros((args.num_steps, args.num_envs), device=device),
        }
    obs = storage["obs"]
    actions = storage["actions"]
    logprobs = storage["logprobs"]
    rewards = storage["rewards"]
    dones = storage["dones"]
    values = storage["values"]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs = None
    if eval_envs is not None:
        eval_obs, _ = eval_envs.reset(seed=args.seed)
        if use_cpu_eval:
            eval_obs = torch.from_numpy(np.asarray(eval_obs)).float().to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    console = Console()
    table = Table(title="PPO Arguments")
    table.add_column("Argument", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in vars(args).items():
        table.add_row(key, str(value))
    console.print(table)

    logger.log(f"####")
    logger.log(
        f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}"
    )
    logger.log(
        f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}"
    )
    logger.log(f"####")

    action_space_low, action_space_high = (
        torch.from_numpy(envs.single_action_space.low).to(device),
        torch.from_numpy(envs.single_action_space.high).to(device),
    )

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            agent.load_state_dict(ckpt["model"], strict=False)
            if not args.evaluate and "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
        else:
            agent.load_state_dict(ckpt)

    # Detached copy for inference
    agent_inference = Agent(envs).to(device)
    if args.tensordict:
        from tensordict import from_module

        agent_inference_p = from_module(agent).data
        agent_inference_p.to_module(agent_inference)
    else:
        agent_inference.load_state_dict(agent.state_dict())

    # Executables
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value
    gae_fn = compute_gae
    update_fn = update_step

    if args.compile:
        policy = torch.compile(policy)
        get_value = torch.compile(get_value, fullgraph=True)
        gae_fn = torch.compile(gae_fn, fullgraph=True)
        # We don't compile update_fn here because we might wrap it in CudaGraphModule
        # Or we can compile it if cudagraphs is false
        if not args.cudagraphs:
            update_fn = torch.compile(update_fn)

    if args.cudagraphs:
        if not args.tensordict:
            raise ValueError("Cudagraphs optimization currently requires --tensordict.")
        from tensordict.nn import CudaGraphModule

        # For CudaGraphModule, we need TensorDict versions
        # This is a bit complex for ppo.py because of the many arguments.
        # For now, let's at least support it for policy and gae if possible
        # or stick to torch.compile which is usually sufficient and more robust.
        policy = CudaGraphModule(policy)
        # gae and update need TensorDict wrapping similar to old_ppo_fast.py
        # Skipping CudaGraphModule for gae/update for now to avoid major refactors
        # unless explicitly requested or if torch.compile isn't enough.
        pass

    latest_metrics = {"success": 0.0, "reward": 0.0}

    # Progress bar setup
    internal_progress = False
    if progress is None and not args.no_progress:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        task_id = progress.add_task("Training", total=args.num_iterations)
        internal_progress = True

    for iteration in range(1, args.num_iterations + 1):
        if not args.tensordict:
            agent_inference.load_state_dict(agent.state_dict())
        # check_agent_nans(agent, logger)
        # print(f"Epoch: {iteration}, global_step={global_step}") # Replaced by pbar
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        if eval_envs is not None and iteration % args.eval_freq == 1:
            # Unified evaluation: single run or multiple runs (eval_num_runs > 1)
            num_runs = args.eval_num_runs
            multi_run = num_runs > 1
            if multi_run:
                logger.log(f"Running {num_runs} evaluation runs for statistical significance")
            else:
                logger.log("Evaluating")
            all_run_metrics = defaultdict(list) if multi_run else None
            total_episodes = 0

            for run_idx in range(num_runs):
                if multi_run:
                    logger.log(f"Evaluation run {run_idx + 1}/{num_runs}")
                eval_obs, _ = eval_envs.reset()
                if use_cpu_eval:
                    eval_obs = torch.from_numpy(np.asarray(eval_obs)).float().to(device)
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(args.num_eval_steps):
                    with torch.no_grad():
                        eval_action = agent.get_action(eval_obs, deterministic=True)
                        if use_cpu_eval and torch.is_tensor(eval_action):
                            eval_action = eval_action.cpu().numpy()
                        (
                            eval_obs,
                            eval_rew,
                            eval_terminations,
                            eval_truncations,
                            eval_infos,
                        ) = eval_envs.step(eval_action)
                        if use_cpu_eval:
                            eval_obs = torch.from_numpy(np.asarray(eval_obs)).float().to(device)

                        if "final_info" in eval_infos and (eval_truncations.any() or eval_terminations.any()):
                            mask = eval_infos["_final_info"]
                            num_episodes += mask.sum()
                            for final_info in eval_infos["final_info"][mask.nonzero()[0]]:
                                for k, v in final_info["episode"].items():
                                    eval_metrics[k].append(torch.as_tensor(v).float())

                total_episodes += num_episodes
                run_label = f"Run {run_idx + 1}: " if multi_run else ""
                logger.log(
                    f"{run_label}Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes"
                )

                if multi_run:
                    for k, v in eval_metrics.items():
                        if len(v) > 0:
                            run_mean = torch.stack(v).float().mean().item()
                            all_run_metrics[k].append(run_mean)
                else:
                    eval_callback_metrics = {}
                    for k, v in eval_metrics.items():
                        mean = torch.stack(v).float().mean()
                        if logger is not None:
                            logger.add_scalar(f"eval/{k}", mean, global_step)
                        logger.log(f"eval_{k}_mean={mean}")
                        eval_callback_metrics[k] = mean.item()
                        if k == "success_once":
                            eval_success = mean.item()

            if multi_run:
                eval_callback_metrics = {}
                logger.log(
                    f"\n#### Aggregated Evaluation Results over {num_runs} runs ({total_episodes} total episodes) ####"
                )
                for k, v in all_run_metrics.items():
                    if len(v) > 0:
                        mean_val = np.mean(v)
                        std_val = np.std(v)
                        eval_callback_metrics[k] = mean_val
                        logger.log(f"eval_{k}: mean={mean_val:.4f}, std={std_val:.4f}")
                        if logger is not None:
                            logger.add_scalar(f"eval/{k}_mean", mean_val, global_step)
                            logger.add_scalar(f"eval/{k}_std", std_val, global_step)
                        if k == "success_once":
                            eval_success = mean_val

            if args.evaluate:
                if not eval_callback_metrics:
                    console.print(
                        "[yellow]No episodes finished during evaluation. No metrics to display.[/yellow]"
                    )
                else:
                    if multi_run:
                        table = Table(title=f"Evaluation Summary ({num_runs} runs)")
                        table.add_column("Metric", style="cyan")
                        table.add_column("Mean", style="magenta")
                        table.add_column("Std", style="yellow")
                        table.add_column("Min", style="green")
                        table.add_column("Max", style="red")
                        for k in eval_callback_metrics.keys():
                            mean_val = eval_callback_metrics[k]
                            std_val = np.std(all_run_metrics[k])
                            min_val = np.min(all_run_metrics[k])
                            max_val = np.max(all_run_metrics[k])
                            table.add_row(k, f"{mean_val:.4f}", f"{std_val:.4f}", f"{min_val:.4f}", f"{max_val:.4f}")
                    else:
                        table = Table(title="Evaluation Summary")
                        table.add_column("Metric", style="cyan")
                        table.add_column("Value", style="magenta")
                        for k, v in eval_callback_metrics.items():
                            table.add_row(k, f"{v:.4f}")
                    console.print(table)
                if callback is not None:
                    callback(global_step, eval_callback_metrics, agent)
                break

        # Use eval_callback_metrics if it exists (prioritized), otherwise latest_metrics
        if callback is not None and iteration % args.eval_freq == 0:
            # Prioritize eval metrics as requested by the user
            callback_metrics = (
                eval_callback_metrics
                if "eval_callback_metrics" in locals()
                else latest_metrics
            )
            callback(global_step, callback_metrics, agent)
        if args.save_model and iteration % args.eval_freq == 0:
            success_suffix = (
                f"_s{eval_success:.3f}" if "eval_success" in locals() else ""
            )
            model_path = (
                f"{args.base_run_dir}/{run_name}/ckpt_{iteration}{success_suffix}.pt"
            )
            latest_path = f"{args.base_run_dir}/{run_name}/latest.pt"
            save_dict = {"model": agent.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(save_dict, model_path)
            torch.save(save_dict, latest_path)
            logger.log(f"model saved to {model_path} and {latest_path}")

            if args.metadata_dir:
                metadata = {
                    "iteration": iteration,
                    "global_step": global_step,
                    "success": float(eval_success)
                    if "eval_success" in locals()
                    else 0.0,
                    "reward": float(latest_metrics["reward"]),
                    "ckpt_path": os.path.abspath(model_path),
                }
                os.makedirs(args.metadata_dir, exist_ok=True)
                metadata_path = os.path.join(
                    args.metadata_dir, f"metadata_{iteration}.json"
                )
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                logger.log(f"metadata saved to {metadata_path}")

        # Intermediate Metrics for Search Coordination
        if args.metadata_dir:
            metrics_data = {
                "iteration": iteration,
                "global_step": global_step,
                "reward": float(latest_metrics.get("reward", 0.0)),
                "success": float(
                    latest_metrics.get(
                        "success_once", latest_metrics.get("success", 0.0)
                    )
                ),
                "time": time.time(),
            }
            metrics_json_path = os.path.join(
                args.metadata_dir, f"metrics_{iteration}.json"
            )
            with open(metrics_json_path, "w") as f:
                json.dump(metrics_data, f, indent=4)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        # region: rollout

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = policy(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                clip_action(action)
            )

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    val = v[done_mask].float().mean().item()
                    logger.add_scalar(f"train/{k}", val, global_step)
                    if k in ["success", "success_once", "reward"]:
                        latest_metrics[k] = val

                success_val = latest_metrics.get(
                    "success_once", latest_metrics.get("success", 0.0)
                )
                reward_val = latest_metrics.get("reward", 0.0)
                if progress is not None:
                    progress.update(
                        task_id,
                        advance=0,
                        description=f"Training [cyan]success={success_val:.2f}[/] [magenta]reward={reward_val:.2f}[/]",
                    )
                with torch.no_grad():
                    final_values[
                        step, torch.arange(args.num_envs, device=device)[done_mask]
                    ] = get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time
        # endregion: rollout

        # bootstrap value according to termination and truncation
        # region: bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = gae_fn(
                rewards,
                values,
                dones,
                next_value,
                next_done,
                final_values,
                args.gamma,
                args.gae_lambda,
                args.finite_horizon_gae,
            )
        # endregion: bootstrap value
        # endregion: bootstrap value

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # region: optimize
        agent.train()
        b_inds = np.arange(
            args.batch_size
        )  # NOTE: batch_size = num_envs * num_steps, so all experiences are used!
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    approx_kl,
                    v_loss,
                    pg_loss,
                    entropy_loss,
                    old_approx_kl,
                    clipfrac,
                    grad_norm,
                ) = update_fn(
                    agent,
                    optimizer,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                    args.clip_coef,
                    args.norm_adv,
                    args.ent_coef,
                    args.vf_coef,
                    args.max_grad_norm,
                    args.clip_vloss,
                )
                clipfracs.append(clipfrac.item())
            # endregion: optimize

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar(
            "time/rollout_fps",
            args.num_envs * args.num_steps / rollout_time,
            global_step,
        )
        logger.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        if progress is not None and task_id is not None:
            progress.update(task_id, completed=iteration)

    if internal_progress and progress is not None:
        progress.stop()
    if not args.evaluate:
        if args.save_model:
            success_val = latest_metrics.get("success_once", latest_metrics["success"])
            success_suffix = f"_s{success_val:.3f}"
            model_path = f"runs/PPO/{run_name}/final_ckpt{success_suffix}.pt"
            latest_path = f"runs/PPO/{run_name}/latest.pt"
            save_dict = {"model": agent.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(save_dict, model_path)
            torch.save(save_dict, latest_path)
            logger.log(f"model saved to {model_path} and {latest_path}")

        if eval_envs is not None:
            # Final Evaluation
            logger.log("Running final evaluation (200 episodes)")
            eval_obs, _ = eval_envs.reset()
            if use_cpu_eval:
                eval_obs = torch.from_numpy(np.asarray(eval_obs)).float().to(device)
            eval_metrics = defaultdict(list)
            num_episodes = 0
            max_final_eval_steps = 200 * (args.num_eval_steps + 1)  # avoid infinite loop when env has no final_info
            step_count = 0
            while num_episodes < 200 and step_count < max_final_eval_steps:
                step_count += 1
                with torch.no_grad():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                    if use_cpu_eval and torch.is_tensor(eval_action):
                        eval_action = eval_action.cpu().numpy()
                    (
                        eval_obs,
                        eval_rew,
                        eval_terminations,
                        eval_truncations,
                        eval_infos,
                    ) = eval_envs.step(eval_action)
                    if use_cpu_eval:
                        eval_obs = torch.from_numpy(np.asarray(eval_obs)).float().to(device)
                    if "final_info" in eval_infos and "episode" in eval_infos["final_info"]:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

            logger.log(f"####")
            logger.log(f"Final Evaluation Results over {num_episodes} episodes:")
            final_success_rate = 0.0
            for k, v in eval_metrics.items():
                mean = torch.cat(v).float().mean().item()
                if logger is not None:
                    logger.add_scalar(f"final_eval/{k}", mean, global_step)
                logger.log(f"final_eval_{k}_mean={mean:.4f}")
                if k == "success_at_end":
                    final_success_rate = mean

            logger.log(f"####")
            logger.log(
                f"Training Finished! Final Average Success Rate: {final_success_rate:.4f}"
            )
            logger.log(f"####")
            if logger is not None:
                logger.add_scalar("final/success_rate", final_success_rate, global_step)

        if args.metadata_dir:
            # Use final_success_rate if evaluation ran, otherwise latest_metrics
            s_rate = (
                final_success_rate
                if "final_success_rate" in locals()
                else float(
                    latest_metrics.get(
                        "success_once", latest_metrics.get("success", 0.0)
                    )
                )
            )
            metadata = {
                "iteration": iteration,
                "global_step": global_step,
                "success": s_rate,
                "reward": float(latest_metrics["reward"]),
                "finished": True,
            }
            os.makedirs(args.metadata_dir, exist_ok=True)
            metadata_path = os.path.join(args.metadata_dir, "final_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.log(f"final metadata saved to {metadata_path}")

    if args.evaluate and args.metadata_dir:
        # Save metadata for evaluation mode
        # In evaluate mode, we ran one eval loop and 'eval_success' should be set
        # provided iteration % args.eval_freq == 1 was true (which it is for iteration 1)

        s_rate = float(eval_success) if "eval_success" in locals() else 0.0
        # If eval_callback_metrics is available, use it for more precision or other metrics
        if "eval_callback_metrics" in locals():
            s_rate = eval_callback_metrics.get(
                "success_once", eval_callback_metrics.get("success", s_rate)
            )

        metadata = {
            "iteration": iteration,
            "global_step": global_step,
            "success": s_rate,
            "reward": float(latest_metrics.get("reward", 0.0)),
            "finished": True,
        }
        os.makedirs(args.metadata_dir, exist_ok=True)
        metadata_path = os.path.join(args.metadata_dir, "final_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.log(f"final metadata saved to {metadata_path}")

    envs.close()
    if eval_envs is not None:
        eval_envs.close()

    final_return_metrics = (
        eval_callback_metrics if "eval_callback_metrics" in locals() else latest_metrics
    )
    return final_return_metrics


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
