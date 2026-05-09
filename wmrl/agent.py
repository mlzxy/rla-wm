"""Hybrid BC + REINFORCE-with-clipping agent built around VLABCPolicy."""

from __future__ import annotations

from collections import defaultdict
import os
import random
import sys
from typing import Any, Dict, Optional, cast

import numpy as np
from rich import print
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIFFUSION_POLICY_DIR = os.path.join(_ROOT_DIR, "third_party", "diffusion_policy")
if _DIFFUSION_POLICY_DIR not in sys.path:
    sys.path.insert(0, _DIFFUSION_POLICY_DIR)

from diffusion_policy.common.pytorch_util import dict_apply

from policies.dataset.maniskill_sequence_dataset import ManiSkillSequenceDataset
from wmrl.ppo_utils import compute_discounted_returns
from wmrl.rng_utils import (
    BC_DATALOADER_STREAM_ID,
    PPO_SHUFFLE_STREAM_ID,
    fold_in_seed,
    seeded_randperm,
)
from wmrl.rl_utils import import_cls
from wmrl.rl_types import RolloutBatch, UpdateMetrics


if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def _seed_dataloader_worker(_worker_id: int) -> None:
    """Align Python and NumPy worker RNG with the DataLoader seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WMRLAgent:
    """BC policy wrapper with REINFORCE-with-clipping (no critic)."""

    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        device: torch.device,
        lr: float,
        chunk_size: int,
        *,
        policy_cls: str = "policies.policy.vla_bc_policy.VLABCPolicy",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        pretrained_ckpt: Optional[str] = None,
        bc_dataset_cfg: Optional[Dict[str, Any]] = None,
        bc_batch_size: int = 64,
        bc_num_workers: int = 4,
        bc_loss_weight: float = 1.0,
        bc_minibatches_per_update: int = 4,
        phase: str = "joint",
        seed: int = 0,
        # Legacy kwargs accepted but unused (for config backward-compat)
        critic_latent_dim: int = 0,
    ):
        self.device = torch.device(device)
        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.bc_loss_weight = float(bc_loss_weight)
        self.bc_minibatches_per_update = int(bc_minibatches_per_update)
        self._last_bc_loss = 0.0

        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs["enable_rl_heads"] = True
        policy_kwargs['mlp_dropout'] = 0.0
        # Remove legacy critic kwargs that may come from old configs
        policy_kwargs.pop('critic_latent_dim', None)
        PolicyCls = import_cls(policy_cls)
        self.policy = PolicyCls(**policy_kwargs).to(self.device)

        if bc_dataset_cfg is None:
            raise ValueError("WMRLAgent requires bc_dataset_cfg for normalizer setup")
        self.bc_dataset_cfg = dict(bc_dataset_cfg)
        self.bc_dataset = ManiSkillSequenceDataset(**self.bc_dataset_cfg)
        self.policy.set_normalizer(self.bc_dataset.get_normalizer())

        if pretrained_ckpt is not None:
            self._load_policy_ckpt(pretrained_ckpt)

        bc_generator = torch.Generator()
        bc_generator.manual_seed(fold_in_seed(self.seed, BC_DATALOADER_STREAM_ID))
        self._bc_loader = DataLoader(
            self.bc_dataset,
            batch_size=int(bc_batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(bc_num_workers),
            pin_memory=True,
            persistent_workers=bool(bc_num_workers > 0),
            generator=bc_generator,
            worker_init_fn=_seed_dataloader_worker,
        )
        self._bc_iter = iter(self._bc_loader)
        self.phase = ""
        self.optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=self.lr)
        self._params: list[nn.Parameter] = []
        self.set_phase(phase)
        self.policy = self.policy.to(self.device)

    def reset(self) -> None:
        reset_fn = getattr(self.policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def set_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        if phase not in {"joint"}:
            raise ValueError(f"Unsupported WMRL phase: {phase}")

        self.policy.setup_rl_tuning()

        self._params = [param for param in self.policy.parameters() if param.requires_grad]
        if not self._params:
            raise RuntimeError(f"No trainable parameters configured for phase={phase}")

        self.optimizer = torch.optim.AdamW(self._params, lr=self.lr)
        self.phase = phase

    def _next_bc_batch(self) -> Dict[str, torch.Tensor]:
        try:
            batch = next(self._bc_iter)
        except StopIteration:
            self._bc_iter = iter(self._bc_loader)
            batch = next(self._bc_iter)
        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

    def _extract_policy_state(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if 'state_dicts' in state:
            return state['state_dicts'].get('ema_model', state['state_dicts']['model'])
        if "policy" in state and isinstance(state["policy"], dict):
            return state["policy"]
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            return state["model_state_dict"]
        return state

    def _load_policy_ckpt(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        policy_state = self._extract_policy_state(state)
        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        if missing:
            print(f"[WMRLAgent] missing policy keys from {ckpt_path}: {missing}")
        if unexpected:
            print(f"[WMRLAgent] unexpected policy keys from {ckpt_path}: {unexpected}")

    def _make_obs_dict(
        self,
        obs: torch.Tensor,
        state_obs: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if state_obs is None:
            raise ValueError("WMRLAgent requires state_obs for VLABCPolicy")
        return {
            "image": obs.to(self.device),
            "state": state_obs.to(self.device),
        }

    @torch.no_grad()
    def get_action_and_logprob(
        self,
        obs: torch.Tensor,
        state_obs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self._make_obs_dict(obs, state_obs)
        nobs = self.policy.normalizer.normalize(obs_dict)
        cond = self.policy.forward_features(nobs)
        mean, std = self.policy.forward_action_dist(cond)
        dist = Normal(mean, std)
        action_flat = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action_flat).sum(-1)
        action = action_flat.view(-1, self.chunk_size, self.action_dim)
        return action, log_prob

    @torch.no_grad()
    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.policy.normalizer["action"].unnormalize(action)

    def _evaluate_actions(
        self,
        obs: torch.Tensor,
        state_obs: Optional[torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self._make_obs_dict(obs, state_obs)
        nobs = self.policy.normalizer.normalize(obs_dict)
        cond = self.policy.forward_features(nobs)
        mean, std = self.policy.forward_action_dist(cond)
        dist = Normal(mean, std)
        flat_actions = actions.reshape(actions.shape[0], -1)
        log_prob = dist.log_prob(flat_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def update(
        self,
        batch: RolloutBatch,
        *,
        gamma: float,
        clip_coef: float,
        ent_coef: float,
        max_grad_norm: float,
        mini_batch_size: int,
        update_epochs: int,
        norm_adv: bool,
        target_kl: float | None,
        update_step: int = 0,
        # Legacy kwargs accepted but unused (for config backward-compat)
        gae_lambda: float = 0.0,
        vf_coef: float = 0.0,
        final_values: Optional[torch.Tensor] = None,
    ) -> UpdateMetrics:
        if batch.state_obs is None or batch.next_state_obs is None:
            raise ValueError("WMRLAgent.update requires state_obs and next_state_obs")

        returns = compute_discounted_returns(
            batch.rewards,
            batch.dones,
            batch.next_done,
            gamma,
        )

        t_steps, n_envs = batch.rewards.shape
        b_obs = batch.obs.reshape(-1, *batch.obs.shape[2:])
        b_state = batch.state_obs.reshape(-1, *batch.state_obs.shape[2:])
        b_actions = batch.actions.reshape(-1, *batch.actions.shape[2:])
        b_logprobs = batch.logprobs.reshape(-1)
        b_advantages = returns.reshape(-1)

        batch_size = t_steps * n_envs
        if mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be > 0, got {mini_batch_size}")
        if mini_batch_size > batch_size:
            raise ValueError(
                f"mini_batch_size ({mini_batch_size}) cannot exceed batch_size ({batch_size})"
            )

        totals = defaultdict(float)
        n_updates = 0
        approx_kl = 0.0
        stop_early = False

        for _epoch in range(update_epochs):
            inds = seeded_randperm(
                batch_size,
                self.seed,
                PPO_SHUFFLE_STREAM_ID,
                int(update_step),
                _epoch,
                device=self.device,
            )
            for start in range(0, batch_size, mini_batch_size):
                mb = inds[start:start + mini_batch_size]
                newlogprob, entropy = self._evaluate_actions(
                    b_obs[mb], b_state[mb], b_actions[mb]
                )

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                adv = b_advantages[mb]
                if norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                ent_loss = entropy.mean()
                if target_kl is not None and approx_kl > target_kl:
                    stop_early = True
                    break
                totals["kl"] += approx_kl
                totals["cf"] += clipfrac

                bc_loss = torch.zeros((), device=self.device)
                if self.bc_loss_weight > 0.0 and self.bc_minibatches_per_update > 0:
                    # BC batches use real images — disable DINO skip.
                    _prev_skip = getattr(self.policy, 'skip_dino_preprocess', False)
                    if _prev_skip:
                        self.policy.skip_dino_preprocess = False
                    for _ in range(self.bc_minibatches_per_update):
                        bc_loss = bc_loss + self.policy.compute_loss(self._next_bc_batch())
                    bc_loss = bc_loss / self.bc_minibatches_per_update
                    if _prev_skip:
                        self.policy.skip_dino_preprocess = True

                loss = pg_loss - ent_coef * ent_loss + self.bc_loss_weight * bc_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, max_grad_norm)
                self.optimizer.step()

                totals["pg"] += float(pg_loss.item())
                totals["ent"] += float(ent_loss.item())
                totals["bc"] += float(bc_loss.item())
                n_updates += 1

            if stop_early:
                break

        denom = max(n_updates, 1)
        self._last_bc_loss = totals["bc"] / denom
        return UpdateMetrics(
            policy_loss=totals["pg"] / denom,
            entropy=totals["ent"] / denom,
            approx_kl=totals["kl"] / denom,
            clipfrac=totals["cf"] / denom,
            bc_loss=totals['bc'] / denom
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "phase": self.phase,
            "last_bc_loss": self._last_bc_loss,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        policy_state = self._extract_policy_state(state)
        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        if missing:
            print(f"[WMRLAgent] missing checkpoint keys: {missing}")
        if unexpected:
            print(f"[WMRLAgent] unexpected checkpoint keys: {unexpected}")
        if isinstance(state, dict) and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except ValueError as exc:
                print(f"[WMRLAgent] optimizer state not loaded: {exc}")
        if isinstance(state, dict) and "last_bc_loss" in state:
            self._last_bc_loss = float(state["last_bc_loss"])


def _run_smoke_test() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_path = os.path.join(_ROOT_DIR, "runs/policy_outputs/0_bc/testing/.hydra/config.yaml")
    ckpt_path = os.path.join(_ROOT_DIR, "runs/policy_outputs/0_bc/testing/epoch=0019-success=0.240.ckpt")
    run_update = True
    smoke_batch_size = 2
    smoke_rollout_steps = 2

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    smoke_cfg = cast(
        Dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.create(
                {
                    "policy": OmegaConf.select(cfg, "policy"),
                    "dataset": OmegaConf.select(cfg, "dataset"),
                    "dataloader": OmegaConf.select(cfg, "dataloader"),
                    "training": OmegaConf.select(cfg, "training"),
                    "optimizer": OmegaConf.select(cfg, "optimizer"),
                    "img_size": OmegaConf.select(cfg, "img_size", default=0),
                }
            ),
            resolve=True,
        )
        or {},
    )
    policy_cfg = dict(smoke_cfg.get("policy", {}))
    dataset_cfg = dict(smoke_cfg.get("dataset", {}))
    dataloader_cfg = dict(smoke_cfg.get("dataloader", {}))
    training_cfg = dict(smoke_cfg.get("training", {}))
    optimizer_cfg = dict(smoke_cfg.get("optimizer", {}))

    dataset_cfg["end_traj_id"] = 10
    num_cameras = int(policy_cfg.get("num_cameras", len(dataset_cfg.get("cameras", ["front_lower_camera"]))))
    n_obs_steps = int(policy_cfg["n_obs_steps"])
    img_size = int(policy_cfg.get("img_size", smoke_cfg.get("img_size", 0)))
    agent_kwargs: Dict[str, Any] = {
        "obs_shape": (n_obs_steps, num_cameras, 3, img_size, img_size),
        "action_dim": int(policy_cfg["action_dim"]),
        "device": torch.device(training_cfg.get("device", "cpu")),
        "lr": float(optimizer_cfg.get("lr", 1e-4)),
        "chunk_size": int(policy_cfg["n_action_steps"]),
        "phase": "joint",
        "policy_cls": str(policy_cfg.pop("_target_", "policies.policy.vla_bc_policy.VLABCPolicy")),
        "policy_kwargs": policy_cfg,
        "bc_dataset_cfg": dataset_cfg,
        "bc_batch_size": int(dataloader_cfg.get("batch_size", 4)),
        "bc_num_workers": int(dataloader_cfg.get("num_workers", 0)),
        "bc_loss_weight": 1.0,
        "bc_minibatches_per_update": 1,
    }
    agent_kwargs["pretrained_ckpt"] = ckpt_path

    print(f"[WMRL smoke] config={config_path}")
    print(f"[WMRL smoke] checkpoint={ckpt_path}")
    print(
        f"[WMRL smoke] phase={agent_kwargs['phase']} "
        f"chunk_size={agent_kwargs['chunk_size']} action_dim={agent_kwargs['action_dim']} "
        f"device={agent_kwargs['device']} end_traj_id={dataset_cfg['end_traj_id']}"
    )

    ############## START ##############
    agent = WMRLAgent(**agent_kwargs)
    bc_batch = cast(Dict[str, Any], agent._next_bc_batch())
    bc_obs = cast(Dict[str, torch.Tensor], bc_batch["obs"])
    image = bc_obs["image"][:smoke_batch_size]
    state = bc_obs["state"][:smoke_batch_size]

    for name, tensor in (
        ("bc.obs.image", image),
        ("bc.obs.state", state),
        ("bc.action", bc_batch["action"][:smoke_batch_size]),
    ):
        print(
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite={bool(torch.isfinite(tensor).all().item())} "
            f"min={tensor.min().item():.6f} max={tensor.max().item():.6f}"
        )

    # Fake critic latent for smoke test.

    action, logprob = agent.get_action_and_logprob(
        image,
        state_obs=state,
        deterministic=True,
    )
    raw_action = agent.unnormalize_action(action)

    for name, tensor in (
        ("policy.action", action),
        ("policy.action_raw", raw_action),
        ("policy.logprob", logprob),
    ):
        print(
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite={bool(torch.isfinite(tensor).all().item())} "
            f"min={tensor.min().item():.6f} max={tensor.max().item():.6f}"
        )

    if run_update:
        obs_steps = []
        state_steps = []
        action_steps = []
        logprob_steps = []
        reward_steps = []
        done_steps = []
        for step_idx in range(smoke_rollout_steps):
            step_action, step_logprob = agent.get_action_and_logprob(
                image,
                state_obs=state,
                deterministic=False,
            )
            obs_steps.append(image)
            state_steps.append(state)
            action_steps.append(step_action)
            logprob_steps.append(step_logprob)
            reward_steps.append(
                torch.full((image.shape[0],), 0.1 * (step_idx + 1), device=agent.device)
            )
            done_steps.append(torch.zeros(image.shape[0], device=agent.device))

        rollout = RolloutBatch(
            obs=torch.stack(obs_steps),
            actions=torch.stack(action_steps),
            logprobs=torch.stack(logprob_steps),
            rewards=torch.stack(reward_steps),
            dones=torch.stack(done_steps),
            next_obs=image.clone(),
            next_done=torch.zeros(image.shape[0], device=agent.device),
            state_obs=torch.stack(state_steps),
            next_state_obs=state.clone(),
        )
        metrics = agent.update(
            rollout,
            gamma=0.95,
            clip_coef=0.2,
            ent_coef=0.0,
            max_grad_norm=0.5,
            mini_batch_size=min(smoke_batch_size, rollout.obs.shape[0] * rollout.obs.shape[1]),
            update_epochs=1,
            norm_adv=True,
            target_kl=0.1
        )
        print(
            "[WMRL smoke] update_metrics="
            f"policy_loss={metrics.policy_loss:.6f} "
            f"entropy={metrics.entropy:.6f} "
            f"approx_kl={metrics.approx_kl:.6f} "
            f"clipfrac={metrics.clipfrac:.6f}"
        )


if __name__ == "__main__":
    _run_smoke_test()