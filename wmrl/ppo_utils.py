"""Reusable REINFORCE-with-clipping math utilities independent of model architecture.

This module isolates policy-gradient training math from any specific policy
network.  The caller only needs to provide:

1) ``evaluate_actions(obs, actions) -> (log_prob, entropy)`` for
   policy evaluation on minibatches.

No value / critic network is used.  Advantages are computed as
normalised discounted returns (REINFORCE with clipping).

Key equations implemented here:

Discounted returns (no baseline):

    R_t = r_t + gamma * (1 - done_{t+1}) * R_{t+1}

Clipped surrogate objective per sample:

    ratio_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))
    L_clip_t = min(ratio_t * A_t,
                   clip(ratio_t, 1-eps, 1+eps) * A_t)

where A_t = R_t (optionally normalised).

Optimization target used in code:

    policy_loss = -E[L_clip_t]
    entropy_bonus = E[H[pi_new(.|s_t)]]
    total_loss = policy_loss - ent_coef * entropy_bonus

Approximate KL used for early stopping:

    approx_kl ~= E[(ratio_t - 1) - log(ratio_t)]

All tensors are expected to already be on the correct device.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor

from wmrl.rl_types import RolloutBatch, UpdateMetrics


EvaluateActionsFn = Callable[
    [Tensor, Tensor],
    tuple[Tensor, Tensor],
]

# Evaluate with critic: returns (log_prob, entropy, value)
EvaluateActionsWithValueFn = Callable[
    [Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]


def compute_discounted_returns(
    rewards: Float[Tensor, "T N"],
    dones: Float[Tensor, "T N"],
    next_done: Float[Tensor, " N"],
    gamma: float,
) -> Float[Tensor, "T N"]:
    """Compute discounted returns without a value baseline (REINFORCE style).

    Args:
        rewards: Tensor of shape ``(T, N)`` with per-step rewards.
        dones: Tensor of shape ``(T, N)`` where non-zero indicates episode end
            at that step.
        next_done: Tensor of shape ``(N,)`` done flag for the final next state.
        gamma: Discount factor.

    Returns:
        Tensor of shape ``(T, N)`` with discounted returns ``R_t``.

    Notes:
        Backward recursion:

            R_t = r_t + gamma * (1 - done_{t+1}) * R_{t+1}

        For ``t = T-1``, ``done_{t+1}`` is ``next_done`` and ``R_{t+1} = 0``.
    """
    t_steps, n_envs = rewards.shape
    returns = torch.zeros_like(rewards)
    future_return = torch.zeros(n_envs, device=rewards.device)
    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            next_nd = 1.0 - next_done.float()
        else:
            next_nd = 1.0 - dones[t + 1].float()
        future_return = rewards[t] + gamma * next_nd * future_return
        returns[t] = future_return
    return returns


def reinforce_clipped_update(
    batch: RolloutBatch,
    *,
    evaluate_actions: EvaluateActionsFn,
    optimizer: optim.Optimizer,
    params: list[nn.Parameter],
    device: torch.device,
    gamma: float = 0.8,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    mini_batch_size: int = 256,
    update_epochs: int = 4,
    norm_adv: bool = True,
    target_kl: float | None = 0.1,
) -> UpdateMetrics:
    """Run REINFORCE with clipped surrogate objective for one rollout batch.

    Uses discounted returns as advantages (no value baseline), with the same
    clipped ratio objective as PPO.

    Args:
        batch: Rollout tensors containing observations, actions, old log-probs,
            rewards, and done flags.
        evaluate_actions: Function mapping ``(obs, actions)`` to
            ``(log_prob, entropy)`` for the current policy.
        optimizer: Optimizer for policy parameters.
        params: Parameter list used for gradient clipping.
        device: Device used for permutation sampling.
        gamma: Discount factor for return estimation.
        clip_coef: Clipping epsilon for ratio clipping.
        ent_coef: Weight on entropy bonus.
        max_grad_norm: Gradient clipping threshold.
        mini_batch_size: Number of flattened samples per SGD minibatch.
        update_epochs: Number of epochs over the same rollout.
        norm_adv: Whether to normalize advantages inside each minibatch.
        target_kl: Optional early-stop threshold on approximate KL.

    Returns:
        ``UpdateMetrics`` with mean losses/statistics over performed updates.
    """
    t_steps, n_envs = batch.rewards.shape

    returns = compute_discounted_returns(
        batch.rewards,
        batch.dones,
        batch.next_done,
        gamma,
    )

    b_obs: Float[Tensor, "B *obs_shape"] = batch.obs.reshape(-1, *batch.obs.shape[2:])
    b_actions: Float[Tensor, "B K A"] = batch.actions.reshape(-1, *batch.actions.shape[2:])  # (B, chunk_size, action_dim)
    b_logprobs: Float[Tensor, " B"] = batch.logprobs.reshape(-1)
    b_advantages: Float[Tensor, " B"] = returns.reshape(-1)

    batch_size = t_steps * n_envs
    if mini_batch_size <= 0:
        raise ValueError(
            f"mini_batch_size must be > 0, got {mini_batch_size}"
        )
    if mini_batch_size > batch_size:
        raise ValueError(
            f"mini_batch_size ({mini_batch_size}) cannot exceed batch_size ({batch_size})"
        )

    total_pg = 0.0
    total_ent = 0.0
    total_kl = 0.0
    total_cf = 0.0
    n_updates = 0

    approx_kl = 0.0
    for _epoch in range(update_epochs):
        inds = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb = inds[start:end]

            newlogprob, entropy = evaluate_actions(b_obs[mb], b_actions[mb])
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            if target_kl is not None and approx_kl > target_kl:
                break

            mb_adv = b_advantages[mb]
            if norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg1, pg2).mean()

            ent_loss = entropy.mean()
            loss = pg_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

            total_pg += pg_loss.item()
            total_ent += ent_loss.item()
            total_kl += approx_kl
            total_cf += clipfrac
            n_updates += 1

        if target_kl is not None and approx_kl > target_kl:
            break

    n = max(n_updates, 1)
    return UpdateMetrics(
        policy_loss=total_pg / n,
        entropy=total_ent / n,
        approx_kl=total_kl / n,
        clipfrac=total_cf / n,
    )


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: Float[Tensor, "T N"],
    values: Float[Tensor, "T N"],
    dones: Float[Tensor, "T N"],
    next_value: Float[Tensor, " N"],
    next_done: Float[Tensor, " N"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[Tensor, "T N"], Float[Tensor, "T N"]]:
    """Compute GAE advantages and returns.

    Args:
        rewards: ``(T, N)`` per-step rewards.
        values: ``(T, N)`` predicted values at each step.
        dones: ``(T, N)`` done flags.
        next_value: ``(N,)`` bootstrap value after the last step.
        next_done: ``(N,)`` done flag for the state after the last step.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        advantages: ``(T, N)`` GAE advantages.
        returns: ``(T, N)`` = advantages + values (regression targets).
    """
    t_steps, n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(n_envs, device=rewards.device)

    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            next_non_terminal = 1.0 - next_done.float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1].float()
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_non_terminal * next_val - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO with critic (value head)
# ---------------------------------------------------------------------------


def ppo_clipped_update_with_critic(
    batch: RolloutBatch,
    *,
    evaluate_actions_with_value: EvaluateActionsWithValueFn,
    policy_optimizer: optim.Optimizer,
    value_optimizer: optim.Optimizer,
    policy_params: list[nn.Parameter],
    value_params: list[nn.Parameter],
    device: torch.device,
    gamma: float = 0.95,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    mini_batch_size: int = 256,
    update_epochs: int = 4,
    norm_adv: bool = True,
    target_kl: float | None = 0.1,
) -> UpdateMetrics:
    """PPO update with a separate value head (critic baseline).

    Like ``reinforce_clipped_update`` but uses GAE for advantages and trains
    a value head alongside the policy.

    Args:
        batch: Must have ``values`` and ``next_value`` populated.
        evaluate_actions_with_value: ``(obs, actions) -> (log_prob, entropy, value)``.
        policy_optimizer: Optimizer for policy parameters.
        value_optimizer: Optimizer for value head parameters.
        policy_params: Policy parameters for gradient clipping.
        value_params: Value head parameters for gradient clipping.
        device: Device for permutation sampling.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_coef: PPO clipping epsilon.
        vf_coef: Value loss weight.
        ent_coef: Entropy bonus weight.
        max_grad_norm: Max gradient norm.
        mini_batch_size: SGD minibatch size.
        update_epochs: Epochs over the rollout.
        norm_adv: Normalize advantages per minibatch.
        target_kl: Early stop on approx KL.

    Returns:
        ``UpdateMetrics`` with losses (including ``value_loss``).
    """
    if batch.values is None or batch.next_value is None:
        raise ValueError(
            "ppo_clipped_update_with_critic requires batch.values and batch.next_value"
        )

    t_steps, n_envs = batch.rewards.shape

    advantages, returns = compute_gae(
        batch.rewards, batch.values, batch.dones,
        batch.next_value, batch.next_done,
        gamma, gae_lambda,
    )

    b_obs = batch.obs.reshape(-1, *batch.obs.shape[2:])
    b_actions = batch.actions.reshape(-1, *batch.actions.shape[2:])
    b_logprobs = batch.logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    batch_size = t_steps * n_envs

    total_pg = 0.0
    total_vf = 0.0
    total_ent = 0.0
    total_kl = 0.0
    total_cf = 0.0
    n_updates = 0

    approx_kl = 0.0
    for _epoch in range(update_epochs):
        inds = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb = inds[start:end]

            newlogprob, entropy, newvalue = evaluate_actions_with_value(
                b_obs[mb], b_actions[mb],
            )
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            if target_kl is not None and approx_kl > target_kl:
                break

            mb_adv = b_advantages[mb]
            if norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # Policy loss (clipped surrogate)
            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg1, pg2).mean()

            # Value loss
            v_loss = 0.5 * (newvalue - b_returns[mb]).pow(2).mean()

            # Entropy
            ent_loss = entropy.mean()

            # Combined loss for policy
            policy_loss = pg_loss - ent_coef * ent_loss
            value_loss = vf_coef * v_loss

            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
            nn.utils.clip_grad_norm_(value_params, max_grad_norm)
            policy_optimizer.step()
            value_optimizer.step()

            total_pg += pg_loss.item()
            total_vf += v_loss.item()
            total_ent += ent_loss.item()
            total_kl += approx_kl
            total_cf += clipfrac
            n_updates += 1

        if target_kl is not None and approx_kl > target_kl:
            break

    n = max(n_updates, 1)
    return UpdateMetrics(
        policy_loss=total_pg / n,
        entropy=total_ent / n,
        approx_kl=total_kl / n,
        clipfrac=total_cf / n,
        value_loss=total_vf / n,
    )
