"""WMRLAgent with a standalone critic (value head) for PPO with GAE.

Wraps the existing :class:`WMRLAgent` and adds a :class:`ValueHead` that
predicts state values from the policy's ``forward_features()`` output.

The value head is **external** to the policy — no policy code is modified.
Value gradients do **not** flow into the frozen policy backbone because
``global_cond`` is ``.detach()``-ed before feeding to the value head during
rollout collection.  During the update, gradients flow through the value
head only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from wmrl.agent import WMRLAgent
from wmrl.critic_value_head import ValueHead
from wmrl.ppo_utils import compute_gae, ppo_clipped_update_with_critic
from wmrl.rng_utils import PPO_SHUFFLE_STREAM_ID, seeded_randperm
from wmrl.rl_types import RolloutBatch, UpdateMetrics


class WMRLAgentWithCritic(WMRLAgent):
    """WMRLAgent extended with a standalone value head for PPO+GAE.

    The value head takes the same ``forward_features()`` output as the policy
    but is trained independently via its own optimizer.

    Extra ``__init__`` kwargs beyond :class:`WMRLAgent`:
        value_hidden_dims: Hidden layer sizes for the value MLP.
        value_lr: Learning rate for the value head optimizer.
        vf_coef: Value loss coefficient in the combined loss.
        gae_lambda: GAE lambda for advantage estimation.
    """

    def __init__(
        self,
        *args: Any,
        value_hidden_dims: tuple[int, ...] = (256, 128),
        value_lr: float = 1e-4,
        vf_coef: float = 0.5,
        gae_lambda: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vf_coef = float(vf_coef)
        self.gae_lambda = float(gae_lambda)

        # Determine the conditioning dimension from the policy.
        cond_dim = int(self.policy._cond_dim)
        self.value_head = ValueHead(
            input_dim=cond_dim,
            hidden_dims=value_hidden_dims,
        ).to(self.device)

        self._value_params = list(self.value_head.parameters())
        self.value_optimizer = torch.optim.AdamW(
            self._value_params, lr=float(value_lr),
        )

    # ------------------------------------------------------------------
    # Rollout: action + value + logprob
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action_value_and_logprob(
        self,
        obs: torch.Tensor,
        state_obs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, compute log-prob and value.

        Returns:
            action: ``(N, chunk_size, action_dim)``
            log_prob: ``(N,)``
            value: ``(N,)``
        """
        obs_dict = self._make_obs_dict(obs, state_obs)
        nobs = self.policy.normalizer.normalize(obs_dict)
        cond = self.policy.forward_features(nobs)

        # Value from detached features (no gradient to policy backbone)
        value = self.value_head(cond.detach())

        # Action distribution
        mean, std = self.policy.forward_action_dist(cond)
        dist = Normal(mean, std)
        action_flat = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action_flat).sum(-1)
        action = action_flat.view(-1, self.chunk_size, self.action_dim)
        return action, log_prob, value

    @torch.no_grad()
    def get_bootstrap_value(
        self,
        obs: torch.Tensor,
        state_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value for GAE bootstrapping after the last step.

        Returns:
            ``(N,)`` value prediction.
        """
        obs_dict = self._make_obs_dict(obs, state_obs)
        nobs = self.policy.normalizer.normalize(obs_dict)
        cond = self.policy.forward_features(nobs)
        return self.value_head(cond.detach())

    # ------------------------------------------------------------------
    # Evaluation (with gradients for training)
    # ------------------------------------------------------------------

    def _evaluate_actions_with_value(
        self,
        obs: torch.Tensor,
        state_obs: Optional[torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions with policy log-prob, entropy, and value.

        Unlike rollout collection, gradients flow through the value head
        (but still not through the frozen policy backbone for value).
        """
        obs_dict = self._make_obs_dict(obs, state_obs)
        nobs = self.policy.normalizer.normalize(obs_dict)
        cond = self.policy.forward_features(nobs)

        # Value: detach cond so value gradients don't flow into policy backbone
        value = self.value_head(cond.detach())

        # Policy logprob + entropy
        mean, std = self.policy.forward_action_dist(cond)
        dist = Normal(mean, std)
        flat_actions = actions.reshape(actions.shape[0], -1)
        log_prob = dist.log_prob(flat_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    # ------------------------------------------------------------------
    # Update (PPO + GAE + critic)
    # ------------------------------------------------------------------

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
        gae_lambda: float = 0.0,
        vf_coef: float = 0.0,
        **kwargs: Any,
    ) -> UpdateMetrics:
        if batch.state_obs is None or batch.next_state_obs is None:
            raise ValueError("WMRLAgentWithCritic.update requires state_obs")
        if batch.values is None or batch.next_value is None:
            raise ValueError(
                "WMRLAgentWithCritic.update requires batch.values and batch.next_value. "
                "Did you use get_action_value_and_logprob during rollout collection?"
            )

        # Use instance defaults if not overridden
        _gae_lambda = gae_lambda if gae_lambda > 0 else self.gae_lambda
        _vf_coef = vf_coef if vf_coef > 0 else self.vf_coef

        t_steps, n_envs = batch.rewards.shape
        b_state = batch.state_obs.reshape(-1, *batch.state_obs.shape[2:])

        def _eval_fn(obs_mb: torch.Tensor, actions_mb: torch.Tensor):
            # Infer minibatch indices to slice state_obs accordingly
            # obs_mb and actions_mb come from flattened (T*N) arrays
            # We need to pass state through, so we capture b_state in closure
            # and use the same index slice. This is handled by the caller
            # in ppo_clipped_update_with_critic via the obs tensor.
            #
            # We pack state into the obs tensor as a workaround:
            # not feasible since they have different shapes.
            #
            # Instead, we do the full update inline here.
            pass

        # --- Inline PPO+critic update (we need state_obs for evaluate) ---
        advantages, returns = compute_gae(
            batch.rewards, batch.values, batch.dones,
            batch.next_value, batch.next_done,
            gamma, _gae_lambda,
        )

        b_obs = batch.obs.reshape(-1, *batch.obs.shape[2:])
        b_actions = batch.actions.reshape(-1, *batch.actions.shape[2:])
        b_logprobs = batch.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        batch_size = t_steps * n_envs
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

                newlogprob, entropy, newvalue = self._evaluate_actions_with_value(
                    b_obs[mb], b_state[mb], b_actions[mb],
                )

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

                if target_kl is not None and approx_kl > target_kl:
                    stop_early = True
                    break

                adv = b_advantages[mb]
                if norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss
                v_loss = 0.5 * (newvalue - b_returns[mb]).pow(2).mean()

                # Entropy
                ent_loss = entropy.mean()

                # BC loss
                bc_loss = torch.zeros((), device=self.device)
                if self.bc_loss_weight > 0.0 and self.bc_minibatches_per_update > 0:
                    _prev_skip = getattr(self.policy, 'skip_dino_preprocess', False)
                    if _prev_skip:
                        self.policy.skip_dino_preprocess = False
                    for _ in range(self.bc_minibatches_per_update):
                        bc_loss = bc_loss + self.policy.compute_loss(self._next_bc_batch())
                    bc_loss = bc_loss / self.bc_minibatches_per_update
                    if _prev_skip:
                        self.policy.skip_dino_preprocess = True

                loss = (
                    pg_loss
                    - ent_coef * ent_loss
                    + _vf_coef * v_loss
                    + self.bc_loss_weight * bc_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, max_grad_norm)
                nn.utils.clip_grad_norm_(self._value_params, max_grad_norm)
                self.optimizer.step()
                self.value_optimizer.step()

                totals["pg"] += pg_loss.item()
                totals["vf"] += v_loss.item()
                totals["ent"] += ent_loss.item()
                totals["kl"] += approx_kl
                totals["cf"] += clipfrac
                totals["bc"] += bc_loss.item()
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
            bc_loss=totals["bc"] / denom,
            value_loss=totals["vf"] / denom,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        base = super().state_dict()
        base["value_head"] = self.value_head.state_dict()
        base["value_optimizer"] = self.value_optimizer.state_dict()
        return base

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "value_head" in state:
            self.value_head.load_state_dict(state["value_head"])
        if "value_optimizer" in state:
            try:
                self.value_optimizer.load_state_dict(state["value_optimizer"])
            except ValueError:
                pass
