import torch
import torch.nn as nn
import numpy as np


class AgentRespeedWrapper(nn.Module):
    """
    Wraps a PPO Agent to scale its output actions by a speed factor.
    Useful for generating data with varying execution speeds (e.g. slow motion).

    Args:
        agent: The PPO Agent to wrap
        min_speed: Minimum speed multiplier (default: 0.25)
        max_speed: Maximum speed multiplier (default: 1.0)
        fixed_speed: If set, use this fixed speed multiplier instead of random sampling
    """

    def __init__(self, agent, min_speed=0.25, max_speed=1.0, fixed_speed=None):
        super().__init__()
        self.agent = agent
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.fixed_speed = fixed_speed

    def get_value(self, x):
        return self.agent.get_value(x)

    def _respeed(self, action):
        if self.fixed_speed is not None:
            return action * self.fixed_speed

        # Sample random speed per batch element
        if action.dim() > 1:
            B = action.shape[0]
            # Uniformly sample speed in [min_speed, max_speed]
            speed = (
                torch.rand(B, 1, device=action.device)
                * (self.max_speed - self.min_speed)
                + self.min_speed
            )
        else:
            # Handle single action case (unbatched) if necessary, though typical usage is batched
            speed = (
                torch.rand(1, device=action.device) * (self.max_speed - self.min_speed)
                + self.min_speed
            )

        return action * speed

    def get_action(self, x, deterministic=False):
        action = self.agent.get_action(x, deterministic=deterministic)
        return self._respeed(action)

    def get_action_and_value(self, x, action=None):
        # We only intervene when sampling a new action (action=None)
        if action is not None:
            return self.agent.get_action_and_value(x, action)

        action, logprob, entropy, value = self.agent.get_action_and_value(
            x, action=None
        )
        scaled_action = self._respeed(action)

        # Note: logprob and entropy correspond to the original action distribution.
        # Scaling the action effectively changes the distribution, but for data collection
        # we typically accept the original logprobs or just discard them.
        return scaled_action, logprob, entropy, value
