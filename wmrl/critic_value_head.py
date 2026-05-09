"""Standalone value head MLP for the critic baseline.

Takes the policy's ``forward_features()`` output (global_cond) as input
and predicts a scalar state value.  Lives outside the policy so no
policy code needs to be modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _ortho_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ValueHead(nn.Module):
    """MLP value head: global_cond → scalar V(s).

    Args:
        input_dim: Dimensionality of the policy's global conditioning vector.
        hidden_dims: Hidden layer sizes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(_ortho_init(nn.Linear(in_dim, h)))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        # Final layer with small gain (PPO convention for value head)
        layers.append(_ortho_init(nn.Linear(in_dim, 1), gain=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, global_cond: torch.Tensor) -> torch.Tensor:
        """Predict state value.

        Args:
            global_cond: ``(B, D)`` from ``policy.forward_features()``.

        Returns:
            ``(B,)`` scalar values.
        """
        return self.net(global_cond).squeeze(-1)
