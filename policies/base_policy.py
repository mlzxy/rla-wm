"""
Base Policy Module for Imitation Learning

Provides an abstract base class `BasePolicy` that all policy implementations
should inherit from, along with typed data interfaces for policy inputs/outputs
and a `ReplayPolicy` for debug testing.

Usage:
    Subclass `BasePolicy` and implement `forward`, `compute_loss`, and
    `configure_optimizers`. Reference your subclass in the YAML config:

    policy:
      module: "policies.my_policy:MyPolicy"
      args:
        hidden_dim: 256
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.datasets.trajectory_dataset import TrajectoryBatch
from policies.action_normalizer import Observation, Action

LossDict = Dict[str, torch.Tensor]

class BasePolicy(ABC, nn.Module):
    """
    Abstract base policy for imitation learning.

    Subclasses must implement:
        - forward(obs): inference pass returning PolicyAction
        - compute_loss(batch): compute training loss from TrajectoryBatch
        - configure_optimizers(): return optimizer (and optional scheduler)
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, obs: Observation, return_loss: bool = False) -> Action | LossDict:
        """
        Run inference to produce actions from observations.

        Args:
            obs: PolicyObservation with current state

        Returns:
            PolicyAction containing the action tensor or the loss dict
        """
        if return_loss:
            return {'total': torch.tensor(0.0, device=obs.actions.device)}
        else:
            return obs.actions 


    def configure_optimizers(self) -> Union[torch.optim.Optimizer, tuple]:
        """
        Return the optimizer for this policy's parameters.

        May also return a tuple (optimizer, scheduler) if a learning rate
        scheduler is desired.
        """
        pass

    def reset(self):
        """
        Reset any internal recurrent state (e.g. for RNN/transformer policies).
        Called at the start of each evaluation episode. Default is a no-op.
        """
        pass
