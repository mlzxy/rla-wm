"""Shared fixtures for the gates. Each `rla/tests/test_*.py` exposes one `run(args)`."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def rule(title: str) -> None:
    print(f"\n=== {title}")


def ok(message: str) -> None:
    print(f"  OK  {message}")


def skip(message: str) -> None:
    print(f"  --  skipped: {message}")


def mock_cfg(module, cfg):
    """Temporarily swap a module's `CFG`, so one process can build several geometries."""
    return mock.patch.object(module, "CFG", cfg)


def head_with(steps: int, **overrides):
    """Build the RLA head as if the environment said so, without re-execing the process."""
    import rla.action_head as action_head_mod
    from rla.config import CFG

    with mock_cfg(action_head_mod, dataclasses.replace(CFG, steps=steps, **overrides)):
        return action_head_mod.RlaL1RegressionActionHead(
            input_dim=896, hidden_dim=896, action_dim=7, use_pro_version=True
        )


def head_inputs(batch: int = 2):
    """One fixed set of head inputs, in the bf16 both the training and eval paths use."""
    import torch

    torch.manual_seed(0)
    return {
        "actions_hidden_states": torch.randn(batch, 25, 512 + 64, 896, dtype=torch.bfloat16),
        "proprio": torch.randn(batch, 8, dtype=torch.bfloat16),
        "proprio_projector": torch.nn.Linear(8, 896).to(torch.bfloat16),
    }


def run_head(head, inputs):
    """`(action, z_hat)`, no gradient. The head must be in **eval** mode for repeated calls to be
    exact -- training mode adds fresh Gaussian noise to both token groups every call."""
    import torch

    with torch.no_grad():
        action = head.predict_action(**inputs)
    return action, head.pop_z_hat()
