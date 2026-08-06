"""
rla/loss.py

The auxiliary loss, as a wrapper around `finetune.run_forward_pass` rather than a copy of it.

    loss = LAMBDA_ACT * L1(action, a*)  +  lambda * L1(z_hat, z*)

`run_forward_pass` is a module global in `vla-scripts/finetune.py` and returns `(loss, metrics)`
where `loss` is exactly the action L1. The wrapper calls it, pops `z_hat` off the head (see
`rla/action_head.py` for why it travels that way rather than in the return value), and combines.

Two details that are easy to get wrong:

  * **`LAMBDA_ACT=0` multiplies the action L1 by zero, it does not skip it.** `0.0 * act_l1` still
    has a defined backward, so `fc2` and every other action-path parameter receives a zero `.grad`
    tensor instead of `None`. Stage 1 supervises only `z`, and without this DDP would see unused
    parameters.
  * **The weight has two modes.** The action L1 falls by roughly an order of magnitude over a run
    while the RLA L1 does not, so a fixed weight that starts as a small auxiliary term drifts
    towards parity. `LAMBDA_RLA=auto:R` recomputes it every step as `R * L_act / L_rla`, holding the
    RLA term at a fixed fraction `R` of the action term for the whole run. `Rla Share` is logged
    either way so the actual ratio is visible rather than assumed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch

from rla.config import CFG

# Metrics for the current step. `finetune.py`'s `recent_metrics` dict is fixed at five keys and
# silently drops anything else (`finetune.py:1007-1013, :1044-1047`), so these are merged in by the
# `log_metrics_to_wandb` wrapper below instead of going through that deque.
_LAST: Dict[str, float] = {}

# W&B is forced offline in finetune.py, so the two loss scales also go to stdout -- they are what
# you set LAMBDA_RLA from, and reading them should not require unpacking a wandb run.
_PRINT_EVERY = int(os.environ.get("RLA_LOG_EVERY", 50))
_STEP = 0

_POSITIONAL = ("vla", "action_head", "proprio_projector", "batch", "action_tokenizer", "device_id")


def _arg(name: str, args: tuple, kwargs: dict) -> Any:
    """`run_forward_pass` is called with keywords from the training loop and could be called
    positionally from anywhere else; resolve either way rather than assuming."""
    if name in kwargs:
        return kwargs[name]
    index = _POSITIONAL.index(name)
    return args[index] if len(args) > index else None


def make_run_forward_pass(orig):
    def run_forward_pass_rla(*args, **kwargs):
        loss, metrics = orig(*args, **kwargs)
        if not CFG.enabled:
            return loss, metrics

        action_head = _arg("action_head", args, kwargs)
        head = action_head.module if hasattr(action_head, "module") else action_head
        z_hat = head.pop_z_hat()
        if z_hat is None:
            # The head was in eval mode, so no z was produced. Validation forwards land here; the
            # action loss alone is the right thing to report for them.
            return loss, metrics

        batch = _arg("batch", args, kwargs)
        device_id = _arg("device_id", args, kwargs)
        if "rla" not in batch:
            raise KeyError(
                "RLA tokens are enabled but the batch has no 'rla' target. Check RLA_SIDECAR, and "
                "that rla/train.py (not vla-scripts/finetune.py) was the entry point so the "
                "dataset hook and collator were installed."
            )

        z_star = batch["rla"].to(device_id).to(z_hat.dtype)          # (B, steps, Q*D)
        if z_hat.shape != z_star.shape:
            raise ValueError(
                f"z_hat {tuple(z_hat.shape)} != target {tuple(z_star.shape)}; the head and the "
                "sidecar disagree about (RLA_STEPS, RLA_QUERIES, RLA_DIM)."
            )

        act_l1 = loss
        rla_l1 = torch.nn.L1Loss()(z_hat, z_star)

        if CFG.lambda_auto:
            # Detached: the weight tracks the loss ratio but carries no gradient of its own.
            lam = CFG.lambda_rla * (act_l1.detach() / rla_l1.detach().clamp(min=1e-6))
        else:
            lam = CFG.lambda_rla

        total = CFG.lambda_act * act_l1 + lam * rla_l1

        act_value, rla_value = act_l1.item(), rla_l1.item()
        lam_value = float(lam) if not torch.is_tensor(lam) else lam.item()
        weighted = lam_value * rla_value
        _LAST.update(
            {
                "rla_loss": rla_value,
                "rla_lambda": lam_value,
                "rla_share": weighted / max(CFG.lambda_act * act_value + weighted, 1e-12),
                "action_l1": act_value,
            }
        )

        global _STEP
        if _PRINT_EVERY and _STEP % _PRINT_EVERY == 0:
            print(
                f"[rla] step {_STEP:6d}  action_l1 {act_value:.4f}  rla_l1 {rla_value:.4f}  "
                f"lambda {lam_value:.4f}  weighted_rla {weighted:.4f}  "
                f"rla_share {_LAST['rla_share']:.3f}  total {(CFG.lambda_act * act_value + weighted):.4f}",
                file=sys.stderr,
            )
        _STEP += 1
        metrics = dict(metrics)
        metrics["loss_value"] = total.item()
        return total, metrics

    return run_forward_pass_rla


def make_log_metrics_to_wandb(orig):
    """Merge the RLA metrics in on the way to W&B.

    Upstream renames `loss_value` to `"{prefix}/Loss"` and title-cases everything else, so
    `rla_loss` shows up as `"VLA Train/Rla Loss"` and needs no special handling here.
    """

    def log_metrics_to_wandb_rla(metrics, prefix, step, wandb_entity):
        if CFG.enabled and _LAST:
            metrics = {**metrics, **_LAST}
        return orig(metrics, prefix, step, wandb_entity)

    return log_metrics_to_wandb_rla


def last_metrics() -> Dict[str, float]:
    """The most recent step's RLA metrics. Used by the smoke test and the verifier."""
    return dict(_LAST)
