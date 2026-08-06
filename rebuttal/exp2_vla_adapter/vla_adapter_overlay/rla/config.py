"""
rla/config.py

Every knob the RLA integration has, resolved once from the environment.

Why environment variables and not `--flags`: `FinetuneConfig` is a frozen dataclass in
`vla-scripts/finetune.py` and this integration does not modify upstream files. Independently, the
*shape* knobs have to be an env var anyway -- `experiments/robot/openvla_utils.get_action_head`
rebuilds the action head from four fixed kwargs and then calls `load_state_dict(...)` **strict**, so
training and evaluation must read the same switch or they disagree about the head's shape and the
load fails (which is the behaviour we want, but only if both sides look at one place).

    RLA_SIDECAR          dir written by the extractor; required when RLA_STEPS > 0 and training
    RLA_STEPS            0 disables everything; otherwise must equal NUM_ACTIONS_CHUNK
    RLA_QUERIES          Q, asserted against the sidecar's manifest.json at startup
    RLA_DIM              D, likewise
    LAMBDA_ACT           weight on the action L1                       (stage 1 sets this to 0)
    LAMBDA_RLA           weight on the RLA L1: a float, or "auto:R"    (see `lambda_auto`)
    RLA_NORM             how the latent targets are scaled: global | center | perdim | none
    RLA_ACTION_TOKENS    1 keeps the action tokens in the policy sequence, 0 removes them
    RLA_INIT_FROM        stage-1 checkpoint dir to warm-start from     (stage 2 only)
    RLA_SHUFFLE_TARGETS  1 permutes z across episodes -- the "is it the content of z?" control
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Default ratio for `LAMBDA_RLA=auto`: hold the RLA term at this fraction of the action term.
AUTO_RATIO_DEFAULT = 0.25

# How the latent-action targets are scaled before the L1. All four are exactly invertible, so none
# of them loses information; what differs is what the loss *weights*.
#
#   global  (z - global_mean) / global_std   two scalars. Keeps the relative scale of the 1024
#                                            latent slots intact, so high-variance slots -- the ones
#                                            carrying the motion -- dominate the loss naturally.
#   center  (z - mean[q,d])   / global_std   same scaling, but also removes each slot's DC offset.
#                                            Those offsets are constant over the whole corpus, carry
#                                            nothing about the motion, and are large on this sidecar
#                                            (|mean| median 1.89, range -13.96..+10.52), so the head
#                                            otherwise spends capacity reproducing them.
#   perdim  (z - mean[q,d])   / std[q,d]     whitened. Every slot contributes equally to the loss
#                                            regardless of how much variance it carries.
#   none    z                                raw encoder units; E|z| ~ 2.8, so lambda needs to be
#                                            about 3.6x smaller to mean the same thing.
NORM_MODES = ("global", "center", "perdim", "none")


def _parse_lambda_rla(raw: str) -> tuple[float, bool]:
    """`"0.05"` -> (0.05, False);  `"auto"` / `"auto:0.3"` -> (0.25 | 0.3, True).

    Fixed and auto mean different things and the difference matters: the action L1 falls by roughly
    an order of magnitude over a run while the RLA L1 does not, so a *fixed* weight that starts as a
    small auxiliary term drifts towards parity by the end. `auto:R` recomputes the weight every step
    as `R * L_act / L_rla`, pinning the RLA term at a fixed fraction of the action term throughout --
    which is what "keep the action loss the main loss" actually requires.
    """
    raw = raw.strip()
    if not raw.lower().startswith("auto"):
        return float(raw), False
    _, _, ratio = raw.partition(":")
    return (float(ratio) if ratio else AUTO_RATIO_DEFAULT), True


def _resolve_step(checkpoint_dir: Path) -> int:
    """Which `--<step>_checkpoint.pt` suffix lives in a checkpoint dir.

    Read off the files rather than parsed out of the directory name: `save_latest_checkpoint_only`
    writes `latest_checkpoint.pt` into the run dir itself, with no step in either name.
    """
    heads = sorted(checkpoint_dir.glob("action_head--*_checkpoint.pt"))
    if not heads:
        raise FileNotFoundError(
            f"no action_head--*_checkpoint.pt in {checkpoint_dir}; RLA_INIT_FROM must point at a "
            "checkpoint directory written by a stage-1 run (outputs/<run_id>--<step>_chkpt)."
        )
    if len(heads) > 1:
        raise RuntimeError(
            f"{len(heads)} action_head checkpoints in {checkpoint_dir}; expected exactly 1. "
            "(experiments/robot/openvla_utils.find_checkpoint_file asserts the same thing.)"
        )
    stem = heads[0].name[len("action_head--"):-len("_checkpoint.pt")]
    return -1 if stem == "latest" else int(stem)


@dataclass(frozen=True)
class RlaConfig:
    sidecar: str
    steps: int
    queries: int
    dim: int
    lambda_act: float
    lambda_rla: float          # the fixed weight, or the ratio R when `lambda_auto`
    lambda_auto: bool
    norm: str                  # target scaling: global | center | perdim | none
    action_tokens: bool        # are the action tokens in the policy sequence at all?
    mask_rla_to_action: bool   # forbid RLA queries from attending to action keys
    shuffle_targets: bool
    init_from: str
    init_step: Optional[int]

    @property
    def enabled(self) -> bool:
        """True when the action head carries RLA tokens. Independent of having a sidecar: evaluation
        needs the same head *shape* but no targets, because z is an output, never an input."""
        return self.steps > 0

    @property
    def has_targets(self) -> bool:
        return self.enabled and bool(self.sidecar)

    @property
    def num_tokens(self) -> int:
        """Extra policy query tokens: one group of `queries` per action-chunk step."""
        return self.steps * self.queries

    @property
    def target_dim(self) -> int:
        """Q*D -- the flattened per-chunk-step target width."""
        return self.queries * self.dim

    @property
    def num_action_tokens(self) -> int:
        return NUM_ACTIONS_CHUNK if self.action_tokens else 0

    @property
    def seq_len(self) -> int:
        """Total policy sequence length: action tokens (if any) followed by RLA tokens.

        `RLA_ACTION_TOKENS=0` shortens this to `num_tokens`, which changes every RLA token's
        absolute RoPE position by `NUM_ACTIONS_CHUNK`. That is not free: `x` enters the policy as
        zeros, so RoPE is the *only* thing distinguishing token positions, and while RLA<->RLA
        relative offsets survive a uniform shift, the cross-attention offsets against the 512 vision
        tokens and the 65 adapter tokens -- where essentially all the information enters -- do not.
        `rla/warmstart.py` splices `gating_factor` across the two widths so the checkpoint still
        loads, but the attention geometry genuinely differs between the stages.
        """
        return self.num_action_tokens + self.num_tokens

    def describe(self) -> str:
        if not self.enabled:
            return "[rla] disabled (RLA_STEPS=0) -- action head is byte-identical to upstream"
        lam = f"auto:{self.lambda_rla}" if self.lambda_auto else f"{self.lambda_rla}"
        layout = (
            f"{self.num_action_tokens} action + {self.num_tokens} rla"
            if self.action_tokens
            else f"{self.num_tokens} rla ONLY (action tokens removed from the sequence)"
        )
        return (
            f"[rla] {self.steps} steps x {self.queries} queries x {self.dim} dim "
            f"-> {self.num_tokens} extra policy tokens\n"
            f"[rla] policy sequence: {layout} = {self.seq_len}, target width {self.target_dim}\n"
            f"[rla] attention: action <- rla "
            + ("ONE-WAY (rla queries cannot see action keys; the rla branch is invariant to the "
               "action tokens)" if self.mask_rla_to_action else "and rla <- action (BIDIRECTIONAL)")
            + f"\n[rla] target scaling: {self.norm}\n"
            + f"[rla] lambda_act={self.lambda_act}  lambda_rla={lam}\n"
            f"[rla] sidecar={self.sidecar or '<unset: inference-only, no targets>'}"
            + (f"\n[rla] warm start from {self.init_from} @ step {self.init_step}" if self.init_from else "")
            + ("\n[rla] WARNING: RLA_SHUFFLE_TARGETS=1 -- targets are permuted across episodes; "
               "this is the control arm, not a real run." if self.shuffle_targets else "")
        )


def _load() -> RlaConfig:
    steps = int(os.environ.get("RLA_STEPS", 0))
    if steps not in (0, NUM_ACTIONS_CHUNK):
        raise ValueError(
            f"RLA_STEPS={steps} must be 0 (disabled) or NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}: the "
            "targets are index-aligned with the action chunk, one latent per chunk step."
        )
    lambda_rla, lambda_auto = _parse_lambda_rla(os.environ.get("LAMBDA_RLA", "0.05"))
    lambda_act = float(os.environ.get("LAMBDA_ACT", 1.0))

    action_tokens = os.environ.get("RLA_ACTION_TOKENS", "1") not in ("0", "", "false", "False")
    mask = os.environ.get("RLA_MASK", "1") not in ("0", "", "false", "False")

    norm = os.environ.get("RLA_NORM", "center").strip().lower()
    if norm not in NORM_MODES:
        raise ValueError(f"RLA_NORM={norm!r} must be one of {NORM_MODES}")
    if steps and not action_tokens and lambda_act != 0.0:
        raise ValueError(
            f"RLA_ACTION_TOKENS=0 removes the action tokens from the policy sequence, so the "
            f"action output is not produced by the trunk and LAMBDA_ACT must be 0 (got {lambda_act}). "
            "This combination is only meaningful for stage-1 pretraining."
        )

    init_from = os.environ.get("RLA_INIT_FROM", "").strip()
    init_step = None
    if init_from:
        init_dir = Path(init_from)
        if not init_dir.is_dir():
            raise FileNotFoundError(f"RLA_INIT_FROM={init_from} is not a directory")
        init_step = _resolve_step(init_dir)

    return RlaConfig(
        sidecar=os.environ.get("RLA_SIDECAR", "").strip(),
        steps=steps,
        queries=int(os.environ.get("RLA_QUERIES", 16)),
        dim=int(os.environ.get("RLA_DIM", 64)),
        lambda_act=lambda_act,
        lambda_rla=lambda_rla,
        lambda_auto=lambda_auto,
        norm=norm,
        action_tokens=action_tokens,
        mask_rla_to_action=mask,
        shuffle_targets=os.environ.get("RLA_SHUFFLE_TARGETS", "0") not in ("0", "", "false", "False"),
        init_from=init_from,
        init_step=init_step,
    )


CFG = _load()
