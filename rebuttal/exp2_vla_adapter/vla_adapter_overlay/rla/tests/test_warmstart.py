"""Gate `warmstart`: stage 2, initialised from stage 1, predicts **exactly** the same z.

The premise of the two-stage run. Stage 2 attaches a freshly-initialised action output head and
starts supervising it, and none of that may move the auxiliary task. Checked through the real
`rla/warmstart.py` file round trip, including `_splice_gates`.

Also measures the same quantity across an `RLA_ACTION_TOKENS=0 -> 1` transition, which is *not*
invariant. That number is the concrete cost of dropping the action tokens from stage 1.
"""

import dataclasses
import tempfile
from pathlib import Path

import torch

from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from rla.tests.harness import head_inputs, head_with, ok, run_head, rule


def run(args) -> None:
    from rla.config import CFG
    from rla.warmstart import _splice_gates, _verify_loaded

    rule("warmstart: stage 2 loaded from stage 1 predicts identical z")

    steps = CFG.steps or NUM_ACTIONS_CHUNK
    inputs = head_inputs()

    stage1 = head_with(steps).to(torch.bfloat16).eval()
    action1, z1 = run_head(stage1, inputs)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "action_head--1_checkpoint.pt"
        # `save_training_checkpoint` saves the DDP-wrapped head, so keys carry a `module.` prefix.
        torch.save({f"module.{k}": v for k, v in stage1.state_dict().items()}, path)

        stage2 = head_with(steps).to(torch.bfloat16).eval()
        state = {k.removeprefix("module."): v
                 for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        assert _splice_gates(state, stage2) == 0, "gates needed splicing in the matched layout"
        stage2.load_state_dict(state)

        # The read-back check that the terminal banner reports. Both directions: it passes on a
        # genuine load, and it *fails* on a corrupted one -- a verifier that cannot fail verifies
        # nothing, and this one is the only thing standing between a silent bad load and a wasted run.
        verified, sampled = _verify_loaded(stage2, state, sample=len(state))
        assert verified == sampled == len(state), f"{verified}/{sampled} of {len(state)} verified"
        ok(f"_verify_loaded confirms all {sampled} head tensors bit-identical after the load")

        tampered = dict(state)
        victim = "model.fc2_rla.weight"
        tampered[victim] = state[victim] + 1.0
        try:
            _verify_loaded(stage2, tampered, sample=len(tampered))
        except RuntimeError as exc:
            assert victim in str(exc)
            ok("_verify_loaded raises when one tensor disagrees, naming it -- the check has teeth")
        else:
            raise AssertionError("_verify_loaded accepted a tampered state dict")

    # "New action tokens and head": stage 2 re-initialises the action output projection.
    with torch.no_grad():
        stage2.model.fc2.weight.normal_(std=0.05)
        stage2.model.fc2.bias.normal_(std=0.05)
    action2, z2 = run_head(stage2, inputs)

    assert torch.equal(z1, z2), (
        f"z moved by {(z1 - z2).abs().max().item():.4g} between the stages. Stage 2 is not a "
        "faithful continuation of stage 1's auxiliary task."
    )
    ok("z_hat bit-identical across the stage-1 -> stage-2 round trip, with fc2 re-initialised")
    ok(f"the action output does change ({(action1 - action2).abs().max().item():.4g}) -- only the "
       "RLA branch is pinned")

    # The cost of RLA_ACTION_TOKENS=0, measured rather than argued.
    import rla.action_head as action_head_mod
    from rla.tests.harness import mock_cfg

    with mock_cfg(action_head_mod, dataclasses.replace(CFG, steps=steps, action_tokens=False)):
        narrow = action_head_mod.RlaL1RegressionActionHead(
            input_dim=896, hidden_dim=896, action_dim=7, use_pro_version=True
        ).to(torch.bfloat16).eval()
    shared = {k: v for k, v in stage1.state_dict().items() if k in narrow.state_dict()}
    assert _splice_gates(shared, narrow) == len(narrow.model.mlp_resnet_blocks)
    narrow.load_state_dict(shared)
    shift = (z1 - run_head(narrow, inputs)[1]).abs().max().item()
    ok(f"RLA_ACTION_TOKENS=0 would move z by {shift:.4g} for the same weights -- the RoPE shift is "
       "real, which is why the action tokens stay in the sequence during stage 1")
