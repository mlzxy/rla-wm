"""Gate `eval`: an RLA checkpoint loads through the real rollout path and predicts actions.

Exercises `experiments/robot/openvla_utils.get_action_head` for real -- `find_checkpoint_file`,
`load_component_state_dict`, the **strict** `load_state_dict`, the `.eval()` -- against a checkpoint
saved the way `save_training_checkpoint` saves one, then makes the exact call
`modeling_prismatic.py:871-874` makes.

Also checks the tripwire fires: a head built with a different geometry must refuse the checkpoint
rather than silently randomise. That is the reason the geometry lives in the environment.
"""

import argparse
import dataclasses
import tempfile
from pathlib import Path

import torch

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from rla.tests.harness import head_inputs, head_with, mock_cfg, ok, rule


def run(args) -> None:
    import experiments.robot.openvla_utils as openvla_utils
    from rla import patch
    from rla.action_head import RlaL1RegressionActionHead
    from rla.config import CFG

    rule("eval: an RLA checkpoint loads through the real eval path and predicts actions")

    steps = CFG.steps or NUM_ACTIONS_CHUNK
    # bf16, because `init_module(..., to_bf16=True)` is what training saves and `get_action_head`
    # casts before loading; a float32 fixture would round on the way in.
    trained = head_with(steps).to(torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        torch.save({f"module.{k}": v for k, v in trained.state_dict().items()},
                   checkpoint_dir / "action_head--1_checkpoint.pt")

        patch.install_eval()
        head = openvla_utils.get_action_head(
            argparse.Namespace(pretrained_checkpoint=str(checkpoint_dir), use_l1_regression=True,
                               use_pro_version=True, save_version="vla-adapter"),
            896,
        )
        assert isinstance(head, RlaL1RegressionActionHead), f"got {type(head).__name__}"
        assert not head.training, "get_action_head must leave the head in eval mode"
        loaded, reference = head.state_dict(), trained.state_dict()
        assert all(torch.equal(loaded[k].cpu(), reference[k].to(loaded[k].dtype))
                   for k in reference), "weights changed across the save/load round trip"
        ok(f"strict load through get_action_head: {len(reference)} tensors, eval mode")

        device = next(head.parameters()).device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v.to(device))
                  for k, v in head_inputs(batch=1).items()}
        with torch.no_grad():
            # Exactly the call in modeling_prismatic.py:871 -- no `phase`, no `return_rla`, and the
            # result is reshaped straight to the action chunk.
            out = head.predict_action(**inputs)
            actions = out.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert not isinstance(out, tuple), (
            "predict_action returned a tuple; modeling_prismatic.py reshapes it directly and would "
            "crash"
        )
        assert torch.isfinite(actions).all(), "non-finite actions"
        ok(f"predict_action -> {tuple(out.shape)} -> reshape{tuple(actions.shape)}, finite")

        import rla.action_head as action_head_mod

        with mock_cfg(action_head_mod, dataclasses.replace(CFG, steps=steps,
                                                           queries=CFG.queries * 2)):
            mismatched = action_head_mod.RlaL1RegressionActionHead(
                input_dim=896, hidden_dim=896, action_dim=7, use_pro_version=True
            )
        try:
            mismatched.load_state_dict(dict(trained.state_dict()))
        except RuntimeError:
            ok("a head built with the wrong RLA_QUERIES refuses the checkpoint (strict load trips)")
        else:
            raise AssertionError(
                "a head with double RLA_QUERIES loaded a checkpoint it should not fit -- the "
                "train/eval shape tripwire is not working"
            )
