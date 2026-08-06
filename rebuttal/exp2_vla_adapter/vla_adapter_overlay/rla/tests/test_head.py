"""Gate `head`: RLA off is upstream; RLA on has the right shapes and gradients.

The first half protects the baseline arm -- with `RLA_STEPS=0` the head must be byte-identical to
upstream so existing `LIBERO-*-Pro` checkpoints still load. The second half checks the RLA head does
what it claims, including the two gradient facts the training loop depends on.
"""

import torch

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from rla.tests.harness import REPO, head_inputs, head_with, ok, run_head, rule


def run(args) -> None:
    from rla.config import CFG

    rule("head: RLA off is upstream; RLA on has the right shapes and gradients")

    torch.manual_seed(0)
    upstream = L1RegressionActionHead(input_dim=896, hidden_dim=896, action_dim=7,
                                      use_pro_version=True)
    torch.manual_seed(0)
    off = head_with(0)

    up_sd, off_sd = upstream.state_dict(), off.state_dict()
    assert set(up_sd) == set(off_sd), (
        f"key sets differ: +{sorted(set(off_sd) - set(up_sd))[:5]} "
        f"-{sorted(set(up_sd) - set(off_sd))[:5]}"
    )
    for key in up_sd:
        assert up_sd[key].shape == off_sd[key].shape, f"{key} shape differs"
        assert torch.equal(up_sd[key], off_sd[key]), f"{key} differs under a fixed seed"
    ok(f"RLA_STEPS=0 state dict identical to upstream ({len(up_sd)} tensors)")

    # bf16 throughout, as in training: `predict_action` casts proprio to bf16 unconditionally, so a
    # float32 projector fails on a dtype mismatch before reaching the trunk.
    inputs = head_inputs()
    upstream = upstream.to(torch.bfloat16).eval()
    off = off.to(torch.bfloat16).eval()
    with torch.no_grad():
        a_up = upstream.predict_action(**inputs)
        a_off = off.predict_action(**inputs)
    assert torch.equal(a_up, a_off), "predict_action differs with RLA_STEPS=0"
    ok(f"RLA_STEPS=0 predict_action bit-identical to upstream {tuple(a_up.shape)}")

    for candidate in sorted(REPO.glob("outputs/REF-*_chkpt/action_head--*_checkpoint.pt")):
        state = {k.removeprefix("module."): v
                 for k, v in torch.load(candidate, map_location="cpu", weights_only=True).items()}
        off.load_state_dict(state)
        ok(f"an existing vanilla checkpoint loads strict into the RLA_STEPS=0 head: {candidate.name}")
        break

    # --- RLA on ---------------------------------------------------------------------------------
    steps = CFG.steps or NUM_ACTIONS_CHUNK
    on = head_with(steps).to(torch.bfloat16)
    new_keys = sorted(k for k in on.state_dict() if k not in up_sd)
    blocks = len(on.model.mlp_resnet_blocks)
    output_keys = ["model.fc2_rla.bias", "model.fc2_rla.weight",
                   "model.layer_norm2_rla.bias", "model.layer_norm2_rla.weight"]
    bridge_keys = [f"model.mlp_resnet_blocks.{i}.gating_factor_rla2act" for i in range(blocks)]
    assert new_keys == sorted(output_keys + bridge_keys), new_keys
    gate = on.model.mlp_resnet_blocks[0].gating_factor
    assert tuple(gate.shape) == (CFG.seq_len, 1), (
        f"gating_factor is {tuple(gate.shape)}, expected ({CFG.seq_len}, 1) -- action and RLA "
        "positions need independent vision gates"
    )
    assert all((b.gating_factor_rla2act == 0).all() for b in on.model.mlp_resnet_blocks), (
        "gating_factor_rla2act must be zero-init, like the vision gate it sits beside"
    )
    ok(f"RLA on adds the 4 output tensors plus one zero-init gating_factor_rla2act per block, and "
       f"widens gating_factor to {tuple(gate.shape)} in each of {blocks} blocks")

    on.train()
    action = on.predict_action(**inputs, phase="Training")
    z_hat = on.pop_z_hat()
    assert action.shape == (2, NUM_ACTIONS_CHUNK, ACTION_DIM), f"action is {tuple(action.shape)}"
    assert z_hat.shape == (2, steps, CFG.target_dim), f"z_hat is {tuple(z_hat.shape)}"
    ok(f"forward: action {tuple(action.shape)}, z_hat {tuple(z_hat.shape)}")

    # lambda_act=0 must still give the action path a *zero* gradient rather than None, or DDP
    # reports an unused parameter. That is why the loss multiplies by 0.0 instead of skipping.
    loss = 0.0 * torch.nn.L1Loss()(action, torch.zeros_like(action)) \
        + torch.nn.L1Loss()(z_hat, torch.zeros_like(z_hat))
    loss.backward()
    assert on.model.fc2_rla.weight.grad.abs().sum() > 0, "no gradient reached fc2_rla"
    assert on.model.fc2.weight.grad is not None, (
        "fc2 has grad=None at lambda_act=0 -- DDP would report an unused parameter"
    )
    assert on.model.fc2.weight.grad.abs().sum() == 0, "fc2 grad is non-zero at lambda_act=0"
    assert torch.isfinite(on.model.mlp_resnet_blocks[0].gating_factor.grad).all()
    ok("backward: fc2_rla grad non-zero, fc2 grad present and exactly zero at lambda_act=0")

    on.eval()
    a_eval, z_eval = run_head(on, inputs)
    assert a_eval.shape == (2, NUM_ACTIONS_CHUNK, ACTION_DIM)
    # The chunk must not be one action repeated. `x` enters the policy as zeros, so if RoPE were
    # absent (the non-Pro block) every query position would be identical and the head would emit the
    # same action eight times -- silently, and only visible as a policy that cannot follow a
    # trajectory. Measured on the real head rather than assumed.
    spread = (a_eval.max(dim=1).values - a_eval.min(dim=1).values).min().item()
    assert spread > 1e-3, (
        f"the 8 chunk steps span only {spread:.3g} -- the head is emitting one action repeated"
    )
    ok(f"the action chunk varies across its 8 steps (min per-dim spread {spread:.3f})")
    assert z_eval is not None, "z_hat should still be produced in eval mode (it is just unused)"
    assert torch.equal(a_eval, run_head(on, inputs)[0]), "eval-mode forward is not deterministic"
    delta = (a_eval - a_up).abs().max().item()
    assert delta > 1e-6, (
        "the RLA head's action output is identical to the RLA_STEPS=0 head's. It should not be -- "
        "the RLA tokens sit in the shared self-attention. If this passes, they are not in it."
    )
    ok(f"eval-mode forward is deterministic; action differs from RLA-off by {delta:.4f}, so the "
       "control arm must be RLA_STEPS=0, not LAMBDA_RLA=0")
