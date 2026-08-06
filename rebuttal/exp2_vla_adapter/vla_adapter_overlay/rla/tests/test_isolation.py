"""Gate `isolation`: attention between the two token groups is one-way.

Action queries may read the RLA tokens; RLA queries may **not** read the action tokens. That is what
keeps the auxiliary task fixed across the stages -- stage 2 newly supervises `fc2`, which changes
what the action tokens carry, and if the RLA tokens could see them the pretrained z predictor would
silently become a different function.
"""

import torch

from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from rla.tests.harness import head_inputs, head_with, ok, run_head, rule, skip


def run(args) -> None:
    from rla.config import CFG

    rule("isolation: rla <- action is masked, action <- rla is not")

    if not CFG.mask_rla_to_action:
        skip("RLA_MASK=0, the two groups attend to each other bidirectionally")
        return

    steps = CFG.steps or NUM_ACTIONS_CHUNK
    head = head_with(steps).to(torch.bfloat16).eval()   # eval: no token noise, so runs are exact
    inputs = head_inputs()

    action_a, z_a = run_head(head, inputs)
    with torch.no_grad():                                # move the action branch, hard
        head.model.fc1.bias.add_(10.0)
        head.model.layer_norm1.bias.add_(3.0)
    action_b, z_b = run_head(head, inputs)
    with torch.no_grad():
        head.model.fc1.bias.sub_(10.0)
        head.model.layer_norm1.bias.sub_(3.0)

    assert torch.equal(z_a, z_b), (
        f"z_hat moved by {(z_a - z_b).abs().max().item():.4g} when only the action branch changed. "
        "The RLA tokens are reading the action tokens -- the mask is not working."
    )
    ok("perturbing fc1.bias by +10 and layer_norm1.bias by +3 leaves z_hat bit-identical")

    delta = (action_a - action_b).abs().max().item()
    assert delta > 0, "the action output ignored a large perturbation of its own input path"
    ok(f"the same perturbation moves the action output by {delta:.4g}")

    # Gradients: the RLA loss alone must not train the action-token path.
    head.train()
    head.zero_grad(set_to_none=True)
    head.predict_action(**inputs, phase="Training")
    z_hat = head.pop_z_hat()
    torch.nn.L1Loss()(z_hat, torch.zeros_like(z_hat)).backward()
    for name in ("fc1", "layer_norm1"):
        for pname, param in getattr(head.model, name).named_parameters():
            total = 0.0 if param.grad is None else param.grad.abs().sum().item()
            assert total == 0, (
                f"model.{name}.{pname} got gradient {total:.4g} from the RLA loss alone; the "
                "auxiliary objective is shaping the action path"
            )
    assert head.model.fc2_rla.weight.grad.abs().sum() > 0, "no gradient reached fc2_rla"
    ok("the RLA loss alone leaves fc1 and layer_norm1 with exactly zero gradient")

    # The reverse direction must be open: removing the RLA tokens changes the action output.
    head.eval()
    bare = head_with(0).to(torch.bfloat16).eval()
    # Same trunk, minus the RLA tokens. `gating_factor` is the one tensor whose shape differs --
    # (136, 1) here, a scalar upstream -- so copy the action row across explicitly rather than
    # skipping it, or the two heads would not be gating vision identically.
    reference = bare.state_dict()
    bare.load_state_dict(
        {k: v for k, v in head.state_dict().items()
         if k in reference and reference[k].shape == v.shape},
        strict=False,
    )
    with torch.no_grad():
        for block, source in zip(bare.model.mlp_resnet_blocks, head.model.mlp_resnet_blocks):
            block.gating_factor.copy_(source.gating_factor.reshape(-1)[:1])
    coupling = (run_head(head, inputs)[0] - run_head(bare, inputs)[0]).abs().max().item()
    assert coupling > 1e-3, (
        f"the action output barely changes ({coupling:.4g}) without the RLA tokens. The coupling "
        "is meant to be one-way, not absent -- action tokens must read the RLA tokens."
    )
    ok(f"removing the RLA tokens moves the action output by {coupling:.4g} -- action <- rla is open")

    # The bridge gate on that surviving direction is learnable and reaches the action output only.
    head.train()
    head.zero_grad(set_to_none=True)
    action = head.predict_action(**inputs, phase="Training")
    z_hat = head.pop_z_hat()
    (torch.nn.L1Loss()(action, torch.zeros_like(action))
     + 0.0 * torch.nn.L1Loss()(z_hat, torch.zeros_like(z_hat))).backward()
    grads = [b.gating_factor_rla2act.grad for b in head.model.mlp_resnet_blocks]
    assert all(g is not None and torch.isfinite(g).all() for g in grads), "bad rla2act gradient"
    assert sum(g.abs().sum().item() for g in grads) > 0, (
        "the action loss produced no gradient on gating_factor_rla2act -- the bridge gate is not "
        "on the path from the RLA keys to the action queries"
    )
    ok(f"the action loss trains gating_factor_rla2act in all {len(grads)} blocks, so the model can "
       "learn how much the predicted latent action should inform the action chunk")
