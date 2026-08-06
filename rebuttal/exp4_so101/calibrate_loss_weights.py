#!/usr/bin/env python
"""
Measure the initial loss magnitudes of a BC-RLA config and print the lambdas that give the
raw-action : latent-action balance you actually want.

    PYTHONPATH=.:./third_party/diffusion_policy CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
      rebuttal/exp4_so101/calibrate_loss_weights.py --config-name bc_rla_so101_100_ladder \\
        latent_encoder_work_dir=runs/so101/16x16_so101_dual/<TS> \\
        rla_latent_tag=<TAG> rla_latent_step=<STEP>

Why this exists
---------------
`lambda_action`, `lambda_latent_l1` and `lambda_latent_mse` fix the *nominal* weights. What matters
is the ratio of the terms they produce, and the two losses live on very different scales: the action
loss is an MSE on actions normalised to [-1, 1], while the latent loss is an L1/MSE on RLA tokens
divided by 10. Measured on an untrained policy, a nominal 2:1 weighting came out around 7:1 in
practice -- the auxiliary signal was ~3x weaker than intended.

So: measure both losses on the real data with the real (frozen) targets, then solve for the lambdas
that put the realised ratio where you asked. Fixing `lambda_action = 1.0` and keeping the two latent
weights equal, the requested ratio R is achieved by

    lambda_latent_l1 = lambda_latent_mse = action_loss / (R * (latent_l1 + latent_mse))

Treat `--ratio 2.0` as a CEILING, not a target to hit exactly: the raw-action loss is what drives
the robot and the latent action is auxiliary, so it is better to under-weight the latent term than
to over-weight it.

Run this for EVERY sidecar, not once. The shipped 0.25 / 0.25 is a heuristic, not a fixed ratio:
the target is `encoder output / 10`, so what those lambdas buy scales with the RLA checkpoint's
latent magnitude -- which the sidecar records as `manifest["value_stats"]["mean_abs"]`. At
mean_abs ~1.9 they realise 6:1 to 8:1; at mean_abs ~6.7 they realise ~1:1, i.e. the auxiliary term
sitting at parity with the objective that actually drives the robot.

This is a *starting* balance either way. The two losses converge at different rates, so training
also logs `train_latent_loss_frac` and `train_action_over_latent` every step -- watch those to see
where the balance actually ends up.

Nothing is written: the script prints YAML to paste into the config, so the run stays reproducible
and the four configs stay comparable.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Repo root is three levels up: <repo>/rebuttal/exp4_so101/<this file>.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "policy")


def build(cfg, device):
    """Dataset + policy, exactly as TrainSO101Workspace would build them."""
    import hydra
    from omegaconf import OmegaConf

    import policies.dataset.few_shot_mixed_dataset as few_shot_mod
    from diffusion_policy.model.common.normalizer import LinearNormalizer  # noqa: F401
    from policies.dataset.few_shot_mixed_dataset import FewShotMixedDataset
    from rebuttal.exp4_so101.dataset.so101_sequence_dataset import SO101SequenceDataset

    # Same rebinding the workspace does, so FewShotMixedDataset builds our subclass.
    few_shot_mod.ManiSkillSequenceDataset = SO101SequenceDataset

    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset = (
        FewShotMixedDataset(**ds_cfg) if "robot_dataset_cfg" in ds_cfg
        else SO101SequenceDataset(**ds_cfg)
    )
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    return dataset, policy.to(device).train()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config-name", required=True, help="e.g. bc_rla_so101_100_ladder")
    ap.add_argument("--config-dir", default=CONFIG_DIR)
    ap.add_argument("--batches", type=int, default=20, help="batches to average over")
    ap.add_argument(
        "--ratio", type=float, default=2.0,
        help="target raw-action : latent-action contribution ratio (default 2 = actions twice the "
             "latent term)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. rla_latent_tag=...")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from diffusion_policy.common.pytorch_util import dict_apply
    from rebuttal.exp4_so101.workspace import resolve_latent_gaps, validate_rla_setup

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=os.path.abspath(args.config_dir), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))
    OmegaConf.resolve(cfg)
    resolve_latent_gaps(cfg)
    validate_rla_setup(cfg)

    if not bool(OmegaConf.select(cfg, "policy.use_latent_head", default=False)):
        raise SystemExit(f"{args.config_name} has no latent head; nothing to balance.")

    device = torch.device(args.device)
    dataset, policy = build(cfg, device)
    loader = DataLoader(
        dataset, batch_size=int(cfg.dataloader.batch_size), shuffle=True,
        num_workers=int(cfg.dataloader.num_workers), drop_last=True,
    )

    print(
        f"\ncalibrating {args.config_name} on {device}\n"
        f"  latent_mode={cfg.latent_mode}  gaps={list(cfg.latent_gaps)}\n"
        f"  rla tag={cfg.get('rla_latent_tag')} step={cfg.get('rla_latent_step')}\n"
        f"  averaging {args.batches} batches of {cfg.dataloader.batch_size} ..."
    )

    acc = {"action_loss": [], "latent_action_l1": [], "latent_action_mse": []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            out = policy.compute_loss(batch)
            for k in acc:
                acc[k].append(float(out[k]))

    if not acc["action_loss"]:
        raise SystemExit("no batches were produced; is the dataset empty?")
    a, l1, mse = (float(np.mean(acc[k])) for k in ("action_loss", "latent_action_l1", "latent_action_mse"))

    lam_a = float(cfg.policy.get("lambda_action", 1.0))
    cur_l1 = float(cfg.policy.get("lambda_latent_l1", 0.0))
    cur_mse = float(cfg.policy.get("lambda_latent_mse", 0.0))
    cur_ratio = (lam_a * a) / max(cur_l1 * l1 + cur_mse * mse, 1e-12)

    lam = a / (args.ratio * max(l1 + mse, 1e-12))

    print(
        f"\n  RAW losses (untrained policy, frozen targets)\n"
        f"    action_loss        {a:.5f}\n"
        f"    latent_action_l1   {l1:.5f}\n"
        f"    latent_action_mse  {mse:.5f}\n"
        f"\n  CURRENT weights  lambda_action={lam_a}  l1={cur_l1}  mse={cur_mse}\n"
        f"    -> realised action:latent = {cur_ratio:.2f} : 1"
        f"   (target {args.ratio:.2f} : 1)\n"
        f"\n  PASTE THIS into {args.config_name}.yaml under `policy:`\n"
        f"    lambda_action: 1.0\n"
        f"    lambda_latent_l1: {lam:.4g}\n"
        f"    lambda_latent_mse: {lam:.4g}\n"
        f"    # calibrated on {args.batches} batches: action {a:.4f} vs latent "
        f"({l1:.4f} + {mse:.4f}) -> {args.ratio:g}:1\n"
        f"\n  Sanity: with those, action:latent = "
        f"{(a) / max(lam * (l1 + mse), 1e-12):.2f} : 1, "
        f"latent_loss_frac = {lam * (l1 + mse) / (a + lam * (l1 + mse)):.3f}\n"
        f"  This is the STARTING balance; watch train_latent_loss_frac during the run.\n"
    )


if __name__ == "__main__":
    main()
