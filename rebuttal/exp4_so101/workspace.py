"""Training workspace that uses `SO101SequenceDataset` instead of `ManiSkillSequenceDataset`.

`TrainVLABCWorkspace.run()` picks its dataset class inline::

    if "robot_dataset_cfg" in dataset_cfg_dict:
        dataset = FewShotMixedDataset(**dataset_cfg_dict)
    else:
        dataset = ManiSkillSequenceDataset(**dataset_cfg_dict)

Both class names are hard-coded, and `FewShotMixedDataset.__init__` likewise constructs
`ManiSkillSequenceDataset` directly, so there is no config hook for swapping the dataset class.

Rather than copy 300 lines of `run()`, this subclass rebinds the two module-level names for the
duration of the call and restores them afterwards. Rebinding `few_shot_mixed_dataset`'s name is
what makes the few-shot config work too: `FewShotMixedDataset` then builds our subclass for both
its robot and pixel-only pools without needing a subclass of its own.

Only needed because the policy configs use `n_obs_steps: 2` -- see
`rebuttal/exp4_so101/dataset/so101_sequence_dataset.py` for why the stock frame sampling is wrong for
observation history. With `n_obs_steps: 1` this class is a no-op and the stock workspace behaves
identically.
"""

import json
import os

from omegaconf import OmegaConf

from policies.workspace.train_vla_bc_workspace import TrainVLABCWorkspace
from rebuttal.exp4_so101.ckpt_tokens import same_step
from rebuttal.exp4_so101.dataset.so101_sequence_dataset import SO101SequenceDataset


def _dataset_nodes(cfg):
    """Every ManiSkillSequenceDataset-shaped config node, flat or few-shot mixed."""
    ds = cfg.dataset
    if "robot_dataset_cfg" in ds:
        return [
            ("dataset.robot_dataset_cfg", ds.robot_dataset_cfg),
            ("dataset.pixel_dataset_cfg", ds.pixel_dataset_cfg),
        ]
    return [("dataset", ds)]


def validate_rla_setup(cfg) -> None:
    """Fail immediately, and with instructions, if a BC-RLA run cannot reach its latent targets.

    Without this the failure modes are all late and confusing: an unset `rla_latent_tag` surfaces as
    an interpolation error, a wrong tag as a FileNotFoundError several minutes into dataset
    preloading, and a dataset that simply never loaded latents as a `KeyError: 'rla_latents'` at the
    first training step -- after the model, the optimizer and 40k samples of metadata are built.

    A BC-RLA run must never fall back to "no latent loss": that would train a plain BC policy under a
    BC-RLA name and the loss curve would look completely healthy.
    """
    policy = cfg.get("policy", {})
    if not bool(policy.get("use_latent_head", False)):
        return                                   # vanilla BC: nothing to check

    # Latent targets are ALWAYS offline. Computing them in the loop would derive them from the
    # augmented pixels the policy sees, while the RLA decoder that must interpret those latents was
    # trained on clean frames -- and nothing about the loss curve would reveal it.
    source = str(policy.get("latent_source", "precomputed"))
    if source != "precomputed":
        raise ValueError(
            f"policy.latent_source must be 'precomputed' (offline), got {source!r}. Generate the "
            "targets with rebuttal/exp4_so101/precompute_rla_latents.py."
        )

    missing = [k for k in ("rla_latent_tag", "rla_latent_step", "latent_encoder_work_dir")
               if OmegaConf.is_missing(cfg, k)]
    if missing:
        raise ValueError(
            f"this BC-RLA config still has unset placeholders: {missing}. Set them from the output "
            "of rebuttal/exp4_so101/precompute_rla_latents.py, e.g.\n"
            '    rla_latent_tag: <rla-run>__step0040000        # any name; must match the sidecar dir\n'
            '    rla_latent_step: "0040000"                    # or 40000, or "0040000.snapshot"\n'
            "    latent_encoder_work_dir: runs/so101/16x16_so101_dual/<TIMESTAMP>\n"
            "either in the yaml or on the command line. QUOTE a padded or `.snapshot` step: YAML "
            "reads a bare 0040000 as octal."
        )

    work_dir = str(cfg.latent_encoder_work_dir)
    if not os.path.exists(os.path.join(work_dir, "config.yaml")):
        raise FileNotFoundError(
            f"latent_encoder_work_dir={work_dir!r} has no config.yaml. It must point at a stage-2 "
            "RLA training run directory (the latent SHAPE is read from it; no weights are loaded)."
        )

    want_step = cfg.get("rla_latent_step", None)
    for name, node in _dataset_nodes(cfg):
        if node is None:
            continue
        latent_dir = node.get("rla_latent_dir", None)
        if not latent_dir:
            raise ValueError(
                f"policy.use_latent_head is true but {name} has no rla_latent_dir, so the "
                "batch would carry no targets and compute_loss would fail at step 1. Set "
                "rla_latent_dir (normally `${dataset_root}/rla_latents/${rla_latent_tag}`)."
            )
        manifest_path = os.path.join(str(latent_dir), "manifest.json")
        if not os.path.exists(manifest_path):
            root = os.path.dirname(os.path.normpath(str(latent_dir)))
            available = []
            if os.path.isdir(root):
                for tag in sorted(os.listdir(root)):
                    mp = os.path.join(root, tag, "manifest.json")
                    if os.path.exists(mp):
                        with open(mp) as fh:
                            available.append((tag, json.load(fh).get("rla_step")))
            raise FileNotFoundError(
                f"{name}.rla_latent_dir={latent_dir!r} has no manifest.json.\n"
                f"Sidecars present under {root} (tag -> rla_step): {available or 'NONE'}\n"
                "Run rebuttal/exp4_so101/precompute_rla_latents.py --rla <RLA_RUN> --step <STEP> first, "
                "then set rla_latent_tag to the tag it prints."
            )
        with open(manifest_path) as fh:
            man = json.load(fh)
        # Token comparison, not int: the step may be padded ("0040000") or carry a checkpoint
        # suffix ("0040000.snapshot"). Padding is not part of a step's identity; the suffix is.
        if want_step is not None and man.get("rla_step") is not None \
                and not same_step(man["rla_step"], want_step):
            raise ValueError(
                f"{name}: rla_latent_step={want_step!r} but {latent_dir} was built from RLA "
                f"step {man['rla_step']!r}. Point rla_latent_tag at the matching sidecar or re-run "
                f"precompute_rla_latents.py --step {want_step}."
            )


def resolve_latent_gaps(cfg) -> list | None:
    """Turn `latent_mode` into the concrete list of t -> t+k gaps, and write it into the config.

    The gaps span the **action chunk the policy predicts**, i.e. `horizon` (32 frames, 1.07 s) --
    not `n_action_steps` (16), which is only how much of that chunk gets executed before the next
    replan. `obs_history_stride` is the one that legitimately tracks `n_action_steps`, because the
    history slot is the previous replan's observation.

        latent_mode: final   ->  [horizon]                       one latent for the whole chunk
        latent_mode: ladder  ->  `latent_ladder_steps` gaps,      the chunk sampled evenly
                                 evenly spaced and ending at
                                 horizon, e.g. 8 -> [4, 8, ..., 32]

    A dense 1..32 ladder would be 32 rungs: a 32x wider head, 32x the stored targets, and adjacent
    rungs only 33 ms apart -- almost the same latent repeated. Eight rungs at 0.13 s spacing carry
    the same trajectory information at a fraction of the cost.

    `latent_gaps` cannot be an `${eval:...}` interpolation: hydra calls `OmegaConf.resolve(cfg)`
    (train_policy.py:39), and an OmegaConf resolver that returns a Python list raises
    `UnsupportedValueType: Value 'list' is not a supported primitive type`. (It only works through
    `to_container(resolve=True)`, which hydra does not use here.) Nested interpolation
    `${_gaps_${latent_mode}}` is rejected by the grammar too. So the mapping is done here instead,
    before the dataset and the policy are constructed.

    The same list is written to `policy.latent_gaps` and to every dataset config's `rla_gaps`, so
    the head and the stored targets can never disagree on order or length.
    """
    mode = cfg.get("latent_mode", None)
    if mode is None:
        return None
    chunk = int(cfg.horizon)
    if mode == "final":
        gaps = [chunk]
    elif mode == "ladder":
        steps = int(cfg.get("latent_ladder_steps", 8))
        if not 1 <= steps <= chunk:
            raise ValueError(f"latent_ladder_steps must be in [1, horizon={chunk}], got {steps}")
        # Evenly spaced, de-duplicated if `steps` does not divide `chunk`.
        gaps = sorted({max(1, round(chunk * i / steps)) for i in range(1, steps + 1)})
    else:
        raise ValueError(f"latent_mode must be 'final' or 'ladder', got {mode!r}")

    # INVARIANT: the ladder must reach the end of the chunk. i == steps gives round(chunk) == chunk,
    # so this holds for every `steps`, but it is asserted rather than assumed: a ladder that stopped
    # short would silently supervise a shorter horizon than the policy acts over, and would make the
    # "final" and "ladder" runs incomparable (their last rungs would describe different transitions).
    if gaps[-1] != chunk:
        raise AssertionError(
            f"latent_gaps must end at the last frame of the chunk ({chunk}), got {gaps}"
        )

    OmegaConf.update(cfg, "latent_gaps", gaps, merge=False)
    if OmegaConf.select(cfg, "policy.latent_gaps", default="__absent__") != "__absent__":
        OmegaConf.update(cfg, "policy.latent_gaps", gaps, merge=False)
    ds = cfg.dataset
    for node in ("robot_dataset_cfg", "pixel_dataset_cfg"):
        if node in ds and ds[node] is not None and "rla_gaps" in ds[node]:
            OmegaConf.update(cfg, f"dataset.{node}.rla_gaps", gaps, merge=False)
    if "rla_gaps" in ds:
        OmegaConf.update(cfg, "dataset.rla_gaps", gaps, merge=False)
    return gaps


class TrainSO101Workspace(TrainVLABCWorkspace):
    def __init__(self, cfg, *args, **kwargs):
        gaps = resolve_latent_gaps(cfg)
        if gaps is not None:
            print(f"[TrainSO101Workspace] latent_mode={cfg.latent_mode} -> latent_gaps={gaps}")
        # Before anything expensive is built, so a misconfigured BC-RLA run dies in seconds with
        # instructions rather than minutes later with a KeyError.
        validate_rla_setup(cfg)
        if bool(OmegaConf.select(cfg, "policy.use_latent_head", default=False)):
            print(
                f"[TrainSO101Workspace] RLA targets: tag={cfg.get('rla_latent_tag')} "
                f"step={cfg.get('rla_latent_step')}"
            )
        super().__init__(cfg, *args, **kwargs)

    def run(self):
        import policies.dataset.few_shot_mixed_dataset as few_shot_mod
        import policies.workspace.train_vla_bc_workspace as workspace_mod

        original = (workspace_mod.ManiSkillSequenceDataset, few_shot_mod.ManiSkillSequenceDataset)
        workspace_mod.ManiSkillSequenceDataset = SO101SequenceDataset
        few_shot_mod.ManiSkillSequenceDataset = SO101SequenceDataset
        try:
            return super().run()
        finally:
            workspace_mod.ManiSkillSequenceDataset, few_shot_mod.ManiSkillSequenceDataset = original
