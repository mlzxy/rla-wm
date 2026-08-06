"""
rla/patch.py

Installs the RLA integration into an already-loaded `finetune` module.

Every patch below replaces a *module global* that `vla-scripts/finetune.py` looks up at call time,
which is why none of this needs the upstream file to change:

    finetune.RLDSBatchTransform                 -> RlaBatchTransform      (adds batch["rla"])
    finetune.PaddedCollatorForActionPrediction  -> RlaCollator            (stacks it)
    finetune.L1RegressionActionHead             -> RlaL1RegressionActionHead
    finetune.run_forward_pass                   -> + lambda * L1(z_hat, z*)
    finetune.log_metrics_to_wandb               -> + the RLA metrics
    finetune.get_peft_model                     -> + stage-1 VLM warm start
    finetune.init_module                        -> + stage-1 head/proprio warm start

plus one outside it:

    prismatic.vla.datasets.rlds.dataset.normalize_action_and_proprio -> + observation["rla_slot"]

`install()` must run before `finetune(cfg)` is called, i.e. before the dataset is built and before
the model is constructed. `rla/train.py` is the entry point that guarantees the ordering.
"""

from __future__ import annotations

from rla import data, loss, term, warmstart
from rla.action_head import RlaL1RegressionActionHead
from rla.config import CFG


def install(finetune_mod) -> None:
    """Patch a loaded `finetune` module in place. Safe to call once per process."""
    term.note(CFG.describe())

    if not CFG.enabled:
        # RLA_STEPS=0: the vanilla arm. Nothing is patched, so the run is upstream's own code path
        # and its checkpoints are interchangeable with a plain `tools/train_eval.sh` run.
        term.note("[rla] nothing patched -- this is the vanilla arm", "yellow")
        return

    if CFG.has_targets:
        data.install_dataset_hook()
        finetune_mod.RLDSBatchTransform = data.RlaBatchTransform
        finetune_mod.PaddedCollatorForActionPrediction = data.RlaCollator
    else:
        raise RuntimeError(
            "RLA_STEPS>0 but RLA_SIDECAR is unset. Training needs targets. (Evaluation does not -- "
            "z is an output, never an input -- but that path goes through rla/eval.py.)"
        )

    finetune_mod.L1RegressionActionHead = RlaL1RegressionActionHead
    finetune_mod.run_forward_pass = loss.make_run_forward_pass(finetune_mod.run_forward_pass)
    finetune_mod.log_metrics_to_wandb = loss.make_log_metrics_to_wandb(
        finetune_mod.log_metrics_to_wandb
    )

    if CFG.init_from:
        finetune_mod.get_peft_model = warmstart.make_get_peft_model(finetune_mod.get_peft_model)
        finetune_mod.init_module = warmstart.make_init_module(finetune_mod.init_module, finetune_mod)

    term.note("[rla] patched: dataset hook, batch transform, collator, action head, loss, logging"
              + (", warm start" if CFG.init_from else ""), "green")


def install_eval() -> None:
    """The evaluation-side patch: one class swap.

    `experiments/robot/openvla_utils.get_action_head` references `L1RegressionActionHead` as a
    module global, constructs it from four fixed kwargs and then calls `load_state_dict(...)`
    strict. Swapping the global is enough -- `find_checkpoint_file`, the strict load and everything
    else stay upstream's, and the strict load becomes the tripwire that catches a train/eval
    disagreement about `RLA_STEPS`.
    """
    import experiments.robot.openvla_utils as openvla_utils

    term.note(CFG.describe())
    if not CFG.enabled:
        term.note("[rla] nothing patched -- evaluating a vanilla checkpoint", "yellow")
        return

    openvla_utils.L1RegressionActionHead = RlaL1RegressionActionHead
    term.note("[rla] patched: openvla_utils.L1RegressionActionHead", "green")
