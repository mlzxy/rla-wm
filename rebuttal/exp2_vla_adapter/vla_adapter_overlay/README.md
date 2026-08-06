# vla_adapter_overlay

Copy `rla/` and `tools/` into your VLA-Adapter clone and you are done:

```bash
REF=<your VLA-Adapter clone>          # ../setup/setup_env.sh builds it
cp -r rebuttal/exp2_vla_adapter/vla_adapter_overlay/rla   "$REF/"
cp -r rebuttal/exp2_vla_adapter/vla_adapter_overlay/tools "$REF/"
```

Nothing under `prismatic/`, `vla-scripts/` or `experiments/` changes. `rla/train.py` loads
`vla-scripts/finetune.py` by path and swaps seven of its module globals at import time — the batch
transform, the collator, the action head, the forward pass, the wandb logger, `get_peft_model` and
`init_module` — plus `normalize_action_and_proprio` one level down in the RLDS pipeline. All of
them are looked up at call time, so replacing the name is enough. `rla/patch.py` has the full list
in its docstring, and `rla/REVIEW.md` is the audit that checks the claim.

With `RLA_STEPS=0` nothing is patched at all and the run is byte-for-byte upstream's code path, so
the vanilla arm and the RLA arm produce interchangeable checkpoints.

`tools/` holds the launchers: `train_eval.sh` (vanilla), `train_eval_rla.sh` (+ RLA),
`plot_loss.py`, and `OOM_DEBUGGING.md`, which is where the host-RAM limits that shaped every batch
size are written down. The environment itself is built once, before any of this, by
[`../setup/setup_env.sh`](../setup/setup_env.sh).

The package is deliberately upstream-clean and is copied here verbatim.
