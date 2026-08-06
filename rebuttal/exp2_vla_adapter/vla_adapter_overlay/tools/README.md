# tools/

Wrappers around this pristine upstream VLA-Adapter clone. Nothing here modifies upstream code —
`vla-scripts/finetune.py` and `experiments/robot/libero/run_libero_eval.py` run exactly as shipped.
These scripts only supply the arguments, the environment variables, and the bookkeeping.

This clone runs on upstream's own pinned stack (torch 2.2.0 / transformers 4.40.1 / timm 0.9.10 /
numpy 1.26.4 / peft 0.11.1 / flash-attn 2.5.6, with robosuite 1.4.1 / mujoco 3.3.0 / LIBERO
`8f1084e3`), in a virtualenv of its own, so nothing about the results is entangled with the host
repository's environment.

## Building the environment

`rebuttal/exp2_vla_adapter/setup/setup_env.sh` in the host repository builds this clone, its
`.venv`, and `LIBERO/`, and links in the datasets and the pretrained VLM. Run it once per machine
(or after the venv is lost). It installs the packages upstream's own `our_envs.txt` and LIBERO's
`requirements.txt` do **not** get you — each commented in the script with the failure it fixes —
and asserts the `peft==0.11.1` and `numpy==1.26.x` pins before it exits.

## train_eval.sh

Train one LIBERO suite, plot the loss, then evaluate.

```bash
tools/train_eval.sh spatial                    # defaults
ACCUM=1 tools/train_eval.sh object             # grad accumulation 2 -> 1
GPUS="0 1 2 3" tools/train_eval.sh goal        # multi-GPU DDP
EVAL_AFTER=0 tools/train_eval.sh long          # train only (still plots the curve)
SAVE_FREQ=500 tools/train_eval.sh object       # object is only 5000 steps; default 5000 gives 1 checkpoint
```

Defaults: batch 8/GPU, accum 2, lr 2e-4, LoRA r64, and per-suite step counts from upstream
issue #35 — spatial 20000, object 5000, goal 50000, long 60000. Those step counts are quoted at
upstream's effective batch of 16 (one GPU, batch 8 × accum 2); the script prints the effective
batch it actually gets so you can see when you have moved off that point.

What it does, in order:

1. train
2. **plot the loss curve** → `train_logs/<suite>--<stamp>-loss.png`, always, even with
   `EVAL_AFTER=0`. Both curr-action and full-chunk L1 are drawn (they differ by ~25%, so mixing
   them up flatters or maligns a run for free). A tail that sits >25% above the run's minimum is
   labelled `** RISING — likely diverged **`.
3. **full-eval the final checkpoint immediately** — the number the run was for
4. rank the remaining checkpoints on a short eval, and full-eval the winner if it beat the final

Step 4 exists because success collapses well before the loss does: measured on LIBERO-Spatial, one
run scored 95% at step 10000 and **0%** at 30000–50000 while its training loss kept improving. The
last checkpoint is not reliably the best one.

`SWEEP=0` skips step 4. `SWEEP_ORDER=desc` walks the sweep newest-first (default `asc`, oldest
first, so you read the curve rise → peak → collapse in order).

**Results are written to `eval_out/<run_id>--<stamp>-results.tsv` as each eval finishes**, not at
the end, so losing the terminal or the node does not lose the numbers. The file carries the run
config in its header and is self-describing.

Use `RUN_TAG=v2` to keep an earlier run's checkpoints: the run id is otherwise fixed per suite and
a rerun overwrites `<run_id>--<step>_chkpt` step for step.

## plot_loss.py

Called by `train_eval.sh`; also usable on any finished run. It re-execs itself under `.venv` if
started by another interpreter, so the shebang works from anywhere.

```bash
tools/plot_loss.py --run-id REF-Spatial --out curve.png --steps 20000
```

Reads the run's wandb record when available (real `_step` axis), else parses `curr:` out of a
training log passed with `--log`.
