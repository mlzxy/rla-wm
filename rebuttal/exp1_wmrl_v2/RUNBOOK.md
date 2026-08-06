# WMRL v2 runbook

Copy-paste commands for the re-run, in order. Background and rationale:
[README.md](README.md).

Everything runs from the repo root. The eval tool sets `PYTHONPATH` itself; the training
script exports it. All long commands are **safe to interrupt and re-run** — that is the
point of the resume design.

```bash
cd <repo root>          # the directory containing train.py
```

> **Eval and training both want the whole machine.** Eval runs one worker per GPU and
> takes `--gpus`; a training run spreads its world-model vec env across every visible GPU
> and must **not** be constrained with `CUDA_VISIBLE_DEVICES` (see §3). Run them one after
> the other, not at the same time.

> **On this 7-GPU box, GPU 1 has an uncorrectable ECC error** — a job on it dies with
> `CUDA error: uncorrectable ECC error encountered`, so eval commands below say
> `--gpus 0,2-6`. Not relevant on another machine: use `--gpus all` (the default) there.

---

## 0. Cheat sheet

| Goal | Command |
|---|---|
| Evaluate a folder of checkpoints on 7 GPUs | `python rebuttal/tools/eval_ckpts.py <dir> --out-dir <results> --gpus 0,2-6` |
| See the plan without running | add `--dry-run` |
| Single-GPU debugging | `--gpus 0` (runs in-process, no subprocess) |
| Just a few checkpoints | `--max-ckpts 4` |
| Resume | re-run the same command |
| Watch progress | `tail -f runs/eval_v2_logs/gpu0_*.log` |
| Headline table | `python rebuttal/tools/analyze_eval.py delta --results <results> --baseline runs/weights/selection_v2.json` |
| Launch RL for one task | `bash rebuttal/exp1_wmrl_v2/train.sh <task> <reward> <seeds...>` — uses **all** visible GPUs; never pin it to one |

Tasks: `pusht rollball pullcube pullcubetool pokecube`.
Reward: `goal` for **pokecube**, `corresponding` for the other four.

---

## 1. Confirm the tooling (~10 min)

The four selected models are already evaluated for 1 set in `runs/eval_smoke`. Re-running
should skip everything — that is the resume check:

```bash
python rebuttal/tools/eval_ckpts.py \
    runs/weights/wmrl_init_v2 runs/weights/rl_best_v2 \
    --num-sets 1 --out-dir runs/eval_smoke --gpus 0,2-6 --no-adopt-legacy
```

Full pass over all 10 selected models (~1 h on 6 GPUs). Optional, but it is the
honest end-to-end test:

```bash
python rebuttal/tools/eval_ckpts.py \
    runs/weights/wmrl_init_v2 runs/weights/rl_best_v2 \
    --out-dir runs/eval_selected --log-dir runs/eval_selected_logs --gpus 0,2-6

python rebuttal/tools/analyze_eval.py summary --results runs/eval_selected
```

Expect these to land within ~1pp of the numbers in [results/](results/); exact agreement is
not expected and is not required.

---

## 2. Re-validate the BC picks (optional, ~2 h on 6 GPUs)

The picks come from the April sweep. That sweep reproduces on current code to within about
1 percentage point (mean |Δ| = 0.007 over six checkpoints), and the epoch-to-epoch gaps that decided the selection are larger than that. So this step
is **not a prerequisite** — run it only if you want the extra confidence.

```bash
# the top-5 BC candidates per task, as recorded in April
python rebuttal/tools/analyze_eval.py top --results runs/eval_lot_results \
    --group BC -k 5 --csv runs/analysis/bc_top5.csv

# turn that into a target list and evaluate all 25 under the current code
tail -n +2 runs/analysis/bc_top5.csv | awk -F, '{print $NF}' > runs/analysis/bc_top5.txt
python rebuttal/tools/eval_ckpts.py runs/analysis/bc_top5.txt \
    --out-dir runs/eval_bc_recheck --log-dir runs/eval_bc_recheck_logs \
    --gpus 0,2-6 --no-adopt-legacy

python rebuttal/tools/analyze_eval.py top --results runs/eval_bc_recheck --group BC -k 5
```

If a different epoch now wins for some task, re-point that task's config:

```bash
# example: PullCube's best is now epoch 22 rather than 19
mkdir -p runs/weights/wmrl_init_v2/2_bc_pullcube/checkpoints
cp -n runs/bc_model_pool_extra/2_bc_s2r_nstate_final/checkpoints/latest_epoch22.ckpt \
      runs/weights/wmrl_init_v2/2_bc_pullcube/checkpoints/
# then edit `pretrained_ckpt:` in rebuttal/exp1_wmrl_v2/configs/pullcube.yaml
```

If you do run this, run it **before** §3 — re-running the RL sweep afterwards costs days.

---

## 3. Train RL: 5 tasks × seeds 1–15

> **Do not set `CUDA_VISIBLE_DEVICES`.** `wmrl/train.py:120-131` switches the world-model
> vec env to `MultiProcessWorldModelVecEnv` whenever it sees more than one GPU and spreads
> the imagined rollouts across **all** visible GPUs — one worker per GPU, with the 112
> envs divided evenly between them. Restricting the variable reduces the number of workers
> and makes each run slower. **One training run is meant to occupy the whole machine, so
> run the tasks sequentially rather than one per GPU.**

### Pre-flight on a new machine (~10 min, do this first)

Everything below was validated on the 7-GPU box, and the training smoke test there ran
under `--debug`, which replaces `num_envs` with 8, `mini_batch_size` with 4, `end_traj_id`
with 10, and turns wandb off. So the real memory footprint, the full dataset, the
multi-GPU world model and wandb are **not** covered by it. Spend ten minutes here before
committing days of compute:

```bash
flags="--total-iterations 2 --run-dir runs/wmrl_v2_smoke" \
    bash rebuttal/exp1_wmrl_v2/train.sh pokecube goal 1
```

Note: **no `--debug`** (that is the point), and `--run-dir` sends it somewhere disposable
— a stray 2-iteration run under `runs/wmrl_v2/` would later be discovered by §4 and
evaluated as if it were a real seed-1 run.

Check, in order:

1. `Using multi-process env with N visible GPUs` — `N` must be every GPU on the box. If
   this line is missing, the single-process env was used and the run will be far slower.
2. `nvidia-smi` during the run. The 112 envs are split across the visible GPUs, so 4 GPUs
   means 28 envs each vs 16 here — roughly 75 % more per card. This is where an OOM would
   appear, and it is the most likely thing to go wrong on a different box.
3. wandb. The configs set `use_wandb: true`, and credentials live in `~/.netrc` on *this*
   machine. If the new one has none, `wandb.init` can block or fail and stall the whole
   chain at run 1. Either `wandb login` first, or `export WANDB_MODE=offline`, or append
   `--no-use-wandb` to `flags`.
4. `runs/wmrl_v2_smoke/**/final.pt` exists, and `config.yaml` sits beside it.

Then clean up and time a longer run to size the sweep:

```bash
rm -rf runs/wmrl_v2_smoke
flags="--total-iterations 5 --run-dir runs/wmrl_v2_smoke" bash rebuttal/exp1_wmrl_v2/train.sh pokecube goal 1
rm -rf runs/wmrl_v2_smoke
```

Then extrapolate to 300 iterations × 75 runs and launch the sweep. Seeds run sequentially
inside the script, and the tasks must run sequentially too:

```bash
bash rebuttal/exp1_wmrl_v2/train.sh pokecube     goal          $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh pusht        corresponding $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh rollball     corresponding $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh pullcube     corresponding $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh pullcubetool corresponding $(seq 1 15)
```

As a single line, to paste into one `tmux`/`screen` session and leave running:

```bash
for t in "pokecube goal" "pusht corresponding" "rollball corresponding" "pullcube corresponding" "pullcubetool corresponding"; do bash rebuttal/exp1_wmrl_v2/train.sh $t $(seq 1 15); done
```

Use `;` (or the loop above), **not** `&&`: each run exits non-zero from the SAPIEN
teardown segfault after its checkpoint is saved, so `&&` would stop the chain after the
first task.

`rebuttal/exp1_wmrl_v2/train.sh` passes exactly the same hyperparameters as `wmrl/train.sh` — `--num-envs 112
--num-steps 4 --mini-batch-size 224 --total-iterations 300 --bc-loss-weight 0.0
--use-critic --policy-kwargs.enable-rl-lora --env-kwargs.terminal-success-bonus 0.0
--no-run-initial-eval`. The only additions are the `v2` tag, the `rebuttal/exp1_wmrl_v2/configs/` config
path, and `--no-eval`. Verify at any time with:

```bash
diff <(grep -o '\.venv/bin/python wmrl/train\.py.*' wmrl/train.sh    | tr ' ' '\n') \
     <(grep -o '\.venv/bin/python wmrl/train\.py.*' rebuttal/exp1_wmrl_v2/train.sh | tr ' ' '\n')
```

Output layout, which is what §4 discovers:

```
runs/wmrl_v2/<task>-<task>-v2-<corr|goal>-lora-seed<N>-run0/<timestamp>/
    config.yaml          # written before training; the eval tool needs it
    ckpt_{25,50,...,275}.pt
    final.pt
```

`--total-iterations 300` runs iterations 0–299 and `save_freq: 25` saves at
`iteration % 25 == 0 and iteration > 0`, so the last periodic checkpoint is **275**, not
300 — 11 periodic + `final.pt` = 12 per run, matching the April pool. Expect ~54 GB total
(75 runs × 12 × 60 MB).

### Running on a different machine

`runs/` and `data/` may be symlinks into a scratch filesystem rather than real directories
in the checkout. On another box, make sure they resolve before launching, or the run dies at
startup:

```bash
ls runs/weights/wmrl_init_v2/*/checkpoints/*.ckpt   # 5 BC initializers (~950 MB)
ls -d runs/weights/rla-wm/maniskill/*/              # world model
ls -d runs/weights/rla/maniskill/*/                 # RLA encoder
ls -d runs/weights/dino-to-image_unet/maniskill/*/  # image decoder
ls -d data/maniskill/ppo/*/                         # BC dataset + env init states
```

Everything is referenced by repo-relative path from the configs, so either mount the same
filer or rsync those five trees and recreate the `runs/` and `data/` symlinks. Results
land under whatever `runs/` resolves to there, so `runs/wmrl_v2/` must be writable.

**A segfault after `Saved final checkpoint` is normal** — SAPIEN teardown, pre-existing,
affects `wmrl/train.sh` identically. The checkpoint is already on disk and the loop
continues to the next seed.

Check what finished:

```bash
ls -d runs/wmrl_v2/*/*/ | wc -l                       # runs started
find runs/wmrl_v2 -name final.pt | wc -l              # runs finished
python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 --dry-run --gpus none --show 20
```

---

## 4. Evaluate everything (~2.5 days on 6 GPUs)

```bash
python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 \
    --out-dir runs/eval_v2_results \
    --log-dir runs/eval_v2_logs \
    --gpus 0,2-6 \
    --min-age-sec 600
```

Run this **after** §3 finishes — a training run already occupies every visible GPU, so
evaluating at the same time just makes both slower.

`--min-age-sec 600` skips checkpoints written in the last 10 minutes. That matters if you
do choose to overlap the two (e.g. evaluating finished tasks while a later task still
trains): it prevents reading a checkpoint mid-write. Re-run the command whenever more
checkpoints appear — finished ones are skipped.

Monitor:

```bash
ls runs/eval_v2_results/*.json | wc -l                     # results so far
tail -f runs/eval_v2_logs/gpu0_*.log | grep -E '^\(run|done|set)'
grep -h EVAL_CKPTS_SUMMARY runs/eval_v2_logs/*.log | tail  # per-worker totals
```

Interrupted? Just re-run the same command. Finished checkpoints are skipped and a
partially-evaluated one continues from its first missing set.

### Useful variations

```bash
# only the early iterations, if the curve says that is where the action is
python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 --epochs 25-100,final --out-dir runs/eval_v2_results --gpus 0,2-6

# one task
python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 --include pokecube --out-dir runs/eval_v2_results --gpus 0,2-6

# more seeds — existing set files are extended, not redone
python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 --num-sets 30 --out-dir runs/eval_v2_results --gpus 0,2-6

# score the EMA weights of a BC ckpt instead (written to a separate *_ema_*.json)
python rebuttal/tools/eval_ckpts.py runs/weights/wmrl_init_v2 --bc-weights ema --out-dir runs/eval_v2_results --gpus 0,2-6
```

---

## 5. Analyse

```bash
R=runs/eval_v2_results

# headline: BC initializer vs what RL reached, per task
python rebuttal/tools/analyze_eval.py delta --results $R \
    --baseline runs/weights/selection_v2.json

# success rate vs RL iteration, mean +/- std over the 15 seeds
python rebuttal/tools/analyze_eval.py curve --results $R

# one row per seed: which iteration was its best
python rebuttal/tools/analyze_eval.py seeds --results $R --task PokeCube

# peak / abs-peak / distribution table (the legacy view)
python rebuttal/tools/analyze_eval.py summary --results $R

# best checkpoints with resolved paths
python rebuttal/tools/analyze_eval.py top --results $R -k 10

# everything, plus a self-contained HTML file
python rebuttal/tools/analyze_eval.py report --results $R \
    --baseline runs/weights/selection_v2.json \
    --html runs/analysis/eval_v2.html
```

Compare against April **only** with both sets labelled, never merged into one number:

```bash
python rebuttal/tools/analyze_eval.py summary \
    --results runs/eval_lot_results:apr --results runs/eval_v2_results:v2
```

Re-select the best checkpoints from the new sweep:

```bash
python rebuttal/tools/analyze_eval.py select --results $R \
    --out runs/weights/selection_v3.json \
    --md runs/weights/selection_v3.md \
    --plan runs/weights/copy_selection_v3.sh
bash runs/weights/copy_selection_v3.sh     # cp -n, never overwrites
```

---

## 6. Reference

### `eval_ckpts.py` — flags worth knowing

| Flag | Effect |
|---|---|
| `--gpus 0,2-6` / `all` / `none` | Which GPUs. One GPU ⇒ in-process, easy to debug. |
| `--dry-run` | Print the plan and each worker's exact command, then exit. |
| `--num-sets N --episodes-per-set M` | Set *k* covers seeds `(k-1)M+1 .. kM`. |
| `--epochs 25-100,final` | Filter by iteration; `final` means `final.pt`. |
| `--include RE` / `--exclude RE` | Regex on the checkpoint path. Repeatable. |
| `--max-ckpts N` | First N jobs only, applied before sharding. |
| `--min-age-sec S` | Skip checkpoints newer than S seconds (live trainer). |
| `--bc-weights model\|ema` | Which weight set of a BC ckpt to score. |
| `--force` | Redo everything, ignoring existing results. |
| `--redo-partial` | Restart partially-evaluated checkpoints at set 1. |
| `--no-adopt-legacy` | Ignore April-style timestamped JSONs when resuming. |
| `--tee --tee-only 0` | Mirror one worker's output to this terminal. |
| `--retries N` | Relaunch a crashed worker; finished sets are already on disk. |

Targets can be a checkpoint file, a run dir, a dir of run dirs, or a `.txt` list of any of
those — mixed freely in one command.

### Where things live

| Path | |
|---|---|
| `runs/weights/wmrl_init_v2/` | BC initializers for the v2 RL runs |
| `runs/weights/rl_best_v2/` | best April RL checkpoint per task (tool testing) |
| `runs/weights/selection_v2.{json,md}` | what was picked, and from where |
| `rebuttal/exp1_wmrl_v2/configs/*.yaml` | v2 training configs |
| `runs/wmrl_v2/` | v2 RL run outputs |
| `runs/eval_v2_results/` | v2 eval results |
| `runs/eval_lot_results/` | April results — **do not merge with the above** |

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA error: uncorrectable ECC error` | GPU 1 on this box only. Eval: `--gpus 0,2-6`. |
| Training much slower than expected | `CUDA_VISIBLE_DEVICES` was set, so fewer world-model workers ran (or the single-process env was used). Unset it and look for `Using multi-process env with N visible GPUs` at startup. |
| Training OOMs on a machine with fewer GPUs | The 112 envs are split across however many GPUs are visible, so fewer GPUs means more envs each. Lower `--num-envs` via `flags="--num-envs 64"`, but note that then no longer matches `wmrl/train.sh`. |
| `Segmentation fault` after `Saved final checkpoint` | Pre-existing SAPIEN teardown crash. Harmless. |
| `plan drift: parent=… worker=…`, exit 3 | Checkpoints appeared or vanished mid-launch. Re-run; or `--min-age-sec 600` while training. |
| `was written with different eval settings` | The results dir has files from a different `--episodes-per-set`. Use a new `--out-dir` or `--force`. |
| Worker log ends with no `EVAL_CKPTS_SUMMARY` | Worker was killed. Re-run; per-set resume loses at most one set. |
| `No module named 'notes'` | You ran the old `run_eval_lot.py`, which is not shipped here. Use `rebuttal/tools/eval_ckpts.py`. |
| A checkpoint scores differently than April | Up to ~1pp on a full-sweep mean is normal ([why). A single 50-episode set can differ far more and means nothing on its own. |
