# exp-1 — WMRL v2: an honest re-measurement

The WMRL numbers in the paper came from a training loop that also picked its own starting
point. That is a bad combination: the BC initializer was chosen by a 50-episode evaluation
run *during* BC training, which is noisy enough that the "best" epoch it picks is often not
the best epoch. v2 fixes the measurement, not the method.

Two things changed, and nothing else:

1. **The BC initializer is re-picked offline** by scoring every saved BC epoch on the full
   episodes instead of 50, and RL starts from the weights that were actually ranked
   (`pretrained_weights: model`, because the offline scorer scores `model`, not `ema_model`).
2. **The reported metric is the offline sweep**, not the in-training evaluation. The
   configs turn in-training eval off entirely (`no_eval: true`) so there is no temptation to
   read it.

Everything else — the world model, the policy architecture, the PPO hyperparameters, the
LoRA setup, the 15 seeds, the 300 iterations — is byte-identical to `wmrl/configs/*.yaml`.
Five settings differ and nothing else; `diff wmrl/configs/pusht.yaml
rebuttal/exp1_wmrl_v2/configs/pusht.yaml` shows those five plus the header comment that
explains them.

| setting | `wmrl/configs/` | here |
|---|---|---|
| `pretrained_ckpt` | the epoch the noisy train-time eval picked | the epoch the offline sweep picked |
| `pretrained_weights` | absent, i.e. `ema` | `model` |
| `run_dir` | `runs/wmrl_critic` | `runs/wmrl_v2` |
| `run_initial_eval` | `true` | `false` |
| `no_eval` | absent, i.e. `false` | `true` |

## What we found

905 checkpoints, the same fixed evaluation seed set each, deterministic actions.
Full tables in [results/v2_sweep_report.md](results/v2_sweep_report.md).

**Table A convention: per seed, the maximum over that seed's 12 checkpoints, then mean ± std
across seeds. This is biased upward** — the BC side has one checkpoint to choose from and
the RL side has twelve. Table A2 in the report is the split-half version that removes the
bias, and it is the number to quote if you only quote one.

| Task | seeds | BC | WMRL, max-of-12 | WMRL, split-half honest | Δ honest |
|---|---:|---:|---:|---:|---:|
| PushT-v2 | 15 | 0.1507 | 0.1732 ± 0.0163 | 0.1641 ± 0.0183 | **+0.0126** |
| RollBall-v1 | 15 | 0.6533 | 0.6557 ± 0.0440 | 0.6428 ± 0.0469 | −0.0115 |
| PullCube-v2 | 15 | 0.8533 | 0.8370 ± 0.0171 | 0.8311 ± 0.0176 | −0.0214 |
| PullCubeTool-v1 | 15 | 0.4000 | 0.3771 ± 0.0411 | 0.3703 ± 0.0414 | −0.0314 |
| PokeCube-v2 | 15 | 0.8973 | 0.9420 ± 0.0106 | 0.9375 ± 0.0103 | **+0.0415** |

Read plainly: **WMRL helps on two of the five tasks, hurts on two, and is a wash on the
fifth.** PokeCube
is a real, large effect (seed-level sign test 15/15, episode-level McNemar p = 6.4e-09 on the
best checkpoint). PushT is a smaller but consistent effect (13/15 seeds above BC, seed-level
t-test p = 5.2e-05). PullCube and PullCubeTool are genuinely negative — RL at these settings
makes the policy slightly worse than the BC initializer it started from. RollBall is a wash.

Pooled across all 75 (task, seed) pairs: mean Δ = +0.0060, 47/75 seeds above BC, sign test
p = 0.0185. That is a real but small aggregate effect that is driven almost entirely by
PokeCube.

One genuinely clean secondary result, from Table B: on the tasks where success rate barely
moves, **the successful episodes get shorter**. Matched on episodes both policies solve,
PokeCube drops 7.9 control steps (15/15 seeds faster, p = 3.1e-05) and RollBall drops 6.3
(14/15, p = 6.1e-05). So RL is doing something even where the headline number is flat.

## Running it

Everything runs from the repo root.

```bash
export PYTHONPATH=.:./third_party/diffusion_policy

# 1. Get the re-picked BC initializers (see ../DATA.md for the download).
#    They land at runs/weights/wmrl_init_v2/<n>_bc_<task>/checkpoints/latest_epoch*.ckpt

# 2. Train. 15 seeds per task, sequentially; each uses all visible GPUs.
bash rebuttal/exp1_wmrl_v2/train.sh pokecube     goal          $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh pusht        corresponding $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh rollball     corresponding $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh pullcube     corresponding $(seq 1 15)
bash rebuttal/exp1_wmrl_v2/train.sh pullcubetool corresponding $(seq 1 15)

# 3. Score every checkpoint. On one machine:
.venv/bin/python rebuttal/tools/eval_ckpts.py runs/wmrl_v2 \
    --out-dir runs/eval_v2/results --num-sets 15 --episodes-per-set 50 --gpus 0-3

# 4. Tables.
.venv/bin/python rebuttal/tools/analyze_sweep.py \
    --results runs/eval_v2/results --out runs/eval_v2/tables
```

Step 3 is ~240 GPU-hours. `eval_ckpts.py` claims work per checkpoint through a directory on
shared storage, so several machines can run it against the same `--out-dir` and join or leave
mid-sweep; a machine that dies needs no cleanup.

## Files

| Path | What |
|---|---|
| [configs/](configs/) | The five v2 configs. Five lines differ from `wmrl/configs/`. |
| [train.sh](train.sh) | Launcher. Same interface as `wmrl/train.sh`. |
| [results/](results/) | The sweep tables as produced, CSV and markdown. |
| [RUNBOOK.md](RUNBOOK.md) | Operational detail: resuming, partial sweeps, what each flag does. |

## Caveats worth stating

- Flow-matching noise is not seed-controlled, so re-running does not reproduce these numbers
  bit for bit.
- Table A cannot be made symmetric: the rest of the BC epoch sweep is no longer on disk, so
  there is no best-of-N for the BC side. That is why A2 exists.
- `ep_best_mcnemar_p` in the report is a description, not a test — the checkpoint was chosen
  for being the best of ~180 on the same episodes it is then scored on.
