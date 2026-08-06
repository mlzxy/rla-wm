# exp-3 — Training the world model on more diverse data

The world model had only ever seen near-success rollouts with the goal marker in one fixed spot.
We rebuilt its training set to cover a much wider range, trained the stack on it, and re-ran WMRL
on Push-T.

## The data

Four collections, all shipped (see [`../DATA.md`](../DATA.md)):

| `data/maniskill/…` | What is in it |
|---|---|
| `ppo_diverse/` | 5 004 successful trajectories, **one distinct goal position each**, drawn over a disc of radius 14 cm with the ten evaluation goals excluded at a 25 mm margin |
| `ppo_noisy/` | the original PPO rollouts replayed with injected action noise and a randomly placed goal (4 500), plus successful demonstrations at six specific shifted goals (6 × 50) |
| `rerender/` | the original trajectories, unchanged, re-rendered with the goal marker moved somewhere else — the block's path is bit-identical while the marker moves up to 194 mm |
| `initframes/` | first frames only, at the ten held-out evaluation goals |

The old training set placed the goal marker at exactly one location, so a shifted-goal frame was
out of distribution as an image and the model fell back on a "the block ends at the marker" prior.
`ppo_diverse` shows the block following the marker; `rerender` shows the marker moving while the
block does *not* follow. Together they pin down the actual rule instead of the prior.

## Training

Three stages, in this order — the world model consumes the autoencoder's latent space, so a later
autoencoder run invalidates every flow model trained against the old one.

```bash
export PYTHONPATH=.:./third_party/diffusion_policy
PY=.venv/bin/python

# 1. the RLA autoencoder
$PY train.py --config rebuttal/exp3_data_diversity/configs/rla_diverse.yaml \
             --output_dir runs/pusht_diverse --num_gpus 4

# 2. the flow-matching world model, against the autoencoder above
$PY train.py --config rebuttal/exp3_data_diversity/configs/wm_diverse.yaml \
             --output_dir runs/pusht_diverse --num_gpus 4
```

Both configs pin their frozen dependencies to a single-checkpoint directory rather than "whatever
is newest" — the headers say why, and the released pins are in [`../DATA.md`](../DATA.md), so you
can skip straight to stage 3 with the shipped checkpoints.

```bash
# 3. WMRL on top of it, one seed
$PY wmrl/train.py --config-file rebuttal/exp3_data_diversity/configs/wmrl_pusht.yaml \
    --tag pusht-diverse-corr-seed1 --seed 1 \
    --env-kwargs.reward-mode corresponding --env-kwargs.terminal-success-bonus 0.0 \
    --total-iterations 300 --num-envs 112 --num-steps 4 --mini-batch-size 224 \
    --bc-loss-weight 0.0 --use-critic --policy-kwargs.enable-rl-lora \
    --no-run-initial-eval --no-eval
```

`--no-eval` is deliberate: the 50-episode eval that runs during training is not the metric. Every
checkpoint is scored offline afterwards on a fixed evaluation set with
[`../tools/eval_ckpts.py`](../tools/eval_ckpts.py), and the tables come from
[`../tools/analyze_sweep.py`](../tools/analyze_sweep.py).

`configs/wmrl_pusht.yaml` differs from [`../exp1_wmrl_v2/configs/pusht.yaml`](../exp1_wmrl_v2/configs/pusht.yaml)
only in which world model and latent decoder it loads, so exp-1's run is a clean control.

## Results

Push-T, the same fixed evaluation seed set per checkpoint, deterministic actions, one harness,
15 RL seeds per arm. Both conventions are reported because they answer slightly different questions.

| convention | old world model | diverse-data world model |
|---|---|---|
| **Ladder mean** — every checkpoint of every seed, pooled, no selection | 0.1362 | **0.1390** |
| **Best-per-seed, averaged** — max over each seed's 12 checkpoints | 0.1732 | **0.1851** |

Training the world model on the wider distribution moves the number up under both conventions,
by +1.2 points on the best-per-seed convention. The BC initializer both arms start from scores
0.1507 on the same evaluation set.

Full tables, including the split-half version that removes the max-over-12 selection bias, are in
[`results/RESULTS.md`](results/RESULTS.md) and [`results/corr_report.md`](results/corr_report.md).
