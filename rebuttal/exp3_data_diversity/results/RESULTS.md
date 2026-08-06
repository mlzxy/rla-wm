# WMRL on Push-T with the diverse-data world model

The exp-1 protocol, unchanged, with one variable moved: the world model.
`rebuttal/exp3_data_diversity/configs/wmrl_pusht.yaml` differs from
`rebuttal/exp1_wmrl_v2/configs/pusht.yaml` only in which world model and latent decoder it loads.

All rows: the same fixed evaluation seed set, deterministic actions, identical harness. The BC initializer
both arms start from re-scores to 0.1507 on the same episodes.

## 1. Ladder mean — no selection of any kind

Per-seed mean over that seed's whole checkpoint ladder, then averaged over seeds.

| arm | world model | seeds | ladder mean |
|---|---|---|---|
| exp-1 `corresponding` | old | 15 | 0.1362 ± 0.0099 |
| **diverse-data `corresponding`** | diverse | 15 | **0.1390 ± 0.0199** |

## 2. Best-per-seed, averaged

Max over each seed's 12 checkpoints, then averaged. Both rows carry the same selection bias, and
`analyze_sweep.py` Table A2 removes it.

| arm | mean-of-best | std | max-of-best |
|---|---|---|---|
| exp-1 `corresponding` | 0.1732 | 0.0163 | 0.2053 |
| **diverse-data `corresponding`** | **0.1851** | 0.0239 | 0.2253 |

## 3. Per-arm tables

`corr_report.md` in this directory is the full report for the diverse-data arm. It answers a
*different* question from the two tables above: `analyze_sweep.py` compares each RL arm against the
BC checkpoint it started from, so it shows +0.0344 over BC with a seed-level p = 3.5e-05. The
tables above instead compare the two world models against each other, both starting from that same
BC checkpoint.

Tables A (headline), A2 (split-half, selection bias removed), B (trajectory length) and
C (McNemar / Wilcoxon / sign test) are produced by `rebuttal/tools/analyze_sweep.py`.
