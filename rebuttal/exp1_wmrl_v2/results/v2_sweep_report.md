# WMRL v2 — evaluation sweep

905 checkpoints, the same fixed evaluation seed set each, deterministic actions.

## A. Headline — max over each seed's checkpoints

Per seed, the best of its checkpoints; then mean ± std and max over the seeds.
`bc_sr` is the initializer those seeds were trained from, with an exact (Clopper–Pearson) interval — it is a proportion, not a constant.

| task            |   n_seeds |   bc_sr |   bc_ci_lo |   bc_ci_hi |   rl_mean_of_best |   rl_std_of_best |   rl_mean_ci_lo |   rl_mean_ci_hi |   rl_max_of_best | argmax     |   d_mean |   d_max |
|:----------------|----------:|--------:|-----------:|-----------:|------------------:|-----------------:|----------------:|----------------:|-----------------:|:-----------|---------:|--------:|
| PushT-v2        |        15 |  0.1507 |     0.1258 |     0.1783 |            0.1732 |           0.0163 |          0.1653 |          0.1812 |           0.2053 | seed14@75  |   0.0225 |  0.0547 |
| RollBall-v1     |        15 |  0.6533 |     0.6181 |     0.6874 |            0.6557 |           0.0440 |          0.6312 |          0.6744 |           0.7133 | seed9@75   |   0.0024 |  0.0600 |
| PullCube-v2     |        15 |  0.8533 |     0.8260 |     0.8779 |            0.8370 |           0.0171 |          0.8274 |          0.8444 |           0.8613 | seed10@25  |  -0.0164 |  0.0080 |
| PullCubeTool-v1 |        15 |  0.4000 |     0.3647 |     0.4361 |            0.3771 |           0.0411 |          0.3564 |          0.3964 |           0.4467 | seed12@250 |  -0.0229 |  0.0467 |
| PokeCube-v2     |        15 |  0.8973 |     0.8734 |     0.9181 |            0.9420 |           0.0106 |          0.9371 |          0.9473 |           0.9587 | seed1@100  |   0.0446 |  0.0613 |


Per-seed values:

| task            | per_seed                                                                                                |
|:----------------|:--------------------------------------------------------------------------------------------------------|
| PushT-v2        | 0.153, 0.188, 0.179, 0.181, 0.165, 0.149, 0.179, 0.148, 0.191, 0.171, 0.185, 0.175, 0.169, 0.205, 0.159 |
| RollBall-v1     | 0.685, 0.601, 0.609, 0.699, 0.656, 0.620, 0.684, 0.667, 0.713, 0.657, 0.668, 0.695, 0.559, 0.625, 0.697 |
| PullCube-v2     | 0.847, 0.840, 0.832, 0.807, 0.843, 0.857, 0.800, 0.841, 0.827, 0.861, 0.829, 0.856, 0.833, 0.835, 0.847 |
| PullCubeTool-v1 | 0.411, 0.299, 0.363, 0.417, 0.343, 0.409, 0.348, 0.361, 0.391, 0.369, 0.335, 0.447, 0.420, 0.337, 0.407 |
| PokeCube-v2     | 0.959, 0.932, 0.928, 0.955, 0.957, 0.936, 0.944, 0.939, 0.925, 0.937, 0.945, 0.933, 0.955, 0.944, 0.940 |


## A2. Same comparison, selection bias removed

Each seed's checkpoint is chosen on one half of the episodes and scored on the other, averaged over both directions; BC is scored on the same reporting half. **The gap between `rl_mean_honest` here and `rl_mean_of_best` in Table A is the size of the max-over-checkpoints bias**, which Table A cannot avoid because the BC side has only one checkpoint to choose from.

| task            |   n_seeds |   bc_sr |   rl_mean_honest |   rl_std_honest |   rl_max_honest |   d_mean |   dirA_mean |   dirB_mean |
|:----------------|----------:|--------:|-----------------:|----------------:|----------------:|---------:|------------:|------------:|
| PushT-v2        |        15 |  0.1514 |           0.1641 |          0.0183 |          0.1886 |   0.0126 |      0.1617 |      0.1665 |
| RollBall-v1     |        15 |  0.6543 |           0.6428 |          0.0469 |          0.6970 |  -0.0115 |      0.6365 |      0.6491 |
| PullCube-v2     |        15 |  0.8525 |           0.8311 |          0.0176 |          0.8607 |  -0.0214 |      0.8510 |      0.8112 |
| PullCubeTool-v1 |        15 |  0.4018 |           0.3703 |          0.0414 |          0.4352 |  -0.0314 |      0.3512 |      0.3895 |
| PokeCube-v2     |        15 |  0.8961 |           0.9375 |          0.0103 |          0.9516 |   0.0415 |      0.9493 |      0.9257 |


## B. Trajectory length of successful episodes

Control steps (`sim_freq=1`, so also environment steps; x0.05 s for simulated seconds). Failures run to the `horizon` and are excluded. RL columns average over the seed-best policies.

`*_all` uses whatever episodes each policy solves — confounded, because a stronger policy also solves harder episodes. `*_matched` uses only episodes **both** solve, paired by seed: that is the comparison that answers whether the policy got faster. Negative `d_len_matched` means WMRL finishes sooner.

| task            |   horizon |   bc_sr |   rl_sr |   bc_len_all |   rl_len_all |   d_len_all |   matched_n |   bc_len_matched |   rl_len_matched |   d_len_matched |   d_len_matched_std |   n_seeds_faster | p_seed_level   |
|:----------------|----------:|--------:|--------:|-------------:|-------------:|------------:|------------:|-----------------:|-----------------:|----------------:|--------------------:|-----------------:|:---------------|
| PushT-v2        |       100 |    0.15 |    0.17 |        41.32 |        44.31 |        2.99 |       51.67 |            35.85 |            36.78 |            0.92 |                2.64 |                6 | 0.849          |
| RollBall-v1     |       100 |    0.65 |    0.66 |        30.87 |        24.71 |       -6.17 |      402.67 |            30.09 |            23.80 |           -6.30 |                3.43 |               14 | 6.1e-05        |
| PullCube-v2     |       100 |    0.85 |    0.84 |        26.07 |        26.76 |        0.69 |      589.20 |            24.63 |            25.89 |            1.26 |                0.83 |                1 | 1              |
| PullCubeTool-v1 |       100 |    0.40 |    0.38 |        54.48 |        52.15 |       -2.33 |      183.47 |            53.20 |            51.75 |           -1.46 |                3.05 |               11 | 0.0844         |
| PokeCube-v2     |       100 |    0.90 |    0.94 |        42.62 |        35.37 |       -7.25 |      659.73 |            42.57 |            34.69 |           -7.88 |                1.80 |               15 | 3.1e-05        |


## C. Significance

Every model saw the same eval seeds, so all comparisons are **paired on the environment's initial state**. Two units of analysis, answering different questions:

- **Seed level (n = training seeds).** Does the *method* beat BC? This is the unit that generalises, and the sign test is the claim that rests on fewest assumptions.
- **Episode level (n = episodes).** Does *this particular checkpoint* beat BC on these problems? `ep_best_*` picks the single best RL checkpoint out of all of them, so its p-value is inflated by the winner's curse and should be quoted as a description, not a test. `ep_honest_*` selects on one half and tests on the other, and is the defensible episode-level number.

| task            |   n_seeds |   bc_sr |   rl_mean |   d_mean | seed_t_p   | seed_wilcoxon_p   | seed_wilcoxon_method   | seed_sign_wins   | seed_sign_p   |   cohens_d |   d_ci_lo |   d_ci_hi |
|:----------------|----------:|--------:|----------:|---------:|:-----------|:------------------|:-----------------------|:-----------------|:--------------|-----------:|----------:|----------:|
| PushT-v2        |        15 |  0.1507 |    0.1732 |   0.0225 | 5.2e-05    | 0.000808          | approx (ties)          | 13/15            | 0.00369       |     1.3786 |   -0.0048 |    0.0487 |
| RollBall-v1     |        15 |  0.6533 |    0.6557 |   0.0024 | 0.418      | 0.325             | approx (ties)          | 10/15            | 0.151         |     0.0546 |   -0.0377 |    0.0428 |
| PullCube-v2     |        15 |  0.8533 |    0.8370 |  -0.0164 | 0.999      | 0.999             | approx (ties)          | 3/15             | 0.996         |    -0.9544 |   -0.0427 |    0.0105 |
| PullCubeTool-v1 |        15 |  0.4000 |    0.3771 |  -0.0229 | 0.976      | 0.968             | exact                  | 6/15             | 0.849         |    -0.5585 |   -0.0632 |    0.0176 |
| PokeCube-v2     |        15 |  0.8973 |    0.9420 |   0.0446 | 8.5e-11    | 0.000361          | approx (ties)          | 15/15            | 3.1e-05       |     4.2075 |    0.0227 |    0.0672 |


`d_ci_*` is a two-stage bootstrap resampling both the seeds and BC's own episodes, so BC's sampling error is not treated as zero.

| task            | ep_best_ckpt   |   ep_best_delta |   ep_best_discordant | ep_best_mcnemar_p   |   ep_honest_mean_delta | ep_honest_median_p   |   ep_honest_n_sig |
|:----------------|:---------------|----------------:|---------------------:|:--------------------|-----------------------:|:---------------------|------------------:|
| PushT-v2        | seed14@75      |          0.0547 |                  143 | 0.000382            |                 0.0217 | 0.13                 |                 2 |
| RollBall-v1     | seed9@75       |          0.0600 |                  207 | 0.00108             |                -0.0035 | 0.753                |                 3 |
| PullCube-v2     | seed10@25      |          0.0080 |                   82 | 0.291               |                -0.0140 | 0.849                |                 0 |
| PullCubeTool-v1 | seed12@250     |          0.0467 |                  247 | 0.0152              |                -0.0238 | 0.901                |                 0 |
| PokeCube-v2     | seed1@100      |          0.0613 |                   68 | 6.4e-09             |                 0.0343 | 0.00531              |                14 |


### Across the five tasks

| test                                       |   n_tasks | max_raw_p   |   holm_max |   n_sig_holm |   bh_max | fisher_p   | stouffer_p   |
|:-------------------------------------------|----------:|:------------|-----------:|-------------:|---------:|:-----------|:-------------|
| seed-level sign test                       |         5 | 0.996       |     1.0000 |            2 |   0.9963 | 8.0e-05    | 0.0365       |
| seed-level Wilcoxon                        |         5 | 0.999       |     1.0000 |            2 |   0.9986 | 0.000342   | 0.168        |
| seed-level t-test                          |         5 | 0.999       |     1.0000 |            2 |   0.9988 | 1.1e-10    | 0.00728      |
| episode-level McNemar (split-half, median) |         5 | 0.901       |     1.0000 |            1 |   0.9011 | 0.11       | 0.381        |


Pooled over all 75 (task, seed) pairs as differences from each task's BC baseline: mean Δ = 0.0060, 47/75 seeds above BC, sign test p = 0.0185, Wilcoxon p = 0.0428.

## Limitations

- **Table A is optimistically biased.** Each RL seed takes a max over its checkpoints; the BC initializer is one checkpoint. Table A2 is the corrected version. A symmetric best-of-N for BC is not possible here — the rest of the BC epoch sweep is no longer on disk.
- **`ep_best_mcnemar_p` is not a hypothesis test.** The checkpoint was chosen for being the best of ~180 on the same episodes it is then tested on.
- The BC baseline is a proportion over the same episodes, with the interval shown; treating it as an exact constant would overstate every p-value's confidence.
- Flow-matching noise is not seed-controlled (`docs/applications.md:51`), so re-running does not reproduce these numbers bit for bit.
