# WMRL with the diverse-data world model — evaluation sweep

181 checkpoints, the same fixed evaluation seed set each, deterministic actions.

## A. Headline — max over each seed's checkpoints

Per seed, the best of its checkpoints; then mean ± std and max over the seeds.
`bc_sr` is the initializer those seeds were trained from, with an exact (Clopper–Pearson) interval — it is a proportion, not a constant.

| task     |   n_seeds |   bc_sr |   bc_ci_lo |   bc_ci_hi |   rl_mean_of_best |   rl_std_of_best |   rl_mean_ci_lo |   rl_mean_ci_hi |   rl_max_of_best | argmax     |   d_mean |   d_max |
|:---------|----------:|--------:|-----------:|-----------:|------------------:|-----------------:|----------------:|----------------:|-----------------:|:-----------|---------:|--------:|
| PushT-v2 |        15 |  0.1507 |     0.1258 |     0.1783 |            0.1851 |           0.0239 |          0.1731 |          0.1963 |           0.2253 | seed11@200 |   0.0344 |  0.0747 |


Per-seed values:

| task     | per_seed                                                                                                |
|:---------|:--------------------------------------------------------------------------------------------------------|
| PushT-v2 | 0.179, 0.189, 0.185, 0.148, 0.157, 0.195, 0.184, 0.144, 0.205, 0.201, 0.225, 0.167, 0.216, 0.176, 0.204 |


## A2. Same comparison, selection bias removed

Each seed's checkpoint is chosen on one half of the episodes and scored on the other, averaged over both directions; BC is scored on the same reporting half. **The gap between `rl_mean_honest` here and `rl_mean_of_best` in Table A is the size of the max-over-checkpoints bias**, which Table A cannot avoid because the BC side has only one checkpoint to choose from.

| task     |   n_seeds |   bc_sr |   rl_mean_honest |   rl_std_honest |   rl_max_honest |   d_mean |   dirA_mean |   dirB_mean |
|:---------|----------:|--------:|-----------------:|----------------:|----------------:|---------:|------------:|------------:|
| PushT-v2 |        15 |  0.1514 |           0.1805 |          0.0289 |          0.2277 |   0.0291 |      0.1727 |      0.1884 |


## B. Trajectory length of successful episodes

Control steps (`sim_freq=1`, so also environment steps; x0.05 s for simulated seconds). Failures run to the `horizon` and are excluded. RL columns average over the seed-best policies.

`*_all` uses whatever episodes each policy solves — confounded, because a stronger policy also solves harder episodes. `*_matched` uses only episodes **both** solve, paired by seed: that is the comparison that answers whether the policy got faster. Negative `d_len_matched` means WMRL finishes sooner.

| task     |   horizon |   bc_sr |   rl_sr |   bc_len_all |   rl_len_all |   d_len_all |   matched_n |   bc_len_matched |   rl_len_matched |   d_len_matched |   d_len_matched_std |   n_seeds_faster | p_seed_level   |
|:---------|----------:|--------:|--------:|-------------:|-------------:|------------:|------------:|-----------------:|-----------------:|----------------:|--------------------:|-----------------:|:---------------|
| PushT-v2 |       100 |    0.15 |    0.19 |        41.32 |        45.47 |        4.15 |       53.80 |            36.49 |            38.61 |            2.11 |                2.91 |                5 | 0.989          |


## C. Significance

Every model saw the same eval seeds, so all comparisons are **paired on the environment's initial state**. Two units of analysis, answering different questions:

- **Seed level (n = training seeds).** Does the *method* beat BC? This is the unit that generalises, and the sign test is the claim that rests on fewest assumptions.
- **Episode level (n = episodes).** Does *this particular checkpoint* beat BC on these problems? `ep_best_*` picks the single best RL checkpoint out of all of them, so its p-value is inflated by the winner's curse and should be quoted as a description, not a test. `ep_honest_*` selects on one half and tests on the other, and is the defensible episode-level number.

| task     |   n_seeds |   bc_sr |   rl_mean |   d_mean | seed_t_p   | seed_wilcoxon_p   | seed_wilcoxon_method   | seed_sign_wins   | seed_sign_p   |   cohens_d |   d_ci_lo |   d_ci_hi |
|:---------|----------:|--------:|----------:|---------:|:-----------|:------------------|:-----------------------|:-----------------|:--------------|-----------:|----------:|----------:|
| PushT-v2 |        15 |  0.1507 |    0.1851 |   0.0344 | 3.5e-05    | 0.000214          | exact                  | 13/15            | 0.00369       |     1.4377 |    0.0056 |    0.0627 |


`d_ci_*` is a two-stage bootstrap resampling both the seeds and BC's own episodes, so BC's sampling error is not treated as zero.

| task     | ep_best_ckpt   |   ep_best_delta |   ep_best_discordant | ep_best_mcnemar_p   |   ep_honest_mean_delta | ep_honest_median_p   |   ep_honest_n_sig |
|:---------|:---------------|----------------:|---------------------:|:--------------------|-----------------------:|:---------------------|------------------:|
| PushT-v2 | seed11@200     |          0.0747 |                  170 | 1.0e-05             |                 0.0327 | 0.0423               |                 9 |


### Across the five tasks

| test                                       |   n_tasks | max_raw_p   |   holm_max |   n_sig_holm |   bh_max | fisher_p   | stouffer_p   |
|:-------------------------------------------|----------:|:------------|-----------:|-------------:|---------:|:-----------|:-------------|
| seed-level sign test                       |         1 | 0.00369     |     0.0037 |            1 |   0.0037 | 0.00369    | 0.00369      |
| seed-level Wilcoxon                        |         1 | 0.000214    |     0.0002 |            1 |   0.0002 | 0.000214   | 0.000214     |
| seed-level t-test                          |         1 | 3.5e-05     |     0.0000 |            1 |   0.0000 | 3.5e-05    | 3.5e-05      |
| episode-level McNemar (split-half, median) |         1 | 0.0423      |     0.0423 |            1 |   0.0423 | 0.0423     | 0.0423       |


Pooled over all 15 (task, seed) pairs as differences from each task's BC baseline: mean Δ = 0.0344, 13/15 seeds above BC, sign test p = 0.00369, Wilcoxon p = 0.000214.

## Limitations

- **Table A is optimistically biased.** Each RL seed takes a max over its checkpoints; the BC initializer is one checkpoint. Table A2 is the corrected version. A symmetric best-of-N for BC is not possible here — the rest of the BC epoch sweep is no longer on disk.
- **`ep_best_mcnemar_p` is not a hypothesis test.** The checkpoint was chosen for being the best of ~180 on the same episodes it is then tested on.
- The BC baseline is a proportion over the same episodes, with the interval shown; treating it as an exact constant would overstate every p-value's confidence.
- Flow-matching noise is not seed-controlled (`docs/applications.md:51`), so re-running does not reproduce these numbers bit for bit.
