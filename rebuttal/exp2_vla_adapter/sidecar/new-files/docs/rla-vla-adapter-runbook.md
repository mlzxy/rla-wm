# RLA → VLA-Adapter runbook

What changed, and the commands to run it. Background and rationale:
[rla-vla-adapter.md](rla-vla-adapter.md).

Everything runs from the repo root, after `source scripts/vla_adapter_env.sh`.

```bash
cd "$REBUTTAL_REPO"   # your rla-wm checkout
source scripts/vla_adapter_env.sh
```

Status as of **2026-07-25**: all seven gates pass with a randomly-initialised `f_enc`, and a
12-step training smoke ran the RLA arm end to end (§3). Nothing real has been trained. The only step
waiting on the RLA autoencoder is §2 — everything else is verified.

---

## 0. Cheat sheet

```bash
# once f_enc exists: extract targets, ~3 h on one GPU (or one suite per GPU, ~45 min)
.venv/bin/python scripts/extract_rla_latents.py \
    --rla-config configs/rla/16x64_libero.yaml \
    --rla-run runs/<rla-run>/<stamp> \
    --out data/libero_rla/16x64-<stamp>

# prove the plumbing, ~10 min (gates 1,2,3,4,5,7; add 6 for the GPU one)
.venv/bin/python scripts/verify_rla_targets.py --sidecar data/libero_rla/16x64-<stamp>
.venv/bin/python scripts/verify_rla_targets.py --gates 6 --rla-run runs/<rla-run>/<stamp>

# train one arm (and auto-eval it)
ARM=baseline                                          scripts/train_libero.sh pro spatial
ARM=rla RLA_SIDECAR=data/libero_rla/16x64-<stamp>     scripts/train_libero.sh pro spatial

# eval an existing checkpoint -- ARM must match how it was trained
ARM=rla CKPT_NAME=LIBERO-Spatial-Pro-rla--100000_chkpt scripts/run_libero_eval.sh spatial
```

**The one thing to get right**: `ARM` selects the action head's *shape*, and
`experiments/robot/openvla_utils.get_action_head()` rebuilds that head and loads it
**strict**. Training and evaluating with different `ARM` values fails at load — deliberately.
`scripts/vla_adapter_env.sh` resolves `ARM` for both scripts so they cannot drift.

---

## 1. What changed

New files:

| File | What |
|---|---|
| `scripts/extract_rla_latents.py` | offline extractor: RLDS → frozen DINOv3 → frozen `f_enc` → per-frame `(A, Q, D)` fp16 sidecar |
| `scripts/verify_rla_targets.py` | the seven gates |
| `datalib/libero_camera.py` | LIBERO camera geometry, moved out of `scripts/convert_libero_to_trajectory.py` (which now re-exports it) so consumers can import it without inheriting that script's `CUDA_VISIBLE_DEVICES=""` |
| `third_party/VLA-Adapter/prismatic/vla/rla_targets.py` | sidecar reader + the `tf.numpy_function` join |
| `docs/rla-vla-adapter.md`, this file | |

Edited (all marked `# v4world:`, all inert when `RLA_STEPS=0`):

| File | Change |
|---|---|
| `prismatic/vla/constants.py` | `RLA_STEPS` / `RLA_QUERIES` / `RLA_DIM` / `RLA_SIDECAR` from the environment |
| `.../rlds/oxe/transforms.py` | `libero_dataset_transform` attaches `trajectory["rla"]` via `tf.numpy_function`, once per episode |
| `.../rlds/dataset.py` | `restructure()` carries `rla` through at the **top level** (not under `observation`) |
| `.../rlds/traj_transforms.py` | `chunk_act_obs` truncates `rla` to `effective_traj_len` |
| `.../datasets/datasets.py` | `RLDSBatchTransform.__call__` passes `rla` into the sample dict |
| `prismatic/util/data_utils.py` | the collator stacks `rla` |
| `prismatic/models/action_heads.py` | RLA output-query tokens, second output head, per-position `h_t` gate |
| `vla-scripts/finetune.py` | `--lambda_rla`, the auxiliary L1 term, `rla_loss` in the metrics |
| `scripts/vla_adapter_env.sh` | the `ARM` switch |
| `scripts/train_libero.sh` | `ARM` / `LAMBDA_RLA`, arm in the run id, refuses `ARM!=baseline` without a sidecar |
| `scripts/run_libero_eval.sh` | documents that `ARM` must match training |
| `AGENTS.md`, `docs/vla-adapter.md` | the "never edit third_party in place" note was stale for VLA-Adapter (vendored since `4e05436`); LIBERO is still a patched clone |

**`window_size` in `datasets.py:185` was deliberately left at 1.** The original design note said to
raise it to `NUM_ACTIONS_CHUNK`; that gathers *past* frames and would silently swap the policy's
input image for frame t−7. See [rla-vla-adapter.md §2](rla-vla-adapter.md) row 4.

---

## 2. Extract the targets

Needs a trained RLA autoencoder run directory containing `ckpts/encoder_step*.pt`.

```bash
.venv/bin/python scripts/extract_rla_latents.py \
    --rla-config configs/rla/16x64_libero.yaml \
    --rla-run runs/<rla-run>/<stamp> \
    --out data/libero_rla/16x64-<stamp> \
    --suites all
```

- **0.20 ep/s ≈ 25 frames/s** on an RTX A6000 → ~3 h for all four suites, **3.9 GB** on disk. To
  parallelise, run one suite per GPU: `CUDA_VISIBLE_DEVICES=$i ... --suites libero_spatial &`
  (~45 min on a 4-GPU box).
- Safe to interrupt and re-run: existing `ep_*.npy` are skipped unless `--overwrite`.
- `--step N` pins a specific checkpoint (default: latest). The exact file lands in `manifest.json`,
  so a sidecar can always be traced back to its `f_enc`.
- `--pairs consecutive` produces the alternative target definition
  (`f_enc(s_{t+i+1} − s_{t+i})` instead of the default anchored `f_enc(s_{t+i+1} − s_t)`) into the
  same `(N, A, Q, D)` layout, so downstream code cannot tell them apart. Write it to a *different*
  `--out` and switch arms with `RLA_SIDECAR`.
- `--random-encoder` skips the checkpoint entirely and leaves `f_enc` randomly initialised. The
  values are noise; use it to exercise the pipeline before the autoencoder exists. `rla_targets.py`
  prints a warning whenever a sidecar built this way is loaded.

Layout:

```
data/libero_rla/<tag>/
  manifest.json   config + the exact checkpoint, checked against RLA_* at load time
  stats.json      per-(q, d) mean/std used to normalise the target on read
  <suite>/keys.json      sha1(state bytes) -> {file, episode_index, traj_len}
  <suite>/ep_000000.npy  (N, A, Q, D) float16
```

## 3. Verify

```bash
.venv/bin/python scripts/verify_rla_targets.py --sidecar data/libero_rla/16x64-<stamp>
.venv/bin/python scripts/verify_rla_targets.py --gates 6 --rla-run runs/<rla-run>/<stamp>   # GPU
```

Default gates are `1,2,3,4,5,7`; gate 6 needs a GPU so it is opt-in. `--suite` picks the suite
(default `libero_spatial`), `--limit` speeds up gates 1 and 2 — it deliberately does **not** shrink
gate 3's synthetic sidecar, which always covers the whole suite.

**Always pass `--sidecar` once you have one.** Without it gate 2 only checks the RLDS side; with it
it also verifies every episode has an entry with the right `episode_index`/`traj_len`, and that
`stats.json` covers `sum(traj_len) × A` rows. That last check exists because a resumed extraction
once computed the normalisation from only the episodes it rewrote — which would have silently
mis-scaled every target.

A complete real-shaped sidecar built with `--random-encoder` lives at
`data/libero_rla/smoke-random` (432 episodes, 804 MB). It is the fixture for gate 2's `--sidecar`
path. **Its values are noise** — `rla_targets.py` prints a warning if anything ever loads it.

Expected output (abridged), with a randomly-initialised `f_enc`:

```
=== Gate 1: N frames <-> N actions, and action[k] drives s[k] -> s[k+1]
  OK  6 episodes: N frames == N actions == N states; gripper-lag peaks at lag 0
=== Gate 2: sha1(observation/state) is a per-episode bijection
  OK  432 distinct keys over 432 episodes in libero_spatial -- bijective
  OK  episode_metadata/file_path is NOT unique (...) -- correctly unused as a key
=== Gate 3: end-to-end (episode, t, i) alignment through the real RLDSDataset
  OK  480 samples, 14 distinct episodes, 125 distinct timesteps (t in [0, 140]), 0 misses
=== Gate 4: RLA_STEPS=0 is byte-identical to upstream
  OK  key sets and init weights identical to 4e05436 (560 tensors)
  OK  predict_action bit-identical to upstream for phase=Inference and phase=Training
  OK  released LIBERO-Spatial-Pro action head loads strict (560 tensors)
=== Gate 5: RLA_STEPS=NUM_ACTIONS_CHUNK forward/backward
  OK  action (2, 8, 7), z_hat (2, 8, 1024)
  OK  action output differs from RLA_STEPS=0 by 1.1964 even at lambda_rla=0
=== Gate 6: the extractor's encode path == RlaAutoencoderMultiViewTrainer.inference_batch
  OK  enc_tokens match on a real batch: max |d| = 0.00e+00 (|z|max = 5.353), shape (2, 16, 64)
=== Gate 7: recomputed dataset statistics match the released reference
  OK  all 2 cached statistics file(s) match LIBERO-Spatial-Pro to <1e-5
```

**Gate 3 is the one that matters.** It writes a synthetic sidecar whose values spell out
`(episode_index, t, i)`, pushes it through the real `RLDSDataset` + collator, and asserts the triple
that comes out the other end. It needs no trained model and it catches every off-by-one, every
shuffle/interleave join break, and the `window_size` trap.

### Smoke-testing real training without `f_enc`

The gates cover the two halves separately (target reaches the collator; head emits `z_hat` with
grads). To exercise the seam inside `run_forward_pass`, build gate 3's synthetic sidecar as a
persistent fixture and train against it:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'scripts')
from pathlib import Path
from verify_rla_targets import build_synthetic_sidecar
print(build_synthetic_sidecar(Path('data/libero_rla/synthetic-spatial'), 'libero_spatial'))"

NCCL_FAST=0 ARM=rla RLA_QUERIES=1 RLA_DIM=4 \
  RLA_SIDECAR=data/libero_rla/synthetic-spatial \
  GPUS=2 EVAL_AFTER=0 MAX_STEPS=12 WANDB_MODE=offline \
  scripts/train_libero.sh smoke spatial
```

Full coverage, 3.4 MB, ~1 min to build. Last run: 12 steps, **0 sidecar misses**,
`VLA Train/Rla Loss = 102.5` alongside `VLA Train/Loss = 102.875` — the target values are raw
`(episode, t, i)` integers under identity normalisation, so a ~100 L1 against a near-zero-init
prediction is exactly right, and it proves the loss is computed against the real targets rather
than zeros.

`NCCL_FAST=0` is needed on hosts without an `ib0` interface (`Bootstrap : no socket interface
found`) — unrelated to RLA. A run shorter than `SAVE_FREQ` exits 2 at the end because no checkpoint
was written to evaluate; that is the launcher working as intended, not a training failure.

## 4. Train

```bash
# control arm -- byte-identical to upstream VLA-Adapter (gate 4 proves it)
ARM=baseline scripts/train_libero.sh pro spatial

# RLA arm
ARM=rla RLA_SIDECAR=data/libero_rla/16x64-<stamp> scripts/train_libero.sh pro spatial

# all four suites, sequentially
ARM=rla RLA_SIDECAR=data/libero_rla/16x64-<stamp> scripts/train_libero.sh pro all
```

- `LAMBDA_RLA=1.0` weights the auxiliary loss. Targets are normalised per-`(q, d)` on read, so this
  is directly comparable to the action L1 and stable across autoencoder runs.
- Checkpoints go to `runs/weights/vla-adapter/LIBERO-<Suite>-Pro-<arm>--<step>_chkpt`
  (`baseline` keeps the historical `-repro` name so existing runs are untouched).
- Watch `VLA Train/Rla Loss` in wandb next to `VLA Train/Loss`.
- The **first** run after these changes recomputes the dataset statistics once per suite (a few
  minutes) — `get_dataset_statistics` hashes the source text of `libero_dataset_transform`. Gate 7
  proves the recomputed values match the released reference exactly.

> The control arm must be `ARM=baseline`, **not** `LAMBDA_RLA=0`. The RLA tokens sit in the shared
> self-attention, so they change the action path even at zero loss weight — gate 5 measures the
> difference (1.1964 on a random batch) precisely so this cannot be forgotten.

## 5. Evaluate

`train_libero.sh` evaluates the final checkpoint automatically (`EVAL_AFTER=0` to skip), passing
`ARM` through. To evaluate an existing checkpoint:

```bash
ARM=rla CKPT_NAME=LIBERO-Spatial-Pro-rla--100000_chkpt scripts/run_libero_eval.sh spatial
.venv/bin/python scripts/summarize_libero_eval.py
```

`RLA_SIDECAR` is not needed here: `z` is an output query, never an input, so evaluation needs the
head's *shape* but no targets. `ARM` alone supplies that.

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| `Error(s) in loading state_dict for L1RegressionActionHead: Missing key(s) ... fc2_rla` | `ARM` at eval ≠ `ARM` at training. Working as designed — set them the same. |
| `RLA_STEPS>0 but the batch has no 'rla' target` | `RLA_SIDECAR` unset or pointing at a directory with no `manifest.json`. |
| `[rla_targets] MISS: no sidecar entry for episode key ...` | The sidecar does not cover this suite. Re-run the extractor with `--suites all`, or check you are training the suite you extracted. Misses return zeros rather than crashing, so **watch for this line** — training will otherwise regress happily against zeros. |
| `RLA_QUERIES=16 does not match the sidecar at ... (1)` | Sidecar/`ARM` shape mismatch; the manifest is the source of truth. |
| `RLA_STEPS>0 requires use_pro_version=True` | The non-Pro block has no positional signal, so every RLA token would predict the same `z`. |
| Statistics recomputed on a run you expected to be cached | The cache key hashes `str(builder.info)`, which embeds the `data_dir` **string**. Run from `third_party/VLA-Adapter` with `--data_root_dir data/libero`, as the launcher does. |
| `WARNING: ... was extracted with --random-encoder` | The sidecar holds noise. Fine for plumbing, meaningless for training. |
