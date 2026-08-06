> Working log, kept as it was written. It records dead ends and superseded numbers on purpose.
> The current summary is in README.md; where the two disagree, the README is right.

# SO-101 real-world runbook

Every command, in order, for: **released dataset → dino_to_image UNet → multi-view RLA →
BC / BC-RLA policies**. Copy-paste the lot.

```bash
cd <repo root>
export PYTHONPATH=.:./third_party/diffusion_policy      # REQUIRED; there is no .env in this checkout
PY=$PWD/.venv/bin/python                                      # system python3 lacks h5py/decord
```

---

## The dependency graph

```
released dataset  ─────────────────────────────────────────────────────►
                        │
        ┌───────────────┼────────────────────────────────┐
        ▼               ▼                                ▼
  Stage 1  UNet    Stage 1'  BC-100 (GPU 2)        Stage 1'  BC-25 (GPU 3)
  (GPU 1, short)        └──────────► independent of RLA, start immediately ◄────────┐
        │                                                                            │
        ▼  paste run dir into the RLA config's `image_decoder_ckpt`                   │
  Stage 2  RLA 16x16 (GPUs 0-3)                                                      │
        │                                                                            │
        ▼  paste run dir into --rla                                                   │
  Stage 2b PRECOMPUTE latent targets  (shard over 5 GPUs, ~10 min)                    │
        │                                                                            │
        ▼  paste the sidecar tag into the BC-RLA config                             │
  Stage 3  BC-RLA (ladder) ──────────────────────────────────────────────────────────┘
```

**The UNet must finish before the RLA starts**: `RlaAutoencoderTrainer.__init__` loads
`image_decoder_ckpt` with `strict=True`, so the RLA config needs a real UNet run directory. The UNet
run only needs a few thousand steps.

Both vanilla-BC runs are completely independent of stages 1, 2 and 2b — launch them at the same
time as the UNet.

**Stage 2b is new and mandatory for BC-RLA.** The latent-action targets are computed once, offline,
from clean frames. Training then never runs DINO or the RLA encoder — far cheaper, and the *only*
reason the colour/lighting augmentation is sound (see "Augmentation and precomputed targets").

5 × RTX 6000 Ada (46 GB) are free, so the whole thing fits in two waves.

---

## Stage 1 — dino_to_image UNet  ✅

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 ./.venv/bin/python train.py   --config rebuttal/exp4_so101/configs/unet/dino_to_image_so101.yaml   --output_dir runs/so101 --num_gpus 4
```

* Trained from scratch at 320 px, `lr 1e-5`. `DinoToImageDecoderV1` is fully convolutional and
  therefore resolution-agnostic, so you *can* warm-start from
  `runs/dino_to_image_v1_libero/20260724_11-59-08` (224 px, LIBERO) by setting
  `trainer.args.load_dir` + `skip_load_misc: true` and dropping the LR to ~5e-6.
* **Watch:** `l1` and `lpips` in wandb, and the `train_/val_gt_vs_pred` snapshot images
  (every `i_sample: 2500` steps).
* **Good enough when:** the reconstructions are recognisable — you can see the cup, the gripper and
  the table edge. It does not need to be sharp; it is a visualisation decoder, not a generative
  model. **~5–10 k steps from scratch.** Checkpoints land every 2 500 steps.
* Kill it with Ctrl-C once the snapshots look right; the last `ckpts/decoder_step*.pt` is what
  stage 2 uses.
* Output: `runs/so101/dino_to_image_so101/<TIMESTAMP>/` ← **write this path down**.

---

## Stage 1′ — Vanilla BC baseline  (GPU 2, in parallel with stage 1)  ✅

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:third_party/diffusion_policy .venv/bin/python policies/train_policy.py  --config-dir=rebuttal/exp4_so101/configs/policy --config-name=bc_so101_100
```

* No dependency on the UNet or the RLA — it can run from the moment the dataset is on disk.
* `bc_so101_100`: 40 epochs × ~590 steps ≈ 24 k steps.
* No DINO in this path, so 320 px is cheap: `batch_size: 64` fits easily.
* **Watch:** `val_loss` (held-out action MSE, 5 % of episodes). There is **no simulator**, so there
  is no `eval_success_rate` — real success rate only comes from the robot-side eval tool.
* Checkpoints: `runs/policy_outputs/bc_so101_100/checkpoints/latest_epoch*.ckpt`, the last 6 kept.

---

## Stage 2 — Multi-view RLA autoencoder, 16 x 16 latent  ✅

**Edit `rebuttal/exp4_so101/configs/rla/16x16_so101_dual.yaml` first** — set `image_decoder_ckpt` to the
stage-1 run directory:

```yaml
    image_decoder_ckpt: "runs/so101/dino_to_image_so101/<TIMESTAMP>"
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python train.py \
  --config rebuttal/exp4_so101/configs/rla/16x16_so101_dual.yaml \
  --output_dir runs/so101 --num_gpus 4
```

* **Watch:** `l1`, `mse`, and per-view `l1_view0` (front) / `l1_view1` (wrist) — the wrist view is
  usually harder. The `gt_vs_pred` snapshots stack the real frame `t+gap` over the RLA
  reconstruction; they become meaningful once the stage-1 UNet is plugged in.
* **Good enough when:** val `l1` has flattened and the reconstructions track the real future frame.
  20–40 k steps is a reasonable target; there is no need to run to `max_steps`.
* **From scratch** — no warm start. The ManiSkill RLA is 32x64, and `load_state_dict` rejects
  shape mismatches even with `strict=False`, so a 16x16 model cannot reuse it.
* Frame gap: re-sampled uniformly in `[1, 63]` every batch. Stage 2b then runs this encoder at
  the gaps the policy needs ([4, 8, …, 32]), so the run must be good across the whole range
  [1, 32], not at one gap — which is
  why `nonlinear_sampling: false` (uniform sampling) rather than the Beta skew toward long gaps.
* At 320 px the sequence is 800 tokens per sample, so a training step peaks at **26.5 GiB for 32
  samples** (measured on a 46 GB RTX 6000 Ada). The config uses `batch_size: 64` with
  `batch_split: 2`, i.e. two 32-sample micro-batches accumulated into one effective batch of 64.
* Output: `runs/so101/16x16_so101_dual/<TIMESTAMP>/` ← **write this path down**.

---

## Stage 2b — Precompute the RLA latent-action targets

Latent targets are **always** computed offline, from clean frames. Computing them inside the
training loop would derive them from the *augmented* pixels the policy sees, while the RLA decoder
that has to interpret those latents was trained on clean frames — and nothing about the loss curve
would reveal it. `policy.latent_source` is asserted to be `precomputed`; anything else aborts.

```bash
RLA=runs/so101/16x16_so101_dual/<TIMESTAMP>
STEP=40000                          # which RLA checkpoint. Drop --step to take the newest.
                                    # See "Step tokens" below -- `40000.snapshot` is also valid.
GAPS=4,8,12,16,20,24,28,32          # the 8-rung ladder; its last rung (32) is also `final`

# Sharded across the 5 GPUs (measured ~10 min for 120 trajectories at 8 gaps):
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i $PY rebuttal/exp4_so101/precompute_rla_latents.py \
    --rla $RLA --step $STEP --gaps $GAPS --num-shards 5 --shard $i &
done; wait

# then once, single process, to write the manifest over the complete set:
$PY rebuttal/exp4_so101/precompute_rla_latents.py --rla $RLA --step $STEP --gaps $GAPS
```

**The final pass is not optional and not expensive.** Each shard refuses to write a manifest for a
set it only partly produced (`Manifest NOT written -- re-run once without --num-shards`), because a
manifest is a claim about the *whole* sidecar. The final pass finds every `.npy` already on disk,
recomputes nothing, and takes ~8 s — it does not even load DINOv3. It is also self-healing: if a
shard died, this pass computes whatever is missing before writing the manifest.

`--tag` is **arbitrary** — it is only the directory name under `rla_latents/`. Leaving it off is
still the recommendation, because the default `<rla-run>__step<TOKEN>` makes two sidecars from
different checkpoints impossible to confuse by eye. But a custom name is equally *safe*: identity
lives in the sidecar's `manifest.json`, never in the directory name, so the "different checkpoint in
an existing tag" guard fires either way (verified: precomputing the snapshot into a tag already
holding the milestone is refused, while re-running the same checkpoint spelled `20000` instead of
`0020000` is accepted).

The last command prints exactly what the policy configs need:

```
[rla-precompute]     rla_latent_tag: 20260802_19-49-36__step0040000
[rla-precompute]     rla_latent_step: "0040000"
```

Writes `data/realworld/so101_converted/rla_latents/<TAG>/traj_*.npy`, each
`(T, 8, 16, 16) float16` = 4 KB per frame, **~170 MB** for the whole dataset:

```
latents[t, j] = encoder( DINO(frame t + gaps[j]) - DINO(frame t) )
```

One file serves both training modes — `final` reads the gap-32 column, `ladder` reads all eight.

* **Gaps span `horizon` (32 frames, 1.07 s)** — the chunk the policy *predicts* — not
  `n_action_steps` (16), which is only how much of it is executed before the next replan. The
  ladder always ends exactly at 32; `resolve_latent_gaps` asserts it.
* `--gaps` computes exactly those columns. Use `--max-gap 32` instead for the dense 1..32 range if
  you want to try other ladders later without recomputing — 4× the time (~40 min sharded) and 4×
  the storage (~680 MB). The dataset selects columns by gap VALUE, so dense and sparse sidecars are
  interchangeable.
* Values are the RAW encoder output; the policy applies its `/10` convention at train time.
* Frames past the end of an episode clamp to the last frame, so the tail yields a zero-motion
  latent rather than a discontinuity.
* **Re-run whenever you retrain the RLA**, with the new `--step`. Old sidecars stay valid and
  addressable by their tag.

### Step tokens: `40000`, `0040000`, `40000.snapshot`

A "step" is a **token**, not necessarily a number: the text between `_step` and `.pt` in the
checkpoint filename. The trainer writes a milestone at every `i_save`, and beside it a resumable
snapshot, so both of these can exist at once:

```
runs/so101/16x16_so101_dual/<TS>/ckpts/
    encoder_step0040000.pt              <- the milestone
    encoder_step0040000.snapshot.pt     <- the snapshot written beside it
```

Every place that names an RLA version (`--step`, `rla_latent_step`, `--rla-step`) accepts any of:

| you write | you get |
|---|---|
| `40000` | `encoder_step0040000.pt` |
| `0040000` | the same file — zero padding is **not** part of a step's identity |
| `40000.snapshot` | `encoder_step0040000.snapshot.pt` — a **different checkpoint** |
| `0040000.snapshot` | the same snapshot |

So `rla_latent_step: 40000` in one config and `rla_latent_step: "0040000"` in another are not a
conflict, while `40000` and `40000.snapshot` are — they are different weights, and the comparison
keeps them apart everywhere. Dropping `--step` takes the highest-numbered checkpoint and prefers
the **milestone** over a snapshot at the same step.

> **Quote it in YAML.** `rla_latent_step: 0040000` is read by YAML 1.1 as **octal** (= 16384). Write
> `rla_latent_step: 40000` (unpadded int) or `rla_latent_step: "0040000"` (quoted). Anything with a
> `.snapshot` suffix is a string and must be quoted. On the command line hydra passes it through as
> a string, so `rla_latent_step="0040000.snapshot"` is fine.

### Version guards

Several RLA checkpoints usually exist. Every hop names one explicitly, and every hop checks:

| when | guard |
|---|---|
| precompute | writing a *different* checkpoint into an existing `--tag` is refused — use the default tag, or `--overwrite`. Works for a custom `--tag` too: the check reads `manifest.json`, not the directory name |
| policy training | `rla_latent_step` is compared against the sidecar's recorded step; a mismatch aborts *before any weights load* and lists every sidecar with its step |

The policy-training guard is the one that matters: if `rla_latent_step` and the sidecar disagree,
the run trains against latents from a *different* encoder. Nothing at training time would
reveal it. A snapshot and its milestone are close enough that the pictures would still look
plausible — which is exactly why they are never treated as the same step.

---

## Stage 3 — The BC + RLA policy

One config, matched to the BC baseline in stage 1′ — same 120 trajectories, same schedule, the
latent objective the only difference:

| config | data | latent supervision |
|---|---|---|
| `bc_rla_so101_100_ladder` | all 120 trajectories | 8 latents, gaps 4…32 |

It has **three mandatory `???` placeholders**. Leaving any unset aborts at startup — a BC-RLA run
must never silently fall back to "no latent loss", because that trains a plain BC policy under a
BC-RLA name and the loss curve looks perfectly healthy.

```yaml
latent_encoder_work_dir: ???   # runs/so101/16x16_so101_dual/<TIMESTAMP>
rla_latent_tag: ???            # printed by stage 2b — any name, e.g. <run>__step0040000
rla_latent_step: ???           # printed by stage 2b — a step TOKEN: 40000, "0040000",
                               # or "0040000.snapshot". Quote anything padded or suffixed.
```

Fill them in the yaml, or pass them on the command line:

```bash
RLA=runs/so101/16x16_so101_dual/<TIMESTAMP>
TAG=<run>__step0040000
STEP=0040000                        # or 40000, or 0040000.snapshot -- all resolve identically
RLA_ARGS="latent_encoder_work_dir=$RLA rla_latent_tag=$TAG rla_latent_step=$STEP"

CUDA_VISIBLE_DEVICES=1 $PY policies/train_policy.py --config-dir=rebuttal/exp4_so101/configs/policy \
  --config-name=bc_rla_so101_100_ladder $RLA_ARGS
```

At startup each run prints, before anything expensive is built:

```
[TrainSO101Workspace] latent_mode=ladder -> latent_gaps=[4, 8, 12, 16, 20, 24, 28, 32]
[TrainSO101Workspace] RLA targets: tag=<run>__step0040000 step=40000
  RLA latents: .../rla_latents/<run>__step0040000 tag=... rla_step=40000 gaps=[...] -> (41399, 8, 16, 16)
```

### What the two modes differ in

| | `final` | `ladder` |
|---|---|---|
| gaps | `[32]` | `[4, 8, …, 32]` (`latent_ladder_steps: 8`) |
| head reads | the shared MLP output — unchanged from the stock policy | the **output query tokens**, 16/8 = 2 per gap, through one small shared MLP |
| head size | `Linear(2048, 256)` | ~21× smaller than a flat `Linear(2048, 8·256)` |

`num_output_queries` (16) must stay divisible by the rung count (8).

### Loss balance

The raw-action loss drives the robot; the latent action is auxiliary and must not dominate.

* Shipped default `lambda_action: 1.0`, `lambda_latent_l1/mse: 0.25`. **Re-measure it for your
  sidecar** — this is not a fixed ratio. The target is `encoder output / 10`, so it scales with the
  RLA checkpoint's latent magnitude, which the sidecar reports as
  `manifest["value_stats"]["mean_abs"]`:

  | sidecar `mean_abs` | what 0.25 / 0.25 realises |
  |---|---|
  | ≈ 1.9 (an early RLA checkpoint) | 6:1 (ladder) to 8:1 (final), `latent_loss_frac` ≈ 0.11–0.14 |
  | ≈ 6.7 (`16x16_so101_dual` @ 20k) | ≈ 1:1 — the latent term is at **parity**, over the ceiling |

  At parity, lower the lambdas: ~0.13 (ladder) / ~0.09 (final) puts it back at 2:1 for a
  `mean_abs ≈ 6.7` sidecar. Confirm with the calibrator rather than taking those numbers on faith.
* Nominal weights do **not** determine the realised ratio — the action loss is an MSE on actions in
  `[-1,1]` while the latent loss is L1/MSE on RLA tokens divided by 10. To target a specific ratio,
  measure it first:

  ```bash
  CUDA_VISIBLE_DEVICES=0 $PY rebuttal/exp4_so101/calibrate_loss_weights.py \
    --config-name bc_rla_so101_100_ladder --ratio 2.0 $RLA_ARGS
  ```

  It runs the untrained policy over ~20 real batches and prints copy-pasteable lambdas. Treat 2:1
  as a ceiling, not a goal.
* Every step logs `train_latent_loss_frac` and `train_action_over_latent`. If `latent_loss_frac`
  climbs past ~0.33 the auxiliary signal is taking over — lower the lambdas.

### Other things to watch

* `train_action_loss`, `val_loss` (validation reports the *action* loss only, so it is directly
  comparable with the vanilla BC runs).
* In `ladder` mode, the per-gap breakdown `train_latent_l1_gap4 … gap32`; near gaps should be easier
  than far ones, and if they are not, something is misaligned.
* `batch_size: 64` — with the targets precomputed there is no DINO in the loop, so a BC-RLA step
  costs barely more than a vanilla BC step.
* Output dirs: `runs/policy_outputs/bc_rla_so101_{100,25}_{final,ladder}`.

---

## Resuming / re-running

* **`train.py` (UNet, RLA):** the CLI `--load_dir` is wired out (`train.py:250`, the line is
  commented). Resume by editing `trainer.args.load_dir` in the YAML to point at the run directory
  you want to continue, and drop `skip_load_misc` if you want the optimizer and step counter back.
  `--output_dir` always gets a fresh timestamp.
* **`policies/train_policy.py`:** `training.resume: true` picks up the newest
  `checkpoints/*_epoch*.ckpt` in the same hydra run directory automatically — just re-run the same
  command. For the BC-RLA runs the `latent_encoder_work_dir` must still exist and be unchanged:
  the frozen RLA encoder is never stored inside a policy checkpoint, it is rebuilt on every load.

## Observation history (why there is a dataset subclass)

The policy configs use `n_obs_steps: 2`, so the policy reads `obs["image"][:, :2]` and
`obs["state"][:, :2]`, treating slot 1 as *now*. The stock `ManiSkillSequenceDataset` places extra
image frames with `np.linspace(0, horizon - 1, ...)`, i.e. spread across the **horizon**, so a naive
`n_obs_steps: 2` would deliver:

```
image[0] = frame t        image[1] = frame t + 31      <- 1.03 s in the FUTURE
```

which is unobtainable on the robot, and — worse — collapses the BC-RLA latent-action target from a
32-frame transition to a **1-frame** one, because `_compute_gt_latent_action` reads
`img_first = raw_images[:, n_obs_steps - 1]` and `img_last = raw_images[:, -1]`. The loss curve
stays perfectly healthy while the RLA supervision is gone.

`rebuttal/exp4_so101/dataset/so101_sequence_dataset.py` fixes it: observation slots are placed at
`now - k * obs_history_stride` and may reach back before the window start (down to the episode
start), and the joint-state history is re-gathered at the same instants. With
`obs_history_stride: 16` (= `n_action_steps`):

```
frames:   [ t-16 ,  t ,  t+31 ]      images loaded per sample: 3 (not 33)
slots:      hist   now   look-ahead
states:   [ t-16 ,  t ]              same instants as the images
```

`rebuttal/exp4_so101/workspace.py` is what makes the training
loop actually use the subclass (`TrainVLABCWorkspace.run()` hard-codes its dataset class, and
`FewShotMixedDataset` constructs `ManiSkillSequenceDataset` directly); it also expands
`latent_mode` into `latent_gaps` for both the policy head and the dataset, so the two can never
disagree on order or length.

Note the look-ahead frame is **gone**: `load_extra_frame: 0` everywhere, because the latent target
no longer has to be computed from pixels. Each sample now loads exactly `n_obs_steps` = 2 images
per camera.

## Augmentation and precomputed targets (why they are one change)

The deployment workspace is lit through a window, so appearance drifts between the recording
session and eval. The policy input therefore gets colour/lighting augmentation
(`augment: true` in every policy config):

| knob | models |
|---|---|
| `brightness`, `contrast` | cloud cover, time of day |
| `color_temp` | daylight warm↔cool (per-channel R/B gain) — a hue rotation is *not* the same thing |
| `illumination_gradient` | direct sun as a smooth ramp across the frame, random direction |
| `noise` | sensor gain in low light |
| `saturation`, `hue` | residual camera/white-balance drift |

One transform per **(sample, camera)**, shared across the observation slots: exposure does not
flicker within the 0.53 s history window, but the two cameras are independent devices. Never
applied to the validation split, so `val_loss` stays comparable across runs.

**This is only sound because the RLA targets are precomputed.** With the old online path, the same
augmented tensor fed both the policy *and* the RLA encoder, so the "ground-truth" latent moved with
the augmentation — while the RLA itself had been trained on clean frames. `SO101SequenceDataset`
raises if you ask for augmentation with online targets rather than let that pass silently.

```
   dataset  ──clean frames──►  precompute_rla_latents.py  ──►  sidecar .npy   (once, offline)
       │                                                            │
       └──frames──► augment (brightness/temp/gradient) ──► policy ──┴──► latent loss
                                                             │
                                                             └──────────► action loss
```

## Two gotchas that cost time

1. **Do not use `training.debug=true` for these policy configs.** `TrainVLABCWorkspace.run()`
   hard-overrides `checkpoint_every = 3` *and* `rollout_every = 3`, ignoring CLI overrides. The
   `rollout_every` override calls `create_eval_env()` → `gym.make("SO101-StackBlackCup")`, which
   does not exist and will crash the run at epoch 3. Debug mode also clamps `end_traj_id` to 10,
   which silently shrinks the training set. For a quick smoke test, use this instead:

   ```bash
   $PY policies/train_policy.py --config-dir=rebuttal/exp4_so101/configs/policy \
     --config-name=bc_so101_100 training.num_epochs=1 training.max_train_steps=3 \
     training.max_val_steps=2 training.checkpoint_every=1 training.val_every=1 \
     logging.enable=false dataloader.num_workers=2 hydra.run.dir=/tmp/smoke
   ```

2. **`utils.misc.fetch_state_dict` picks the lexicographically last checkpoint.** Your RLA runs do
   write both `encoder_step0020000.pt` and `encoder_step0020000.snapshot.pt`, and `.snapshot.` sorts
   *after* the milestone — so its "latest" is the snapshot. It also composes filenames as
   `{name}_step{int(step):07d}.pt`, so it cannot open a snapshot even if you ask for one.

   Nothing in `rebuttal/exp4_so101/` goes through it: `ckpt_tokens.find_ckpt` resolves by
   (step number, milestone-before-snapshot) instead, and that one function backs precompute, the
   policy's frozen encoder load, the exporter and the bundle loader. If you load a run directory by
   hand with `fetch_state_dict`, check which file you got.

---

## What is in this folder

| file | purpose |
|---|---|
| `policy/rla_unified_mv.py` | `VLABCPolicyRLAUnifiedMV`, so the policy can load a multi-view RLA encoder |
| `dataset/so101_sequence_dataset.py` | `SO101SequenceDataset`: strided observation history, precomputed RLA targets, colour/lighting augmentation |
| `workspace.py` | `TrainSO101Workspace`: makes the loop use `SO101SequenceDataset`; expands `latent_mode` → `latent_gaps` |
| `precompute_rla_latents.py` | stage 2b: RLA latents for every frame at the requested gaps, per RLA checkpoint step |
| `ckpt_tokens.py` | resolves `encoder_step0040000[.snapshot].pt` from a step token, and compares two spellings of a step. The single source of truth for "which checkpoint" across precompute and training |
| `calibrate_loss_weights.py` | measures the initial action/latent loss magnitudes and prints lambdas for a target ratio |
| `configs/unet/dino_to_image_so101.yaml` | stage 1 |
| `configs/rla/16x16_so101_dual.yaml` | stage 2 |
| `configs/policy/bc_so101_100.yaml` | vanilla BC on all 120 trajectories |
| `configs/policy/bc_rla_so101_100_ladder.yaml` | BC + RLA, 8 latents at gaps 4…32 |

## Key design decisions

| decision | value | rationale |
|---|---|---|
| action chunk | `horizon 32`, `n_action_steps 16`, `n_obs_steps 1` | 30 fps → predict 1.07 s, execute 0.53 s, ~22 replans/episode. Measured median max-joint drift over 16 steps is 16.2 units on a ±100 range (8 %); over 32 steps it is 27.1 (p90 53.5), too much to commit blind through a grasp. |
| RLA max frame gap | 64 (2.13 s) | 4× the executed chunk; the policy's fixed 32-frame latent target sits mid-range. |
| RLA latent | **16 tokens × 16 dim** | 256 numbers per latent action: a tight bottleneck for the policy to regress, and small enough that the precomputed sidecars are only ~340 MB for the whole dataset (16 gaps × 256 values × 2 bytes = 8 KB/frame). Rules out the ManiSkill warm start, which is 32×64. |
| latent targets | precomputed offline | removes DINOv3 + the RLA encoder from the training loop, and decouples the target from the policy's pixels so augmentation is sound. |
| latent gaps | `final` = `[32]`, `ladder` = `[4,8,…,32]` | one config per cell (4 total); `latent_ladder_steps` sets the rung count and the ladder always ends at `horizon`. Ladder additionally gives a per-gap predicted-future *video* at deploy time. |
| latent head | `final`: shared MLP. `ladder`: per-query | in ladder mode each gap reads its own group of output query tokens through a small shared MLP — 21× fewer parameters than a flat head, and each gap gets its own slice of attention capacity. |
| loss balance | `lambda_latent_* = 0.25`, **verify per sidecar** | raw actions must dominate; 2:1 is the ceiling, not the target. The realised ratio depends on the RLA checkpoint's latent scale (the target is `encoder output / 10`), so 0.25 gave ~6–8:1 against an early checkpoint but ~1:1 against a `mean|z| ≈ 6.7` one. Run `calibrate_loss_weights.py` against the sidecar you will train on; `train_latent_loss_frac` logs it every step. |
| latent source | always offline | asserted. Online targets would be derived from augmented pixels while the RLA decoder was trained on clean frames. |
| augmentation | on for BC and BC-RLA, off on val | the deployment workspace is lit through a window. |
| image size | **320** everywhere | 20×20 DINOv3 patches/view, 800 tokens for 2 views (~4× the attention cost of 224). The policy's latent target runs DINO on the *dataset* images, so the policy `img_size` **must** equal the RLA `img_size`. |
| observation history | `n_obs_steps 2`, `obs_history_stride 16` | obs = [frame t−16, frame t]. The stride equals `n_action_steps`, so the history slot is the previous replan's observation — free at deploy time, and 0.53 s back rather than a 33 ms velocity cue. Needs `SO101SequenceDataset`; see below. |
| latent-action gap | 32 frames (1.07 s) | `horizon`, the chunk the policy *predicts*. `n_action_steps` (16) is only how much of it executes before the next replan, and is what `obs_history_stride` tracks. Measured from the *current* frame (obs slot 1); well inside the RLA's trained `[1, 63]`. |
| positional encoding | `learned`, not `plucker` | no camera intrinsics/extrinsics exist for this dataset. |
| foreground masks | constant 255 | no segmentation source; `rgb * (mask > 0)` is the identity. |
