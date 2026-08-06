# exp-4 — real-world SO-101 arm

**There are no robot evaluation results in this release.** No `sessions/`, no `so101_eval.py`, no
success rates. So the strongest claim this experiment supports is: **the data exists, the RLA
autoencoder exists, two policies trained and load.** Whether BC-RLA helps on this robot is
untested.

Everything below is about what you can actually reproduce, which is the pipeline, not a result.

---

## What exists

**Data.** 120 trajectories of one task, "stack the black cup", teleoperated on an SO-101 follower
arm. Two cameras (front and wrist), 6 DoF, 30 fps, 640×480, 41 399 frames total, 11.5 s per episode
on average. A 121st episode was recorded, went wrong, and was dropped — the converter excludes
anything the recorder marked `failure`, which was exactly lerobot episode 43. Episodes were
shuffled with seed 42 before being renumbered, so any prefix of the numbering
is a random sample of the session rather than the first thirty, sloppiest recordings.
`episode_map.json` keeps the `traj_id ↔ lerobot episode` mapping.

**RLA autoencoder.** A 2-view model, 16 latent tokens × 16 dim, at 320 px:
`configs/rla/16x16_so101_dual.yaml`, shipped at `encoder_step0100000.snapshot.pt`. Positional
embeddings are **learned, not Plücker** — there is no camera calibration for this rig, so the
converter writes synthetic intrinsics and identity extrinsics purely so that
`TrajectoryDataset.__getitem__` finds the keys it indexes unconditionally. With
`pos_embed_mode: learned` nothing ever reads their values. Do not switch that to `plucker` without
calibrating the cameras first; it will run and it will be meaningless.

**Two policies**, a matched pair under `runs/policy_outputs/final_realworld/` — same data, same
schedule, the latent objective the only difference:

| config | robot data | latent supervision |
|---|---|---|
| `bc_so101_100` | 120 traj | none (baseline) |
| `bc_rla_so101_100_ladder` | 120 traj | 8 latents, gaps 4…32 |

There is no simulator for this dataset, so there is no `eval_success_rate` anywhere in these runs.
Model selection was held-out BC loss (`val_ratio: 0.05`), `checkpoint.enable_topk` is off, and the
runs only keep rolling `latest_epoch*.ckpt` saves. That is why the honest claim at the top is the
one it is.

---

## The pipeline

Run from the repo root, with:

```bash
export PYTHONPATH=.:./third_party/diffusion_policy
PY=$PWD/.venv/bin/python    # absolute: two steps below run from another directory
```

1. **Get the dataset.** It ships already converted to this repo's trajectory layout, as mp4 —
   see [`../DATA.md`](../DATA.md). It unpacks to `data/realworld/so101_converted/`, which is
   what every config below points at. See "Data format" for what is inside it.
2. **Train the dino_to_image UNet** (visualisation decoder; the RLA needs it to exist).
   ```bash
   CUDA_VISIBLE_DEVICES=1,2,3,4 $PY train.py \
     --config rebuttal/exp4_so101/configs/unet/dino_to_image_so101.yaml \
     --output_dir runs/so101 --num_gpus 4
   ```
   Shipped: `runs/so101/dino_to_image_so101/20260802_19-25-13`, `decoder_step0065000.pt`.
3. **Train the RLA autoencoder.** Paste the stage-2 run directory into `image_decoder_ckpt` in the
   config first — it is loaded with `strict=True`, so it has to be real.
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 $PY train.py \
     --config rebuttal/exp4_so101/configs/rla/16x16_so101_dual.yaml \
     --output_dir runs/so101 --num_gpus 4
   ```
   Shipped: `runs/so101/16x16_so101_dual/20260802_20-10-15`.
4. **Precompute the latent targets.** Always offline, from clean frames — computing them inside the
   training loop would derive them from augmented pixels that the RLA decoder was never trained on,
   and nothing in the loss curve would show it.
   ```bash
   RLA=runs/so101/16x16_so101_dual/20260802_20-10-15
   for i in 0 1 2 3 4; do
     CUDA_VISIBLE_DEVICES=$i $PY rebuttal/exp4_so101/precompute_rla_latents.py \
       --rla $RLA --step 0100000.snapshot --gaps 4,8,12,16,20,24,28,32 \
       --num-shards 5 --shard $i &
   done; wait
   $PY rebuttal/exp4_so101/precompute_rla_latents.py \
     --rla $RLA --step 0100000.snapshot --gaps 4,8,12,16,20,24,28,32   # writes the manifest
   ```
   The last, unsharded pass is not optional: a shard refuses to write a manifest for a set it only
   partly produced. It recomputes nothing and takes about eight seconds.
5. **The BC baseline** (independent of everything above — it needs only the dataset).
   ```bash
   CUDA_VISIBLE_DEVICES=2 $PY policies/train_policy.py \
     --config-dir=rebuttal/exp4_so101/configs/policy --config-name=bc_so101_100
   ```
6. **The BC-RLA run**, with the override line from the next section:
   ```bash
   CUDA_VISIBLE_DEVICES=3 $PY policies/train_policy.py \
     --config-dir=rebuttal/exp4_so101/configs/policy \
     --config-name=bc_rla_so101_100_ladder $RLA_ARGS
   ```
`RUNBOOK.md` has the same sequence with the reasoning, the things to watch in wandb, and the
memory numbers.

---

## The three `???` placeholders

The BC-RLA config deliberately does not resolve on its own. `latent_encoder_work_dir`,
`rla_latent_tag` and `rla_latent_step` are all `???`, and hydra aborts at startup if any is left
unset. That is the point: a BC-RLA run that silently fell back to "no latent loss" would train a
plain BC policy under a BC-RLA name, and its loss curve would look perfectly healthy. A crash is
much cheaper than that.

For the checkpoints this release ships, the override line is:

```bash
RLA_ARGS='latent_encoder_work_dir=runs/so101/16x16_so101_dual/20260802_20-10-15
          rla_latent_tag=20260802_20-10-15__step0100000-snapshot
          rla_latent_step="0100000.snapshot"'
```

or on one line:

```bash
latent_encoder_work_dir=runs/so101/16x16_so101_dual/20260802_20-10-15 rla_latent_tag=20260802_20-10-15__step0100000-snapshot 'rla_latent_step="0100000.snapshot"'
```

**Keep the quotes on `rla_latent_step`.** A step is a *token*, not a number, and the moment one of
these values lands in a YAML file — you edit the config instead of overriding it, or you copy the
line into a script that writes YAML — a bare `0100000` is read by YAML 1.1 as **octal, i.e. 32768**,
and the run then fails looking for a checkpoint that does not exist. (Measured with this repo's
pinned PyYAML/OmegaConf; the same trap gives 8192 for `0020000`, which is the example the config
comments use.) On hydra 1.3's command line a leading zero happens to force a string already, so the
quotes are belt-and-braces there — but they cost nothing and the failure they prevent is silent
until it is loud in the wrong place. `ckpt_tokens.py` explains the token grammar: zero padding is
not part of a checkpoint's identity, the `.snapshot` suffix is.

---

## Data format, honestly

The pipeline above was run on **JPEG q95 frame directories**, themselves decoded from the
recorder's AV1 stream. That is a lot of small files — around 41 k frame pairs per camera — so the
**released dataset ships H.264 CRF-18 mp4 instead**, transcoded from those JPEGs. Per trajectory:

```
front_camera_rgb.mp4                        libx264, yuv420p, crf 18      (no sidecar, by contract)
front_camera_foreground_mask.mp4            libx264, yuv444p, qp 0
front_camera_foreground_mask_encoding.json  {"encoding": "h264_lossless"}
```

`datalib.dataset` reads both forms natively. One thing to know: a frames directory **shadows** an
mp4 of the same name in `_resolve_stream_source_for_key`, so never let both exist for the same key.

Consequences worth being explicit about:

* The **RLA latent sidecar that ships alongside was computed from the JPEGs**, so relative to the
  pipeline that produced the policies it is exact.
* Re-running `precompute_rla_latents.py` on the released mp4 gives **near-identical but not
  bit-identical** latents. crf 18 on top of q95 is a second generation of loss; it is invisible and
  it is not zero.
* The masks are the constant-255 placeholder (there is no segmentation source for this rig), and
  qp 0 is lossless, so they survive the transcode exactly.
* The **raw LeRobot recording is also released** (transcoded AV1 → H.264) if you would rather
  rebuild the whole chain from the original.

---

## A loss-balance warning worth reading

The BC-RLA config ships `lambda_action: 1.0`, `lambda_latent_l1: 0.25`,
`lambda_latent_mse: 0.25`. **Those numbers are not a fixed ratio and they were not chosen for the
checkpoint that shipped.**

The latent target is `encoder_output / 10`, so the realised balance scales with the RLA
checkpoint's latent magnitude, which the sidecar reports as `manifest["value_stats"]["mean_abs"]`.
0.25/0.25 was picked against a sidecar with `mean_abs ≈ 1.9`, where it lands around 6:1–8:1 in
favour of the action loss. The sidecar that actually shipped
(`20260802_20-10-15__step0100000-snapshot`) reports **`mean_abs = 5.05`** — roughly 2.7× larger,
which pushes the latent term to near parity with the action term and past the runbook's own stated
2:1 ceiling. (The RUNBOOK quotes ≈ 6.7 for the 20 k checkpoint of the same run and calls that
parity; 5.05 is the same problem, slightly milder.)

Nothing about a run's loss curve makes this visible. If you retrain, re-derive the lambdas:

```bash
CUDA_VISIBLE_DEVICES=0 $PY rebuttal/exp4_so101/calibrate_loss_weights.py \
  --config-name bc_rla_so101_100_ladder --ratio 2.0 $RLA_ARGS
```

It runs the untrained policy over ~20 real batches and prints copy-pasteable lambdas. Treat 2:1 as
a ceiling, not a target. `train_latent_loss_frac` is logged every step; past ~0.33 the auxiliary
signal is taking over.

---

## RUNBOOK.md

[`RUNBOOK.md`](RUNBOOK.md) is an unedited working log, carried over as it was written: the full
pipeline with the reasoning, GPU memory figures, and the version-guard design. It contains dead
ends and superseded numbers on purpose. Where it disagrees with this README, this README is
right.
