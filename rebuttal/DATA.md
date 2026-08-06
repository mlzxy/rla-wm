# Data and checkpoints for the rebuttal experiments

> **Not uploaded yet.** These archives have been built and verified locally but are not on
> Hugging Face, so the download commands below will not work until they are.

Everything extracts from the repository root and lands on the same repo-relative paths the
shipped configs already reference, so there is nothing to edit after unpacking.

## What is where

Same Hugging Face dataset repo as the main release, `xyzhang368/RLA-WM`.

| Archive | Extracts to | Needed for | Size |
|---|---|---|---|
| `rebuttal_pusht_diverse_data.tar` | `data/maniskill/{ppo_noisy,ppo_diverse,rerender,initframes}` | exp-3 | 7.0 GB |
| `rebuttal_libero_converted/data.tar.part_*` | `data/libero_converted/` | exp-2 | 10 GB (2 parts) |
| `rebuttal_so101.tar` | `data/realworld/so101_converted/` | exp-4 | 816 MB |
| `rebuttal_ckpt_exp1_wmrl_v2.tar` | `runs/weights/wmrl_init_v2/` | exp-1 | 946 MB |
| `rebuttal_ckpt_exp2_libero_rla.tar` | `runs/{16x64_libero,dino_to_image_v1_libero}/` | exp-2 | 1.4 GB |
| `rebuttal_ckpt_exp3_pusht_diverse.tar` | `runs/pusht_diverse/frozen/` | exp-3 | 2.7 GB |
| `rebuttal_ckpt_exp4_so101/data.tar.part_*` | `runs/{so101,policy_outputs}/` | exp-4 | 5.1 GB (2 parts) |

Data comes to about 18 GB, checkpoints to about 10 GB. Nothing here is required to read the
code or the results — the numbers are already in each experiment's `results/` directory.

### You also need two things from the base release

These are not rebuttal artefacts, so they are not in the table above, but exp-1 and exp-3 both
fail without them. They come from
[docs/data-and-checkpoints.md](../docs/data-and-checkpoints.md):

| What | Where it has to land | Who needs it |
|---|---|---|
| The `maniskill` data split | `data/maniskill/ppo/ur10e_stick/…` | every WMRL config's `bc_dataset_cfg` and `env_kwargs.dataset_dir`, in both exp-1 and exp-3 |
| The released DINO→RGB decoder | `runs/weights/dino-to-image_unet/maniskill/20260404_22-53-36` | the BC policy's frozen encode→decode bridge. exp-3 keeps this one **on purpose** even though it ships a newer UNet — swapping it would change the policy's input distribution and the v2 control would stop being a control |

## Downloading

```bash
cd <repo root>

# exp-3 -- the Push-T noisy and diverse collections.
hf download xyzhang368/RLA-WM --repo-type dataset --include "rebuttal_pusht_diverse_data.tar" --local-dir .
tar -xf rebuttal_pusht_diverse_data.tar

# exp-2 -- LIBERO converted to the trajectory format.
hf download xyzhang368/RLA-WM --repo-type dataset --include "rebuttal_libero_converted/*" --local-dir .
cat rebuttal_libero_converted/data.tar.part_* | tar -xvf -

# exp-4 -- the real-robot recording.
hf download xyzhang368/RLA-WM --repo-type dataset --include "rebuttal_so101.tar" --local-dir .
tar -xf rebuttal_so101.tar

# checkpoints -- one archive per experiment, all extracting under runs/
hf download xyzhang368/RLA-WM --repo-type dataset --include "rebuttal_ckpt_*" --local-dir .
tar -xf rebuttal_ckpt_exp1_wmrl_v2.tar
tar -xf rebuttal_ckpt_exp2_libero_rla.tar
tar -xf rebuttal_ckpt_exp3_pusht_diverse.tar
cat rebuttal_ckpt_exp4_so101/data.tar.part_* | tar -xvf -
```

## What each archive actually contains

**`rebuttal_pusht_diverse_data.tar`** — everything exp-3 collected:

| Directory | Trajectories | What |
|---|---|---|
| `ppo_noisy/ur10e_stick/PushT-v2/all` | 4 500 | noisy open-loop replays, goal randomised per trajectory |
| `ppo_noisy/ur10e_stick/PushT-v2/novel/goal{0..5}` | 6 × 50 | successful demos at six shifted goals |
| `rerender/ur10e_stick/PushT-v2/{success,fail,noisy}` | 1 000 / 500 / 4 500 | the same trajectories re-rendered with a randomised goal marker |
| `ppo_diverse/ur10e_stick/PushT-v2/success` | 5 004 | one success per distinct goal, ten evaluation goals excluded at a 25 mm margin |
| `initframes/ur10e_stick/PushT-v2/goal{0..9}` | 10 × 400 | initial frames only, for the ten held-out goals |

**No archive contains a symlink or a hardlink.** Two things were left out for that reason,
and neither costs you any information:

- `ppo_noisy/.../views/` — a browsing tree that grouped `all/` by noise σ and by how far the
  goal moved. Every trajectory already stores that provenance in its own `metadata.h5`, so the
  tree is derived, not data.
- The `*__shard*` and `_shard*` directories from parallel collection — hardlink farms
  identical to the merged directories they sit beside.

**`rebuttal_so101.tar`** — 120 trajectories, 41 399 frames, two cameras. H.264 mp4 per camera
(`crf 18`, `yuv420p`) plus lossless `yuv444p` masks, `metadata.h5`, `index.json`, and one
precomputed RLA latent sidecar (`rla_latents/20260802_20-10-15__step0100000-snapshot/`).

The frames were originally written as JPEG q95 image directories, themselves decoded from the
recorder's AV1 stream. The shipped mp4s are a transcode of those JPEGs, so the latent sidecar
that travels with them (computed from the JPEGs) is exact, while **re-running
`precompute_rla_latents.py` on the released mp4 gives near-identical but not bit-identical
latents**. Train against the shipped sidecar and this does not arise.

**`rebuttal_ckpt_exp2_libero_rla.tar` ships RLA step 90000, but the exp-2 numbers were measured
against a sidecar built from step 40000** (`16x64-a8-step040000-snapshot`, named in every file in
`exp2_vla_adapter/results/`). That checkpoint is not in this archive, and the rebuild does not
fail loudly: `extract_rla_sidecar.py --step` defaults to the newest checkpoint in the directory,
so it quietly produces `16x64-a8-step090000` instead. If you rebuild, you are producing different
latent targets from the ones behind the published numbers.

**`rebuttal_ckpt_exp3_pusht_diverse.tar`** — three single-checkpoint directories. They are pinned rather than "whatever is newest" because
`fetch_state_dict` takes the last file in a directory and there was no way for a config to
name a step until the rebuttal:

| Directory | Consumed by |
|---|---|
| `runs/pusht_diverse/frozen/unet_pusht_noisy_step50000` | the frozen image decoder, everywhere in exp-3 |
| `runs/pusht_diverse/frozen/rla_diverse_step0006000` | the RLA encoder for the WMRL v3 sweep |
| `runs/pusht_diverse/frozen/wm_diverse_step0025000` | the world model for the WMRL v3 sweep |

## Not shipped — and how to rebuild it

| What | Why not | How to get it |
|---|---|---|
| LIBERO source RLDS (9.6 GB) | Public already, and `data/libero_converted/` above is the form everything here consumes | OpenVLA's `modified_libero_rlds` export |
| `data/libero_rla/` sidecars (8.4 GB) | Derived, and tied to one RLA checkpoint | `rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py` against the RLA checkpoint in `rebuttal_ckpt_exp2_libero_rla.tar` |
| VLA-Adapter training checkpoints (52 GB) | Too large, and reproducible | `rebuttal/exp2_vla_adapter/setup/setup_vla_adapter_ref.sh`, then `tools/train_eval.sh` / `tools/train_eval_rla.sh` inside the clone |
| exp-3's intermediate world-model and UNet run directories (17 GB) | Mostly optimizer state; the checkpoints they produced are the three pins shipped above | The configs in `rebuttal/exp3_data_diversity/configs/` already point at the pins |
