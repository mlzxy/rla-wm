# exp-2: RLA latents inside VLA-Adapter, evaluated on LIBERO

## What this is

We took our RLA latent-action targets and put them into a real VLA policy —
[VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter), a 0.5B Qwen2.5 VLA — and trained it
on LIBERO. The policy predicts our latents alongside its actions. That extra loss term is the only
difference from the baseline arm; everything else is upstream's code, upstream's recipe.

## Results

Success rate (%) on the four LIBERO suites, 50 trials per task.

| suite | VLA-Adapter | + RLA |
|---|---|---|
| Spatial | 97.0 | **97.6** |
| Object | 97.4 | **99.2** |
| Goal | 96.6 | 96.6 |
| Long | 93.4 | 93.4 |

Adding the RLA objective improves **Object** and **Spatial**; Goal and Long are unchanged. The
per-run result files these come from are in [`results/`](results/), written by the launcher as each
run progressed.

All RLA arms used `rla=8x16x64`, `lambda_act=1.0`, `lambda_rla=auto:0.25`, sidecar
`16x64-a8-step040000-snapshot`.

---

## How the integration works

`vla_adapter_overlay/rla/` is a pure overlay. Nothing under `prismatic/`, `vla-scripts/` or
`experiments/` is edited by it.

`rla/train.py` loads `vla-scripts/finetune.py` by path and rebinds seven of its module globals
before `finetune(cfg)` runs: the batch transform (adds `batch["rla"]`), the collator (stacks it),
the action head (emits the extra latent tokens), the forward pass (the same loss plus
`lambda * L1(z_hat, z*)`), the wandb logger, `get_peft_model` and `init_module` (warm-start aware).
One more swap sits a level down, on `prismatic.vla.datasets.rlds.dataset.normalize_action_and_proprio`.
All of these are looked up at call time, so rebinding the name is enough.

The full list is in the docstring of
[`vla_adapter_overlay/rla/patch.py`](vla_adapter_overlay/rla/patch.py). With `RLA_STEPS=0` nothing
is patched at all and the run is upstream's own code path, so the two arms produce interchangeable
checkpoints. [`vla_adapter_overlay/rla/REVIEW.md`](vla_adapter_overlay/rla/REVIEW.md) is the audit
that checks the "no upstream file touched" claim.

---

## Setting it up

**VLA-Adapter runs in a directory and a virtualenv of its own, on upstream's own pinned
versions** — torch 2.2, transformers 4.40.1, timm 0.9.10, **peft 0.11.1**. Nothing from that
stack is installed into this repository's environment. Both arms then run on the authors'
versions, so neither number depends on anything we did to the environment.

```bash
source rebuttal/config.sh
rebuttal/exp2_vla_adapter/setup/setup_env.sh
```

That clones upstream at `23fa0c9c`, builds `<clone>/.venv`, installs upstream's stack plus the
five packages their `our_envs.txt` is missing or pins impossibly (the script names each one and
why), clones LIBERO at `8f1084e3` into `<clone>/LIBERO` with its one patch, writes LIBERO's
`.libero/config.yaml`, and links the datasets and the pretrained VLM from this repo so nothing is
copied twice. The clone lands next to this repository by default; `REF_DIR=/somewhere/else`
moves it.

It ends by asserting the two pins that fail *silently* if they drift:

- **`peft == 0.11.1`.** Newer peft expands `target_modules="all-linear"` to a different module
  set, so LoRA would wrap a different part of the network — no error, just a run that is no
  longer comparable.
- **`numpy == 1.26.x`.** The LIBERO stack installs after the training stack, and letting the
  resolver move numpy breaks TensorFlow 2.15.

Then copy the overlay in and train:

```bash
REF=<your VLA-Adapter clone>
cp -r rebuttal/exp2_vla_adapter/vla_adapter_overlay/rla   "$REF/"
cp -r rebuttal/exp2_vla_adapter/vla_adapter_overlay/tools "$REF/"

cd "$REF"
tools/train_eval.sh     spatial     # baseline arm
tools/train_eval_rla.sh spatial     # + RLA
.venv/bin/python -m rla.tests       # the overlay's own gates
```

Each run appends its result to a TSV like the ones in [`results/`](results/). `rla.tests` is the
overlay's gate runner; `--gates head` is the seconds-long subset that needs no data.
[`vla_adapter_overlay/README.md`](vla_adapter_overlay/README.md) and
[`vla_adapter_overlay/rla/README.md`](vla_adapter_overlay/rla/README.md) document the launchers.

---

## Data and the autoencoder

The RLA autoencoder trains on `data/libero_converted` — LIBERO converted to our trajectory format,
with per-episode camera intrinsics and extrinsics so Plücker ray embeddings work. It ships ready to
use; see [`../DATA.md`](../DATA.md).

Two configs:

- [`configs/rla/16x64_libero.yaml`](configs/rla/16x64_libero.yaml) — the RLA autoencoder. Two
  views (agentview + wrist), 16 latent tokens of 64 channels, and **Plücker-grounded positional
  embeddings** on both encoder and decoder, so tokens from the two cameras that observe the same
  region get similar codes.
- [`configs/unet/dino_to_image_v1_libero.yaml`](configs/unet/dino_to_image_v1_libero.yaml) — the
  DINO→image decoder, used only to render snapshots.

`libero_prep/convert_libero_to_trajectory.py` is the converter that produced it. It is here for
one reason: it owns the LIBERO camera geometry, and `sidecar/extract_rla_sidecar.py` imports those
helpers from it rather than keeping a second copy that could drift.

Their internal paths are repo-root-relative and are left exactly as they were run. The models and
trainer these configs name — `MultiViewTokenTransformer`, `DinoToImageDecoderV1`,
`RlaAutoencoderMultiViewTrainer` — live in [`../src/`](../src/) and are registered into
`src.models` / `src.trainers`.

---

## The RLA sidecar

Training the VLA does not run our autoencoder. The latents are extracted **once** into a sidecar
directory keyed by (episode, timestep), and the VLA just looks them up.

The extractor is the one piece of exp-2 that runs in **this** repo's environment, because it
needs our RLA autoencoder. It reads the RLDS export, so it wants TensorFlow, which the base
environment does not carry:

```bash
uv pip install "tensorflow==2.20.*" "tensorflow-datasets>=4.9.10" \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```

```bash
# Once. Needs a GPU and the trained autoencoder. --rla-run is the autoencoder run directory
# holding ckpts/; --out defaults to a name derived from the latent shape, chunk size and
# checkpoint step, which is what you want.
export PYTHONPATH=.:./third_party/diffusion_policy
.venv/bin/python rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py \
    --rla-run runs/16x64_libero/<stamp> --gpus 0,1,2,3
```

Do a small one first — `--suites libero_spatial --limit 3 --viz 3 --out data/libero_rla/dbg` takes
about 30 seconds and is what tells you the targets are sane before you spend an hour.

```bash
# Read it anywhere: numpy only, no torch, none of this repo.
python rebuttal/exp2_vla_adapter/sidecar/read_rla_sidecar.py describe <sidecar-dir>
```

`read_rla_sidecar.py` also has `peek`, `join` and `selftest` subcommands (`--help` lists them);
`describe`, `peek` and `selftest` never import TensorFlow. It is deliberately dependency-free so
the format outlives this codebase.

[`sidecar/README.md`](sidecar/README.md) documents the layout,
[`sidecar/MANIFEST.md`](sidecar/MANIFEST.md) lists what is in the bundle with hashes, and
[`sidecar/samples/`](sidecar/samples/) has a handful of decoded frames to eyeball.

`sidecar/patches/` and `sidecar/apply.sh` are a **record**, not a step: they are the integration as
originally written, diffed against the development repo. The runnable path is the two commands
above.
