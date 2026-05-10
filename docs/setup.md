# Setup

## Requirements

- **Python** 3.10
- **CUDA** 12.8 (`nvcc --version`)
- ~200 GB RAM recommended for training (less for inference / eval)

## Install

```bash
MAX_JOBS=1 uv sync
source .venv/bin/activate
export PYTHONPATH=.:./third_party/diffusion_policy
```

Raise `MAX_JOBS` on a large server; `1` is the safest default for compiling CUDA extensions on a memory-constrained machine.

`PYTHONPATH` must include the repo root **and** [third_party/diffusion_policy/](../third_party/diffusion_policy/) — the BC / BC-RLA training in [policies/](../policies/) imports from the vendored copy.

All entry points are run from the repo root.

## Hugging Face token

Required to download DINOv3 weights (and the dataset / checkpoint repos):

```bash
hf login          # interactive
# or
export HF_TOKEN=<your-token>
```

DINOv3 is a gated model — make sure your account has been granted access at <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>. Verify with:

```bash
hf download facebook/dinov3-vitl16-pretrain-lvd1689m
```

If this fails with a 401/403, request access on the model page and re-run after approval.

## Weights & Biases

Every config has `wandb.enabled: true` with `entity: <username>` as a placeholder. Three ways to handle it:

```bash
# 1. Override the entity per run
.venv/bin/python train.py --config <cfg> ... wandb.entity=<your-entity>

# 2. Disable wandb entirely (edit the config or override)
.venv/bin/python train.py --config <cfg> ... wandb.enabled=false

# 3. Run offline
WANDB_MODE=offline .venv/bin/python train.py --config <cfg> ...
```

## Troubleshooting

- **`uv sync` fails compiling `gsplat` (or another CUDA extension)** — verify CUDA 12.8 and gcc 15.2 (recommended); lower `MAX_JOBS` if RAM-bound.
- **`ModuleNotFoundError: diffusion_policy`** — `PYTHONPATH` is missing `./third_party/diffusion_policy`.
- **DINOv3 download fails** — check `HF_TOKEN` is set and you have access to the gated DINOv3 repo.
- **Entry point can't find a module** — make sure you ran it from the repo root (some scripts inject the repo into `sys.path`).
