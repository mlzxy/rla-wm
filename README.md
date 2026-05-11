# Learning Visual Feature-Based World Models via Residual Latent Action

<a href="https://arxiv.org/abs/2605.07079" target="_blank" rel="noopener noreferrer">Arxiv</a> &nbsp;|&nbsp; <a href="http://mlzxy.github.io/rla-wm" target="_blank" rel="noopener noreferrer">Project page</a> &nbsp;|&nbsp; <a href="https://colab.research.google.com/github/mlzxy/rla-wm/blob/main/notebooks/colab_demo.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/img/colab_favicon_256px.png" height="20" style="vertical-align:middle;"> Colab demo</a>

Code for the paper. Four artifacts in this repo:

- **RLA** — Residual Latent-Action autoencoder over DINOv3 patch tokens.
- **RLA-WM** — flow-matching world model that predicts future RLA latents conditioned on the current frame and robot actions.
- **BC / BC-RLA** — learning-from-actionless-video policies, optionally augmented with the RLA latent objective on videos-only data.
- **WMRL** — on-policy visual RL inside the RLA-WM environment.

Plus **Maniskill3DWorld** — multi-modal ManiSkill trajectories (7-camera RGB, depth, masks, animated robot meshes, voxel point clouds). Standalone dataset for 3D / multi-view research; see [docs/applications.md](docs/applications.md#3d--multi-view-visualization-optional).

## Quickstart

**Easiest path — try in Colab.** The [Colab demo](https://colab.research.google.com/github/mlzxy/rla-wm/blob/main/notebooks/colab_demo.ipynb) runs a pretrained PushT RLA-WM rollout and trains 50 WMRL iterations (400 updates) on a free T4. No local setup.

**RLA-WM inference demo:**

```bash
# Environment
MAX_JOBS=1 uv sync
source .venv/bin/activate
export PYTHONPATH=.:./third_party/diffusion_policy

# Pretrained weights → runs/weights/
hf download xyzhang368/RLA-WM --local-dir runs/weights

# Minimal dataset for inference → data/maniskill/ + data/eval_handles/
mkdir -p data && cd data
hf download xyzhang368/RLA-WM --repo-type dataset --include "maniskill.tar"  --local-dir . && tar -xf maniskill.tar
cd ..

jupyter notebook notebooks/inference_demo.ipynb
```

This setup covers RLA-WM inference only. For training, evaluation, BC/BC-RLA, or WMRL, see the docs below — each may need additional splits (see [docs/data-and-checkpoints.md](docs/data-and-checkpoints.md)). If `uv sync` fails or you need detail on environment requirements, see [docs/setup.md](docs/setup.md).

## Docs

| Goal | Doc |
|---|---|
| Set up the environment | [docs/setup.md](docs/setup.md) |
| Download datasets & weights | [docs/data-and-checkpoints.md](docs/data-and-checkpoints.md) |
| Train RLA / RLA-WM / decoder | [docs/training.md](docs/training.md) |
| RLA-WM Evaluation | [docs/evaluation.md](docs/evaluation.md) |
| Learn from Actionless Videos, WMRL | [docs/applications.md](docs/applications.md) |

## Repository layout

| Path | Purpose |
|---|---|
| [train.py](train.py) | Single entry point for RLA, RLA-WM, and UNet training |
| [configs/](configs/) | YAMLs for `rla/`, `rla_wm/`, `unet/` |
| [policies/](policies/) | BC / BC-RLA training (Hydra) |
| [wmrl/](wmrl/) | World-model RL training loop and configs |
| [eval/](eval/) | RLA-WM evaluation wrapper and predictor modules |
| [datalib/](datalib/) | Trajectory readers, augmentation, Rerun export |
| [src/](src/) | Models, datasets, trainers shared across entry points |
| [third_party/diffusion_policy/](third_party/diffusion_policy/) | Vendored diffusion-policy code; required on `PYTHONPATH` |
| `data/` | Datasets (gitignored, populated per [docs/data-and-checkpoints.md](docs/data-and-checkpoints.md)) |
| `runs/weights/` | Pretrained checkpoints (gitignored, populated per [docs/data-and-checkpoints.md](docs/data-and-checkpoints.md)) |

## Citation

```bibtex
@article{zhang2026learning,
  title={{Learning Visual Feature-Based World Models via Residual Latent Action}},
  author={Zhang, Xinyu and Xu, Zhengtong and Tao, Yutian and Wang, Yeping and She, Yu and Boularias, Abdeslam},
  journal={arXiv preprint arXiv:2605.07079},
  year={2026},
  eprint={2605.07079},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
