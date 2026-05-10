# Learning Visual Feature-Based World Models via Residual Latent Action

[Arxiv](https://arxiv.org/abs/XXXX.XXXXX) &nbsp;|&nbsp; [Project page](http://mlzxy.github.io/rla-wm) &nbsp;|&nbsp; [<img src="https://colab.research.google.com/img/colab_favicon_256px.png" height="20" style="vertical-align:middle;"> Colab demo](https://colab.research.google.com/github/mlzxy/rla-wm/blob/main/notebooks/colab_demo.ipynb)



This repository contains the implementation, configs, and training/evaluation entry points for the models described in the paper:

- **RLA** — Residual Latent-Action autoencoder over DINOv3 patch tokens.
- **RLA-WM** — Flow-matching world model that predicts future RLA latents conditioned on the current frame and robot actions.
- **BC / BC-RLA** — Learning-from-Actionless-Video policies, optionally augmented with the RLA latent objective on videos-only data.
- **WMRL** — On-policy Visual RL inside the RLA-WM environment.


> We also provide a Maniskill3DWorld Dataset (`maniskill_full`), which includes RGB from 7 cameras, robot and foreground masks, depth, animated robot meshes, and point clouds with voxelization. We initially attempted to build a world model from 3D representations, but ultimately did not use the full dataset. Nonetheless, feel free to use it if it suits your needs! Visualization code and an example are provided below in [§10](#10-3d--multi-view-visualization-optional).



## 1. Setup

```bash
MAX_JOBS=1 uv sync       # CUDA 12.8 + gcc 15.2 recommended; Increase MAX_JOBS if you are running on a large server, "1" is most reliable
source .venv/bin/activate
export PYTHONPATH=.:./third_party/diffusion_policy
```

`PYTHONPATH` needs to include the repo root **and** [third_party/diffusion_policy/](third_party/diffusion_policy/). 

<details>
<summary>Troubleshooting</summary>

- **`uv sync` fails compiling `flash-attn`/`kaolin`/`gsplat`** — verify `nvcc --version` reports CUDA 12.8 and `gcc --version` reports 15.2. Lower `MAX_JOBS` if RAM-bound.
- **`ModuleNotFoundError: diffusion_policy`** — `PYTHONPATH` is missing `./third_party/diffusion_policy`.
- **wandb** — every config has `wandb.enabled: true` with our placeholder `entity:`. Override the entity, set `wandb.enabled: false`, or run with `WANDB_MODE=offline`.
- All entry points must be invoked from the repo root (some inject the repo into `sys.path`).
</details>


## 2. Datasets and pretrained weights

We host both on Hugging Face under [xyzhang368/RLA-WM](https://huggingface.co/xyzhang368/RLA-WM) (model) and [xyzhang368/RLA-WM](https://huggingface.co/datasets/xyzhang368/RLA-WM) (dataset). 


### 2.1 Pretrained weights → `runs/weights/`

```bash
hf download xyzhang368/RLA-WM --local-dir runs/weights
```

Resulting layout:

```
runs/weights/
├── dino-to-image_unet/{maniskill,iws}/<ts>/
├── rla/{maniskill,iws,iws_pusht}/<ts>/
├── rla-wm/maniskill/{panda,xarm,ur10e}/<ts>/
├── rla-wm/iws/{pusht,box,rope}/<ts>/
└── wmrl_checkpoints/{0,1,2,3,5}_bc_*_nstate/checkpoints/*.ckpt
```

### 2.2 Datasets → `data/`

The dataset repo has four splits. Download only what each recipe needs:

| Split | Required for | Size | HF path |
|---|---|---|---|
| `data/maniskill` | RLA, RLA-WM, eval (Maniskill) | 6.2 GB | `maniskill.tar` |
| `data/iws_converted` | RLA, RLA-WM, eval (IWS) | 2.8 GB | `iws_converted.tar` |
| `data/eval_handles` | RLA-WM eval | < 1 MB | `eval_handles/` |
| `data/maniskill_jpgs` | **Optional** — only for [§8](#8-bc--bc-rla-learning-from-actionless-video) BC / BC-RLA policies (JPG-decoded for cheap dataloading) | ≈ 25 GB | `maniskill_jpgs/data.tar.part_a{a,b,c}` |
| `data/maniskill_full` | **Optional** — only for [§10](#10-3d--multi-view-visualization-optional) 3D + multi-view rerun visualization | ≈ 130 GB | `maniskill_full/data.tar.part_a{a..m}` |




```bash
mkdir -p data && cd data

# Required
hf download xyzhang368/RLA-WM --repo-type dataset --include "maniskill.tar"     --local-dir .
tar -xf maniskill.tar     # → data/maniskill/
hf download xyzhang368/RLA-WM --repo-type dataset --include "iws_converted.tar" --local-dir .
tar -xf iws_converted.tar # → data/iws_converted/
hf download xyzhang368/RLA-WM --repo-type dataset --include "eval_handles/*"    --local-dir .
                          # → data/eval_handles/{maniskill,iws}/handles.*.json

# Optional (BC / BC-RLA policies)
hf download xyzhang368/RLA-WM --repo-type dataset --include "maniskill_jpgs/*" --local-dir .
cat maniskill_jpgs/data.tar.part_* | tar -xvf -   # → data/maniskill_jpgs/

# Optional (3D rerun visualization, ≈ 130 GB)
hf download xyzhang368/RLA-WM --repo-type dataset --include "maniskill_full/*" --local-dir .
cat maniskill_full/data.tar.part_* | tar -xvf -   # → data/maniskill_full/

cd ..
```

<details>
<summary>ManiSkill JPGs is also optional</summary>

Note that you can skip `data/maniskill_jpgs` by modifying the policy configs in `policies/config` to load directly from `data/maniskill`. The program will then decode frames from videos within the dataloader (higher CPU usage, but saves storage space and bandwidth).Note that you can skip the `data/maniskill_jpgs` by changing the policy configs at `policies/config` to load from `data/maniskill` directly. The program will decode frames from videos in dataloader, (higher CPU usage, but saving space and bandwidth).

</details>



## 3. Released checkpoints

#### RLA Encoders

| Domain | Path | Consumed by |
|---|---|---|
| Maniskill (play+ppo) | `rla/maniskill/20260323_21-41-25` | [configs/rla/32x64.yaml](configs/rla/32x64.yaml), all [rla_wm](configs/rla_wm/) and [wmrl](wmrl/configs/) configs |
| IWS (box+rope+sweep+grasp) | `rla/iws/20260403_13-27-02` | [configs/rla_wm/iws_box.yaml](configs/rla_wm/iws_box.yaml), [configs/rla_wm/iws_rope.yaml](configs/rla_wm/iws_rope.yaml) |
| IWS PushT-only | `rla/iws_pusht/20260406_13-33-29` | [configs/rla_wm/iws_pusht.yaml](configs/rla_wm/iws_pusht.yaml) |

#### DINO→Image UNet Decoders

| Domain | Path | Consumed by |
|---|---|---|
| Maniskill | `dino-to-image_unet/maniskill/20260404_22-53-36` | every Maniskill RLA / RLA-WM config |
| IWS | `dino-to-image_unet/iws/20260402_18-21-21` | every IWS RLA / RLA-WM config |

#### RLA-WM (Flow World Models)

| Robot / Scene | Path | Consumed by |
|---|---|---|
| Panda | `rla-wm/maniskill/panda/20260405_11-00-59` | [eval/run_eval.sh](eval/run_eval.sh) `panda`, [wmrl/configs/{pullcube,pullcubetool}.yaml](wmrl/configs/) |
| xArm | `rla-wm/maniskill/xarm/20260404_15-49-08` | [eval/run_eval.sh](eval/run_eval.sh) `xarm`, [wmrl/configs/pokecube.yaml](wmrl/configs/pokecube.yaml) |
| UR10e | `rla-wm/maniskill/ur10e/20260404_15-49-19` | [eval/run_eval.sh](eval/run_eval.sh) `ur10e`, [wmrl/configs/{pusht,rollball}.yaml](wmrl/configs/) |
| IWS PushT | `rla-wm/iws/pusht/20260409_00-49-21` | [eval/run_eval_iws.sh](eval/run_eval_iws.sh) `pusht` |
| IWS Box | `rla-wm/iws/box/20260406_22-51-48` | [eval/run_eval_iws.sh](eval/run_eval_iws.sh) `box` |
| IWS Rope | `rla-wm/iws/rope/20260406_00-23-24` | [eval/run_eval_iws.sh](eval/run_eval_iws.sh) `rope` |

#### WMRL BC Starting Points

| Path | Consumed by |
|---|---|
| `wmrl_checkpoints/{0,1,2,3,5}_bc_*_nstate/checkpoints/*.ckpt` | [wmrl/configs/*.yaml](wmrl/configs/) `pretrained_ckpt:` |



<details>
<summary>❓ Why a separate RLA for IWS PushT?</summary>

We notice that the background motion in PushT is much stronger than the motion of the T object itself (the T remains quite static across many frames). This may cause the RLA to misinterpret object motion as background motion, making the world model learning difficult (despite the RLA reconstruction still shows good quality).

To address this, we train a PushT‑only RLA ([configs/rla/32x64_iws_pusht_only.yaml](configs/rla/32x64_iws_pusht_only.yaml) → `runs/weights/rla/iws_pusht/`), and the [iws_pusht RLA‑WM config](configs/rla_wm/iws_pusht.yaml) consumes this model. With this approach, the PushT RLA‑WM works well. Meanwhile, `iws_box` and `iws_rope` share the combined IWS RLA at `runs/weights/rla/iws/`.

</details>


## 4. UNet (DINO → image) decoder training

```bash
.venv/bin/python train.py --config configs/unet/dino_to_image_v2.yaml      --output_dir runs --num_gpus -1   # Maniskill
.venv/bin/python train.py --config configs/unet/dino_to_image_v1_iws.yaml  --output_dir runs --num_gpus -1   # IWS
```

Outputs land under `runs/<config_name>/<timestamp>/`. The shipped checkpoints under `runs/weights/dino-to-image_unet/` are the result of these recipes.


## 5. RLA training

```bash
.venv/bin/python train.py --config configs/rla/32x64.yaml --output_dir runs --num_gpus -1
```

| Config | Domain |
|---|---|
| [configs/rla/32x64.yaml](configs/rla/32x64.yaml) | Main Maniskill (play + PPO, 7 cameras) |
| [configs/rla/32x64_play.yaml](configs/rla/32x64_play.yaml) | Maniskill play data only |
| [configs/rla/32x64_iws.yaml](configs/rla/32x64_iws.yaml) | IWS, all 6 tasks combined |
| [configs/rla/32x64_iws_pusht_only.yaml](configs/rla/32x64_iws_pusht_only.yaml) | IWS PushT only (see [§3](#3-released-checkpoints) note) |
| [configs/rla/8x8.yaml](configs/rla/8x8.yaml) | Small-codebook ablation (8 latent tokens × 8 dim) |


## 6. RLA-WM training

Same entry point. Each config pins its own frozen RLA encoder and DINO→image decoder.

```bash
.venv/bin/python train.py --config configs/rla_wm/panda.yaml --output_dir runs --num_gpus -1
```

| Config | Domain / Robot or Scene |
|---|---|
| [configs/rla_wm/panda.yaml](configs/rla_wm/panda.yaml) | Maniskill, Franka Panda |
| [configs/rla_wm/xarm.yaml](configs/rla_wm/xarm.yaml) | Maniskill, xArm6 + Robotiq |
| [configs/rla_wm/ur10e.yaml](configs/rla_wm/ur10e.yaml) | Maniskill, UR10e + stick |
| [configs/rla_wm/iws_pusht.yaml](configs/rla_wm/iws_pusht.yaml) | IWS PushT (uses PushT-only RLA) |
| [configs/rla_wm/iws_box.yaml](configs/rla_wm/iws_box.yaml) | IWS Box |
| [configs/rla_wm/iws_rope.yaml](configs/rla_wm/iws_rope.yaml) | IWS Rope |


## 7. RLA-WM evaluation

The eval handles ([data/eval_handles/](data/eval_handles/)) ship pre-computed; no regeneration needed.

```bash
bash eval/run_eval.sh     {panda|xarm|ur10e}   # Maniskill
bash eval/run_eval_iws.sh {pusht|box|rope}     # IWS
```

Outputs land at `runs/eval_output/<robot>/` or `runs/eval_output/iws_<scene>/` with `eval_summary.{json,md}`, per-handle metrics, and rollout videos. Internals: [eval/eval_wrapper.py](eval/eval_wrapper.py) loads the cached handles, spawns workers, and dispatches each sample to the predictor module ([eval/predictors/rla_wm_predictor.py](eval/predictors/rla_wm_predictor.py) for Maniskill, [eval/predictors/rla_wm_predictor_iws.py](eval/predictors/rla_wm_predictor_iws.py) for IWS).


## 8. BC / BC-RLA (Learning from Actionless Video)

Requires `data/maniskill_jpgs/` (see [§2.2](#22-datasets--data)).

```bash
# Robot-data BC baseline
bash policies/train.sh fs_bc      setting=1 n_shots=50

# BC + RLA latent objective on pixel-only data
bash policies/train.sh fs_bc_rla  setting=1 n_shots=50 
```

`setting` selects a (task, robot) pair from [policies/config/fs_bc.yaml](policies/config/fs_bc.yaml#L20-L29):

| `setting` | Task | Robot |
|---|---|---|
| 0 | PushT-v2 | ur10e_stick |
| 1 | RollBall-v1 | ur10e_stick |
| 2 | PullCube-v2 | panda |
| 3 | PullCubeTool-v1 | panda |
| 5 | PokeCube-v2 | xarm6_robotiq |

For PushT (setting=0), please use `n_shots=150`. Note that we exclude the insertion task (setting=4), as it is difficult to learn side insertion using only a front-view camera.


## 9. WMRL (Visual RL inside the world model)

```bash
# bash wmrl/train.sh <task> <reward_mode> <seed1> [seed2 ...]
bash wmrl/train.sh pusht corresponding 6 7
```

| Config | Task | Robot | Pretrained BC checkpoint |
|---|---|---|---|
| [wmrl/configs/pusht.yaml](wmrl/configs/pusht.yaml) | PushT-v2 | ur10e_stick | `0_bc_s2r_nstate` |
| [wmrl/configs/rollball.yaml](wmrl/configs/rollball.yaml) | RollBall-v1 | ur10e_stick | `1_bc_s2e_nstate` |
| [wmrl/configs/pullcube.yaml](wmrl/configs/pullcube.yaml) | PullCube-v2 | panda | `2_bc_s2r_nstate` |
| [wmrl/configs/pullcubetool.yaml](wmrl/configs/pullcubetool.yaml) | PullCubeTool-v1 | panda | `3_bc_s2r_nstate` |
| [wmrl/configs/pokecube.yaml](wmrl/configs/pokecube.yaml) | PokeCube-v2 | xarm6_robotiq | `5_bc_s2r_nstate` |

PokeCube uses reward mode `goal`, all other tasks use `corresponding`.

All seeds 1–15 runs with eval results on wandb: https://wandb.ai/ryx/wmrl_final?nw=nwuserryx.

> Note: In those runs, we do not control the flow matching seeds. The code now seeds the process as in [`_sample_flow_noise`](wmrl/world_model_env.py). Running with the same seed could yield a different result, but the overall conclusion remains valid.


## 10. 3D + multi-view visualization (optional)

`data/maniskill_full/` carries the full multi-camera + depth tensors used to build the 3D voxel/point-cloud views. Render a trajectory to a `.rrd` file viewable in [Rerun](https://rerun.io/):

```bash
.venv/bin/python -m datalib.traj2rdd \
    data/maniskill_full/ppo/xarm6_robotiq/PegInsertionSide-v1/success 0 \
    --use-qpos --voxelize --output runs/viz
rerun runs/viz/0.rrd
```

Replace the dataset path / `traj_id` as needed. Run `python -m datalib.traj2rdd --help` for the full flag list (`--limit`, `--img-size`, `--resolution`, `--vis-masks`, …).


https://github.com/user-attachments/assets/b0aac57c-7352-4ad7-a99a-c071ed9668df

preview of our 3d dataset 


## 11. Repository layout

| Path | Purpose |
|---|---|
| [train.py](train.py) | Single entry point for RLA, RLA-WM, and UNet training |
| [configs/](configs/) | YAMLs for `rla/`, `rla_wm/`, `unet/` |
| [policies/](policies/) | BC / BC-RLA training (Hydra) |
| [wmrl/](wmrl/) | World-model RL training loop and configs |
| [eval/](eval/) | RLA-WM evaluation wrapper and predictor modules |
| [datalib/](datalib/) | Trajectory readers, augmentation, rerun export |
| [src/](src/) | Models, datasets, trainers shared across entry points |
| [third_party/diffusion_policy/](third_party/diffusion_policy/) | Vendored diffusion-policy code; required on `PYTHONPATH` |
| `data/` | Datasets (gitignored, populated in [§2.2](#22-datasets--data)) |
| `runs/weights/` | Pretrained checkpoints (gitignored, populated in [§2.1](#21-pretrained-weights--runsweights)) |
