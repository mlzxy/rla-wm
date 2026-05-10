# Data and Checkpoints

Both are hosted on Hugging Face under [xyzhang368/RLA-WM](https://huggingface.co/xyzhang368/RLA-WM) (model) and [xyzhang368/RLA-WM](https://huggingface.co/datasets/xyzhang368/RLA-WM) (dataset).

## Pretrained weights

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

## Datasets

The dataset repo has four splits. Download only what each recipe needs:

| Split | Required for | Size | HF path |
|---|---|---|---|
| `data/maniskill` | RLA, RLA-WM, ManiSkill eval | 6.2 GB | `maniskill.tar` |
| `data/iws_converted` | RLA, RLA-WM, IWS eval | 2.8 GB | `iws_converted.tar` |
| `data/eval_handles` | RLA-WM eval | < 1 MB | `eval_handles/` |
| `data/maniskill_jpgs` | **Optional** — [BC / BC-RLA](applications.md#bc--bc-rla--learning-from-actionless-video) only | ≈ 25 GB | `maniskill_jpgs/data.tar.part_a{a,b,c}` |
| `data/maniskill_full` | **Optional** — [3D viz](applications.md#3d--multi-view-visualization-optional) only | ≈ 130 GB | `maniskill_full/data.tar.part_a{a..m}` |

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

# Optional (3D viz, ≈ 130 GB)
hf download xyzhang368/RLA-WM --repo-type dataset --include "maniskill_full/*" --local-dir .
cat maniskill_full/data.tar.part_* | tar -xvf -   # → data/maniskill_full/

cd ..
```

You can skip `data/maniskill_jpgs` by editing the policy configs in [policies/config/](../policies/config/) to load directly from `data/maniskill`. The dataloader will then decode frames from the videos at runtime — higher CPU usage, but saves ~25 GB of disk.

## Released checkpoints

The tables below list paths under `runs/weights/` and the configs that consume each one.

### RLA encoders

| Domain | Path | Consumed by |
|---|---|---|
| ManiSkill (play + PPO) | `rla/maniskill/20260323_21-41-25` | [configs/rla/32x64.yaml](../configs/rla/32x64.yaml), all [configs/rla_wm/](../configs/rla_wm/) and [wmrl/configs/](../wmrl/configs/) |
| IWS (box + rope + sweep + grasp) | `rla/iws/20260403_13-27-02` | [configs/rla_wm/iws_box.yaml](../configs/rla_wm/iws_box.yaml), [configs/rla_wm/iws_rope.yaml](../configs/rla_wm/iws_rope.yaml) |
| IWS PushT only | `rla/iws_pusht/20260406_13-33-29` | [configs/rla_wm/iws_pusht.yaml](../configs/rla_wm/iws_pusht.yaml) |

### DINO → image UNet decoders

| Domain | Path | Consumed by |
|---|---|---|
| ManiSkill | `dino-to-image_unet/maniskill/20260404_22-53-36` | every ManiSkill RLA / RLA-WM config |
| IWS | `dino-to-image_unet/iws/20260402_18-21-21` | every IWS RLA / RLA-WM config |

### RLA-WM (flow world models)

| Robot / Scene | Path | Consumed by |
|---|---|---|
| Panda | `rla-wm/maniskill/panda/20260405_11-00-59` | [eval/run_eval.sh](../eval/run_eval.sh) `panda`, [wmrl/configs/{pullcube,pullcubetool}.yaml](../wmrl/configs/) |
| xArm | `rla-wm/maniskill/xarm/20260404_15-49-08` | [eval/run_eval.sh](../eval/run_eval.sh) `xarm`, [wmrl/configs/pokecube.yaml](../wmrl/configs/pokecube.yaml) |
| UR10e | `rla-wm/maniskill/ur10e/20260404_15-49-19` | [eval/run_eval.sh](../eval/run_eval.sh) `ur10e`, [wmrl/configs/{pusht,rollball}.yaml](../wmrl/configs/) |
| IWS PushT | `rla-wm/iws/pusht/20260409_00-49-21` | [eval/run_eval_iws.sh](../eval/run_eval_iws.sh) `pusht` |
| IWS Box | `rla-wm/iws/box/20260406_22-51-48` | [eval/run_eval_iws.sh](../eval/run_eval_iws.sh) `box` |
| IWS Rope | `rla-wm/iws/rope/20260406_00-23-24` | [eval/run_eval_iws.sh](../eval/run_eval_iws.sh) `rope` |

### WMRL BC starting points

| Path | Consumed by |
|---|---|
| `wmrl_checkpoints/{0,1,2,3,5}_bc_*_nstate/checkpoints/*.ckpt` | [wmrl/configs/*.yaml](../wmrl/configs/) `pretrained_ckpt:` |

> **Why a separate RLA for IWS PushT?** Background motion in PushT is much stronger than the motion of the T object itself (the T remains static across many frames). A combined IWS RLA can misinterpret object motion as background motion, which makes world-model learning unstable. We train a PushT-only RLA ([configs/rla/32x64_iws_pusht_only.yaml](../configs/rla/32x64_iws_pusht_only.yaml) → `runs/weights/rla/iws_pusht/`) and consume it from [configs/rla_wm/iws_pusht.yaml](../configs/rla_wm/iws_pusht.yaml). `iws_box` and `iws_rope` share the combined IWS RLA at `runs/weights/rla/iws/`.
