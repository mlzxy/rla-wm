# Training

All three stages share the same entry point:

```bash
.venv/bin/python train.py --config <yaml> --output_dir runs --num_gpus -1
```

Outputs land under `runs/<config_name>/<timestamp>/`. `--num_gpus -1` uses all visible GPUs.

The stages depend on each other in this order:

1. [UNet decoder](#1-unet-decoder-dino--image) — needed to render predicted DINO tokens back to RGB for evaluation and visualization.
2. [RLA autoencoder](#2-rla-autoencoder) — produces the latent action used by the world model. Each RLA-WM config pins a frozen RLA encoder.
3. [RLA-WM](#3-rla-wm) — the flow-matching world model on top of frozen RLA + decoder.

Each stage takes ~3 days on 4× A6000 (48 GB) for the released checkpoints. See the paper appendix for full hyperparameters.

## 1. UNet decoder (DINO → image)

```bash
.venv/bin/python train.py --config configs/unet/dino_to_image_v2.yaml      --output_dir runs --num_gpus -1   # ManiSkill
.venv/bin/python train.py --config configs/unet/dino_to_image_v1_iws.yaml  --output_dir runs --num_gpus -1   # IWS
```

The shipped checkpoints under `runs/weights/dino-to-image_unet/` come from these recipes.

## 2. RLA autoencoder

```bash
.venv/bin/python train.py --config configs/rla/32x64.yaml --output_dir runs --num_gpus -1
```

| Config | Domain |
|---|---|
| [configs/rla/32x64.yaml](../configs/rla/32x64.yaml) | Main ManiSkill (play + PPO, 7 cameras) |
| [configs/rla/32x64_play.yaml](../configs/rla/32x64_play.yaml) | ManiSkill play data only |
| [configs/rla/32x64_iws.yaml](../configs/rla/32x64_iws.yaml) | IWS, all 6 tasks combined |
| [configs/rla/32x64_iws_pusht_only.yaml](../configs/rla/32x64_iws_pusht_only.yaml) | IWS PushT only — see the [PushT note](data-and-checkpoints.md#why-a-separate-rla-for-iws-pusht) |
| [configs/rla/8x8.yaml](../configs/rla/8x8.yaml) | Small-codebook ablation (8 latent tokens × 8 dim) |

## 3. RLA-WM

Each config pins its own frozen RLA encoder and DINO → image decoder.

```bash
.venv/bin/python train.py --config configs/rla_wm/panda.yaml --output_dir runs --num_gpus -1
```

| Config | Domain / Robot or Scene |
|---|---|
| [configs/rla_wm/panda.yaml](../configs/rla_wm/panda.yaml) | ManiSkill, Franka Panda |
| [configs/rla_wm/xarm.yaml](../configs/rla_wm/xarm.yaml) | ManiSkill, xArm6 + Robotiq |
| [configs/rla_wm/ur10e.yaml](../configs/rla_wm/ur10e.yaml) | ManiSkill, UR10e + stick |
| [configs/rla_wm/iws_pusht.yaml](../configs/rla_wm/iws_pusht.yaml) | IWS PushT (uses PushT-only RLA) |
| [configs/rla_wm/iws_box.yaml](../configs/rla_wm/iws_box.yaml) | IWS Box |
| [configs/rla_wm/iws_rope.yaml](../configs/rla_wm/iws_rope.yaml) | IWS Rope |
