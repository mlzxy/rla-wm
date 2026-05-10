# Applications

Three downstream uses of RLA / RLA-WM, in the order they appear in the paper.

- [BC / BC-RLA — learning from actionless video](#bc--bc-rla--learning-from-actionless-video) — paper §4.2, Table 2
- [WMRL — visual RL inside the world model](#wmrl--visual-rl-inside-the-world-model) — paper §4.3, Table 3, Fig. 6
- [3D + multi-view visualization](#3d--multi-view-visualization-optional) — paper §A and the Maniskill3DWorld dataset

## BC / BC-RLA — learning from actionless video

Requires `data/maniskill_jpgs/` (see [data-and-checkpoints.md](data-and-checkpoints.md#datasets)).

```bash
# Robot-data BC baseline
bash policies/train.sh fs_bc      setting=1 n_shots=50

# BC + RLA latent objective on pixel-only data
bash policies/train.sh fs_bc_rla  setting=1 n_shots=50
```

`setting` selects a (task, robot) pair from [policies/config/fs_bc.yaml](../policies/config/fs_bc.yaml#L20-L29):

| `setting` | Task | Robot | Notes |
|---|---|---|---|
| 0 | PushT-v2 | ur10e_stick | use `n_shots=150` (PushT is harder) |
| 1 | RollBall-v1 | ur10e_stick | |
| 2 | PullCube-v2 | panda | |
| 3 | PullCubeTool-v1 | panda | |
| 4 | PegInsertionSide-v1 | xarm6_robotiq | excluded from the paper — side insertion is hard with a single front-view camera |
| 5 | PokeCube-v2 | xarm6_robotiq | |

## WMRL — visual RL inside the world model

```bash
# bash wmrl/train.sh <task> <reward_mode> <seed1> [seed2 ...]
bash wmrl/train.sh pusht corresponding 6 7
```

| Config | Task | Robot | Pretrained BC checkpoint |
|---|---|---|---|
| [wmrl/configs/pusht.yaml](../wmrl/configs/pusht.yaml) | PushT-v2 | ur10e_stick | `0_bc_s2r_nstate` |
| [wmrl/configs/rollball.yaml](../wmrl/configs/rollball.yaml) | RollBall-v1 | ur10e_stick | `1_bc_s2e_nstate` |
| [wmrl/configs/pullcube.yaml](../wmrl/configs/pullcube.yaml) | PullCube-v2 | panda | `2_bc_s2r_nstate` |
| [wmrl/configs/pullcubetool.yaml](../wmrl/configs/pullcubetool.yaml) | PullCubeTool-v1 | panda | `3_bc_s2r_nstate` |
| [wmrl/configs/pokecube.yaml](../wmrl/configs/pokecube.yaml) | PokeCube-v2 | xarm6_robotiq | `5_bc_s2r_nstate` |

PokeCube uses reward mode `goal`; all other tasks use `corresponding`.

All seeds 1–15 with eval results on wandb: <https://wandb.ai/ryx/wmrl_final?nw=nwuserryx>.

> The wandb runs do not control the flow-matching seeds. The current code seeds the process in [`_sample_flow_noise`](../wmrl/world_model_env.py). Re-running with the same seed may yield slightly different numbers, but the overall conclusion is unchanged.

## 3D + multi-view visualization (optional)

Alongside the trimmed `maniskill` split used for RLA / RLA-WM, we release **Maniskill3DWorld** (`data/maniskill_full/`) — a richer, multi-modal version of the same trajectories: RGB from 7 cameras, robot and foreground masks, depth maps, animated robot meshes, and voxelized point clouds. It's the dataset we built while exploring 3D world models. The paper ended up using a 2D feature-space approach, so the full 3D bundle didn't make it into our final pipeline, but it's a standalone resource that may be useful for 3D / multi-view research and we're shipping it as-is.

The viz tool needs `open3d`, which is not in the base environment (it's only used here). Install it once:

```bash
uv pip install open3d
```

Render a trajectory to a `.rrd` file viewable in [Rerun](https://rerun.io/):

```bash
.venv/bin/python -m datalib.traj2rdd \
    data/maniskill_full/ppo/xarm6_robotiq/PegInsertionSide-v1/success 0 \
    --use-qpos --voxelize --output runs/viz
rerun runs/viz/0.rrd
```

Replace the dataset path / `traj_id` as needed. Run `python -m datalib.traj2rdd --help` for the full flag list (`--limit`, `--img-size`, `--resolution`, `--vis-masks`, …).

Preview clip:

https://github.com/user-attachments/assets/b0aac57c-7352-4ad7-a99a-c071ed9668df
