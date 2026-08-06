> This file belongs to `sidecar/new-files/`, a tree meant to be copied onto a repository
> root (`rsync -a sidecar/new-files/ <repo>/`). Its relative links are written for that
> destination, so they do not resolve from where the file currently sits.

# RLA → VLA-Adapter — findings, design, and the traps

What the integration actually looks like once you read the code, why each decision was made, and
the six places the original design note
([`third_party/VLA-Adapter/rla-vla-adapter-integration.md`](../third_party/VLA-Adapter/rla-vla-adapter-integration.md),
v2) is wrong about this tree. For the commands, see [rla-vla-adapter-runbook.md](rla-vla-adapter-runbook.md).

Status as of **2026-07-25**: the RLA autoencoder is **not trained yet**. Everything below is built
and gated against a randomly-initialised `f_enc` and a synthetic sidecar; only the *numbers* wait on
`f_enc`. When it lands, one re-run of `scripts/extract_rla_latents.py` is the whole delta.

---

## 1. The goal

Get a LIBERO number where RLA measurably improves VLA-Adapter, with an ablation that separates
"RLA specifically" from "any auxiliary future-prediction target" (reviewer 9yrp) and gives an
apples-to-apples comparison against the LAPA/UniVLA line (reviewer msVp). LIBERO's wrist camera is
also the direct rebuttal to msVp's "residual-space assumption breaks under camera motion".

The mechanism, unchanged from the design note: give the policy head **extra output-query tokens**
that are trained to predict the RLA latent action `z`, alongside the tokens that predict actions.
`z` is an *output*, never an input, so nothing needs privileged future information at inference.

---

## 2. Corrections to the v2 design note

Verified against this tree, not against the pinned learning clone the note was written from.

| # | The note says | Actually |
|---|---|---|
| 1 | `Q = 32`, `\|z\| = 2048` | [`configs/rla/16x64_libero.yaml:9-10`](../configs/rla/16x64_libero.yaml) → `latent_num_tokens: 16`, `token_dim: 64` → **Q=16, \|z\|=1024**. There is no `vq` block in the config, so `use_vq=False` ([`rla_autoencoder_trainer.py:63`](../src/trainers/rla_autoencoder_trainer.py)) and `norm_output_tokens: false` — **`z` is continuous and unnormalised**. Continuity is a feature: it is exactly what distinguishes RLA from LAPA/UniVLA's VQ codes, and `L1RegressionActionHead` is already a continuous head. Unnormalised is a problem — see §6. |
| 2 | Edit `MLPResNetBlock` | [`finetune.py:134`](../third_party/VLA-Adapter/vla-scripts/finetune.py) defaults `use_pro_version=True` and [`scripts/train_libero.sh`](../scripts/train_libero.sh) passes it explicitly, so the block that runs is **`MLPResNetBlock_Pro`** ([`action_heads.py:287-410`](../third_party/VLA-Adapter/prismatic/models/action_heads.py)). Different code: the gated group is `k_task`/`h_t` at `:391`, and it applies **RoPE** over the `x` sequence at `:381-382`. |
| 3 | (not mentioned) | `x` entering the policy is **all zeros** (`action_heads.py:60-63`), and `MLPResNetBlock_Pro` has no learned positional embedding — so **RoPE is the only thing that distinguishes token positions in block 0**. This is load-bearing for us: appending RLA tokens at positions 8.. gives them distinct identities for free, with no new embedding table. |
| 4 | Bump `window_size` to `NUM_ACTIONS_CHUNK` so observations chunk to length `T`, index-aligned with the action chunk | **Wrong, and it fails silently.** `chunk_act_obs` gathers `tf.range(-window_size + 1, 1)` ([`traj_transforms.py:26-30`](../third_party/VLA-Adapter/prismatic/vla/datasets/rlds/traj_transforms.py)) — that is frames **t−7..t**, the *past*. Actions get their own `range(-W+1, 1+future_action_window_size)` index tensor; observations never see the future. Worse, `RLDSBatchTransform` reads `observation["image_primary"][0]` ([`datasets.py:161`](../third_party/VLA-Adapter/prismatic/vla/datasets/datasets.py)), so the policy's input image would quietly become frame t−7, and every image would be decoded 8×. **Leave `window_size=1`.** |
| 5 | Write the rebuilt dataset to `data/libero_converted` | That path is already taken. `data/libero_converted` is the **`TrajectoryDataset` copy** (per-episode `*.mp4` + `metadata.h5`) that the RLA autoencoder itself reads, written by [`scripts/convert_libero_to_trajectory.py`](../scripts/convert_libero_to_trajectory.py). `data/libero` is the RLDS/TFDS source VLA-Adapter reads. Two different formats, two different consumers. |
| 6 | "the vla-adaptor clone is pinned — do not modify it in place" | No longer true. [`docs/vla-adapter.md:12-15`](vla-adapter.md): VLA-Adapter is **vendored committed source** since `4e05436`; edit it like any other code and mark the edit `# v4world:` (existing convention, e.g. `finetune.py:61`). `scripts/patches/vla-adapter/` is kept only as a record of the delta from upstream. |

### An upstream corollary worth knowing

Follow row 3 through and the *non*-Pro `MLPResNetBlock` has no positional signal at all — no RoPE,
no embedding, and an all-zero `x`. Every query position is then mathematically identical, so the
head emits the **same action eight times**. Measured, same seed and inputs, `max |action[i] −
action[0]|` across the chunk:

| | `use_pro_version=False` | `use_pro_version=True` |
|---|---|---|
| chunk spread | **0.000e+00** | 2.470e-01 |

So the non-Pro path cannot represent a varying action chunk at all. Nothing here depends on fixing
that — `use_pro_version=True` is the default and what `train_libero.sh` passes — but it is worth
knowing before comparing any non-Pro number against a Pro one. It is also why enabling RLA on the
non-Pro block raises instead of training a head whose RLA tokens would all predict the same `z`.

### Two facts the note did not have, both load-bearing

- **`episode_metadata/file_path` is not unique per episode.** It names the source HDF5 *task* file,
  and every demo of a task shares it — spatial episodes 1 and 2 both report
  `..._wooden_cabinet_and_place_it_on_the_plate_demo.hdf5`. It cannot be a join key. (This is the
  check the note's §5.3-b flagged as "not yet verified"; the answer is no.)
- **The extractor does not need `data/libero_converted` at all.** `data/libero` RLDS carries both
  views at 256², `observation/state` and `language_instruction`; the camera geometry comes from
  `data/libero_converted/camera_params.json` through `episode_camera_arrays()`, which derives
  per-frame `K` and camera-to-world from `state` + the instruction. Reading RLDS directly makes
  episode indexing and the join key exact.

  That function used to live in `scripts/convert_libero_to_trajectory.py`, which sets
  `CUDA_VISIBLE_DEVICES=""` at module scope (it only decodes TFRecords and must keep TF off the
  GPU) — importing it would have blinded torch, and the extractor needs TF-on-CPU *and*
  torch-on-GPU in one process. So the pure-numpy geometry moved to
  [`datalib/libero_camera.py`](../datalib/libero_camera.py); the converter re-exports it, and
  `scripts/backfill_libero_camera_params.py` / `scripts/test_libero_camera_geometry.py` are
  unchanged.

---

## 3. Frame / action alignment — measured

The question that gates everything downstream: does step `k` pair one frame with one action, and
which transition does `action[k]` cause?

Measured on `data/libero/libero_spatial_no_noops`, first 6 episodes:

- **N steps ↔ N frames (both views) ↔ N actions ↔ N states.** e.g. `steps=110`,
  `image=(110,256,256,3)`, `wrist=(110,256,256,3)`, `action=(110,7)`, `state=(110,8)`.
- **`action[k]` drives `s[k] → s[k+1]`.** The discriminating test is the gripper: correlate the
  gripper command `action[:,6]` against the change in finger separation
  `width[k+1+lag] − width[k+lag]`. It peaks at **lag 0**:

  | lag | −2 | −1 | **0** | +1 | +2 |
  |---|---|---|---|---|---|
  | corr | −0.307 | −0.308 | **−0.316** | −0.309 | −0.275 |

  (The end-effector version, `cos(a[k], s[k+1]−s[k]) = +0.92`, agrees but does not discriminate:
  consecutive OSC deltas are too correlated for the lagged variant to look different.)
- Therefore there are **N frames but only N−1 real transitions**. The last action leads nowhere.
  Targets at the tail must clamp to `N−1` — which is exactly what `chunk_act_obs` already does to
  the action chunk (`floored_action_chunk_indices = min(max(idx, 0), traj_len − 1)`), so clamping
  keeps the two in lockstep.

---

## 4. Why the targets are anchored, not consecutive

For frame `t` and `i = 0 … A−1` (`A = NUM_ACTIONS_CHUNK = 8`):

```
z_i(t) = f_enc( s_{min(t+i+1, N-1)} − s_t )        # both views concatenated, as the trainer does
```

Every one of the `A` latents is anchored on **frame `t`**, the observation the policy conditions on.
`z_i` is therefore the cumulative consequence of actions `a_t … a_{t+i}`, and RLA token group `i`
lines up with action token `i`.

The alternative — consecutive pairs `f_enc(s_{t+i+1} − s_{t+i})`, which is what the note's §3
describes — was rejected for two reasons:

1. **The decoder diagnostic only works anchored.** `f_dec` is trained as `f_dec(x_t, z) → x_{t+1}`,
   with `x_t` the *first* frame of the pair. Anchored, all `A` predicted latents decode against the
   current frame, giving the `A`-step "imagined rollout" figure the note's §7 wants. With
   consecutive pairs, decoding `z_i` needs `s_{t+i}` — a frame the policy does not have at
   inference. (Consecutive pairs are also 8× cheaper to store, because `z_i(t) = z_0(t+i)` is
   shareable. That is the only argument in their favour.)
2. **Gap-1 residuals barely carry signal on the static camera.** Mean `|img[k+g] − img[k]|` in
   0–255 units, same 6 episodes:

   | gap | 1 | 2 | 4 | 8 | 16 |
   |---|---|---|---|---|---|
   | agentview | 2.5 | 3.8 | 5.4 | **7.0** | 8.7 |
   | wrist | 7.5 | 10.6 | 14.9 | **20.4** | 26.1 |

   Anchored spans gaps 1..8, i.e. 2.5→7.0 on agentview. All-consecutive sits at 2.5 forever.

There is a third reason, from the autoencoder's own side: it trains on pairs with gap
`~U{1..63}` (`trajectory_dataset.py:1375` and `_sample_horizon`, driven by `horizon: [2, 64]` and
`num_frames: 2` — frames `[frame_id, frame_id + h − 1]`). Gap 1 is the extreme tail of that
distribution; gaps 1..8 sit comfortably inside it.

`scripts/extract_rla_latents.py --pairs consecutive` still emits the other variant — the DINO cache
makes it free — so "anchored vs consecutive" stays an ablation rather than a rebuild.

---

## 5. The join: sidecar + content hash

`data/libero` stays untouched. The extractor writes one `.npy` per episode; the tf.data graph looks
it up once per episode inside `libero_dataset_transform`.

**The key is `sha1(observation["state"].astype(float32).tobytes())`.** `state` is a
`Tensor(shape=(8,), float32)` in the TFRecord, so the bytes are identical on both sides —
the extractor reads the same records the trainer does. It is unique per episode (Gate 2 asserts the
bijection), unlike `file_path` (§2).

Why not rebuild the TFDS dataset with `rla` as a native feature (the note's recommended §5.3-a):
~10 GB rewrite across four suites, a JPEG re-encode of every frame unless `SkipDecoding`
passthrough works, and a hand-written `features.json`/`dataset_info.json` — for a join that a
32-byte hash solves exactly. The sidecar is also versioned per autoencoder run, which matters
because `f_enc` will be retrained.

Where each piece attaches, and why it has to be there:

| File | Change | Why there |
|---|---|---|
| `oxe/transforms.py` `libero_dataset_transform` | `tf.numpy_function` lookup → `trajectory["rla"]` | The per-dataset standardize hook; runs **once per trajectory**, before flattening. `observation["state"]` is still present here. |
| `rlds/dataset.py` `restructure()` | pop `rla` before the dict is rebuilt, put it back at **top level** | `restructure` builds a fresh dict and drops everything not listed. Top level, *not* `observation` — observation keys get windowed with past indices by `chunk_act_obs`, and `add_pad_mask_dict`/`goal_relabeling.uniform` also walk `observation`. |
| `traj_transforms.py` `chunk_act_obs` | truncate to `effective_traj_len` | The sidecar is already windowed per frame, so there is nothing to gather — just the same truncation `absolute_action_mask` gets at `:56`. |
| `datasets.py` `RLDSBatchTransform.__call__` | pass `rlds_batch["rla"]` through | After `.flatten()` each element is one frame, so this is `(A, Q·D)`. |
| `util/data_utils.py` collator | stack like `actions` | Nothing else stacks unknown keys. |

Frame transforms (`apply_frame_transforms`) only touch `observation` and `task`, so a top-level key
passes through decode/resize/augment untouched.

> ⚠️ **Editing `libero_dataset_transform` invalidates the dataset-statistics cache.**
> `get_dataset_statistics(hash_dependencies=(..., inspect.getsource(standardize_fn)))`
> (`rlds/dataset.py:206`) hashes the function's *source text*, so any edit — even one that is inert
> when RLA is off — costs one full recompute pass per suite (a few minutes; the result is cached as
> `data/libero/<suite>/1.0.0/dataset_statistics_<hash>.json`).
>
> Two things make this benign, both **measured**:
> - The hash depends on the source text, not on `RLA_STEPS`, so **all four arms share one statistics
>   file** and are normalisation-comparable by construction.
> - The recomputed statistics match the ones the released checkpoints were trained with. Against
>   `runs/weights/vla-adapter/LIBERO-Spatial-Pro/dataset_statistics.json`: `action`/`proprio`
>   `mean`/`std`/`min`/`max` differ by exactly **0.0**, `q01`/`q99` by ≤ 2.2e-7 (float round-off),
>   and `num_transitions=52970` / `num_trajectories=432` match. That is Gate 7.

---

## 6. Target normalisation

`z` is continuous and unnormalised (§2.1), so its scale is whatever the autoencoder's training
happened to settle on. Feeding that straight into an L1 term next to a normalised-action L1 term
makes `lambda_rla` meaningless and unstable across autoencoder runs.

The extractor therefore accumulates **per-`(q, d)` mean/std** (1024 pairs — the token slots are
distinct, so per-slot statistics are the right granularity, not a single global scalar) plus a
global RMS, and writes both to `stats.json`. `rla_targets.py` normalises on read, and the same
`stats.json` un-normalises for the `f_dec` diagnostic. `lambda_rla` is then comparable across runs.

---

## 7. Architecture: where the RLA tokens go

Unchanged in spirit from the note's §3, adapted to `MLPResNetBlock_Pro` and Q=16.

```
x_action : (B, 8, ACTION_DIM*D) -> layer_norm1 -> fc1 -> relu -> (B, 8, D)
x_rla    : (B, RLA_STEPS*RLA_QUERIES, D)  zeros (+ noise in training), starts at D — no fc1
x        = cat([x_action, x_rla], dim=1)          # (B, 8 + 128, D) = (B, 136, D)
   ... 24 × MLPResNetBlock_Pro on the combined sequence ...
action   = fc2(layer_norm2(x[:, :8]))                                  # (B, 8, 7)
z_hat    = fc2_rla(layer_norm2_rla(x[:, 8:].reshape(B, 8, 16, D)))     # (B, 8, 16, 64)
```

- **One combined self-attention pass.** Splitting into "action-as-Q" and "rla-as-Q" passes would
  silently drop cross-group attention unless `K/V` were re-merged in both. Don't.
- **Gating is vectorised, not duplicated.** `MLPResNetBlock_Pro` multiplies the `h_t` scores by a
  scalar `tanh(gating_factor)`; that becomes a `(1, 1, T, 1)` vector built from `gating_factor`
  (positions `< 8`) and a new zero-init `gating_factor_rla` (positions `>= 8`), so each group learns
  its own "how much Raw feature to let in" schedule. Per-future-step gates (`T+1` of them) stay a
  later refinement.
- **RoPE now runs at `seq_len = 136`.** Action and RLA groups share one ladder. Fine, and it is what
  gives the RLA tokens their identity (§2.3).
- **`gating_factor_rla`, `fc2_rla` and `layer_norm2_rla` only exist when `RLA_STEPS > 0`**, so with
  RLA off the state dict is byte-identical to upstream and released `LIBERO-*-Pro` checkpoints still
  load. Gate 4 asserts this.
- **RLA tokens stay in `x` at inference.** The attention distributions were trained with them
  present; removing them is a train/test mismatch. Only the `z_hat` projection is skipped.

Cost: 8 → 136 policy tokens. With Q=16 this is half the note's v2 estimate (~7 GFLOP/forward across
24 blocks against `D=896`, `K=512`) — low single digits of a percent against the VLM forward pass.
Parameter count is unaffected by sequence length.

### The strict-load trap

`experiments/robot/openvla_utils.py:510-542` rebuilds `L1RegressionActionHead` with four fixed
kwargs and then calls `load_state_dict(state_dict)` — **strict**. A `finetune.py` CLI flag would let
train and eval disagree about the head's shape and fail (or worse, if it were non-strict, silently
randomise). So `RLA_STEPS` / `RLA_QUERIES` / `RLA_DIM` live in `prismatic/vla/constants.py` and are
read from the environment, the same way `NUM_ACTIONS_CHUNK` is derived from `sys.argv`. Both
launchers export them; `ARM=` picks the set.

### Why `predict_action` keeps its return type

`modeling_prismatic.py:877` does `action_head.predict_action(...)` then
`.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)`. That file also exists as a **copy** at
`pretrained_models/configs/modeling_prismatic.py`, which `check_model_logic_mismatch` installs over
the checkpoint's `auto_map` copy at startup — so any change there has to be made twice and survives
into every checkpoint dir. Adding `return_rla: bool = False` and returning the bare action tensor by
default means **neither file is touched**.

---

## 8. The extractor, in one paragraph

Per episode: decode both views, resize 256→224 with `resize_image(..., cv2.INTER_LINEAR)` and
rescale `K` with `resize_intrinsics(K, 256, 256, 224, 224)` — both from `datalib/dataset.py:41,60`,
because that is the exact path `read_frames` uses and therefore what the autoencoder trains on (do
**not** substitute `tf.image.resize`, which VLA-Adapter's own eval path uses with `lanczos3`). Run
frozen DINOv3-L once over all `N × 2` frames and **keep the tokens in memory for the trajectory
only** (`N × 2 × 196 × 1024` fp32 ≈ 220 MB at N=140); they are never written to disk. Build Plücker
rays per anchor frame. Then run `f_enc` over the `N × A` anchored pairs in batches and write
`(N, A, Q, D)` fp16.

Budget, **measured** on one RTX A6000 with a randomly-initialised `f_enc` (200 `libero_spatial`
episodes, steady state after torch.compile warmup): **0.20 ep/s ≈ 25 frames/s**, GPU pegged at 100%.
Across ~272k frames that is **~3 h** on one GPU, or **~45 min** with one suite per GPU on a 4-GPU
box. Disk: 16.4 KB/frame → **3.9 GB**.

`--batch-pairs 256` is no faster than the default 64, so it is compute-bound, not launch-bound.
`f_enc` dominates DINO roughly 4:1 (`N·A ≈ 1080` encoder forwards per episode against `N·2 ≈ 280`
DINO forwards), which is also why the fp32 token cache below costs ~1.8× — it widens the two
gathers feeding every one of those `N·A` encoder calls.

> ⚠️ **Keep the cached DINO tokens in fp32.** `DINOv3FeatureExtractor.forward` runs the ViT under
> `autocast(fp16)`, but autocast keeps LayerNorm in fp32, so `last_hidden_state` comes back **fp32**
> — and that is what `f_enc` was trained on. Caching them as fp16 to halve memory (~220 MB → ~110 MB
> per trajectory) shifts `z` by ~0.6% (max |d| = 6.1e-3 against `|z|max = 4.7`). Small enough to
> look like noise, systematic enough to bias every target. Gate 6 is what caught it; after the fix
> the extractor matches the trainer **exactly** (max |d| = 0.0).

TF must stay on CPU while torch keeps the GPU: call `tf.config.set_visible_devices([], "GPU")`
immediately after importing TF and before torch touches CUDA. (`prismatic/vla/datasets/rlds/dataset.py:35`
does the same at module scope; the extractor does it explicitly rather than relying on that.)

---

## 9. Ablation matrix

Same injection point, same recipe; only the auxiliary target changes. `ARM=` in
`scripts/train_libero.sh` selects one.

| Arm | Aux target | Answers |
|---|---|---|
| `baseline` | none (`RLA_STEPS=0`, upstream state dict) | floor |
| `dino` | per-step `s_{t+i+1}` regression (the DINO-WM-style strawman) | 9yrp: "would any future-prediction target do this?" |
| `univla` | discrete VQ latent-action code | msVp: apples-to-apples vs LAPA-style |
| `rla` | `z*_i` from frozen `f_enc` | headline |

Two extra arms the extractor makes nearly free: `--pairs consecutive` (anchored vs consecutive
targets), and per-view extraction (static-only / wrist-only / fused) for the camera-motion rebuttal.

**The baseline arm must be `RLA_STEPS=0`, not `lambda_rla=0`.** The RLA tokens sit in the shared
self-attention, so they change the action path even with zero loss weight. Gate 5 asserts exactly
this, so nobody reaches for the cheaper-looking knob.

---

## 10. Verification

`scripts/verify_rla_targets.py`, gate style of `scripts/verify_vla_adapter_port.py`. Gates 1–5 and 7
run today, with no trained `f_enc`.

| Gate | What it proves |
|---|---|
| 1 | Alignment: `N == len(actions) == len(images) == len(states)`; gripper-lag correlation peaks at lag 0 (§3 as a regression test). |
| 2 | `sha1(state)` is a bijection over each suite and matches `keys.json`; `file_path` is *not* unique. With `--sidecar`, also: every episode has an entry with matching `episode_index`/`traj_len`, and `stats.json` covers `sum(traj_len) × A` rows — the last one caught a real bug where a *resumed* extraction computed the normalisation from only the episodes it happened to rewrite. |
| 3 | **End-to-end index alignment.** A synthetic sidecar with `rla[t, i, :3] = (episode_index, t, i)` run through the real `RLDSDataset` + collator, asserting the decoded triple matches each frame's own `observation["timestep"]` and its episode. Catches every off-by-one, every shuffle/interleave join break, and the `window_size` trap — without `f_enc`. |
| 4 | `RLA_STEPS=0` gives the upstream state-dict key set; a released `LIBERO-Spatial-Pro` action head loads strict; `predict_action` is bit-identical under a fixed seed. |
| 5 | `RLA_STEPS=8` gives `(B,8,7)` + `(B,8,1024)` with finite grads on `fc2_rla` and `gating_factor_rla`; and the action output *does* differ from `RLA_STEPS=0` at `lambda_rla=0`. |
| 6 | Extractor's encode path equals the trainer's own `inference_batch` on a shared batch, using the same encoder object (runs with `--random-encoder`). Currently **max \|d\| = 0.0**. This gate has already earned its keep: it caught the fp16 token-cache bias above. |
| 7 | The recomputed `dataset_statistics_<hash>.json` matches the released `runs/weights/vla-adapter/LIBERO-<Suite>-Pro/dataset_statistics.json` — i.e. our edit reproduces the normalisation the published checkpoints were trained with. |

---

## 11. Not doing

- **§8's two-stage LAPA-style pretrain** (regress `z*` on actionless video, then swap the head back
  and finetune). Reuses the same extracted targets, so it stays cheap to add later; out of scope now.
- **RLA as a fourth bridge KV condition.** Rejected in the note's §9 and still rejected: `z*` depends
  on a future frame that does not exist at inference. As output queries there is no asymmetry.
- **Rebuilding the TFDS dataset.** §5.
