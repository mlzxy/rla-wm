# RLA → VLA-Adapter Integration Design (LIBERO Rebuttal Experiment)

> Lived at `third_party/VLA-Adapter/rla-vla-adapter-integration.md` until it was moved here with the
> rest of the session's output. It was never tracked by git. Move it back if you want it there.
>
> **Superseded by [`docs/rla-vla-adapter.md`](new-files/docs/rla-vla-adapter.md).** This note was
> written against a different (pinned, read-only) VLA-Adapter clone and gets six things wrong about
> the vendored tree — most importantly §5.2's `window_size` bump, which would silently feed the
> policy *past* frames as its image input, and §3.2's target block, which is not the one that runs
> (`use_pro_version=True` selects `MLPResNetBlock_Pro`). It also assumes `Q=32`, while
> `configs/rla/16x64_libero.yaml` is `Q=16`. Kept for the motivation (§1), the ablation matrix (§6),
> and the rejected alternatives (§8, §9), which all still stand. Everything implementation-facing
> lives in the `docs/` pair.

Status: design converged, not yet implemented. Target: NeurIPS 2026 rebuttal (reviewers msVp, 9yrp).

v2 changes from v1: RLA tokens are now per-future-step (`(B, T, Q, D)`, one RLA representation
per action-chunk step) rather than one global `(B, Q, D)` block; targets are precomputed offline
and baked into a new converted LIBERO dataset instead of extracted online during training.

## 1. Motivation

- **Reviewer msVp**: RLA's core idea is close to the LAPA line of work (unlabeled-video latent
  actions). Since UniVLA's paper already reports LIBERO success rates for itself and LAPA, we
  should evaluate on LIBERO for an apples-to-apples comparison. Our own rebuttal draft already
  commits to this: "explain the difference between LAPA and RLA ... need to add quantitative
  results on LIBERO."
- **Reviewer msVp (weakness 2)**: the residual-space assumption breaks under camera motion /
  background motion. Our own note: "the LIBERO experiment can be important though because it
  includes wrist camera" — i.e. LIBERO's moving wrist cam is a chance to directly rebut this.
- **Reviewer 9yrp**: is the WAM gain from RLA specifically, or would any reasonable auxiliary
  future-prediction target give similar regularization? Needs a controlled ablation, not a single
  number.
- **Internal rebuttal todo**: "Libero (VLA-adaptor) — try to start it today (baseline running +
  start learning the RLA)". VLA-Adapter (arXiv 2509.09372) was already the planned target backbone.

Goal: get a LIBERO number where RLA measurably improves a modern, LIBERO-competitive VLA
(VLA-Adapter), using an ablation design that isolates RLA's contribution from "any auxiliary
target," and ideally using the wrist camera to push back on the camera-motion weakness.

## 2. Recap: what we're plugging into

VLA-Adapter's policy head (`prismatic/models/action_heads.py` in the vla-adaptor clone) is a
24-block `MLPResNet` (`L1RegressionActionHead`). Each block (`MLPResNetBlock`) does:

- Q from the policy tokens `x` (currently `NUM_ACTIONS_CHUNK` tokens, e.g. 8 for LIBERO, each
  dim `D` = VLM hidden size, 896 for Qwen2.5-0.5B).
- Self-attention K/V from `x` itself.
- Cross-attention K/V from `h_a + p` (ActionQuery + proprio, un-gated) and `h_t` (VLM Raw/vision
  patch feature at layer `i+1`, gated by `tanh(gating_factor_i)`, zero-init per block).
- All three score groups are concatenated and softmax'd jointly, then combined V is projected
  back and passed through a residual FFN.

Critically, `T = x.size(1)` is read dynamically inside the block — nothing about the attention
math assumes a fixed number of query tokens. This is what makes the design below a clean
extension rather than a rewrite.

## 3. Design: RLA as dedicated, per-future-step policy tokens

Rejected simpler alternatives:
- Pool some existing hidden state and predict one global `z` with a linear head — only shares
  information through a single upstream pooled vector.
- One global `(B, Q, D)` RLA block for the whole chunk (v1 of this doc) — treats the future as
  one blob instead of mirroring the action chunk's own per-step structure.

Current design: **one RLA representation per action-chunk step**, i.e. action `a_{t+i}` (which
takes the episode from frame `t+i` to `t+i+1`) gets a matching latent action
`z_i = f_enc(s_{t+i+1} - s_{t+i})`, predicted by its own group of `Q` tokens. Shape: `(B, T, Q, D)`
with `T = NUM_ACTIONS_CHUNK` (so the RLA sequence has the same length as, and is index-aligned
with, the action chunk) and `Q = 32` (matching `f_enc`'s own internal query-token convention).

### 3.1 Token construction

```python
T = NUM_ACTIONS_CHUNK   # one RLA group per action step, index-aligned with the action chunk
Q = 32                  # matches the RLA autoencoder's own convention: 32 queries x 64 dim = |z|=2048

x_action = torch.zeros(B, ACTION_DIM*NUM_ACTIONS_CHUNK, D)
           .reshape(B, NUM_ACTIONS_CHUNK, ACTION_DIM*D)
           # [+ learnable_random_perturbations, training only — existing behavior, unchanged]
           -> fc1 -> (B, NUM_ACTIONS_CHUNK, D)

x_rla = torch.zeros(B, T, Q, D)
        # [+ its own freshly-sampled learnable_random_perturbations, training only,
        #    symmetric treatment with x_action]
        # no ACTION_DIM-style reshape/fc1 needed — starts directly at D.
x_rla_flat = x_rla.reshape(B, T*Q, D)   # MLPResNetBlock expects a flat (B, seq_len, D) sequence

x = torch.cat([x_action, x_rla_flat], dim=1)   # (B, NUM_ACTIONS_CHUNK + T*Q, D)
```

For LIBERO (`T=8`, `Q=32`): total sequence length is `8 + 256 = 264` tokens through the Policy —
noticeably more than v1's `8+32=40`. See §4 for the revised (still favorable, but no longer
"basically free") cost analysis.

### 3.2 Joint self-attention, separated gating

`x` (combined) flows through all 24 blocks unchanged *except* the Raw-feature (`h_t`) gate, which
becomes per-query-position instead of a single scalar. This preserves full self-attention
interaction between action tokens and RLA tokens (all groups keep attending to each other every
layer) while letting each group learn its own "how much Raw feature to let in" schedule:

```python
# was: self.gating_factor = nn.Parameter(torch.zeros(1)); ratio_g = tanh(g)  (one scalar, whole block)
gating_factor_action = nn.Parameter(torch.zeros(1))
gating_factor_rla    = nn.Parameter(torch.zeros(1))   # v1: shared across all T*Q rla tokens
ratio_g_action = torch.tanh(gating_factor_action)
ratio_g_rla    = torch.tanh(gating_factor_rla)

gate_per_position = torch.cat([
    ratio_g_action.expand(NUM_ACTIONS_CHUNK),
    ratio_g_rla.expand(T*Q),
])  # (T_seq,) with T_seq = NUM_ACTIONS_CHUNK + T*Q

attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) \
                        * gate_per_position.view(1, 1, T_seq, 1)   # only change vs current code
```

Important: do **not** implement "separate gates" by running two independent attention passes
(action-as-Q and rla-as-Q separately). That silently drops the cross-group self-attention unless
`K_self/V_self` are manually re-merged in both passes. Keeping one combined self-attention op and
only vectorizing the gate avoids that trap entirely.

v1 default above still uses **one shared `gating_factor_rla` across all `T` future steps** (2
distinct gates total: action, rla). Now that there's a natural per-step axis, a v3 refinement
could give each of the `T` future-step groups its own gate (`T+1` distinct gates), letting the
network learn e.g. "trust near-term RLA more than far-future RLA." Not needed for a first pass —
flag as a later refinement if the shared-gate version already shows a gain.

### 3.3 Output head

```python
x_action_out = x[:, :NUM_ACTIONS_CHUNK]                       # (B, NUM_ACTIONS_CHUNK, D)
x_rla_out    = x[:, NUM_ACTIONS_CHUNK:].reshape(B, T, Q, D)    # unflatten back to per-step groups

action_chunk = fc2(layer_norm2(x_action_out))                  # existing path, unchanged shape
z_hat        = fc2_rla(layer_norm2_rla(x_rla_out))             # Linear(D, 64), per-token
                                                                 # -> (B, T, Q, 64)
z_hat_flat   = z_hat.reshape(B, T, Q*64)                        # per-step (B, T, 2048), compare
                                                                 # against T offline-precomputed
                                                                 # targets z*_0 .. z*_{T-1}
```

Loss: `total_loss = l1_loss(action_chunk, gt_actions) + lambda_rla * recon_loss(z_hat_flat, z_star.detach()).mean(dim=1)`
(mean/sum over the `T` per-step RLA losses).

### 3.4 Train vs. inference behavior

Keep the RLA tokens in `x` at inference too — do not drop them. Reasoning: action-token attention
distributions were trained assuming these extra tokens are present; removing them at test time is
a train/test mismatch. Simply skip loading/computing `z_star` at inference — nothing else changes.

## 4. Compute cost (revised — no longer negligible, but still bounded)

v1's argument ("self-attn ≪ cross-attn, so it's free") undersold the cost of the bigger `T*Q`
token count; worth being precise here instead of repeating a reassurance by reflex.

Rough FLOPs per block (`D=896`, cross-attention against `K≈512` Raw-feature tokens per
`finetune.py:899`'s
`NUM_PATCHES`):
- v1 (`T_seq=40`): cross ≈ 2·40·512·896 ≈ 37M, self ≈ 2·40²·896 ≈ 3M, ffn ≈ 40·896² ≈ 32M →
  ~72M/block, ~1.7B total — roughly 0.2% of a single VLM forward pass (~550+ tokens through a
  0.5B-parameter transformer, on the order of several hundred billion FLOPs).
- v2 (`T_seq=264`): cross ≈ 2·264·512·896 ≈ 242M, self ≈ 2·264²·896 ≈ 125M, ffn ≈ 264·896² ≈ 212M →
  ~579M/block, ~14B total — roughly 1–2% of the VLM forward pass.

So the honest statement is: **the Policy's own compute grows roughly 8x from v1 to v2, but stays
in the low single digits of a percent relative to the VLM backbone that dominates the forward
pass.** Parameter count is unaffected by `T_seq` (weights are shared across token positions; only
activations grow). Still comfortably affordable, just not literally a rounding error anymore —
worth watching if `Q` or `T` grow further (e.g. don't naively also scale `Q` up per step without
re-checking this budget).

## 5. Data pipeline: offline RLA extractor + a rebuilt LIBERO dataset

Key realization driving this section: since every action step now needs its own RLA target
(`T=8` targets per training sample, each requiring a frozen DINOv3 forward + a frozen `f_enc`
forward), computing these **online inside the VLA-Adapter training loop is wasteful** (8x the
frozen-encoder calls, every step, forever) and unnecessary, since none of this depends on
VLA-Adapter's own weights. Precompute once, offline, and store the targets alongside the dataset.

### 5.1 New deliverable: the RLA extractor

A standalone script/tool (frozen DINOv3 + the trained-on-LIBERO `f_enc`, per the internal rebuttal
todo "start learning the RLA") that walks every LIBERO trajectory and computes, for each step `i`,
`z_i = f_enc(s_{i+1} - s_i)` where `s` are DINOv3 tokens. Output: one `(traj_len, Q, 64)` (or
flattened `(traj_len, 2048)`) array per episode.

### 5.2 Where this plugs into the existing pipeline (verified against the actual code)

The base LIBERO data is not built by a TFDS builder inside this repo — it's downloaded
pre-converted from `openvla/modified_libero_rlds` to `data/libero` (per
`README.md:132-135`). The per-dataset
standardization hook already exists and is exactly the right injection point:

- `prismatic/vla/datasets/rlds/oxe/transforms.py:827`
  `libero_dataset_transform(trajectory)` — a per-trajectory dict transform (whole-episode tensors,
  shape `(traj_len, ...)`), already mutates `trajectory["observation"]` (adds `EEF_state`,
  `gripper_state`). Adding `trajectory["observation"]["rla"] = <precomputed (traj_len, Q, 64)>`
  here is the natural, minimal-footprint hook.
- `prismatic/vla/datasets/rlds/dataset.py:131-200`
  `restructure()` runs after the standardize_fn and only forwards specific named keys
  (`image_*`, `depth_*`, `proprio`, `timestep`) from `old_obs` into `new_obs` — needs one added
  line, `new_obs["rla"] = old_obs["rla"]`, or the field gets silently dropped here.
- `prismatic/vla/datasets/datasets.py:185`
  hardcodes `window_size=1` (comment already anticipates this: "If we wanted to feed / predict
  more than one step"), while `future_action_window_size=NUM_ACTIONS_CHUNK-1` only extends the
  *action* window. Bump `window_size` to `NUM_ACTIONS_CHUNK` so **all** observation sub-keys
  (images, proprio, and the new `rla` field alike) get chunked to length `T`, automatically
  index-aligned with the action chunk — no custom join-key logic needed once the field lives
  inside `observation`.
- `RLDSBatchTransform.__call__` then just reads `rlds_batch["observation"]["rla"]` (shape
  `(T, Q, 64)` per sample) the same way it already reads `image_primary`, and threads it through
  as a training target. `finetune.py` no longer needs any inline frozen-encoder call — the
  expensive part already happened offline.

### 5.3 How to physically get the field into the TFDS records — two options

- **(a) Recommended: rebuild the TFDS dataset with `rla` as a native feature**, producing a new
  dataset at `data/libero_converted` (matching the user's own naming instinct — literally a
  converted copy of `data/libero` plus one feature). Read `data/libero`'s existing trajectories,
  run the extractor, write a new TFDS dataset with `rla` as a genuine feature-spec entry. Once
  it's a native feature, `libero_dataset_transform`'s edit above is a plain passthrough — no
  runtime file I/O, no identifier bookkeeping, robust under shuffling/interleaving. This is the
  properly "TF-graph-native" answer given the pipeline is a `tf.data` graph, not an ad hoc index
  join.
- **(b) Faster fallback, more fragile: runtime lookup via `tf.py_function`.** Keep `data/libero`
  untouched, store precomputed arrays in per-episode files, and load them inside
  `libero_dataset_transform` via `tf.py_function`/`tf.numpy_function` keyed by whatever stable
  per-trajectory identifier the raw TFDS record exposes. Not yet verified whether
  `modified_libero_rlds` carries such an identifier at the trajectory level — check before
  committing to this path. Also needs care that `tf.data` shuffling/interleaving doesn't break the
  join. Only worth it if (a) proves too slow to build under the rebuttal deadline.

### 5.4 DINO version mismatch (still applies, now scoped to the offline extractor only)

VLA-Adapter's own vision backbone is DINOv2-L
(`vit_large_patch14_reg4_dinov2.lvd142m`, `dinosiglip_vit.py:29`),
while the RLA autoencoder is trained on **DINOv3**-L
(`7.appendix.tex:21`). Different pretrained weights — the extractor must
run its own frozen DINOv3 forward pass and must not reuse VLA-Adapter's DINOv2 tokens. Since this
now happens entirely offline in the extractor (§5.1), it no longer has any bearing on
VLA-Adapter's training-time or inference-time cost at all — a strict improvement over v1, where
this decoupled DINOv3 pass still ran once per training step.

## 6. Ablation matrix (answers msVp + 9yrp with one shared infra)

Same injection point, same training recipe, only the auxiliary target changes (all precomputed
offline the same way):

| Arm | Aux target for the RLA tokens | Answers |
| --- | --- | --- |
| Baseline | none (vanilla VLA-Adapter) | floor |
| Raw-DINO-regress | direct per-step `s_{i+1}` regression (the DINO-WM-style strawman already in our own paper) | 9yrp: "any future-prediction target?" |
| UniVLA-code | discrete VQ latent action code (their own encoder, swapped in) | msVp: apples-to-apples vs. LAPA-style baseline |
| RLA (ours) | `z*_i` from frozen `f_enc` | headline number |

Report per-suite LIBERO success rate (Spatial/Object/Goal/Long or whichever suites match
UniVLA's reported numbers, for direct comparability) for all four arms.

### Wrist-camera angle

LIBERO provides both a static third-person camera and a wrist/eye-in-hand camera (the latter
moves with the gripper — a non-fixed camera). Deliberately extract RLA from static-only /
wrist-only / fused views (the extractor in §5.1 needs to run per-view, or on a chosen fused
input). If wrist-only or fused still helps, that's a direct, quantitative rebuttal to msVp's
"non-fixed camera breaks the residual-sparsity assumption" weakness, not just a verbal concession.

## 7. Bonus qualitative diagnostic

Because `z_hat` stays in the same `(T, Q=32, 64)` token layout as `f_enc`'s real per-step output
(not flattened until the last step), each of the `T` steps can be fed into the paper's own frozen
`f_dec` to decode a predicted future DINO reconstruction — i.e., a full `T`-step "imagined
rollout" of what VLA-Adapter's policy implicitly expects to happen while it predicts actions, not
just a single frame. Free (both networks already exist, frozen), and reads as a strip of
predicted frames next to the real ones — stronger rebuttal figure material than a single-frame
version, and mirrors our own Fig. rla-predictive style.

## 8. Alternative path considered (not primary, higher narrative payoff, more engineering)

Two-stage LAPA-style pretrain → finetune, made easy by `L1RegressionActionHead` already being a
**continuous** regression head (unlike LAPA/UniVLA which need VQ discretization to fit an
autoregressive token head):

1. Stage 1: temporarily swap the head's output dim from `ACTION_DIM` to `Q*64` per step (or
   `T*Q*64` if predicting the whole per-step sequence at once), train the whole VLA-Adapter (VLM
   LoRA + Policy) to regress `z*` on actionless LIBERO video — no real action labels needed.
2. Stage 2: swap the output dim back to `ACTION_DIM`, warm-start from stage 1, finetune normally
   on labeled LIBERO actions.

This is structurally the closest match to what msVp is literally comparing against (LAPA/UniVLA's
own two-stage paradigm), and would make a stronger apples-to-apples claim. Higher engineering cost
than the per-step RLA-tokens design (needs a two-phase training wrapper + head-dim swap/warm-start
logic), though it reuses the exact same offline-extracted targets from §5. Treat as a stretch goal
or camera-ready follow-up if the primary design lands early.

## 9. Idea considered and rejected: RLA as a 4th bridge KV condition

Originally considered treating `z*` as a new gated KV condition alongside `h_t`/`h_a`/`p` (reusing
the exact Bridge Attention gate mechanism). Rejected because `z*` depends on the future frame,
which doesn't exist at inference — this would require privileged-information handling (modality
dropout, train/test mismatch mitigation à la "Learning by Cheating"), adding real risk for a
rebuttal timeline. The chosen design (§3) avoids this entirely: RLA tokens are *output queries*
that cross-attend to predict `z`, not an *input condition* requiring the unknown target — same
pattern already used for action tokens, so no asymmetry to patch around. Worth one sentence in
limitations/future-work, not worth implementing now.

## 10. Implementation touch points & logistics

- **New, separate component — the RLA extractor** (§5.1): frozen DINOv3 + frozen `f_enc`, walks
  `data/libero` trajectories, writes per-step `z_i` arrays.
- **New, separate component — dataset rebuild**: consumes the extractor's output, produces
  `data/libero_converted` as a TFDS dataset with `rla` baked in as a native per-step observation
  feature (§5.3-a), or a `tf.py_function`-based runtime loader as a faster fallback (§5.3-b).
- `prismatic/vla/datasets/rlds/oxe/transforms.py:827` (`libero_dataset_transform`) — surface the
  `rla` field into `trajectory["observation"]`.
- `prismatic/vla/datasets/rlds/dataset.py:131-200` (`restructure()`) — forward `new_obs["rla"] =
  old_obs["rla"]`.
- `prismatic/vla/datasets/datasets.py:185` — bump `window_size` to `NUM_ACTIONS_CHUNK`.
- `RLDSBatchTransform.__call__` — read `rlds_batch["observation"]["rla"]` and thread it through
  as a training target (no frozen-encoder call needed here anymore — that moved offline).
- `prismatic/util/data_utils.py` (`PaddedCollatorForActionPrediction`) — stack the new `rla`
  target field.
- `prismatic/models/action_heads.py` — `MLPResNetBlock`/`_Pro` gate vectorization (§3.2);
  `MLPResNet`/`L1RegressionActionHead` — RLA token construction, split, new output head
  (§3.1, 3.3).
- `vla-scripts/finetune.py` — read the precomputed `z*` target from the batch, add the
  `lambda_rla`-weighted per-step loss term. No frozen-encoder call needed here (moved offline vs.
  v1 of this doc).

**The vla-adaptor clone at `<VLA-Adapter clone>` is a pinned learning
clone (see its CLAUDE.md) — do not modify it in place.** Any real implementation should happen in
a separate branch or a fresh copy.

**Suggested order given rebuttal time pressure**: the RLA extractor + dataset rebuild (§5) is now
the long pole — it gates everything else, including the earlier "two-stage pretraining"
alternative (§8), so start there. Once `data/libero_converted` exists, §3's architecture change is
a contained, mechanical edit to `action_heads.py`. Treat §8, the per-step gate refinement (§3.2
v3), and the wrist-camera-only breakdown as stretch goals if time remains.
