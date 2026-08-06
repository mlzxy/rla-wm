# Review guide

Everything is new files under `rla/`, `tools/train_eval_rla.sh` and `.vscode/launch.json`. **No file
under `prismatic/`, `vla-scripts/`, `experiments/` or `tools/train_eval.sh` was touched** —
`rla/train.py` loads `vla-scripts/finetune.py` and swaps out module globals at import time.

---

## Read in this order

Six files, ~600 lines total. The rest is tests and docs.

### 1. `rla/action_head.py` (~230 lines) — start here, this is the architecture

```
x_action : (B, 8, 7·D) → layer_norm1 → fc1 → relu → (B, 8, D)
x_rla    : (B, 8·16, D) zeros (+ training noise); starts at D, so no fc1
x        = cat([x_action, x_rla], dim=1)            # (B, 136, D)
   ... 24 × RlaMLPResNetBlockPro over the combined sequence ...
action   = fc2(layer_norm2(x[:, :8]))               # (B, 8, 7)
z_hat    = fc2_rla(layer_norm2_rla(x[:, 8:]))       # (B, 128, 64) → (B, 8, 1024)
```

Three classes. Look at them in this order:

- **`RlaMLPResNetBlockPro.forward`** — the only place upstream code is copied. It is
  `MLPResNetBlock_Pro.forward` (`action_heads.py:337-410`) verbatim with **two added statements**,
  marked `>>> <<<`. Everything else in the file is a genuine subclass. The two additions are the
  attention design (§ below); read them first and check the copy against the original.
- **`RlaMLPResNetBlockPro.__init__`** — widens `gating_factor` from `(1,)` to `(seq_len, 1)` so
  action positions and RLA positions gate vision independently, and adds the scalar
  `gating_factor_rla2act`. The two masks are `persistent=False` buffers, so only parameters enter
  the state dict.
- **`RlaMLPResNet.forward`** — builds the token sequence, splits it, runs the two output heads.
  Note it returns **only** `action` and stashes `z_hat` on the module. That is deliberate:
  `predict_action` then needs no override at all, and `modeling_prismatic.py:871`'s
  `.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)` keeps working, so neither that file nor its
  `pretrained_models/configs/` copy has to change.

### 2. `rla/data.py` (~130 lines) — how the target reaches `batch["rla"]`

The whole join is **one wrapped module-global function**. Read `_normalize_with_rla` and the
docstring above it. The two facts that make that hook point the only correct one:

1. `traj["observation"]["proprio"]` is still raw there, and for LIBERO is bit-identical to
   `observation/state` (`libero_dataset_transform` splits `state` into `EEF_state`/`gripper_state`,
   `restructure` concatenates exactly those two back). That is the array the join key hashes.
2. It runs *after* `restructure`, which rebuilds the trajectory dict from scratch and drops every
   key it does not know. Anything written into `observation` from here survives.

Only a 4-byte slot id crosses the tf.data graph; the `(8, 1024)` payload is fetched from the mmap in
`RlaBatchTransform`. That keeps the shuffle buffer at 400 KB instead of 3.2 GB per rank, and — the
part worth checking — leaves the per-dataset `standardize_fn` untouched, so the cached
`dataset_statistics_<sha>.json` is reused byte-identically and RLA runs stay normalisation-comparable
to your existing `REF-*` numbers.

### 3. `rla/loss.py` (~120 lines) — the combined loss

`loss = LAMBDA_ACT · L1(action, a*) + λ · L1(z_hat, z*)`, as a wrapper around
`finetune.run_forward_pass` rather than a copy. Two things to check:

- `LAMBDA_ACT=0` **multiplies by zero, it does not skip**. `0.0 * act_l1` still has a defined
  backward, so `fc2` gets a zero `.grad` tensor instead of `None` and DDP does not see an unused
  parameter. `test_head` asserts exactly this.
- the `auto:R` weight mode — see the measurement below for why a fixed λ does not work.

### 4. `rla/warmstart.py` (~150 lines) — stage 1 → stage 2

Two wrapped globals. `get_peft_model` loads stage 1's **merged** VLM into the still-plain model
before the fresh LoRA goes on (the adapter alone does not carry `action_queries`, which is why the
merge path re-injects it by hand at `finetune.py:584`). `init_module` strict-loads the head and
proprio projector.

The other half of this file is the **load report** — a coloured block per module, on rank 0, that
answers "did stage 2 actually start from stage 1" on screen. Three of its rows are checks rather than
prints, and `_verify_loaded` is the one to look at: it re-reads the weights out of the live module
and byte-compares them against the file, because `load_state_dict` not complaining is a weaker claim
than the weights being right. `test_warmstart` runs it both ways — passes on a genuine load, raises
naming the tensor on a tampered one. `rla/README.md` §7 has the annotated output.

### 5. `rla/config.py` (~180 lines) — every knob, one place

Environment variables, not `--flags`, and not only because `FinetuneConfig` is frozen:
`openvla_utils.get_action_head` rebuilds the head from four fixed kwargs and loads **strict**, so
train and eval have to read one switch or they disagree about the head's shape.

### 6. `rla/patch.py` (~90 lines) — the list of everything that gets swapped

Short. Read it last as the index of the whole integration.

---

## The four design decisions worth arguing about

**1. Attention between the groups is one-way.** Action queries read RLA keys, gated per block by a
zero-init `gating_factor_rla2act`. RLA queries **cannot** read action keys at all (`-inf`).

That asymmetry is what makes the two-stage run sound: stage 2 attaches a fresh action head and
starts supervising it, changing what the action tokens carry, and none of that may move the
auxiliary task. Measured: stage 2 loaded from stage 1 predicts **bit-identical** `z`, and the RLA
loss alone leaves `fc1`/`layer_norm1` with *exactly* zero gradient.

Inherited nuance: `tanh(g) = 0` flattens logits to a constant, so a closed gate makes the RLA keys
*uninformative* rather than unattended — the same semantics upstream's zero-init vision gate has at
step 0. That is why the control arm has to be `RLA_STEPS=0`, not a closed gate.

**2. The action tokens stay in the sequence during stage 1**, unsupervised (`λ_act=0`). No action
labels are used either way, so the actionless-pretraining claim is unaffected. Keeping them pins the
RoPE geometry identical across the stages. `RLA_ACTION_TOKENS=0` gives the RLA-only sequence and is
supported (the gates splice), but measured cost: **it moves `z` by 0.89 for the same weights**,
because the RLA tokens' absolute positions shift by 8 and the cross-attention offsets against the
512 vision tokens shift with them.

**3. `z_hat` is computed in eval too** and discarded. One `Linear(896, 64)` over 128 tokens is
nothing, and it makes the module a pure function of its inputs in eval mode — which is what lets the
`isolation` and `warmstart` gates compare runs bit-for-bit.

**4. A join miss raises.** `RlaStore.frame` runs in plain python inside `RlaBatchTransform`, so a
missing episode is a traceback, never a silent block of zeros. The slot lookup itself sits in a
`tf.numpy_function` — the one place that must not raise — so it returns `-1` and lets `frame` fail.

---

## What was verified, with numbers

All gates pass: `.venv/bin/python -m rla.tests`

| gate | result |
|---|---|
| `join` | **1693/1693 episodes, bijection on all four suites**, 0 uncovered, 0 orphaned |
| `align` | 480 samples, 38 episodes, 132 timesteps, **0 misses**; the gripper trajectory of `a_t..a_{t+7}` matches the joined target on every sample |
| `head` | RLA off is **byte-identical to upstream (560 tensors)**; an existing `REF--10000` checkpoint loads strict; chunk varies across its 8 steps (spread 0.223); action differs from RLA-off by 1.92 |
| `isolation` | perturbing `fc1.bias` by +10 leaves `z_hat` **bit-identical** while moving the action output by 1.72; RLA loss alone leaves `fc1` at exactly zero gradient; the action loss trains `gating_factor_rla2act` in all 24 blocks |
| `warmstart` | `z_hat` **bit-identical** across the stage-1 → stage-2 round trip with `fc2` re-initialised; `RLA_ACTION_TOKENS=0` would move it by 0.89 |
| `eval` | strict load through the real `get_action_head` (588 tensors); `predict_action → (1,8,7) → reshape(8,7)`; a wrong-geometry head refuses the checkpoint |

Why `align` is not circular: comparing the target's own encoded `t` against `observation["timestep"]`
would prove nothing — the target was *fetched* using that timestep. The gripper command travels a
completely different path (RLDS → `chunk_act_obs` → collator), so agreement is real evidence.

**End-to-end pipeline run** (2 GPUs, DDP): stage 1 (60 steps, all four suites) → stage 2 (300 steps,
spatial, warm-started) → eval (10 episodes). Warm start loaded 982 tensors, only the tied
`lm_head.weight` left fresh. Eval produced real rollouts — the arm moves through all 220 frames
(`mean|frame_t − frame_0| ≈ 4–5` in 0–255 units). 0% success at 300 training steps is expected.

**Not yet done:** no real training run, so no policy numbers. The `stats` gate needs a finished
stage-2 run on a suite you already have a `REF-*` run for.

---

## Loss scales, measured — and why the default is `auto:0.25`

Robot actions are normalised by upstream: `BOUNDS_Q99` maps `[q01,q99] → [-1,1]`, except the gripper
dim which `action_normalization_mask` leaves raw.

RLA latents are scaled by `RLA_NORM`, default **`center`**: `(z − mean[q,d]) / global_std` — a
single global divisor, so the relative scale of the 1024 latent slots is preserved, with each slot's
static offset removed. All four modes are exactly invertible, so none loses information; what differs
is what the L1 weights. Measured on real latents (gate `norm`):

| mode | `E|z|` | std | what it does |
|---|---|---|---|
| `center` **(default)** | 0.389 | 0.50 | per-slot offsets removed, single global divisor |
| `global` | 0.797 | 1.01 | two scalars, offsets left in |
| `perdim` | 0.750 | 0.94 | whitened; every slot weighted equally |
| `none` | 2.840 | 3.60 | raw encoder units |

Per-channel std spread is only **4.5× max/min** (median 1.70, range 0.95–4.34), so whitening buys
little — `global` and `perdim` agree on the loss scale to 0.046, which is why every λ number below
holds under either.

**One thing worth your decision.** Decomposing the raw latent variance:

| | variance | share |
|---|---|---|
| between-channel (the static per-`(q,d)` offsets) | 9.085 | **71.7%** |
| within-channel (actual frame-to-frame motion) | 3.586 | 28.3% |

So under `global`, about **0.85 of the unit-variance target is a corpus-wide constant** the head has
to memorise, and only ~0.53 is the part that varies with the frame. `fc2_rla.bias` can only absorb
the 64 `d`-marginal offsets, not all 1024 `(q,d)` values, so the rest has to come out of the token
representations. It is learnable and it converges, but it means the reported RLA loss — and hence
`auto:R`'s λ, which tracks that loss — is mostly measuring offset memorisation rather than motion
prediction.

`center` removes exactly that term and nothing else — still one global divisor, so relative slot
scales are preserved, but the DC offsets go — which is why it is the default. Its loss is 0.389 at
chance and all of it is motion signal.

### The stage-1 loss curve

1000 steps, 4 GPUs (effective batch 32), all four suites, `λ_act=0`, `λ_rla=1`, `RLA_NORM=center`:

| step | `rla_l1` | vs chance (0.3885) |
|---|---|---|
| 0 | 0.6104 | +57% — random `fc2_rla` init, i.e. *worse* than predicting the mean |
| 20 | 0.4092 | +5% — has learned to output roughly the mean |
| 100 | 0.3901 | +0.4% — at chance |
| 300 | 0.3555 | **−8.5%** |
| 600 | 0.3340 | **−14.0%** |
| 1000 | 0.3115 | **−19.8%** |

The reference to read against is **0.3885**, the L1 from predicting the constant mean — not step 0,
which is inflated by the random output head. The model crosses chance around step 100 and is ~20%
below it by step 1000, still descending with no sign of a floor, so the 20k-step budget has room.
That is genuine frame-dependent latent-action prediction from image + language, which is the whole
premise of the pretraining stage.

`action_l1` stayed flat at 0.59–0.70 across the entire run with no trend — the confirmation that
`λ_act=0` really does leave the action head unsupervised.

From the stage-2 run (`LAMBDA_RLA=auto:0.25`):

| step | `action_l1` | `rla_l1` | equivalent fixed λ |
|---|---|---|---|
| 0 | 0.57 | 0.77 | 0.185 |
| 25 | 0.31 | 0.80 | 0.098 |
| 100 | 0.27 | 0.79 | 0.085 |
| 300 | 0.24 | 0.77 | 0.077 |

**The action loss falls; the RLA loss does not.** `action_l1` more than halved in 300 steps and will
keep falling toward ~0.05 over a real run, while `rla_l1` sat flat at 0.77–0.81 — the 1024-dim
latent regression is simply much harder. So the equivalent fixed λ has to span roughly 0.185 → 0.016
to hold a constant balance, a >10× range. **A fixed λ cannot do it**: at λ=0.05 the RLA share would
drift from 6% at the start to ~44% at convergence, and the auxiliary term would end up dominating
exactly when you least want it to.

`auto:R` recomputes `λ_t = R · L_act/L_rla` every step, so the RLA term stays at a fixed fraction of
the action term. At `R=0.25` the RLA share is pinned at **0.200** — measured, every step, on both
ranks — which is "comparable contributions, robot action loss larger". `Rla Share` and `Rla Lambda`
are logged to W&B and printed to stderr every `RLA_LOG_EVERY` steps, so this is visible rather than
assumed.

---

## Running it

```bash
# gates first -- ~5 min, no GPU for most of them, and they default their own env
.venv/bin/python -m rla.tests

# stage 1: RLA-only, all four suites. One run, shared by every stage-2 arm.
STEPS=20000 SAVE_FREQ=5000 tools/train_eval_rla.sh pretrain

# stage 2: per suite, warm-started from a stage-1 checkpoint *directory*
ls -d outputs/RLA-P1*--*_chkpt                       # pick one
RLA_INIT_FROM=outputs/RLA-P1--20000_chkpt \
  STEPS=10000 SAVE_FREQ=2500 tools/train_eval_rla.sh long
```

`RLA_SIDECAR` defaults to `data/libero_rla/16x64-a8-step040000-snapshot`; pass it only for a
different sidecar. `tools/train_eval.sh` is the untouched vanilla arm.

The launcher validates `RLA_INIT_FROM` — directory, exactly one `action_head--*_checkpoint.pt`,
merged `*.safetensors` present — and prints the resolved step in its banner *before* the two-minute
join preflight, so a wrong path costs a second. Then the per-module load report (§4) confirms what
landed.

Add `ACCUM=1` to match the REF baselines (`batch=8 accum=1 eff_batch=32`); the launcher defaults to 2.

**Host RAM, not GPU, is what limits run length here** — `tools/OOM_DEBUGGING.md` records a job
SIGKILLed at 194.1 / 193.8 GiB. Stage 1's 4-suite corpus turns out *not* to be the risk: measured at
4 ranks, the mixture costs only ~1.7 GiB/rank over a single suite, because `.repeat()` precedes
`.shuffle()` and there is one frame buffer for the whole mixture. Steps are the risk. So the launcher
now inherits `train_eval.sh`'s two memory settings (`SHUFFLE_BUFFER=80000`, `MALLOC_ARENA_MAX=2`,
together worth 14.7 GiB — 80.4 → 65.7 GiB cgroup), defaults `pretrain` to 20000 steps rather than
80000, warns above 40000 steps with the projection arithmetic, and samples per-rank RSS into
`train_logs/rla-<stage>--<stamp>-rss.log`. `rla/README.md` §8 has the full table.

`.vscode/launch.json` has a config per gate plus single-process smoke runs for both stages;
breakpoints work in `rla/*` **and** in the untouched `prismatic/` + `vla-scripts/` sources.
`rla/README.md` is the reference doc — knobs, architecture, traps.
