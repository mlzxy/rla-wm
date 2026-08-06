# RLA → VLA-Adapter: two-stage pretrain → co-train

Stage 1 pretrains on **all four LIBERO suites** by predicting the RLA latent action `z` and nothing
else — no action labels. Stage 2 warm-starts from it and does suite-specific **action + RLA
co-training**, with the action L1 as the dominant term.

**No upstream file is modified.** `rla/train.py` loads `vla-scripts/finetune.py` and patches its
module globals; `tools/train_eval.sh` stays the untouched vanilla arm.

---

## 0. Cheat sheet

```bash
# 1. gates -- no GPU, ~5 min. Do not skip; the align gate is what catches a silent misalignment.
#    RLA_STEPS and RLA_SIDECAR default themselves here (rla/tests/__init__.py), so this is complete.
.venv/bin/python -m rla.tests                 # all 7, ~5 min
.venv/bin/python -m rla.tests --gates head    # the no-data subset, seconds

# 2. stage 1 -- RLA-only pretraining on all four suites. One run, shared by every stage-2 arm.
STEPS=20000 SAVE_FREQ=5000 tools/train_eval_rla.sh pretrain

# 3. stage 2 -- per suite, warm-started from a stage-1 checkpoint *directory*.
RLA_INIT_FROM=outputs/RLA-P1--20000_chkpt \
  STEPS=10000 SAVE_FREQ=2500 tools/train_eval_rla.sh long
```

`RLA_SIDECAR` defaults to `data/libero_rla/16x64-a8-step040000-snapshot` in the launcher, so it only
needs passing for a different sidecar. Suites: `pretrain | spatial | object | goal | long`.

Add `ACCUM=1` to match the REF baselines, which all ran `batch=8 accum=1 eff_batch=32`; the launcher
defaults to `ACCUM=2`. Host RAM, not GPU, is what limits run length here — §8 has the measured
numbers, and the launcher warns before a run that would hit the wall.

**Which stage-1 checkpoint.** `RLA_INIT_FROM` is a checkpoint *directory*, one of the
`outputs/RLA-P1--<step>_chkpt` that stage 1 writes every `SAVE_FREQ` steps (`ls -d
outputs/RLA-P1*--*_chkpt`). The step is read off the `action_head--<step>_checkpoint.pt` inside it,
not parsed out of the directory name, so `latest_checkpoint.pt` layouts work too. The launcher
validates the directory, the head file and the merged `*.safetensors` *before* the two-minute
preflight, and prints the resolved step in its banner; then `rla/warmstart.py` prints a coloured
block per module as each one loads — see §7.

The vanilla baseline is `tools/train_eval.sh long` — unchanged, and its numbers are already in
`eval_out/`.

---

## 1. What is where

| file | what |
|---|---|
| `rla/config.py` | every knob, read once from the environment |
| `rla/sidecar.py` | process-wide sidecar handle; episode↔slot index; `frame(slot, t)` |
| `rla/data.py` | the tf.data hook, `RlaBatchTransform`, `RlaCollator` |
| `rla/action_head.py` | `RlaMLPResNetBlockPro`, `RlaMLPResNet`, `RlaL1RegressionActionHead` |
| `rla/loss.py` | combined loss + the λ policy + W&B metrics |
| `rla/warmstart.py` | stage-1 → stage-2 VLM / head / proprio loading, and the load report |
| `rla/term.py` | the ANSI helper that makes that report visible in a scrolling log |
| `rla/patch.py` | installs all of the above into the loaded `finetune` module |
| `rla/train.py` | **training entry point** |
| `rla/eval.py` | **evaluation entry point** |
| `rla/tests/` | the gates |
| `tools/train_eval_rla.sh` | launcher; `pretrain` and `<suite>` modes |
| `.vscode/launch.json` | debug configs for every gate and both stages (single-process) |

`rla/read_rla_sidecar.py` is reused unchanged — it is the reader, and `preflight` in it is the join
gate. Nothing about the sidecar format is reimplemented here.

## 2. Knobs

| var | default | meaning |
|---|---|---|
| `RLA_SIDECAR` | — | sidecar dir; required for training |
| `RLA_STEPS` | `0` | `0` disables everything; otherwise must equal `NUM_ACTIONS_CHUNK` (8) |
| `RLA_QUERIES` / `RLA_DIM` | `16` / `64` | asserted against `manifest.json` at startup |
| `LAMBDA_ACT` | `1.0` | stage 1 forces `0.0` |
| `LAMBDA_RLA` | `0.05` | a float, or `auto:R`; stage 1 forces `1.0`, stage 2 defaults to `auto:0.25` |
| `RLA_ACTION_TOKENS` | `1` | stage 1 only: keep the action tokens in the policy sequence (§3) |
| `RLA_MASK` | `1` | one-way attention: action ← rla allowed, rla ← action forbidden (§3) |
| `RLA_NORM` | `center` | target scaling: `global` \| `center` \| `perdim` \| `none` (§6) |
| `RLA_LOG_EVERY` | `50` | print the two loss scales and their ratio to stderr every N steps |
| `RLA_INIT_FROM` | — | stage-1 checkpoint dir (§7) |
| `RLA_COLOR` | `1` | colour the banners; `0` (or `NO_COLOR=1`) for plain text |
| `RLA_PREFLIGHT` | `1` | run the join check inside `rla/train.py` (the launcher does it once up front instead) |
| `RLA_SHUFFLE_TARGETS` | `0` | permute `z` across episodes — the "is it the content of `z`?" control |

They are environment variables and not `--flags` because `openvla_utils.get_action_head` rebuilds the
head from four fixed kwargs and then loads **strict** — training and evaluation have to read one
switch or they disagree about the head's shape.

### λ: keeping the action loss dominant

The action L1 falls by roughly an order of magnitude over a run while the RLA L1 does not, so a
*fixed* weight that starts as a small auxiliary term drifts towards parity. `auto:R` recomputes it
every step as `R · L_act / L_rla`, holding the RLA term at fraction `R` of the action term for the
whole run. `VLA Train/Rla Share` is logged either way, so the actual ratio is visible rather than
assumed — check it after ~200 steps and adjust rather than guessing.

## 3. Architecture: what the two stages share

Both stages run the **same policy sequence**, `[8 action tokens | 128 RLA tokens]` = 136. The only
difference is the loss:

| | stage 1 | stage 2 |
|---|---|---|
| policy sequence | 8 action + 128 RLA | 8 action + 128 RLA |
| `λ_act` | **0** — action tokens present, unsupervised | 1.0 |
| `λ_rla` | 1.0 | `auto:0.25` |
| action labels used | **none** | yes |

So stage 1 is genuinely actionless — no action label enters the loss — while the *architecture* is
unchanged. Weight sharing is therefore total: the state dict is identical between the stages and
loads strict, nothing is spliced.

| component | params | trained in stage 1? |
|---|---|---|
| `layer_norm1`, `fc1` (896×6272) | 5.6M | **yes** — the action tokens feed the RLA tokens through self-attention |
| 24 × block (`q_proj`, `k/v_self`, `k/v_adapter`, `k/v_task`, `o_proj`, `ffn`) | ~145M | yes |
| 24 × `gating_factor` `(136,1)` | 3.3K | rows 8–135 directly, rows 0–7 via the action tokens |
| `layer_norm2_rla`, `fc2_rla` | 59K | yes |
| `layer_norm2`, `fc2` (7×896) | 8K | **no** — action-output-only path |
| 24 × `gating_factor_rla2act` | 24 | **no** — only the action queries use it |
| VLM LoRA, `action_queries`, `proprio_projector` | — | yes (carried by the merged VLM) |

### Attention between the two groups is one-way

Action queries may read the RLA keys, gated per block by a zero-init scalar
`gating_factor_rla2act`, so the network learns how much the predicted latent action should inform
the action chunk. RLA queries **cannot** read the action keys at all (`-inf`).

That asymmetry is what makes the two-stage run sound: stage 2 attaches a fresh action output head
and starts supervising it, which changes what the action tokens carry — and none of that may move
the auxiliary task. With the mask, `z` is a function of (vision, adapter, proprio, other RLA tokens)
only, so stage 2 initialised from stage 1 predicts **bit-identical** `z`. The `isolation` and
`warmstart` gates check exactly this.

One inherited nuance: `tanh(g) = 0` flattens the gated logits to a constant, so a fully-closed gate
makes the RLA keys *uninformative* rather than unattended. That is the same semantics upstream's
zero-init vision gate has at step 0, and it is why the control arm has to be `RLA_STEPS=0` rather
than a closed gate.

Only `fc2` + `layer_norm2` — 8K of ~150M — arrive at stage 2 untrained, which is exactly LAPA's
"re-initialise the action head". (They do get AdamW's decoupled weight decay applied against a zero
gradient in stage 1, shrinking them by ~4% over 20k steps. Harmless on a random init.)

### `RLA_ACTION_TOKENS=0`: pretraining on the RLA tokens alone

Set it and stage 1's sequence becomes the 128 RLA tokens only. It is supported — `rla/warmstart.py`
splices the `(128,1)` gates into stage 2's `(136,1)` — but it is **not** the default, for two
measurable reasons:

1. **RoPE offsets shift.** `x` enters the policy as zeros, so RoPE is the only thing distinguishing
   token positions. Without the action tokens the RLA group sits at positions 0–127 in stage 1 and
   8–135 in stage 2. RLA↔RLA relative offsets survive a uniform shift, but the cross-attention terms
   `q_m · k_task_n` (512 vision tokens) and `q_m · k_adapter_n` (65 action-query + proprio tokens)
   all shift by +8 — and since `x` is zeros, that cross-attention is where essentially *all* the
   information enters. The high-frequency RoPE channels rotate substantially over 8 positions.
2. **`fc1` never trains.** `Linear(6272, 896)`, 5.6M params, the largest single matrix in the head,
   is absent from stage 1's graph entirely and arrives at stage 2 at init.

Neither costs anything to avoid, and keeping the tokens does not weaken the actionless claim: no
action labels are used either way. In stage 1 those 8 positions act as register/summary tokens the
RLA tokens read from; in stage 2 they become the action tokens, handing `fc2` a representation
already shaped by 20k steps of dynamics prediction.

`RLA_ACTION_TOKENS=0` requires `LAMBDA_ACT=0` (enforced in `rla/config.py`): with the action tokens
out of the sequence there is no trunk output to project them from.

### What changes at evaluation

**One thing, and it is already wired.** `rla/eval.py` swaps
`openvla_utils.L1RegressionActionHead` for the RLA class; everything downstream is untouched,
because `predict_action`'s return type is unchanged — it still returns a bare `(1, 8, 7)` tensor, so
`modeling_prismatic.py:871-874`'s `.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)` keeps working and
neither that file nor its `pretrained_models/configs/` copy needs editing.

Concretely, at rollout time:

- the 128 RLA tokens **stay in the sequence** — the attention distributions were trained with them
  present, and dropping them would be a train/test mismatch;
- `z_hat` is still computed and simply discarded. One `Linear(896, 64)` over 128 tokens per step is
  nothing next to the VLM forward, and making the module a pure function of its inputs in eval mode
  is what lets the `isolation` and `warmstart` gates compare runs bit-for-bit;
- no sidecar is needed — `z` is an output, never an input;
- `RLA_STEPS`/`RLA_QUERIES`/`RLA_DIM`/`RLA_ACTION_TOKENS` must match training. They are read from
  the environment by both sides, and `get_action_head` loads **strict**, so a mismatch fails at load
  instead of silently randomising half the head.

Gate `evalhead` proves all of this against the real `get_action_head` — save with DDP prefixes,
strict load, the exact call the rollout makes — and checks that the tripwire actually trips on a
geometry mismatch.

## 4. How the join works, and why it cannot fail quietly

The sidecar is joined to RLDS by `sha1(observation/state)` per episode. Three independent nets:

1. **`preflight` before training.** The launcher runs `rla.verify --gates join` as its own process
   and refuses to launch unless every suite is a bijection with `require_full=True`. Costs seconds
   to fail, ~2 min to pass.
2. **A miss during training raises.** `RlaStore.frame` runs inside `RlaBatchTransform`, which is
   plain python in the training process, so a missing episode is a traceback — never a block of
   zeros. (The slot lookup itself sits inside a `tf.numpy_function`, the one place that must not
   raise, so it returns `-1` and lets `frame` do the failing.)
3. **Gate `align`.** A synthetic sidecar whose latents encode
   `(episode, t, i, gripper(a_{t+i}), N)` is pushed through the *real* dataloader, and the decoded
   gripper trajectory is compared against `batch["actions"]` for the same batch element. That
   comparison is not circular — the two values travel completely different paths — so it catches a
   wrong episode, an off-by-one in `t`, and a reversed chunk axis.

Two properties the design leans on, both checked rather than assumed:

- **`observation/proprio` is bit-identical to `observation/state`** at the hook point, because
  `libero_dataset_transform` splits `state` into `EEF_state`/`gripper_state` and `restructure`
  concatenates exactly those two back. That is why the hook goes on
  `normalize_action_and_proprio` — the last place the raw states exist.
- **The dataset-statistics cache is untouched.** `get_dataset_statistics` hashes
  `inspect.getsource(standardize_fn)`, and the standardize_fn is not modified, so the existing
  `dataset_statistics_<sha>.json` is reused byte-identically and RLA runs are normalisation-
  comparable to the baselines by construction. Gate `stats` confirms it after a run.

## 5. Traps

- **The control arm is `RLA_STEPS=0`, not `LAMBDA_RLA=0`.** The RLA tokens sit in the shared
  self-attention and change the action path even at zero loss weight. Gate `head` asserts the two
  differ, so nobody reaches for the cheaper-looking knob.
- **`use_pro_version=True` is required.** `x` enters the policy as zeros and the non-Pro block has
  no RoPE and no positional embedding, so all 128 RLA tokens would be identical and predict one `z`.
  The head raises rather than training that.
- **`use_minivlm` must stay `True` in both stages.** Flipping it switches `RLDSBatchTransform` to
  the OpenVLA branch (`action_chunk_len=1`, different prompt) — a silently different task.
- **Stage 1 needs `merge_lora_during_training True`.** The warm start loads the merged VLM, because
  PEFT's adapter does not contain `action_queries`.
- **Training noise on the RLA tokens keys off `self.training`, not `cfg.phase`.** These agree in
  every configuration the launcher produces; they would diverge only under `--phase Inference`
  during training, which nothing does.

## 6. Target scaling (`RLA_NORM`)

All four modes are exactly invertible — `RlaStore.denormalize` is the inverse of whichever is in
force, so a predicted `z` can always be fed back through `f_dec`. What differs is what the L1
*weights*, and the `norm` gate pins the resulting loss scale so a change of sidecar cannot move it
out from under a tuned λ:

| mode | `(z − offset) / scale` | measured E&#124;z&#124; |
|---|---|---|
| `global` | two scalars | 0.782 |
| **`center`** (default) | per-slot mean, one global std | **0.389** |
| `perdim` | fully whitened | 0.828 |
| `none` | raw encoder units | 2.80 |

`center` is the default because the per-`(q,d)` offsets are **71.7% of the total variance** on this
sidecar and are constant over the whole corpus: under `global` most of the loss would be memorising
constants rather than predicting motion. It still divides by a single scalar, so the relative scale
of the 1024 slots — which is where the motion signal lives — is preserved.

Consequence for λ: `center` puts the RLA L1 near 0.31–0.39 instead of ~0.79, so a *fixed*
`LAMBDA_RLA` means about half of what it did under `global`. `auto:R` is unaffected, since it tracks
the ratio rather than the absolute scale.

## 7. The warm-start report

Stage 2 prints one block per module as it loads, so "did it actually start from stage 1" is answered
on screen rather than inferred:

```
+-- RLA warm start: stage 1 -> stage 2 -----------------------------------
| from       outputs/RLA-P1--20000_chkpt   step 20000
| vlm        982 tensors / 1252.6M params from 1 shard(s)
| coverage   OK  982/983 of the model's tensors (99.9%), 0 unexpected
| absent     OK  1 tied weight(s), populated via shared storage ['language_model.lm_head.weight']
| read back  OK  24/24 sampled tensors bit-identical to the file after load
| lora       fresh adapter next, lora_B=0 -> identity at step 0
+-------------------------------------------------------------------------
```

then the same for `proprio_projector` and `action_head` (strict loads, so `load` being `OK` already
means every key matched by name and shape). Each row is a real check, not a print:

- **coverage** — raises below 99%. A genuine merged VLA-Adapter save covers 982 of 983 tensors.
- **absent** — the one absent key is `lm_head.weight`, which HF omits because `tie_word_embeddings`
  shares it with `embed_tokens`. Tie-ness is established by comparing `data_ptr()` against the
  tensors that *were* loaded, so this row distinguishes "populated through shared storage" from
  "left at random init" instead of trusting HF's `_tied_weights_keys` (which lives on the inner
  `Qwen2ForCausalLM`, not on the wrapper).
- **read back** — the weights are re-read out of the live module and byte-compared against the file
  in the dtype the parameter holds. `load_state_dict` not complaining is a weaker claim than this;
  a silent cast or a broadcast would pass it. The `warmstart` gate checks this check both ways: it
  passes on a genuine load and raises, naming the tensor, on a tampered one.
- **gates** (head only) — whether `gating_factor` needed splicing, which happens only across an
  `RLA_ACTION_TOKENS` change.

Rank 0 prints; the other ranks perform the identical load and raise on the identical checks.
`RLA_COLOR=0` or `NO_COLOR=1` gives plain text — colour is on by default even when stderr is a pipe,
because the launcher runs everything through `tee` and that is exactly when this needs to be visible.

## 8. Host RAM — the binding constraint on this cluster

`tools/OOM_DEBUGGING.md` records a job SIGKILLed by the SLURM cgroup at **194.1 / 193.8 GiB** with
GPU memory fine. Stage 1 reads all four suites, so the obvious worry is that a 4× bigger corpus is a
4× bigger risk. **It is not** — measured here at 4 ranks, cgroup usage at step 300:

| | `B=100000`, arenas default | `B=80000`, `MALLOC_ARENA_MAX=2` |
|---|---|---|
| 4-suite mixture (stage 1) | 80.4 GiB (19.2/rank) | **65.7 GiB (15.5/rank)** |
| single suite (stage 2) | 73.6 GiB (17.5/rank) | — |

The mixture costs **~1.7 GiB/rank** over a single suite, not 4×, for two structural reasons:
`.repeat()` sits *before* `.shuffle()` (`dataset.py:552, 569`), so the frame buffer fills to `B`
whatever the corpus size; and there is **one** buffer for the whole mixture, applied after
`sample_from_datasets`. The extra is three more concurrent readers — `Threads per Dataset: [1 1 1 1]`
instead of `[1]`.

**What actually drives the risk is steps.** The incident's own numbers give the slope: 73 → 194 GiB
over 52,857 steps ≈ **2.3 MiB/step** summed across 4 ranks. From the 65.7 GiB baseline that puts the
wall near **57k steps**, and reproduces the observed death at 52.8k from the un-tuned baseline. So:

- `pretrain` defaults to **20000** steps → projects to ~110 GiB, 57% of the limit.
- The launcher **warns** above 40000 steps at ≥4 ranks, with the arithmetic and the escape hatches.
  Note this fires on `long`'s own default of 60000 — that default mirrors upstream, and stage 2 is
  meant to run at 5k–10k to match the REF eval points anyway.
- `SHUFFLE_BUFFER` defaults to **80000** and `MALLOC_ARENA_MAX=2` is exported, matching
  `tools/train_eval.sh`. The 80000 is also required for *comparability*: every REF baseline ran at
  80000, and tf.data's shuffle is a sampling knob, so 100000 would not be sampling-comparable.
- Every run writes `train_logs/rla-<stage>--<stamp>-rss.log` (per-rank RSS + cgroup, every 60 s).
  `OOM_DEBUGGING.md` recommends exactly this but its snippet greps `[f]inetune.py`, which never
  matches an RLA run — the entry point is `rla/train.py`. `RSS_WATCH=0` disables it.

Cost of the two memory settings, measured on the same probe: **1.45 it/s vs 1.53 it/s (~5%)**, which
answers one of that document's open questions (it lists the `MALLOC_ARENA_MAX` throughput cost as
unmeasured) — though this probe cannot separate the arena cap from the smaller buffer.

**What the RLA code itself adds** is small and mostly reclaimable:

- the sidecar is **4.2 GB of file-backed mmap** (`np.load(mmap_mode="r")`, all 1693 episodes are
  `.npy`). It is charged to the cgroup as page cache, but shared by every rank on the node and
  **evictable under pressure** rather than OOM-triggering.
- `RlaStore` caches all 1693 handles — cheap for `.npy`. For a **`.npz`** sidecar there is no mmap
  and each entry would be the decompressed block in anonymous memory, i.e. the whole corpus in
  *every* rank, so the cache bound drops to 64 in that case (`rla/sidecar.py`).
- `batch["rla"]` is 32 KB/sample and the slot carried through the tf.data graph is **4 bytes/frame**
  (400 KB at `B=100000`) — the reason the payload is fetched in numpy rather than shuffled (§4).

## 9. Debugging

`.vscode/launch.json` has a config per gate and a single-process smoke run for each stage.
Breakpoints work in `rla/*` **and** in the untouched `prismatic/` + `vla-scripts/` sources.
Useful first stops:

| where | what you see |
|---|---|
| `rla/data.py` `_normalize_with_rla` | the tf.data hook, once per trajectory |
| `rla/data.py` `RlaBatchTransform.__call__` | `slot` and `t` for one frame |
| `rla/sidecar.py` `RlaStore.frame` | where the `(A, Q*D)` target is materialised |
| `rla/action_head.py` `RlaMLPResNet.forward` | token append/split, both output heads |
| `rla/loss.py` `run_forward_pass_rla` | the combined loss and the resolved λ |

The single-process configs set `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_*` by hand: `finetune.py`
calls `dist.barrier()` unconditionally, and accelerate's `PartialState()` only initialises the
process group when it sees those.
