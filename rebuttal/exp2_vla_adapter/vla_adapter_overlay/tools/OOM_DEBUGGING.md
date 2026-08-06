# Host-RAM OOM in `tools/train_eval.sh` — incident notes

Written 2026-07-26 after job `59120608` was killed. Kept next to `train_eval.sh` because
everything here is about running *that* script on Amarel. Language is English to match the
rest of the repo.

---

## TL;DR

`tools/train_eval.sh long` died at step **52,857 / 60,000** (88%, 9h24m in) with
`exitcode: -9 (SIGKILL)` on rank 0. **It was not NCCL and not GPU memory.** The SLURM cgroup
OOM-killer took rank 0 because the four DDP ranks together reached **194.1 GiB against a
193.8 GiB limit**.

The torchrun traceback is misleading: `ChildFailedError` with `<NO_OTHER_FAILURES>` and
SIGTERMs to ranks 1-3 is just torchrun cleaning up after rank 0 vanished. Exit code `-9` is
the whole signal — a process cannot catch SIGKILL, so there is never a Python traceback.

**Rule of thumb: `-9` ⇒ something outside the process killed it ⇒ check the OOM killer first.
`-6`/`-11` or a NCCL timeout message ⇒ then start suspecting NCCL/CUDA.**

---

## Diagnosis playbook

Run these in order. The first one is usually decisive.

```bash
# 1. Did the kernel OOM-killer fire? Match the PID against the one in the torchrun traceback.
dmesg -T | grep -iE 'oom|killed process|out of memory' | tail -30

# 2. Full dump: per-process RSS table + cgroup accounting.
dmesg -T | awk '/python invoked oom-killer/,/Memory cgroup out of memory/' | tail -80

# 3. What did the job actually ask for, and what did it peak at?
sacct -j <jobid> --format=JobID,State,ExitCode,ReqMem,MaxRSS,MaxVMSize,AllocCPUS,Elapsed
sstat -j <jobid>.0 --format=MaxRSS,AveRSS      # live jobs only

# 4. Live cgroup counters (only readable from a node inside the job).
cat /sys/fs/cgroup/memory/slurm_<node>/uid_<uid>/job_<jobid>/memory.{usage_in_bytes,limit_in_bytes,failcnt,max_usage_in_bytes}

# 5. Sweep your other running jobs for the same problem before they die too.
squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %R"
```

### Reading the dump

```
oom-kill:constraint=CONSTRAINT_MEMCG, oom_memcg=/slurm_gpu021/uid_1609111/job_59120608,
         task=python, pid=567337
memory: usage 203161592kB, limit 203161592kB, failcnt 340170
```

- `constraint=CONSTRAINT_MEMCG` ⇒ **your job's cgroup limit**, not the node. The node had
  236 GB free at the time. Do not go looking at `free -g`; it will mislead you.
- `failcnt 340170` ⇒ the job had been scraping the ceiling for a long time before the kernel
  gave up. A high failcnt means slow growth into the wall, not a sudden spike.
- Match `pid=` against the traceback's `(pid: NNNNNN)` to confirm it is the same process.

### The two lines that tell you fragmentation vs. real leak

```
total_inactive_anon 205511114752    (191.4 GiB)   <- allocated but cold
total_active_anon     2039738368    (  1.9 GiB)   <- actual working set
total_rss_huge       45933920256    ( 42.8 GiB)   <- transparent huge pages
```

99% of resident anon memory was **inactive** — allocated and never touched again. That is the
signature of allocator fragmentation (freed blocks the allocator never returned to the OS),
not of a live data structure growing. THP amplifies it: one live 4 KB object pins a whole
2 MB page.

If `active_anon` had been the large number, it would be a genuine leak and `MALLOC_ARENA_MAX`
would not have helped.

---

## What the numbers were

Per-rank RSS at kill time (dump reports pages; ×4096 for bytes):

| rank | pid | pages | RSS |
|---|---|---|---|
| 0 | 567337 | 12,777,269 | 48.74 GiB ← killed |
| 1 | 567338 | 12,714,625 | 48.50 GiB |
| 2 | 567339 | 12,740,307 | 48.60 GiB |
| 3 | 567340 | 12,651,067 | 48.26 GiB |
| | | | **194.1 GiB** vs **193.8 GiB limit** |

Overshoot was **0.2%**. Note there are no dataloader worker processes in the table —
`num_workers=0` (`vla-scripts/finetune.py:993`), because RLDS runs its own parallelism. The
whole tf.data pipeline lives **inside each rank's process**, and every rank keeps its **own
full copy** of the shuffle buffer.

`SLURM_MEM_PER_NODE=200000` (200000 MiB requested; the cgroup limit lands at ~198,400 MiB
after SLURM's own reservation). Node has 251 GB total, so there is little room to just ask
for more.

---

## Dataset facts (the per-suite numbers)

From `data/libero/<dataset>/1.0.0/dataset_statistics_*.json` and `dataset_info.json`:

| suite | dataset dir | transitions | trajectories | frames/episode | on disk | bytes/transition | script default steps |
|---|---|---|---|---|---|---|---|
| spatial | `libero_spatial_no_noops` | 52,970 | 432 | 122.6 | 1.8 G | ~36.5 KB | 20,000 |
| object | `libero_object_no_noops` | 66,984 | 454 | 147.5 | 2.7 G | ~43.3 KB | 20,000 |
| goal | `libero_goal_no_noops` | 52,042 | 428 | 121.6 | 1.8 G | ~37.1 KB | 50,000 |
| long | `libero_10_no_noops` | **101,469** | 379 | **267.7** | 3.5 G | ~36.0 KB | **60,000** |

Images: `256×256×3` uint8, **JPEG-encoded**, two per frame (`image` + `wrist_image`), per
`features.json`. `libero_10` is ~2× the frames of the others but has the *fewest* episodes —
its trajectories are much longer, which matters for eval wall-clock too.

To re-derive:

```bash
for d in libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops libero_10_no_noops; do
  f=$(find data/libero/$d -name "dataset_statistics_*.json" | head -1)
  python3 -c "import json;d=json.load(open('$f'));print('$d',d['num_transitions'],d['num_trajectories'])"
done
```

---

## The shuffle buffer: how it actually works

`shuffle_buffer_size` defaults to `100_000` (`vla-scripts/finetune.py:78`) and is **per rank**.

### It is a sampling knob, not a speed knob

`tf.data.Dataset.shuffle(B)` is **not** a global permutation:

```
1. fill buffer with the first B elements of the stream
2. per output: pick a uniform random index i in [0,B), emit buffer[i],
   refill that slot from the stream
```

Consequences:

- Input element at stream position `p` **cannot** be emitted before output position `p - B`.
  Two elements more than ~`B` apart in the stream essentially never share a batch.
- Residence time is geometric with mean `B`. **`B` is the width of the reorder window**;
  outside it, order is untouched.

Contrast with `torch.utils.data.DataLoader(shuffle=True)`, which uses `RandomSampler` over
`__getitem__` and gives a true global permutation — no buffer, no knob. TFRecords are a
*sequential stream with no random access*, so shuffling must be approximate and the buffer
size **is** the shuffle quality. This is why the usual "buffers only affect speed" intuition
(true for `prefetch`, `with_ram_budget`) is wrong here.

### Why it matters especially in this pipeline

`dlimp/dataset.py:151` sets `interleave_cycle_length = num_parallel_reads`, which traces to
`traj_read_threads=len(mixture_spec)` in `prismatic/vla/datasets/datasets.py:198` — **1** for
a single-dataset mix. So one shard is read at a time, episodes come out in order, and
`.flatten()` feeds the buffer *consecutive timesteps of one trajectory*. The frame-level
shuffle is the **only** thing decorrelating time-adjacent frames.

### Pipeline order (verify before reasoning about memory)

`prismatic/vla/datasets/rlds/dataset.py`:

```
make_dataset_from_rlds        -> trajectories, images still JPEG (SkipDecoding, dlimp/dataset.py:147)
.repeat()                     -> line 552   *** BEFORE the shuffle ***
apply_trajectory_transforms   -> action chunking, goal relabeling
.flatten()                    -> frames
sample_from_datasets
.shuffle(shuffle_buffer_size) -> line 569   <- buffer holds ENCODED bytes, ~36 KB/frame
apply_frame_transforms        -> line 407: decode_and_resize + augment   <- decode happens HERE
.batch(batch_size)
.with_ram_budget(1)
```

Two things fall out:

1. The buffer holds **JPEG bytes, not pixels** — roughly `B × 36 KB` ≈ **3.6 GB/rank** at
   `B=100_000`, not the ~39 GB it would be if decoded.
2. **`.repeat()` precedes the shuffle**, so the stream is infinite and the buffer **always
   fills to `B` regardless of dataset size**.

> ⚠️ **Correction worth recording.** During this investigation I first claimed the smaller
> suites only fill the buffer to 52-67% and that this was why `long` specifically died. That
> is **wrong** — because of `.repeat()`, all four suites fill to the full 100,000 frames, and
> their buffer memory is nearly identical (~3.5-4 GB/rank; bytes/transition barely varies).
> If you reason about tf.data memory, always locate `.repeat()` relative to `.shuffle()` first.

### So why did `long` die and not the others?

**Steps, not dataset size.** Same 4-rank config across the board:

- `goal` ran **50,000** steps and finished cleanly.
- `long` died at **52,857**.
- `spatial` / `object` only run 20,000 — nowhere near the cliff.

The growth is roughly step-driven and suite-independent; `goal` simply stopped just before the
wall. Any run past ~50k steps at 4 ranks is at risk.

### Coverage at different buffer sizes

Reorder window in epochs = `B / transitions`. Once it exceeds 1.0, sampling is effectively
uniform over the whole dataset and further increases buy nothing:

| suite | B=100,000 | B=80,000 | B=50,000 |
|---|---|---|---|
| spatial | 1.89 ep | 1.51 ep | 0.94 ep |
| object | 1.49 ep | 1.19 ep | 0.75 ep |
| goal | 1.92 ep | 1.54 ep | 0.96 ep |
| long | 0.99 ep | **0.79 ep** | 0.49 ep |

For `long` at `B=80,000`: ~299 episodes resident, all 10 tasks represented in every batch;
P(two of 8 samples share an episode) rises from ~7.5% to ~9.4%. Episode *order* is
independently shuffled anyway (`shuffle_files=True`, `dlimp/dataset.py:146`), so tasks are
mixed before reaching the buffer.

---

## Fixes applied

| # | Change | Where | Changes training results? |
|---|---|---|---|
| 1 | `SHUFFLE_BUFFER` env var, default **80000** (upstream: 100000) | `tools/train_eval.sh` | Yes, marginally — see coverage table |
| 2 | LoRA merge restricted to rank 0 + `del` + `gc.collect()` | `vla-scripts/finetune.py:588` | **No** |
| 3 | `export MALLOC_ARENA_MAX=2` | `tools/train_eval.sh` | **No** |
| 4 | `SAVE_FREQ` default 5000 → 10000 | `tools/train_eval.sh` | **No** (fewer merges) |

### Fix 2 — the merge was running on every rank

`save_training_checkpoint()` guarded only `merged_vla.save_pretrained(...)` with
`is_main_process`. Everything above it — `AutoModelForVision2Seq.from_config/from_pretrained`,
`PeftModel.from_pretrained`, `merge_and_unload()` — ran on **all four ranks**. Three quarters
of that was pure waste, repeated at every checkpoint.

Measured from a merged checkpoint: **982 tensors, 1.253 B params**, i.e. **~2.4 GB** per bf16
CPU copy, and the merge holds the base plus the PEFT-wrapped model at once. So each checkpoint
was allocating and freeing roughly 4 × 2.4 GB of CPU model that only rank 0 ever used.

Sanity-check a merged checkpoint with:

```python
from safetensors import safe_open
with safe_open("outputs/<ckpt>/model.safetensors", framework="pt") as f:
    keys = list(f.keys())
    print(len(keys), any("lora_" in k.lower() for k in keys))   # -> N, False
```

`False` on the second value is the important one: any surviving `lora_*` key means
`merge_and_unload()` did not actually fold the adapter in.

**`dist.barrier()` must stay outside the `is_main_process` guard** (it is at
`finetune.py:613`, 8-space indent, inside the `use_lora` guard). All ranks evaluate
`cfg.use_lora and cfg.merge_lora_during_training` identically since `cfg` comes from the same
argv, so no rank can skip the barrier. Getting this wrong deadlocks all ranks at the first
checkpoint — ~1.8h into a `long` run.

### Fix 3 — `MALLOC_ARENA_MAX=2`

glibc `malloc` keeps multiple independent heaps ("arenas") to cut lock contention: when a
thread finds the main arena locked, it spawns/uses another rather than blocking. The 64-bit
default cap is `8 × ncores` — **240 arenas** on a 30-core node. Freed blocks in an arena are
effectively never returned to the OS (a non-main heap is only released when the entire 64 MB
region is free, which fragmentation prevents), so a thread-heavy process — tf.data pipeline,
`num_parallel_calls=16` frame transforms, TF thread pools, torch, NCCL — accumulates arenas
full of unreusable holes and RSS only climbs.

Capping at 2 forces reuse within two arenas. Cost is more malloc lock contention;
**throughput impact here is unmeasured**.

---

## Bug found while smoke-testing: `ACCUM > 1` saves every checkpoint twice

`vla-scripts/finetune.py` training loop:

```python
gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
...
if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
    save_training_checkpoint(...)
```

The guard tests `gradient_step_idx`, but the loop iterates over `batch_idx`. With
`ACCUM=2`, `batch_idx=10` and `batch_idx=11` both map to `gradient_step_idx=5`, so the
condition fires **twice for the same step** — two full LoRA merges, two `save_pretrained`
calls, one overwriting the other.

Reproduced directly (`ACCUM=2, SAVE_FREQ=5`):

```
Saving Model Checkpoint for Step 5
Saved merged model for Step 5 at: outputs/REF-Long-smoke--5_chkpt
Saving Model Checkpoint for Step 5        <- same step, again
```

**This did not affect the run that OOMed** — it used `ACCUM=1` (10 `Saved merged model`
lines for 10 checkpoint dirs), where `gradient_step_idx == batch_idx` and the bug is dormant.
It matters now because `tools/train_eval.sh` currently defaults `ACCUM=2`, so the next run
would merge twice per checkpoint.

### Fixed by deduplicating on `log_step`, *not* by gating on the closing micro-batch

The obvious one-liner is **wrong**:

```python
# DO NOT USE -- silently drops the final checkpoint when ACCUM > 1
if gradient_step_idx > 0 and log_step % cfg.save_freq == 0 \
        and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
```

`if log_step == cfg.max_steps: break` sits at the *end* of the loop body, so it fires on the
**first** micro-batch of step `max_steps`. The closing micro-batch of that step never runs,
so the gate is never satisfied there and the final checkpoint is never written.

What is actually in the code (`finetune.py:1108`, with `last_saved_step = -1` initialized
before the loop):

```python
if gradient_step_idx > 0 and log_step % cfg.save_freq == 0 and log_step != last_saved_step:
    last_saved_step = log_step
    save_training_checkpoint(...)
```

Verified by simulating the loop for `max_steps=60000, save_freq=10000`:

| variant | ACCUM=1 | ACCUM=2 |
|---|---|---|
| original | 6 ckpts, no dup | 6 ckpts, **10000-50000 each saved twice** |
| closing-micro-batch gate | 6 ckpts, no dup | 5 ckpts, **60000 lost** |
| `last_saved_step` (shipped) | 6 ckpts, no dup | 6 ckpts, no dup |

Behaviour is **bit-identical at ACCUM=1**, which is what every completed reference run used
(`goal`, `spatial`, `object`, and the `long` run that OOMed).

Also confirmed on real 2-GPU runs (`SHUFFLE_BUFFER=2000` to keep startup short):

| run | code | config | `Saving Model Checkpoint` lines | final ckpt |
|---|---|---|---|---|
| `RUN_TAG=smoke` | before fix | `ACCUM=2 STEPS=12 SAVE_FREQ=5` | **4** (steps 5 and 10 twice each) | n/a (12 not a save step) |
| `RUN_TAG=smoke2` | after fix | `ACCUM=2 STEPS=10 SAVE_FREQ=5` | **2** (once each) | `--10_chkpt` present ✅ |

`smoke2` deliberately puts `max_steps` **on** a save boundary, which is the exact case the
rejected one-liner would have broken.

## Operational gotchas

**Always pass `RUN_TAG` when re-running a suite.** `RUN_ID="REF-${Suite}"` with no tag, so a
re-run overwrites `outputs/REF-<Suite>--<step>_chkpt` in place. Worse: the eval sweep globs
`outputs/${RUN_ID}--*_chkpt`, so if the old run used `SAVE_FREQ=5000` and the new one uses
`10000`, the odd-numbered old checkpoints (5000, 15000, 25000, …) survive as orphans and get
swept and written into `RESULTS` **as if they belonged to the new run**. Silent cross-run
contamination.

```bash
RUN_TAG=v2 tools/train_eval.sh long     # -> REF-Long-v2--*_chkpt, no collision
```

**Time budget for `long`:** ~10.7h training (60k steps @ ~1.55 it/s on 4× A100-PCIE-40GB) + ~4-5h eval
(500 episodes final + sweeps; libero_10 rollouts are longer than the other suites).
Budget ~16h of remaining walltime.

**`vla-scripts/finetune.py --help` is broken** — a stray `%` in a field comment makes argparse
raise `ValueError: unsupported format character 't'` while formatting help. Pre-existing
upstream bug, unrelated to any of this. To check a flag parses, bypass `--help`:

```python
import importlib.util, draccus
spec = importlib.util.spec_from_file_location("ft", "vla-scripts/finetune.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(draccus.parse(config_class=m.FinetuneConfig, args=["--shuffle_buffer_size", "80000"]))
```

---

## Not verified / open questions

- **The growth slope was never measured.** Baseline RSS at step 0 is unknown, so how much
  headroom the fixes actually buy is an estimate. With ~2.9 GB saved from the buffer alone,
  whether that covers the remaining 7,143 steps depends on the baseline: it works out if the
  step-0 baseline was ≥180 GiB and fails if it was ≤170 GiB. **Sample per-rank RSS during the
  next run** rather than trusting the estimate.
- **`MALLOC_ARENA_MAX=2` throughput cost** — unmeasured.
- **Resume path looks fragile**: `cfg.resum_vla_path` (note the typo, `finetune.py:72`)
  defaults to `openvla/openvla-7b`, and `finetune.py:276` uses it for `load_checkpoint`.
  Untested; do not assume `--resume True` works without checking.
- Whether the remaining growth after fixes 2-4 is fully explained by allocator fragmentation,
  or whether something in the tf.data pipeline also genuinely accumulates.

## Suggested monitoring for the next long run

```bash
# per-rank RSS every 60s alongside the training log
while pgrep -f finetune.py >/dev/null; do
  echo "$(date +%s) $(ps -eo pid,rss,cmd | grep '[f]inetune.py' | awk '{s+=$2; printf "%s:%.1fGB ", $1, $2/1048576} END {printf "total:%.1fGB", s/1048576}')"
  sleep 60
done | tee train_logs/rss.log
```

An hour of that extrapolates to whether the run clears 60k, instead of finding out at hour 10.
