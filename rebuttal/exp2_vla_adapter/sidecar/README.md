# RLA → VLA-Adapter — session bundle

> **Read this first.** Everything below is the bundle exactly as it was written on 2026-07-25,
> against the development repo at commit `4e05436`. It is kept as the record of how the sidecar
> integration was built, **not** as a step you run — the patches will not apply to this repo. The
> runnable path is the two commands in [`../README.md`](../README.md) §6: extract a sidecar with
> `extract_rla_sidecar.py`, read it with `read_rla_sidecar.py`. The folder was called
> `rla-vla-adapter-patch/` then and is `rebuttal/exp2_vla_adapter/sidecar/` now; the paths in the
> commands below are the old ones and have been left alone.

Self-contained snapshot of everything one session (2026-07-25) changed, packaged so that **outside
this folder the working tree is identical to `HEAD`**. Nothing here is applied; `git status` in the
repo shows only this directory.

- **Base commit**: `4e05436` — *Vendor VLA-Adapter source in-tree instead of cloning + patching it*
- **Branch**: cut on `vla-adapter-libero`, now checked out as `rla-integration` — same commit, so
  the patches do not care which
- **Scope**: 19 files, `+2573 / −162`. 6 new files, 13 edited.
- **Status**: all seven verification gates pass with a randomly-initialised `f_enc`; a 12-step
  training smoke ran the RLA arm end to end. No RLA autoencoder has been trained yet, so no
  real targets and no policy numbers exist.

---

## Apply

```bash
cd <repo root>
rla-vla-adapter-patch/apply.sh --check     # dry run, touches nothing
rla-vla-adapter-patch/apply.sh             # apply patches/00-all.patch
```

Each patch is tried with a plain `git apply` first — that leaves the git index untouched — and only
falls back to `--3way` if the surrounding context has drifted (`--3way` does stage what it merges,
and the output says so).

If the tree has diverged and `00-all.patch` conflicts, apply the pieces instead — each is
independent except for the two dependencies noted in the table:

```bash
rla-vla-adapter-patch/apply.sh --split           # 01 … 07 in order, stops at the first failure
rla-vla-adapter-patch/apply.sh --split --only 03 # just one
```

The six **new** files can never conflict, so if a patch fails you can always fall back to copying
them straight out of `new-files/` (the tree there mirrors repo paths):

```bash
rsync -a rla-vla-adapter-patch/new-files/ ./
```

`apply.sh` refuses to run unless it is invoked from the repo root, and warns (does not stop) if
`HEAD` is not `4e05436`.

---

## What is in each patch

| # | Patch | Files | Depends on |
|---|---|---|---|
| 01 | `01-datalib-libero-camera.patch` | **new** `datalib/libero_camera.py`; `scripts/convert_libero_to_trajectory.py` re-exports from it | — |
| 02 | `02-extractor.patch` | **new** `scripts/extract_rla_latents.py` | 01 |
| 03 | `03-vla-adapter-data-path.patch` | **new** `prismatic/vla/rla_targets.py`; `constants.py`, `rlds/oxe/transforms.py`, `rlds/dataset.py`, `rlds/traj_transforms.py`, `datasets/datasets.py`, `util/data_utils.py` | — |
| 04 | `04-action-head-and-loss.patch` | `prismatic/models/action_heads.py`, `vla-scripts/finetune.py` | 03 (imports `RLA_STEPS`) |
| 05 | `05-launchers.patch` | `scripts/vla_adapter_env.sh`, `scripts/train_libero.sh`, `scripts/run_libero_eval.sh` | — |
| 06 | `06-verifier.patch` | **new** `scripts/verify_rla_targets.py` (the seven gates) | 01–05 |
| 07 | `07-docs.patch` | **new** `docs/rla-vla-adapter.md`, `docs/rla-vla-adapter-runbook.md`; cross-ref in `docs/vla-adapter.md` | — |

`00-all.patch` is exactly 01–07 concatenated (verified: same 3070 diff lines, and applying either
way produces a byte-identical tree).

Everything in 03/04 is inert when `RLA_STEPS=0`, and gate 4 proves the `RLA_STEPS=0` action head is
byte-identical to upstream — released `LIBERO-*-Pro` checkpoints still load strict.

---

## The v2 design note

`rla-vla-adapter-integration.md` in this folder is the original design note, **moved** here from
`third_party/VLA-Adapter/`. It was never tracked by git, so it is not in any patch — it travels as
a plain file. A "superseded by" header was added to it this session listing the six things it gets
wrong about the vendored tree. Its motivation (§1), ablation matrix (§6) and rejected alternatives
(§8, §9) all still stand; everything implementation-facing was rewritten into `docs/`.

Move it back with `mv rla-vla-adapter-patch/rla-vla-adapter-integration.md third_party/VLA-Adapter/`
if you want it at its old path (the header's relative link assumes it stays here).

---

## What is deliberately *not* here

| | Why |
|---|---|
| `AGENTS.md` | gitignored and never tracked, so it cannot be a patch and never syncs. One paragraph was corrected — see "AGENTS.md delta" below. Still applied on this machine, because reverting it would restore a note that is now wrong. |
| `data/libero_rla/smoke-random/` (804 MB) | throwaway sidecar built with `--random-encoder` to exercise the gates. Regenerate in ~45 min, or delete. |
| Any commit | nothing was committed; `HEAD` is untouched. |

### AGENTS.md delta

The "Install/update dependencies" bullet said VLA-Adapter must never be edited in place. That
stopped being true at `4e05436`. Current text:

> `third_party/VLA-Adapter` is **vendored committed source** (since `4e05436`) — edit it like any
> other code in the tree and mark the edit `# v4world:`; `scripts/patches/vla-adapter/` is kept only
> as a record of the delta from upstream, do not add to it. `third_party/LIBERO` is still a
> gitignored clone patched by `scripts/patches/libero/` at setup time — that one you must not edit
> in place. See `docs/vla-adapter.md`.

---

## Where the real documentation is

Both docs are inside this bundle under `new-files/docs/`, and land in `docs/` when patch 07 is
applied:

- **`docs/rla-vla-adapter.md`** — findings: the six corrections to the v2 design note, measured
  frame/action alignment, why targets are anchored rather than consecutive, the content-hash join,
  the strict-load trap, the ablation matrix.
- **`docs/rla-vla-adapter-runbook.md`** — what changed file by file, and the exact commands:
  extract → verify → train an arm → eval, with every env var spelled out.

Read the runbook's §0 cheat sheet first; it is four commands.

---

## Verified, with numbers

All measured on this tree, RTX A6000:

| Gate | Result |
|---|---|
| 1 frame/action alignment | N steps ↔ N frames (both views) ↔ N actions ↔ N states; gripper-lag correlation peaks at lag 0 |
| 2 join key | `sha1(state)` bijective over all 432 spatial episodes; `stats.json` covers all 423,760 rows (52,970 frames × 8) |
| 3 end-to-end index alignment | synthetic sidecar decodes to the right `(episode, t, i)` through the real `RLDSDataset` + collator, 0 misses |
| 4 baseline byte-equivalence | 560-tensor key set identical to `4e05436`; released `LIBERO-Spatial-Pro` head loads strict; `predict_action` bit-identical |
| 5 RLA forward/backward | `action (2,8,7)`, `z_hat (2,8,1024)`, `\|grad fc2_rla\| = 47.114`; action output differs from `RLA_STEPS=0` by 1.1964 even at λ=0 |
| 6 extractor ↔ trainer equality | `max \|d\| = 0.00e+00` |
| 7 statistics cache | recomputed statistics match the released `LIBERO-Spatial-Pro` reference (mean/std/min/max exactly 0.0, q01/q99 ≤ 2.17e-7) |

Training smoke: 12 steps, 0 sidecar misses, `Rla Loss = 102.5`.
Extraction: 0.20 ep/s ≈ 25 frames/s → **~3 h single-GPU for four suites**; 804 MB for
`libero_spatial`.

---

## Two open decisions, before you spend GPU hours

1. **`--pairs anchored` vs `--pairs consecutive`** is a real research fork, not an implementation
   detail. Anchored is the default, argued from the `f_dec(x_t, z)` inference constraint and the
   residual-magnitude table in `rla-vla-adapter.md` §4. Consecutive is 8× cheaper to store and is
   what the original v2 note described. Both come off the same DINO cache, so running both is cheap
   — worth doing if the headline number is close.
2. **`lambda_rla=1.0` is an untested guess.** Targets are per-`(q, d)` normalised so it is on the
   same scale as the action L1, but nothing has calibrated it. Expect to sweep.

---

## Bundle contents

```
rla-vla-adapter-patch/
  README.md                          this file
  MANIFEST.md                        sha256 of every file here, plus the tree state it was cut from
  apply.sh                           apply / dry-run helper
  patches/                           00-all.patch + 01..07
  new-files/                         verbatim copies of the 6 new files, mirroring repo paths
  rla-vla-adapter-integration.md     the v2 design note, moved out of third_party/VLA-Adapter/
```
