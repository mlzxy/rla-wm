# Sample output — `extract_rla_sidecar.py`

28 filmstrips from a verified run, spread over **all four suites** and over early / middle / late
anchors within each episode. Kept so you can see what "working" looks like without running anything.
The `.jpg`s and `manifest.json` ship with the release (12.6 MB) — during development they were
gitignored and only this file was tracked. Regenerate any time:

```bash
export PYTHONPATH=.:./third_party/diffusion_policy
IMPL=rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py

# extract a small sidecar over all four suites, 4 GPUs, ~1 min
# (--out is explicit only because this is a throwaway; a real run omits it and lands in
#  data/libero_rla/<Q>x<D>-a<chunk>-step<step>, which is why a --chunk 16 sidecar can never
#  overwrite a --chunk 8 one)
.venv/bin/python $IMPL --rla-run runs/16x64_libero/20260725_00-39-52 \
    --out data/libero_rla/sample --gpus 0,1,2,3 --suites all --limit 8 --viz 4

# then render more anchors from it, no re-extraction
.venv/bin/python $IMPL --viz-only data/libero_rla/sample \
    --rla-run runs/16x64_libero/20260725_00-39-52 --suites all --viz 5
cp data/libero_rla/sample/viz/*.jpg rebuttal/exp2_vla_adapter/sidecar/samples/
```

`f_enc` = `encoder_step0032500.pt`, still mid-training. See `manifest.json` for the sha256.

---

## Reading a filmstrip

```
row 1 (GT)     [  t  ][ t+1 ][ t+2 ][ t+3 ][ t+4 ][ t+5 ][ t+6 ][ t+7 ][ t+8 ]
row 2 (RECON)  [ ---- ][ z_0 ][ z_1 ][ z_2 ][ z_3 ][ z_4 ][ z_5 ][ z_6 ][ z_7 ]
```

Every cell is **agentview | wrist** side by side. Row 2 is `image_decoder(f_dec(x_t, z_i))` with `z`
read back from the `.npy` that was written — you are looking at the artifact, not a fresh forward
pass. Row 2's first cell is intentionally black: `z_0` already predicts `t+1`, so nothing decodes
to `t` itself.

| | good | bad |
|---|---|---|
| tracking | row 2 follows row 1 one step at a time | row 2 frozen, or shifted the same amount everywhere |
| sharpness | objects and gripper legible | mush → undertrained `f_enc`/`f_dec`, or wrong checkpoint |
| handedness | row 2 matches row 1 left-right | mirrored → intrinsics `fx` sign / camera order |
| cameras | left half agentview, right half wrist, in **both** rows | swapped → `CAMERA_ORDER` |

## The alignment line

Each filmstrip also answers the same question numerically: for each `i`, which real frame is
`recon(z_i)` actually closest to? It should be `t+i+1`.

```
ep0 t=22: chunk alignment 8/8, offsets [0,0,0,0,0,0,0,0],
          recon err 0.017 vs per-step motion 0.023 -> exact
ep2 t=172: chunk alignment 0/8, offsets [-2,-3,-4,-5,-6,-7,-8,-10],
          recon err 0.034 vs per-step motion 0.009
          -> unresolvable: recon error >= per-step motion, argmin is noise
```

Three verdicts, and only one of them is a bug:

- **`exact`** — every `z_i` decodes closest to its own target.
- **`unresolvable`** — the reconstruction error exceeds how much the scene changes per step, so the
  argmin is picking between frames that are more similar to each other than the reconstruction is
  to any of them. Says nothing either way. Common late in an episode, after the task is done and
  the arm has stopped.
- **`CHUNK AXIS OFF BY k`** — *this* is the failure. An indexing bug shifts **every** step by the
  same amount, so it only fires when all eight offsets are identical and non-zero. Ramps like
  `[-2,-3,-4,…]` are `f_dec` hedging toward the anchor under motion it cannot fully predict, which
  is a property of the checkpoint, not of the indexing.

Across the 20 anchors in this set: **10 exact, 9 unresolvable, 1 drift, 0 chunk-axis shifts.**
The exact ones are the anchors where the arm is actually moving (`per-step motion` > `recon err`) —
i.e. the test resolves exactly when it has the signal to, and never reports a constant shift.

## Console output from the extraction run

```
[parent] rank 0 -> GPU 0 ... rank 3 -> GPU 3
[rank 0] [libero_spatial] episodes [0, 2) of 8 -> data/libero_rla/sample/libero_spatial
[rank 0] [libero_spatial] selfcheck vs inference_batch: structural 0.00e+00 (exact),
                          end-to-end 1.65e-02 = 0.10% of |z|max 16.6  OK
[rank 0] [libero_spatial] ep0 t=22: chunk alignment 8/8, ... -> exact
[rank 0] [libero_spatial] 1/2  N=110  z=(110, 8, 16, 64)  0.25 ep/s
...
computing normalisation statistics
  stats over 32 episodes, 43528 rows
done -> data/libero_rla/sample  (1.0 min)
32 episode files, 0.08 GiB
```

- **`structural 0.00e+00`** — the hard gate, on every rank and every suite. This file's encode call,
  fed `inference_batch`'s own DINO tokens and Plücker rays, reproduces the trainer exactly. Non-zero
  aborts the run.
- **`end-to-end 4e-03 … 4e-02` (0.03–0.26%)** — fp16 batch-shape noise in `f_enc`, not an error;
  smaller than one fp16 ulp at `|z| ~ 17`. Fails only above 2%.
- **`0.14–0.32 ep/s` per GPU**, lower on `libero_10` (episodes are 2–3x longer). ~1.1 ep/s on four
  GPUs → roughly 26 min for all four suites in full (~1700 episodes).

## Verified invariants

```
episode indices contiguous per suite         no rank gaps or overlap in the sharding
stats.json covers all 43528 rows             = 5441 frames x 8 chunk steps
z dtype float16, shape (N, 8, 16, 64)
per-(q,d) mean in [-14.9, +11.8], std in [0.88, 4.68], global rms 3.86
dead slots (std < std_floor 1e-3): 0 / 1024  the autoencoder uses its whole 16x64 budget
normalised targets: mean -0.009, std 1.023
```

The last line comes from running `prismatic/vla/rla_targets.py`'s exact lookup logic, unmodified,
against this sidecar — including `mmap_mode="r"`. It works as is, which is why `--compress` is off
by default.
