#!/usr/bin/env python
"""
Precompute RLA latent actions for every frame, so BC-RLA training never runs DINO or the RLA
encoder itself.

    PYTHONPATH=.:./third_party/diffusion_policy .venv/bin/python \\
      rebuttal/exp4_so101/precompute_rla_latents.py \\
        --rla runs/so101/16x16_so101_dual/<TIMESTAMP> \\
        --tag rla16x16

Why
---
Training used to call `_compute_gt_latent_action` inside the loop, which meant:

  * DINOv3-L ran on 4 images per sample, every step, forever -- by far the most expensive part of
    BC-RLA training;
  * the latent target was computed from the *same* image tensor the policy sees, so any photometric
    augmentation applied to the policy input also corrupted the target. The RLA was trained on
    clean frames, so an augmented target is simply wrong.

Precomputing fixes both. The target is derived once from clean, un-augmented frames, and the
training loop becomes a plain regression against a stored array -- which is what lets
`SO101SequenceDataset` apply colour/lighting augmentation freely.

What is written
---------------
For every frame t of every trajectory, the latent action at each requested gap:

    latents[t, j] = encoder( DINO(frame min(t+gaps[j], T-1)) - DINO(frame t) )

`gaps` defaults to the dense range 1..max_gap (32 = the policy `horizon`, the action chunk it
predicts). One file then serves every training mode:

    latent_mode "final"   ->  gap 32                     the transition over the whole chunk
    latent_mode "ladder"  ->  gaps [4, 8, ..., 32]       the chunk sampled evenly (8 rungs)

If you already know the ladder you want, `--gaps 4,8,12,16,20,24,28,32` computes only those and is
4x faster and 4x smaller than the dense default. The manifest records which gaps are present and
the dataset selects them by VALUE, so a sparse sidecar is used exactly like a dense one.

Layout (sidecars, deliberately OUTSIDE the traj_* directories so datalib's key scanner never sees
them and so several RLA versions can coexist):

    <dataset>/rla_latents/<tag>/manifest.json
    <dataset>/rla_latents/<tag>/traj_000000.npy      (T, max_gap, num_tokens, token_dim) float16
    ...

Values are the RAW encoder output. The /10 scaling that `compute_loss` applies is a *policy*
convention and is deliberately not baked in here.

Versions
--------
`--tag` is an arbitrary directory name; `--step` is a checkpoint *token*, not necessarily a number:

    --step 40000              encoder_step0040000.pt          (milestone)
    --step 0040000            the same file -- padding is not part of a step's identity
    --step 40000.snapshot     encoder_step0040000.snapshot.pt (the resumable snapshot beside it)

Whatever it resolves to is written to `manifest["rla_step"]` as the canonical padded token, and the
policy configs pin that exact string. So a custom `--tag` is safe: identity lives in the manifest,
never in the directory name.

Cost: one DINO pass over every frame (~83k images) plus T x len(gaps) encoder forwards, which
dominates. Measured ~26 s per trajectory per 16 gaps on an RTX 6000 Ada, so the dense 32-gap
default is ~52 s/trajectory (~100 min for 120 on one GPU) and the 8-gap ladder is ~13 s (~26 min).
Shard across GPUs to cut that down:

    for i in 0 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$i .venv/bin/python rebuttal/exp4_so101/precompute_rla_latents.py \
        --rla <RUN> --tag <TAG> --num-shards 5 --shard $i &
    done; wait
    # then once, to write the manifest over the complete set:
    .venv/bin/python rebuttal/exp4_so101/precompute_rla_latents.py --rla <RUN> --tag <TAG>
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Repo root is three levels up: <repo>/rebuttal/exp4_so101/<this file>.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

from datalib.dataset import ManiSkillTrajectoryDataset  # noqa: E402
from rebuttal.exp4_so101.ckpt_tokens import ckpt_token, find_ckpt, parse_step_token, same_step  # noqa: E402
from rebuttal.exp4_so101.policy.rla_unified_mv import build_encoder_from_work_dir  # noqa: E402
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels  # noqa: E402

DEFAULT_DATASET = "data/realworld/so101_converted"
CAMERAS = ["front_camera", "wrist_camera"]


def resolve_encoder_step(rla_run: str, step=None) -> tuple[str, str]:
    """Pick which RLA encoder checkpoint to use and return `(token, filename)`.

    `step` is a token, not necessarily a number: `40000`, `"0040000"` and `"40000.snapshot"` all
    name a real file (see `rebuttal/exp4_so101/ckpt_tokens.py`). The return value is the CANONICAL token
    the file itself carries, even when the caller asked for "latest", so the sidecar records exactly
    which weights produced it and the policy configs can pin that string verbatim.
    """
    path = find_ckpt(rla_run, "encoder", step)
    return ckpt_token(path, "encoder"), os.path.basename(path)


def default_tag(rla_run: str, token: str) -> str:
    """`<run-dir-name>__step0040000`, or `..._step0040000-snapshot` for a snapshot checkpoint.

    The suffix is folded into the directory name so a snapshot and the milestone at the same step
    cannot land in the same sidecar -- they are different weights.
    """
    number, suffix = parse_step_token(token)
    base = os.path.basename(os.path.normpath(rla_run))
    name = f"{base}__step{number:07d}" if number >= 0 else f"{base}__step{token}"
    if suffix:
        name += "-" + suffix.strip(".").replace(".", "-")
    return name


@torch.no_grad()
def dino_tokens_for_trajectory(dino, traj_ds, traj_id, cameras, img_size, device, batch=32):
    """All frames of one trajectory -> (T, num_views * Lp, C) fp16 on `device`, plus (pH, pW).

    Per-view blocks are contiguous, matching what MultiViewTokenTransformer slices on and what
    `VLABCPolicyRLAUnified._extract_dino_tokens` produces.
    """
    rgb_keys = [f"{cam}_rgb" for cam in cameras]
    traj = traj_ds.read_trajectory(
        traj_id, video_keys=rgb_keys, metadata_keys=[], img_size=img_size
    )
    per_cam = [np.asarray(traj.video_streams[k]) for k in rgb_keys]   # each (T, H, W, 3) uint8
    T = per_cam[0].shape[0]
    frames = np.stack(per_cam, axis=1)                                # (T, Cam, H, W, 3)
    del per_cam, traj

    out, patch_hw = [], None
    for s in range(0, T, batch):
        chunk = torch.from_numpy(frames[s : s + batch]).to(device)    # (b, Cam, H, W, 3) uint8
        b, cam = chunk.shape[:2]
        x = chunk.permute(0, 1, 4, 2, 3).reshape(b * cam, 3, *chunk.shape[2:4]).float() / 255.0
        _, grid = dino(x, return_spatial_grid=True)                   # (b*cam, C, pH, pW)
        _, C, pH, pW = grid.shape
        patch_hw = (pH, pW)
        out.append(grid.flatten(2).transpose(1, 2).reshape(b, cam * pH * pW, C).half())
    return torch.cat(out, dim=0), patch_hw, T


@torch.no_grad()
def latents_for_trajectory(encoder, tokens, gaps, num_views, batch=64):
    """(T, L, C) DINO tokens -> (T, len(gaps), num_tokens, token_dim) fp16 on CPU.

    Frames past the end of the episode clamp to the last frame, the same convention the dataset
    uses, so the tail of an episode yields a zero-motion latent rather than a discontinuity.
    """
    T = tokens.shape[0]
    expects_views = bool(getattr(encoder, "expects_num_views", False))
    result = None

    for j, k in enumerate(gaps):
        tgt_idx = torch.clamp(torch.arange(T, device=tokens.device) + int(k), max=T - 1)
        for s in range(0, T, batch):
            e = min(s + batch, T)
            diff = tokens[tgt_idx[s:e]].float() - tokens[s:e].float()
            z = encoder(diff, num_views=num_views)[0] if expects_views else encoder(diff)[0]
            if result is None:
                result = torch.empty(
                    (T, len(gaps), z.shape[1], z.shape[2]), dtype=torch.float16, device="cpu"
                )
            result[s:e, j] = z.half().cpu()
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rla", required=True, help="trained RLA run dir (stage 2)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument(
        "--step", default=None,
        help="RLA encoder checkpoint to use, as a step TOKEN: '40000', '0040000', or "
             "'40000.snapshot' for the resumable encoder_step0040000.snapshot.pt written beside a "
             "milestone. Default: the newest, preferring the milestone over a snapshot at the same "
             "step. Always recorded in the manifest so a policy can pin it.",
    )
    ap.add_argument(
        "--tag", default=None,
        help="sidecar subdirectory name. ARBITRARY -- any filesystem-safe string works, it is only "
             "a directory name. Default '<rla-run>__step<TOKEN>', so sidecars from different "
             "checkpoints never collide; a custom tag is still checked against the manifest, so it "
             "cannot silently mix two checkpoints either.",
    )
    ap.add_argument(
        "--max-gap", type=int, default=32,
        help="precompute the DENSE range of gaps 1..max-gap. Default 32 = the policy `horizon`, "
             "so every latent_mode works without recomputing.",
    )
    ap.add_argument(
        "--gaps", default=None,
        help="comma-separated gaps to compute INSTEAD of the dense range, e.g. "
             "'4,8,12,16,20,24,28,32' for the default 8-rung ladder plus final. Much faster; the "
             "policy configs then must only ask for gaps in this set.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--enc-batch", type=int, default=64, help="RLA encoder batch")
    ap.add_argument("--dino-batch", type=int, default=32, help="frames per DINO forward")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=0, help="only the first N trajectories (smoke test)")
    ap.add_argument("--num-shards", type=int, default=1, help="split the work across N processes")
    ap.add_argument("--shard", type=int, default=0, help="which shard this process handles")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(args.rla, "config.yaml"))
    img_size = int(cfg.vars.img_size)
    num_views = int(cfg.vars.get("num_views", len(CAMERAS)))
    dino_channels = int(cfg.vars.dino_channels)
    if num_views != len(CAMERAS):
        raise ValueError(f"RLA num_views={num_views} but {len(CAMERAS)} cameras configured")

    if args.gaps:
        gaps = sorted({int(g) for g in args.gaps.split(",") if g.strip()})
        if not gaps or gaps[0] < 1:
            raise ValueError(f"--gaps must be positive frame counts, got {args.gaps!r}")
    else:
        gaps = list(range(1, args.max_gap + 1))

    enc_step, enc_file = resolve_encoder_step(args.rla, args.step)
    tag = args.tag or default_tag(args.rla, enc_step)
    out_dir = os.path.join(args.dataset, "rla_latents", tag)
    os.makedirs(out_dir, exist_ok=True)

    # Refuse to mix checkpoints inside one tag: the sidecar would silently contain latents from two
    # different encoders, and nothing downstream could detect it. This is what makes an arbitrary
    # --tag safe -- the guard reads the manifest, not the directory name.
    existing_manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(existing_manifest) and not args.overwrite:
        with open(existing_manifest) as fh:
            prev = json.load(fh)
        if prev.get("rla_step") is not None and not same_step(prev["rla_step"], enc_step):
            raise SystemExit(
                f"[rla-precompute] {out_dir} already holds latents from step {prev['rla_step']}, "
                f"but you asked for step {enc_step}. Use a different --tag (the default encodes "
                "the step) or pass --overwrite."
            )

    device = torch.device(args.device)

    # Built on FIRST USE, not up front. The final "write the manifest over the complete set" pass
    # finds every .npy already on disk and computes nothing, so loading DINOv3 (and torch.compile-ing
    # it) would be ~a minute of pure waste on the one invocation you always have to run. The step
    # token is already resolved above, so a bad --step still fails immediately.
    _models = []

    def get_models():
        if not _models:
            enc, _ = build_encoder_from_work_dir(args.rla, device=str(device), step=enc_step)
            d = DINOv3FeatureExtractor(model_name=get_dinov3_model_for_channels(dino_channels))
            _models.extend([enc.to(device).eval(), d.to(device).eval()])
        return _models

    traj_ds = ManiSkillTrajectoryDataset(args.dataset)
    traj_ids = traj_ds.list_trajectories()
    if args.limit:
        traj_ids = traj_ids[: args.limit]
    all_traj_ids = list(traj_ids)
    if args.num_shards > 1:
        traj_ids = traj_ids[args.shard :: args.num_shards]

    print(
        f"[rla-precompute] {args.rla}\n"
        f"[rla-precompute] encoder: {enc_file}  (step {enc_step})\n"
        f"[rla-precompute] -> {out_dir}\n"
        f"[rla-precompute] {len(traj_ids)} trajectories | img_size={img_size} | "
        f"num_views={num_views} | {len(gaps)} gaps {gaps if len(gaps) <= 12 else f'1..{gaps[-1]}'} "
        f"| dtype={args.dtype}"
    )

    entries, t0, total_frames = {}, time.time(), 0
    stat_min, stat_max, stat_abs = np.inf, -np.inf, []

    def record_stats(arr):
        """Value stats over `arr`, accumulated whether it was just computed or read back.

        Reading them back matters: on the manifest-writing pass every trajectory already exists, so
        stats gathered only on the compute path would leave `value_stats` null in the one manifest
        anyone actually reads. `mean_abs` is what sets the BC-RLA loss balance (the latent target is
        `encoder output / 10`), so a null there is not cosmetic.
        """
        nonlocal stat_min, stat_max
        f = np.asarray(arr, dtype=np.float32)
        stat_min, stat_max = min(stat_min, float(f.min())), max(stat_max, float(f.max()))
        stat_abs.append(float(np.abs(f).mean()))

    for i, traj_id in enumerate(traj_ids, 1):
        path = os.path.join(out_dir, f"traj_{traj_id}.npy")
        if os.path.exists(path) and not args.overwrite:
            arr = np.load(path)          # not mmap: record_stats reads all of it anyway (~1.4 MB)
            entries[traj_id] = {"path": os.path.basename(path), "shape": list(arr.shape)}
            total_frames += arr.shape[0]
            record_stats(arr)
            continue

        encoder, dino = get_models()
        tokens, patch_hw, T = dino_tokens_for_trajectory(
            dino, traj_ds, traj_id, CAMERAS, img_size, device, batch=args.dino_batch
        )
        lat = latents_for_trajectory(encoder, tokens, gaps, num_views, batch=args.enc_batch)
        del tokens
        torch.cuda.empty_cache()

        arr = lat.numpy().astype(np.float16 if args.dtype == "float16" else np.float32)
        # Write through a file handle: np.save(path_str, ...) would append a second ".npy".
        tmp = path + ".partial"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)

        entries[traj_id] = {"path": os.path.basename(path), "shape": list(arr.shape)}
        total_frames += T
        record_stats(arr)

        if i % 10 == 0 or i == len(traj_ids):
            el = time.time() - t0
            print(
                f"[rla-precompute] {i}/{len(traj_ids)}  {el:6.1f}s  "
                f"({el / i:.2f}s/traj, eta {(len(traj_ids) - i) * el / i:6.1f}s)  "
                f"last shape {arr.shape}"
            )

    if args.num_shards > 1 and len(entries) < len(all_traj_ids):
        print(
            f"\n[rla-precompute] shard {args.shard}/{args.num_shards} done "
            f"({len(entries)} trajectories, {time.time() - t0:.0f}s). Manifest NOT written -- "
            "re-run once without --num-shards to write it over the complete set."
        )
        return

    manifest = {
        "tag": tag,
        "rla_run": os.path.abspath(args.rla),
        "rla_step": enc_step,
        "encoder_ckpt": enc_file,
        "encoder_class": str(cfg.models.encoder.name),
        "num_tokens": int(cfg.vars.latent_num_tokens),
        "token_dim": int(cfg.vars.token_dim),
        "num_views": num_views,
        "img_size": img_size,
        "gaps": gaps,
        "max_gap": max(gaps),
        "dtype": args.dtype,
        "layout": "latents[t, j] = encoder(DINO(frame min(t+gaps[j], T-1)) - DINO(frame t)); "
                  "`gaps` gives the frame count for each column j",
        "scaling": "RAW encoder output. The policy applies its own /10 convention at train time and "
                   "x10 in predict_latent_action; do not pre-scale here.",
        "end_of_episode": "frames past the episode end clamp to the last frame (zero-motion latent)",
        "num_trajectories": len(entries),
        "total_frames": int(total_frames),
        "value_stats": {
            "min": None if stat_min == np.inf else round(stat_min, 4),
            "max": None if stat_max == -np.inf else round(stat_max, 4),
            "mean_abs": round(float(np.mean(stat_abs)), 4) if stat_abs else None,
        },
        "trajectories": entries,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    size_gb = sum(
        os.path.getsize(os.path.join(out_dir, e["path"])) for e in entries.values()
    ) / 1e9
    print(
        f"\n[rla-precompute] DONE: {len(entries)} trajectories, {total_frames} frames, "
        f"{size_gb:.2f} GB in {time.time() - t0:.0f}s\n"
        f"[rla-precompute] value range [{manifest['value_stats']['min']}, "
        f"{manifest['value_stats']['max']}], mean|z| = {manifest['value_stats']['mean_abs']}\n"
        f"[rla-precompute] point the policy configs at:\n"
        f"[rla-precompute]     rla_latent_tag: {tag}\n"
        # QUOTED deliberately. The token is padded, and PyYAML (YAML 1.1) reads a bare 0040000 as
        # OCTAL -> 16384. It is also a string proper whenever it carries a `.snapshot` suffix.
        f'[rla-precompute]     rla_latent_step: "{enc_step}"'
    )


if __name__ == "__main__":
    main()
