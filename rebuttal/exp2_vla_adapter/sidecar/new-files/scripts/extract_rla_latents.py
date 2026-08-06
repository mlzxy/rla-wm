#!/usr/bin/env python
"""Precompute RLA latent-action targets for every LIBERO frame, for VLA-Adapter to regress.

Reads the RLDS datasets VLA-Adapter itself trains on (`data/libero/<suite>_no_noops/1.0.0`) and
writes, for each frame `t`, the `A = NUM_ACTIONS_CHUNK` latent actions that the action chunk
starting at `t` is supposed to bring about:

    z_i(t) = f_enc( s_{min(t+i+1, N-1)} - s_t )        i = 0 .. A-1        ("anchored", default)

Every latent is anchored on frame `t` -- the observation the policy conditions on -- so `z_i` is the
cumulative consequence of actions `a_t .. a_{t+i}`, and RLA token group `i` lines up with action
token `i`. `--pairs consecutive` instead emits `f_enc(s_{t+i+1} - s_{t+i})`; see
docs/rla-vla-adapter.md §4 for why anchored is the default (short version: `f_dec(x_t, z)` can only
decode an anchored `z` at inference, and gap-1 agentview residuals are ~2.5/255).

Why RLDS and not data/libero_converted: the RLDS records carry both views, `observation/state` and
`language_instruction`, which is everything needed -- camera geometry comes from
`data/libero_converted/camera_params.json` through `datalib.libero_camera.episode_camera_arrays`.
Reading the same records the trainer reads also makes the episode index and the join key exact.

Output (see docs/rla-vla-adapter.md §5; consumed by prismatic/vla/rla_targets.py):

    <out>/manifest.json          config + provenance, so a sidecar can never be silently mismatched
    <out>/stats.json             per-(q, d) mean/std and global RMS over the whole corpus
    <out>/<suite>/keys.json      sha1(state bytes) -> {file, episode_index, traj_len}
    <out>/<suite>/ep_000000.npy  (N, A, Q, D) float16

DINO tokens are cached in memory for the current trajectory only (fp32, ~220 MB at N=140 -- see
`RlaLatentEncoder.dino_tokens` for why not fp16) and are never written to disk; only `z` is
persisted (~16 KB/frame, ~3.9 GB for all four suites). Throughput is ~0.2 episodes/s
(~25 frames/s) on an RTX A6000, so budget ~3 h for all four suites or run one suite per GPU.

Usage (after `source scripts/vla_adapter_env.sh`):

    # before f_enc exists -- exercises the whole pipeline with a randomly-initialised encoder
    .venv/bin/python scripts/extract_rla_latents.py --random-encoder \
        --out data/libero_rla/smoke --suites libero_spatial --limit 4

    # the real thing
    .venv/bin/python scripts/extract_rla_latents.py \
        --rla-config configs/rla/16x64_libero.yaml \
        --rla-run runs/<rla-run>/<stamp> \
        --out data/libero_rla/16x64-<stamp>
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# TensorFlow decodes the TFRecords; torch runs DINOv3 and f_enc on the GPU. TF must be pinned to CPU
# *before* it grabs any device, or it takes the whole GPU and torch OOMs. Note this is
# `set_visible_devices`, not CUDA_VISIBLE_DEVICES="" -- torch still needs to see the GPU.
# (prismatic/vla/datasets/rlds/dataset.py:35 does the same at module scope; doing it explicitly here
# keeps the extractor independent of that import.)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds  # noqa: E402
import torch  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402

from datalib.dataset import resize_image, resize_intrinsics  # noqa: E402
from datalib.libero_camera import (  # noqa: E402
    FRONT_CAM,
    WRIST_CAM,
    CAMERA_PARAMS_DEFAULT,
    axis_angle_to_matrix,
    episode_camera_arrays,
    load_camera_params,
)
from src import models  # noqa: E402
from src.models.plucker_embedding import compute_plucker_rays  # noqa: E402
from utils.cam import convert_extrinsics_3x4_to_4x4  # noqa: E402
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels  # noqa: E402
from utils.misc import fetch_state_dict, load_config  # noqa: E402

# RLDS directory name -> suite name, same mapping scripts/convert_libero_to_trajectory.py uses.
SUITES = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

# The camera order the RLA autoencoder was trained with
# (configs/rla/16x64_libero.yaml: cameras: ["agentview_camera", "wrist_camera"]). The per-view token
# blocks are concatenated along the sequence in this order, and MultiViewTokenTransformer's view
# embedding / Plucker rays are indexed by it, so it must not be permuted.
CAMERA_ORDER = (FRONT_CAM, WRIST_CAM)
RLDS_IMAGE_KEY = {FRONT_CAM: "image", WRIST_CAM: "wrist_image"}


def episode_key(states: np.ndarray) -> str:
    """Stable per-episode join key: sha1 of the raw `observation/state` bytes.

    `state` is a float32 Tensor feature, so the bytes here are byte-identical to what
    `libero_dataset_transform` sees inside the tf.data graph -- that is what makes the join exact.
    `episode_metadata/file_path` cannot be used: it names the source HDF5 *task* file, which every
    demo of a task shares.
    """
    return hashlib.sha1(np.ascontiguousarray(states, dtype=np.float32).tobytes()).hexdigest()


class RlaLatentEncoder:
    """Frozen DINOv3 + frozen `f_enc`, driving the same math RlaAutoencoderMultiViewTrainer does.

    Deliberately not the trainer itself: constructing one builds an optimizer, the decoders and a
    TrajectoryDataset, none of which extraction needs. `scripts/verify_rla_targets.py` Gate 6
    asserts this path is numerically identical to `RlaAutoencoderMultiViewTrainer.inference_batch`
    on a shared batch, which is the safety a shared base class would have given.
    """

    def __init__(
        self,
        config_path: str,
        run_dir: Optional[str],
        step: Optional[int] = None,
        device: str = "cuda",
        random_encoder: bool = False,
    ) -> None:
        self.cfg = edict(load_config(config_path))
        self.device = torch.device(device)
        self.config_path = config_path
        self.run_dir = run_dir
        self.step = step

        enc_cfg = self.cfg.models.encoder
        self.encoder = getattr(models, enc_cfg.name)(**enc_cfg.args).to(self.device)
        if random_encoder:
            print("[extract_rla] --random-encoder: f_enc is randomly initialised, z is MEANINGLESS")
            self.loaded_ckpt = None
        else:
            if not run_dir:
                raise ValueError("--rla-run is required unless --random-encoder is given")
            self.encoder.load_state_dict(
                fetch_state_dict("encoder", run_dir, self.device, step=step), strict=True
            )
            # Record which file actually got loaded: `step=None` means "latest", and the manifest
            # has to pin the exact checkpoint or a sidecar cannot be traced back to its f_enc.
            ckpts = sorted(Path(run_dir, "ckpts").glob("encoder_step*.pt"))
            self.loaded_ckpt = str(
                ckpts[-1] if step is None else Path(run_dir, "ckpts", f"encoder_step{step:07d}.pt")
            )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.num_tokens = int(enc_cfg.args.num_tokens)
        self.token_dim = int(enc_cfg.args.out_channels)
        self.dino_channels = int(self.cfg.trainer.args.get("dino_channels", 1024))
        self.num_views = int(self.cfg.trainer.args.get("num_views", len(CAMERA_ORDER)))
        self.inverse_input_mode = str(
            self.cfg.trainer.args.get("inverse_input_mode", "sub")
        ).lower()
        if self.inverse_input_mode == "append":
            # Same restriction RlaAutoencoderMultiViewTrainer.__init__ enforces: "append"
            # concatenates along the sequence, so the per-view blocks stop being equally sized.
            raise ValueError("inverse_input_mode='append' is not supported with multiple views")

        # Probe on the inner module, mirroring RlaAutoencoderMultiViewTrainer._forward_views.
        self.expects_num_views = bool(getattr(self.encoder, "expects_num_views", False))
        self.expects_plucker = bool(getattr(self.encoder, "expects_plucker", False))

        self.dino = DINOv3FeatureExtractor(
            model_name=get_dinov3_model_for_channels(self.dino_channels)
        ).to(self.device)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def dino_tokens(self, images: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """[N, Cam, 3, H, W] uint8 -> ([N, Cam, Lp, C] on device, (ph, pw)).

        Mirrors RlaAutoencoderTrainer._extract_dino_tokens, including its uint8 -> [0, 1] rescale.

        The tokens stay in DINO's own output dtype -- **fp32**. DINOv3FeatureExtractor.forward runs
        the ViT under `autocast(fp16)`, but autocast keeps LayerNorm in fp32, so `last_hidden_state`
        comes back fp32 and that is what f_enc was trained on. Caching them as fp16 to halve memory
        shifts z by ~0.6% (max |d| = 6.1e-3 on a |z|max = 4.7 batch); gate 6 of
        scripts/verify_rla_targets.py caught exactly that. ~220 MB per trajectory at N=140, which is
        the price of matching training bit-for-bit.

        LIBERO's foreground masks are a constant 255 (the converter has no segmentation to write --
        verified: `rgbs * (foreground_masks > 0).float() == rgbs.float()`), so the trainer's masking
        step is the identity and is skipped here. If real masks ever land, this is the line that has
        to change with them.
        """
        num, cams = images.shape[:2]
        flat = torch.from_numpy(images).to(self.device).reshape(num * cams, *images.shape[2:])
        flat = flat.float() / 255.0
        _, patch_grid = self.dino(flat, return_spatial_grid=True)
        _, ch, ph, pw = patch_grid.shape
        tokens = patch_grid.flatten(2).transpose(1, 2).reshape(num, cams, ph * pw, ch)
        return tokens, (ph, pw)

    def compose(self, x_t: torch.Tensor, x_T: torch.Tensor) -> torch.Tensor:
        """RlaAutoencoderTrainer._compose_inverse_input, minus the unsupported 'append' mode.

        Dtype-preserving on purpose -- see `encode`.
        """
        if self.inverse_input_mode == "sub":
            return x_T - x_t
        if self.inverse_input_mode == "concat":
            return torch.cat([x_T, x_t], dim=2)
        raise RuntimeError(f"Unhandled inverse_input_mode={self.inverse_input_mode!r}")

    @torch.no_grad()
    def encode(
        self,
        x_anchor: torch.Tensor,
        x_future: torch.Tensor,
        plucker: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """([B, Cam*Lp, C], [B, Cam*Lp, C], [B, Cam*Lp, 6]) -> [B, Q, D] latent actions.

        Composition is dtype-preserving, so with `dino_tokens`'s fp32 output the residual is formed
        in fp32 exactly as RlaAutoencoderTrainer.inference_batch does.
        """
        enc_input = self.compose(x_anchor, x_future)
        if not self.expects_num_views:
            tokens, _ = self.encoder(enc_input, tokens=None)
        elif self.expects_plucker:
            tokens, _ = self.encoder(
                enc_input, tokens=None, num_views=self.num_views, plucker=plucker
            )
        else:
            tokens, _ = self.encoder(enc_input, tokens=None, num_views=self.num_views)
        if tokens.ndim != 3:
            raise ValueError(f"Encoder token output must be rank-3 [B, Q, D], got {tokens.shape}")
        return tokens


def decode_episode(episode) -> Dict[str, np.ndarray]:
    """One RLDS episode -> stacked numpy arrays. N frames <-> N actions <-> N states (verified)."""
    frames = {cam: [] for cam in CAMERA_ORDER}
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    instruction = ""
    for step in episode["steps"]:
        obs = step["observation"]
        for cam in CAMERA_ORDER:
            frames[cam].append(obs[RLDS_IMAGE_KEY[cam]].numpy())
        states.append(obs["state"].numpy())
        actions.append(step["action"].numpy())
        if not instruction:
            instruction = step["language_instruction"].numpy().decode("utf-8")

    out = {cam: np.stack(frames[cam]) for cam in CAMERA_ORDER}
    out["state"] = np.stack(states).astype(np.float32)
    out["action"] = np.stack(actions).astype(np.float32)
    out["instruction"] = instruction
    return out


def prepare_views(
    episode: Dict[str, np.ndarray], img_size: int, camera_params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize both views to `img_size` and build the matching per-frame K / camera-to-world.

    Resizing goes through `datalib.dataset.resize_image` (cv2 INTER_LINEAR) and
    `resize_intrinsics`, because that is the exact path `ManiSkillTrajectoryDataset.read_frames`
    takes -- i.e. what the RLA autoencoder was trained on. Do NOT swap in `tf.image.resize`
    (lanczos3), which is what VLA-Adapter's own eval preprocessing uses.

    Returns ([N, Cam, 3, S, S] uint8, [N, Cam, 3, 3] float32, [N, Cam, 4, 4] float32).
    """
    orig_h, orig_w = episode[CAMERA_ORDER[0]].shape[1:3]
    states = episode["state"]
    cams = episode_camera_arrays(
        task_description=episode["instruction"],
        eef_pos=states[:, :3].astype(np.float64),
        eef_rot=axis_angle_to_matrix(states[:, 3:6].astype(np.float64)),
        params=camera_params,
        height=orig_h,
        width=orig_w,
    )

    rgbs, intrinsics, c2w = [], [], []
    for cam in CAMERA_ORDER:
        video = episode[cam]  # [N, H, W, 3] uint8
        resized = np.stack([resize_image(frame, img_size) for frame in video])
        rgbs.append(resized.transpose(0, 3, 1, 2))  # [N, 3, S, S]

        k = cams[f"{cam}_intrinsics"][:, 0]  # [N, 3, 3]
        intrinsics.append(resize_intrinsics(k, orig_h, orig_w, img_size, img_size))
        # TrajectoryDataset stores camera-to-world under the key `w2c` and never inverts it;
        # compute_plucker_rays expects camera-to-world, so pass it straight through.
        c2w.append(convert_extrinsics_3x4_to_4x4(cams[f"{cam}_extrinsics"][:, 0]))

    return (
        np.stack(rgbs, axis=1),
        np.stack(intrinsics, axis=1).astype(np.float32),
        np.stack(c2w, axis=1).astype(np.float32),
    )


def anchored_pairs(num_frames: int, chunk: int) -> np.ndarray:
    """[N, A] future frame index for each (anchor, chunk-step), clamped to the last frame.

    `min(t + i + 1, N - 1)` matches `chunk_act_obs`'s own
    `floored_action_chunk_indices = min(max(idx, 0), traj_len - 1)`, so the RLA targets clamp at the
    tail exactly the way the action chunk does. (There are N frames but only N-1 real transitions:
    the last action leads nowhere.)
    """
    t = np.arange(num_frames)[:, None]
    i = np.arange(chunk)[None, :]
    return np.minimum(t + i + 1, num_frames - 1)


@torch.no_grad()
def encode_episode_anchored(
    enc: RlaLatentEncoder,
    tokens: torch.Tensor,
    plucker: torch.Tensor,
    chunk: int,
    batch_pairs: int,
) -> torch.Tensor:
    """[N, A, Q, D] with z[t, i] = f_enc(s_{min(t+i+1, N-1)} - s_t), anchored on frame t."""
    num_frames, cams, lp, ch = tokens.shape
    flat = tokens.reshape(num_frames, cams * lp, ch)
    future = torch.from_numpy(anchored_pairs(num_frames, chunk)).to(tokens.device)  # [N, A]
    anchor_idx = torch.arange(num_frames, device=tokens.device, dtype=future.dtype)
    anchor_idx = anchor_idx[:, None].expand_as(future)

    a_flat, f_flat = anchor_idx.reshape(-1), future.reshape(-1)
    out = torch.empty(
        a_flat.numel(), enc.num_tokens, enc.token_dim, device=tokens.device, dtype=torch.float32
    )
    for lo in range(0, a_flat.numel(), batch_pairs):
        hi = min(lo + batch_pairs, a_flat.numel())
        a, f = a_flat[lo:hi], f_flat[lo:hi]
        out[lo:hi] = enc.encode(flat[a], flat[f], plucker[a] if plucker is not None else None)
    return out.reshape(num_frames, chunk, enc.num_tokens, enc.token_dim)


@torch.no_grad()
def encode_episode_consecutive(
    enc: RlaLatentEncoder,
    tokens: torch.Tensor,
    plucker: torch.Tensor,
    chunk: int,
    batch_pairs: int,
) -> torch.Tensor:
    """[N, A, Q, D] with z[t, i] = f_enc(s_{k+1} - s_k), k = min(t+i, N-2).

    Consecutive latents are shareable across anchors (`z_i(t) == z_0(t+i)`), so only N-1 are
    computed and then gathered into the same [N, A, ...] layout the anchored mode produces --
    downstream code cannot tell the two apart.
    """
    num_frames, cams, lp, ch = tokens.shape
    flat = tokens.reshape(num_frames, cams * lp, ch)
    num_pairs = max(num_frames - 1, 1)
    base = torch.empty(
        num_pairs, enc.num_tokens, enc.token_dim, device=tokens.device, dtype=torch.float32
    )
    idx = torch.arange(num_pairs, device=tokens.device)
    for lo in range(0, num_pairs, batch_pairs):
        hi = min(lo + batch_pairs, num_pairs)
        k = idx[lo:hi]
        nxt = torch.clamp(k + 1, max=num_frames - 1)
        base[lo:hi] = enc.encode(flat[k], flat[nxt], plucker[k] if plucker is not None else None)

    t = torch.arange(num_frames, device=tokens.device)[:, None]
    i = torch.arange(chunk, device=tokens.device)[None, :]
    return base[torch.clamp(t + i, max=num_pairs - 1)]


class RunningStats:
    """Per-(q, d) mean/std accumulated in float64 across the whole corpus."""

    def __init__(self, num_tokens: int, token_dim: int) -> None:
        self.count = 0
        self.total = np.zeros((num_tokens, token_dim), dtype=np.float64)
        self.total_sq = np.zeros((num_tokens, token_dim), dtype=np.float64)

    def update(self, z: np.ndarray) -> None:
        flat = z.reshape(-1, *z.shape[-2:]).astype(np.float64)
        self.count += flat.shape[0]
        self.total += flat.sum(axis=0)
        self.total_sq += (flat**2).sum(axis=0)

    def finalize(self) -> dict:
        if self.count == 0:
            raise RuntimeError("no latents were accumulated; nothing to write stats for")
        mean = self.total / self.count
        var = np.maximum(self.total_sq / self.count - mean**2, 0.0)
        std = np.sqrt(var)
        return {
            "count": int(self.count),
            "mean": mean.tolist(),
            "std": std.tolist(),
            # Floor the std so a dead latent slot cannot blow up the normalised target.
            "std_floor": 1e-3,
            "global_rms": float(np.sqrt((self.total_sq.sum()) / (self.count * mean.size))),
        }


def compute_stats(out_root: Path, num_tokens: int, token_dim: int) -> dict:
    """Per-(q, d) statistics over every `ep_*.npy` present under `out_root`.

    A final pass over the files, not an accumulator threaded through extraction, because extraction
    skips episodes that already exist: accumulating inline would silently compute the statistics
    from only the episodes a *resumed* run happened to write, and every target would then be
    normalised against the wrong mean/std.
    """
    stats = RunningStats(num_tokens, token_dim)
    files = sorted(out_root.glob("*/ep_*.npy"))
    for idx, path in enumerate(files):
        stats.update(np.load(path))
        if idx % 200 == 0:
            print(f"  stats {idx + 1}/{len(files)}", flush=True)
    print(f"  stats over {len(files)} episodes, {stats.count} frames")
    return stats.finalize()


def write_keys(path: Path, keys: Dict[str, dict]) -> None:
    """Atomic-ish write, so an interrupt cannot leave a half-written keys.json behind."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(keys, indent=1, sort_keys=True))
    tmp.replace(path)


def extract_suite(
    suite: str,
    enc: RlaLatentEncoder,
    args: argparse.Namespace,
    camera_params: dict,
) -> dict:
    rlds_dir = REPO_ROOT / args.rlds_root / SUITES[suite] / "1.0.0"
    if not rlds_dir.is_dir():
        print(f"[skip] {suite}: {rlds_dir} does not exist")
        return {}

    out_dir = Path(args.out) / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    keys_path = out_dir / "keys.json"
    keys: Dict[str, dict] = {}
    if keys_path.exists() and not args.overwrite:
        keys = json.loads(keys_path.read_text())

    builder = tfds.builder_from_directory(str(rlds_dir))
    total = builder.info.splits["train"].num_examples
    if args.limit > 0:
        total = min(total, args.limit)
    print(f"[{suite}] {total} episodes -> {out_dir}")

    encode = encode_episode_anchored if args.pairs == "anchored" else encode_episode_consecutive
    started = time.time()
    num_written = 0
    for ep_idx, raw in enumerate(builder.as_dataset(split=f"train[:{total}]", shuffle_files=False)):
        rel = f"ep_{ep_idx:06d}.npy"
        target = out_dir / rel
        episode = decode_episode(raw)
        key = episode_key(episode["state"])
        num_frames = len(episode["state"])
        if key in keys and keys[key]["file"] != rel:
            raise RuntimeError(
                f"[{suite}] state hash collision: {key} maps to both {keys[key]['file']} and {rel}"
            )
        keys[key] = {"file": rel, "episode_index": ep_idx, "traj_len": num_frames}

        if target.exists() and not args.overwrite:
            continue

        rgbs, intrinsics, c2w = prepare_views(episode, args.img_size, camera_params)
        tokens, patch_hw = enc.dino_tokens(rgbs)

        plucker = None
        if enc.expects_plucker:
            rays = compute_plucker_rays(
                intrinsics=torch.from_numpy(intrinsics).to(enc.device),
                cam2world=torch.from_numpy(c2w).to(enc.device),
                patch_hw=patch_hw,
                image_hw=(args.img_size, args.img_size),
            )  # [N, Cam, Lp, 6]
            plucker = rays.reshape(rays.shape[0], -1, 6)

        z = encode(enc, tokens, plucker, args.chunk, args.batch_pairs)
        z_np = z.detach().float().cpu().numpy()
        np.save(target, z_np.astype(np.float16))
        num_written += 1
        if ep_idx % 25 == 0 or ep_idx == total - 1:
            rate = (ep_idx + 1) / max(time.time() - started, 1e-6)
            print(
                f"  [{suite}] {ep_idx + 1}/{total}  N={num_frames}  "
                f"z={tuple(z_np.shape)}  {rate:.2f} ep/s",
                flush=True,
            )

        del tokens, plucker, z
        if ep_idx % 50 == 0:
            torch.cuda.empty_cache()
            # Flush the index as we go: extraction takes hours, and a run interrupted before the
            # end would otherwise leave the .npy files with no keys.json to join them by --
            # unusable, and invisible to `verify_rla_targets.py --sidecar`.
            write_keys(keys_path, keys)

    write_keys(keys_path, keys)
    print(f"[{suite}] wrote {num_written} new episodes ({len(keys)} keys total)")
    return {"num_episodes": len(keys), "num_written": num_written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rla-config", default="configs/rla/16x64_libero.yaml")
    parser.add_argument("--rla-run", default=None, help="RLA autoencoder run dir holding ckpts/")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    parser.add_argument("--out", required=True, help="sidecar root, e.g. data/libero_rla/16x64-<stamp>")
    parser.add_argument("--rlds-root", default="data/libero")
    parser.add_argument("--camera-params", default=CAMERA_PARAMS_DEFAULT)
    parser.add_argument("--suites", nargs="+", default=["all"], choices=["all"] + list(SUITES))
    parser.add_argument("--chunk", type=int, default=8, help="A; must equal NUM_ACTIONS_CHUNK")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--pairs", choices=["anchored", "consecutive"], default="anchored")
    parser.add_argument("--batch-pairs", type=int, default=64, help="encoder pairs per forward")
    parser.add_argument("--limit", type=int, default=-1, help="only the first N episodes per suite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--random-encoder",
        action="store_true",
        help="skip the checkpoint and keep f_enc randomly initialised -- for exercising the "
        "pipeline before the autoencoder is trained. z is meaningless.",
    )
    parser.add_argument("--overwrite", action="store_true", help="re-extract existing episodes")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA device: DINOv3 + f_enc need a GPU (pass --device cpu to force)")

    enc = RlaLatentEncoder(
        config_path=args.rla_config,
        run_dir=args.rla_run,
        step=args.step,
        device=args.device,
        random_encoder=args.random_encoder,
    )
    camera_params = load_camera_params(args.camera_params)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    suites = list(SUITES) if "all" in args.suites else args.suites

    per_suite = {}
    for suite in suites:
        result = extract_suite(suite, enc, args, camera_params)
        if result:
            per_suite[suite] = result

    # Authoritative final pass over everything on disk -- see compute_stats.
    print("\ncomputing normalisation statistics")
    (out_root / "stats.json").write_text(
        json.dumps(compute_stats(out_root, enc.num_tokens, enc.token_dim))
    )
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "rla_config": args.rla_config,
                "rla_config_sha1": hashlib.sha1(
                    Path(args.rla_config).read_bytes()
                ).hexdigest(),
                "rla_run": args.rla_run,
                "checkpoint": enc.loaded_ckpt,
                "random_encoder": bool(args.random_encoder),
                "chunk": args.chunk,
                "num_tokens": enc.num_tokens,
                "token_dim": enc.token_dim,
                "img_size": args.img_size,
                "pairs": args.pairs,
                "cameras": list(CAMERA_ORDER),
                "inverse_input_mode": enc.inverse_input_mode,
                "dino_channels": enc.dino_channels,
                "rlds_root": args.rlds_root,
                "suites": per_suite,
            },
            indent=1,
        )
    )
    print(f"\ndone -> {out_root}")
    if args.random_encoder:
        print("WARNING: --random-encoder was used; these latents are noise, not RLA targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
