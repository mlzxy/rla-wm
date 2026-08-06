#!/usr/bin/env python
"""Extract RLA latent-action targets for every LIBERO frame, as a sidecar VLA-Adapter can join.

For each frame `t` of each RLDS episode this writes the `A = 8` latent actions that the action
chunk starting at `t` is supposed to bring about:

    z_i(t) = f_enc( s_{min(t+i+1, N-1)} - s_t )      i = 0 .. A-1      ("anchored", the default)

Every latent is anchored on frame `t` -- the observation the policy conditions on -- so `z_i` is the
cumulative consequence of actions `a_t .. a_{t+i}` and RLA token group `i` lines up with action
token `i`. `--pairs consecutive` instead emits `f_enc(s_{t+i+1} - s_{t+i})`, which is 8x cheaper to
store but cannot be decoded at inference (`f_dec(x_t, z)` only accepts an anchored `z`); it exists
as an ablation arm, not as the default.

Correctness rests on one idea: **the math is not reimplemented here.** `RlaSidecarExtractor`
subclasses `RlaAutoencoderMultiViewTrainer` and calls its `_extract_dino_tokens`,
`_compose_inverse_input`, `_plucker_rays`, `_forward_views`, `inference_batch` and
`_decode_tokens_to_vis` directly, so a target can only differ from what the autoencoder trained on
if the trainer itself changed. Three checks back that up on every run:

  * `--selfcheck` (on by default) runs the trainer's own `inference_batch` on a handful of pairs.
    Its *structural* term -- this file's encode call fed the reference's own tokens -- is exactly 0.
  * `--viz N` renders chunk filmstrips *from the files just written*, so the picture validates the
    artifact -- fp16 cast, chunk-axis order, camera order -- not merely a fresh forward pass.
  * each filmstrip also reports, numerically, which ground-truth frame each `recon(z_i)` is actually
    closest to. It should be `t+i+1`; a systematic lead or lag means the chunk axis is off by one.

Measured on `runs/16x64_libero/20260725_00-39-52` @ step 32500, `libero_spatial`: structural
0.00e+00, end-to-end 1.7e-02 (0.10% of |z|max 16.6), chunk alignment 8/8 with median offset +0.

================================================================================================
USAGE
================================================================================================

Everything runs from the repo root, in THIS repo's .venv (it needs our RLA autoencoder), with

    export PYTHONPATH=.:./third_party/diffusion_policy

It reads the RLDS export, so it also needs TensorFlow -- see ../README.md for the one-line
install. Shorthand below:

    IMPL=rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py
    RUN=runs/16x64_libero/20260725_00-39-52          # the RLA autoencoder run dir

**1. Debug first -- one GPU, a few episodes, both checks on.** Always do this after retraining
`f_enc`; it takes about 30 s and it is what tells you the targets are sane before you spend an hour.

    .venv/bin/python $IMPL --rla-run $RUN --out data/libero_rla/dbg \
        --gpus 0 --suites libero_spatial --limit 3 --viz 3

**2. The real run -- four GPUs, all four suites.** ~1.1 episodes/s aggregate, so ~26 min for the
~1700 episodes across the four suites.

    .venv/bin/python $IMPL --rla-run $RUN --gpus 0,1,2,3
    # -> data/libero_rla/16x64-a8-step032500

Leave `--out` off and the directory names itself after everything that changes the contents:
latent shape, **chunk size**, and checkpoint step (plus `--pairs`/`--img-size` when non-default).
A sidecar built with `--chunk 8` is not interchangeable with one built with `--chunk 16` -- `A` is
the second axis of every array, and `RlaTargetStore` rejects a manifest whose `chunk` disagrees
with `NUM_ACTIONS_CHUNK` -- so they must not land in the same directory or be told apart by memory.
Pass `--out` explicitly only for throwaway runs; if its name does not mention the chunk size, the
run says so and continues.

**3. Re-render filmstrips later, without re-extracting.**

    .venv/bin/python $IMPL --viz-only data/libero_rla/16x64-a8-step032500 --rla-run $RUN --viz 8

Interrupted runs resume: episodes whose file already exists are skipped, `keys.json` is flushed
every 50 episodes, and `stats.json` is always recomputed from everything on disk at the end.

### What the output should look like

    [parent] --out not given, writing to data/libero_rla/16x64-a8-step032500
    [rank 0] [libero_spatial] episodes [0, 108) of 432 -> .../16x64-a8-step032500/libero_spatial
    [rank 0] [libero_spatial] selfcheck vs inference_batch: structural 0.00e+00 (exact),
                              end-to-end 1.65e-02 = 0.10% of |z|max 16.6  OK
    [rank 0] [libero_spatial] ep0 t=36: chunk alignment 8/8 exact, offsets [0,0,0,0,0,0,0,0],
                              median +0 (no systematic lead/lag)
    [rank 0] [libero_spatial] 1/108  N=110  z=(110, 8, 16, 64)  0.27 ep/s

Read it in this order:

  * `structural` must be **exactly 0.00e+00**. Anything else means this file feeds the encoder
    differently from the trainer and the targets are wrong -- the run aborts.
  * `end-to-end` is fp16 batch-shape noise, expect ~0.1%. It only fails above 2% of |z|.
  * `median +0` on the chunk alignment. Individual steps landing +-1 is normal while the arm is
    nearly stationary; a *median* of +-1 means the chunk axis is off by one.
  * `z=(N, 8, 16, 64)` -- N frames, A=8 chunk steps, Q=16 queries, D=64.

Then open `<out>/viz/*.jpg`: top row is ground truth `t, t+1 .. t+8`, bottom row is
`f_dec(x_t, z_i)` decoded to pixels, both cameras side by side in each cell. The bottom row should
track the top row one step at a time, sharp, correct handedness. Sample filmstrips from a verified
run are in `samples/` next to this file.

### Flags you will actually touch

    --rla-run PATH     required; the autoencoder run dir holding ckpts/
    --step TOKEN       pin a checkpoint: `37500`, `0037500`, or `37500.snapshot` for the resumable
                       `encoder_step0037500.snapshot.pt` written beside the milestone. Default is
                       the highest-numbered checkpoint (milestone over snapshot), resolved once in
                       the parent and passed down, so every rank encodes with the same f_enc even
                       if training saves a new one mid-extraction.
    --chunk A          action chunk size (default 8 = VLA-Adapter's NUM_ACTIONS_CHUNK). Changes the
                       data, so it lands in a different auto-named directory.
    --out PATH         sidecar root (default: auto-named from Q, D, chunk and step)
    --gpus 0,1,2,3     one worker subprocess per id (default "0")
    --suites ...       libero_spatial|libero_object|libero_goal|libero_10, or all (default)
    --limit N          only the first N episodes per suite -- for debugging
    --viz N            filmstrips to render, rank 0 only (default 4; 0 disables and skips
                       loading f_dec and the image decoder entirely)
    --viz-only PATH    render filmstrips for an existing sidecar and exit
    --pairs MODE       anchored (default) | consecutive -- see above; different research question
    --compress         .npz instead of .npy; only 1.12x, off by default (see above)
    --overwrite        re-extract episodes that already have a file
    --no-selfcheck     skip the inference_batch comparison
    --dino-batch N     frames per DINO forward (default 16); batches are padded to exactly this,
                       so it trades tail compute against peak memory, never accuracy
    --batch-pairs N    encoder pairs per forward (default 64)
    --dino-compile     torch.compile the ViT as the trainer does; makes output batch-shape
                       dependent, so the sidecar stops being reproducible. Off by default.

Output, under `data/libero_rla/<Q>x<D>-a<A>-step<step>/` unless `--out` overrides it:

    manifest.json          config, run, checkpoint sha256, A/Q/D, img_size, pairs, format
    stats.json             per-(q, d) mean/std over the whole corpus + std_floor
    <suite>/keys.json      sha1(observation/state bytes) -> {file, episode_index, traj_len}
    <suite>/_keys_rank*.json   per-worker shards, merged into keys.json; kept so a run resumes
    <suite>/ep_000000.npy  (N, A, Q, D) float16, mmap-able -- one file per episode
    viz/*.jpg              chunk filmstrips

A*Q*D*2 = 16 KB/frame at A=8, so 4.3 GiB for all four suites (1693 episodes, mean N 119/144/142/275
for spatial/object/goal/10) -- and proportionally more for a larger `--chunk`. The reader in the overlay
(`rla/sidecar.py`, and `prismatic/vla/rla_targets.py` in ../sidecar/patches/03) consumes this as
is -- no change needed.

### Why compression is off by default

`--compress` writes DEFLATE-compressed `.npz` instead, and it is **not worth it**: measured on real
latents it buys **1.12x**, because fp16 mantissa bytes are near-uniform (256 distinct low bytes,
only 2.5% distinct values overall) so DEFLATE finds almost nothing. Byte-shuffling first (what
blosc does for float arrays) reaches 1.21x -- still not enough to justify a non-standard container.
Meanwhile `.npz` costs CPU on write, loses `mmap_mode` on read, and needs two extra lines in the
reader, because `np.load` hands back an `NpzFile`:

    cached = np.load(self.root / suite / meta["file"], mmap_mode="r")
    if isinstance(cached, np.lib.npyio.NpzFile):        # v4world: only if --compress was used
        cached = cached["z"]

If disk is genuinely tight the real levers are `--pairs consecutive` (8x fewer distinct latents) or
int8 quantisation with a per-(q, d) scale, not zip. `keys.json` records each file name with its
extension, so the two formats can even coexist in one sidecar.

Caveats worth knowing before you trust a number that came out of this:

  * The RLA run uses `save_latest_only: true`, so `ckpts/encoder_step*.pt` is replaced as training
    continues. The encoder is loaded once at startup and the manifest pins its path *and* sha256,
    but a sidecar extracted mid-training is a snapshot of a moving model. Re-extract from a final
    checkpoint before the headline numbers.
  * That directory holds two kinds of file: milestones (`encoder_step0037500.pt`) and resumable
    snapshots (`encoder_step0037500.snapshot.pt`). `--step` takes either -- `37500`, `0037500` or
    `37500.snapshot` -- and the default picks the highest-numbered one, preferring the milestone
    when both exist at that step. Note this differs from `utils.misc.fetch_state_dict`, whose
    "latest" is a text sort and therefore lands on `.snapshot`.
  * `--pairs anchored` vs `consecutive` is an open research fork, not an implementation detail.
  * LIBERO ships no segmentation, so `foreground_masks` are a constant 255 and the trainer's masking
    step is the identity. The self-check goes through `inference_batch`, which does apply the mask,
    so this claim is re-verified numerically on every run.
  * **Both DINOv3 and f_enc are batch-shape sensitive under fp16 autocast** -- the same frame run in
    a batch of 16 and a batch of 32 can differ. `episode_tokens` pads every DINO call to a fixed
    shape to kill this for the ViT (it was worth 2.8e-2 on tokens of scale 3.3, confined to the
    partial tail batch of each episode); f_enc's residual sensitivity is 8.0e-3 on |z|max 18.2,
    which is smaller than one fp16 ulp at that magnitude, so it is left alone. The upshot: a sidecar
    is reproducible for a given `--dino-batch`, and the manifest records it.

Supersedes the single-GPU `extract_rla_latents.py` carried in patches/02-extractor.patch,
which stays there as the record of the session that first built this pipeline.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# rebuttal/exp2_vla_adapter/sidecar/ -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# TensorFlow decodes the TFRecords; torch runs DINOv3, f_enc and f_dec on the GPU. TF has to be
# pinned to CPU before it grabs a device, or it takes the whole GPU and torch OOMs. Note this is
# `set_visible_devices`, not CUDA_VISIBLE_DEVICES="" -- torch still needs to see the GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds  # noqa: E402
import torch  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402

from datalib.dataset import resize_image, resize_intrinsics  # noqa: E402
from src import models as model_zoo  # noqa: E402
from rebuttal.src.trainers.rla_autoencoder_multiview_trainer import (  # noqa: E402
    RlaAutoencoderMultiViewTrainer,
)
from src.utils.general_utils import save_image_with_notes  # noqa: E402
from utils.cam import convert_extrinsics_3x4_to_4x4  # noqa: E402
from utils.dino import DINOv3FeatureExtractor, get_dinov3_model_for_channels  # noqa: E402
from utils.misc import load_config  # noqa: E402


def _import_libero_camera_helpers():
    """The LIBERO camera geometry, imported from the converter rather than copied.

    `../libero_prep/convert_libero_to_trajectory.py` owns `episode_camera_arrays` and friends, but it also
    does `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")` at module scope to keep TensorFlow off
    the GPU during conversion. Importing it naively would hide the GPU from torch too. TF is already
    pinned above, so: pre-seed the variable to make the `setdefault` a no-op, import, then restore
    the original environment. torch initialises CUDA lazily, on first use, which is well after this.

    Copying the ~150 lines instead would let the two drift apart silently; the converter's only
    other module-scope import is `datalib.dataset`, which this file needs anyway.
    """
    import importlib.util

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = saved if saved is not None else "__placeholder__"
    try:
        path = (REPO_ROOT / "rebuttal/exp2_vla_adapter/libero_prep"
                / "convert_libero_to_trajectory.py")
        spec = importlib.util.spec_from_file_location("_libero_converter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return module


_CONV = _import_libero_camera_helpers()
FRONT_CAM = _CONV.FRONT_CAM
WRIST_CAM = _CONV.WRIST_CAM
CAMERA_PARAMS_DEFAULT = _CONV.CAMERA_PARAMS_DEFAULT
load_camera_params = _CONV.load_camera_params
episode_camera_arrays = _CONV.episode_camera_arrays
axis_angle_to_matrix = _CONV.axis_angle_to_matrix

# RLDS directory name -> suite name, the same mapping the converter uses.
SUITES = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

# The camera order the autoencoder was trained with
# (rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml:
# cameras: ["agentview_camera", "wrist_camera"]). Per-view token blocks are concatenated along the
# sequence in this order and the Plücker rays are indexed by it, so it must not be permuted.
CAMERA_ORDER = (FRONT_CAM, WRIST_CAM)
RLDS_IMAGE_KEY = {FRONT_CAM: "image", WRIST_CAM: "wrist_image"}


def episode_key(states: np.ndarray) -> str:
    """Stable per-episode join key: sha1 of the raw `observation/state` bytes.

    Must stay byte-for-byte identical to `prismatic/vla/rla_targets.episode_key`. `state` is a
    float32 Tensor feature, so these bytes are exactly what `libero_dataset_transform` sees inside
    the tf.data graph -- that is what makes the join exact. `episode_metadata/file_path` cannot be
    used: it names the source HDF5 *task* file, which every demo of a task shares.
    """
    return hashlib.sha1(np.ascontiguousarray(states, dtype=np.float32).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def ckpt_token(path: Path, name: str) -> str:
    """`.../encoder_step0037500.snapshot.pt` -> `0037500.snapshot`."""
    return path.name[len(f"{name}_step"):-len(".pt")]


def parse_step_token(step: str) -> Tuple[int, str]:
    """`37500` / `0037500` / `37500.snapshot` -> `(37500, "")`, `(37500, "")`, `(37500, ".snapshot")`.

    Splits a step token into its leading number and whatever the trainer appended. Zero padding is
    not part of a checkpoint's identity, so it must not be part of the comparison -- otherwise
    `--step 37500` would fail to find `encoder_step0037500.pt`. A token with no leading digits at
    all (a hand-renamed checkpoint) falls back to comparing its literal text.
    """
    text = str(step).strip()
    cut = len(text) - len(text.lstrip("0123456789"))
    if cut == 0:
        return -1, text
    return int(text[:cut]), text[cut:]


def load_state_dict_file(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """`utils.misc.fetch_state_dict`'s tail (`misc.py:383-388`), against an explicit file.

    Not `fetch_state_dict` itself, for two reasons. It composes the filename as
    `{name}_step{int(step):07d}.pt`, which cannot express `..._step0037500.snapshot.pt`; and given
    `step=None` it resolves "latest" with a second, independent `sorted()`, so the file it opened
    could in principle differ from the one `_ckpt_path` recorded. Resolving once and loading
    exactly what we resolved is what makes `manifest["checkpoint"]` a statement about the weights
    that were actually used.
    """
    print(f"loading {path}")
    blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


# --------------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------------


class RlaSidecarExtractor(RlaAutoencoderMultiViewTrainer):
    """The trainer's encode/decode path, without the training scaffolding.

    `__init__` deliberately does **not** call `super().__init__()`: the base builds an optimizer,
    both TrajectoryDatasets, a dataloader and wandb, none of which extraction needs, and doing so
    would tie extraction to a training recipe that is still running. Instead it sets exactly the
    attributes the inherited methods read -- each one below names its consumer. If the trainer grows
    a new dependency this fails loudly with `AttributeError` rather than diverging in silence.
    """

    def __init__(
        self,
        config_path: str,
        run_dir: str,
        step: Optional[int] = None,
        device: str = "cuda",
        need_decoder: bool = True,
        need_image_decoder: bool = False,
        dino_compile: bool = False,
    ) -> None:
        self.cfg = edict(load_config(config_path))
        targs = self.cfg.trainer.args

        self._device = torch.device(device)                       # see the `device` property below
        self.rgb_key = str(targs.get("rgb_key", "rgbs"))          # _get_masked_rgb_sequence
        self.inverse_input_mode = str(targs.get("inverse_input_mode", "sub")).lower()
        self.num_views = int(targs.get("num_views", len(CAMERA_ORDER)))  # inference_batch
        self.dino_channels = int(targs.get("dino_channels", 1024))
        self.decoder_only_mode = False                            # inference_batch
        self.use_vq = False                                       # inference_batch (no `vq` model)
        self.step = 0                                             # inference_batch, VQ revival only
        self.revival_every = -1                                   # inference_batch, VQ revival only

        if self.inverse_input_mode == "append":
            # Same restriction RlaAutoencoderMultiViewTrainer.__init__ enforces: "append"
            # concatenates along the sequence, so per-view blocks stop being equally sized.
            raise ValueError("inverse_input_mode='append' is not supported with multiple views")

        self.models: Dict[str, torch.nn.Module] = {}
        self.models["encoder"] = self._build("encoder")
        self._load(self.models["encoder"], "encoder", run_dir, step)
        self.encoder_ckpt = self._ckpt_path("encoder", run_dir, step)

        if need_decoder or need_image_decoder:
            # inference_batch always runs f_dec, so the self-check needs it too, not just the viz.
            self.models["decoder"] = self._build("decoder")
            self._load(self.models["decoder"], "decoder", run_dir, step)
        if need_image_decoder:
            self.models["image_decoder"] = self._build("image_decoder")
            img_ckpt = str(targs.get("image_decoder_ckpt", ""))
            if not img_ckpt:
                raise ValueError(
                    f"{config_path} has no trainer.args.image_decoder_ckpt, so DINO tokens cannot "
                    "be rendered back to pixels. Train "
                    "rebuttal/exp2_vla_adapter/configs/unet/dino_to_image_v1_libero.yaml "
                    "and point the config at its run dir, or pass --viz 0."
                )
            # The image decoder run saves under the name "decoder" -- the same lookup
            # `RlaAutoencoderTrainer.__init__` does, at its own run's latest step.
            self.image_decoder_ckpt = self._ckpt_path("decoder", img_ckpt, None)
            self.models["image_decoder"].load_state_dict(
                load_state_dict_file(self.image_decoder_ckpt, self.device), strict=True
            )

        for module in self.models.values():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        # _plucker_rays. Probed on the inner modules exactly as the parent's __init__ does.
        self._needs_plucker = any(
            getattr(self.models.get(name), "expects_plucker", False)
            for name in ("encoder", "decoder")
        )

        # `use_compile=False`, unlike the trainer, which takes the default True. Measured on this
        # tree: compiled DINOv3 is **batch-shape dependent** -- the same frame run at batch 32 and
        # batch 64 differs by 7.5e-3 on tokens of scale 2.89 (~0.3%), because Inductor picks
        # different fused kernels per shape. Uncompiled it is exactly invariant (0.0 across batch
        # 4/8/32/64) and deterministic. That matters twice over:
        #   * every frame gets identical treatment regardless of where it landed in a batch, so a
        #     partial tail batch cannot give the last few frames of an episode different targets;
        #   * `selfcheck` can then be a hard equality instead of a tolerance.
        # The cost is a ~0.3% token difference from the exact kernels training used, which f_enc
        # turns into ~0.25% on z -- far below the fp16 storage quantisation, and compile is a
        # semantics-preserving transform anyway. `--dino-compile` restores the trainer's setting.
        self.dino_compile = dino_compile
        self.dino_extractor = DINOv3FeatureExtractor(       # _extract_dino_tokens
            model_name=get_dinov3_model_for_channels(self.dino_channels),
            use_compile=dino_compile,
        ).to(self.device)
        self.dino_extractor.eval()
        for param in self.dino_extractor.parameters():
            param.requires_grad = False

        enc_args = self.cfg.models.encoder.args
        self.num_tokens = int(enc_args.num_tokens)
        self.token_dim = int(enc_args.out_channels)

    @property
    def device(self) -> torch.device:
        """Fixed at construction.

        The base class derives this from `self.models` (`base.py:243`), which is empty while the
        models are still being built and moved, so it cannot be used during `__init__`. Overriding
        rather than assigning because the base declares it read-only.
        """
        return self._device

    # -- construction helpers -----------------------------------------------------------------

    def _build(self, name: str) -> torch.nn.Module:
        spec = self.cfg.models[name]
        return getattr(model_zoo, spec.name)(**spec.args).to(self.device)

    def _load(self, module: torch.nn.Module, name: str, run_dir: str, step: Optional[str]) -> None:
        path = self._ckpt_path(name, run_dir, step)
        module.load_state_dict(load_state_dict_file(path, self.device), strict=True)

    @staticmethod
    def _ckpt_path(name: str, run_dir: str, step: Optional[str]) -> Path:
        """Resolve a checkpoint. `step` is a *token*, not necessarily a number.

        The trainer writes milestone `{name}_step0037500.pt` files and, on its own cadence,
        resumable `{name}_step0037500.snapshot.pt` ones beside them, so the text between `_step`
        and `.pt` is only sometimes an integer. Tokens are compared as (number, suffix) via
        `parse_step_token`, so `--step 37500`, `--step 0037500` and `--step 37500.snapshot` all
        name a real file without the caller having to reproduce the zero padding.

        `step=None` takes the **highest-numbered** checkpoint, preferring the plain milestone when
        a snapshot sits at the same step -- unlike `fetch_state_dict`, which sorts the names as
        text and would hand back `...0037500.snapshot.pt` over `...0037500.pt`.
        """
        ckpt_dir = Path(run_dir) / "ckpts"
        available = sorted(ckpt_dir.glob(f"{name}_step*.pt"))
        if not available:
            raise FileNotFoundError(f"no {name}_step*.pt under {ckpt_dir}")

        if step is None:
            def rank(path: Path) -> Tuple[int, bool, str]:
                number, suffix = parse_step_token(ckpt_token(path, name))
                return number, suffix == "", suffix   # plain beats snapshot at the same step
            return max(available, key=rank)

        want = parse_step_token(str(step))
        for path in available:
            if parse_step_token(ckpt_token(path, name)) == want:
                return path
        tokens = ", ".join(ckpt_token(p, name) for p in available)
        raise FileNotFoundError(
            f"no checkpoint for --step {step!r} in {ckpt_dir}; available: {tokens}"
        )

    # -- the fast path ------------------------------------------------------------------------

    @torch.no_grad()
    def episode_tokens(
        self, rgbs: np.ndarray, batch_frames: int = 32
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """[N, Cam, 3, S, S] uint8 -> ([N, Cam, Lp, C] on device, (ph, pw)).

        Straight through the trainer's own `_extract_dino_tokens`, including its uint8 -> [0, 1]
        rescale, in frame batches so peak memory stays bounded for long episodes.

        The tokens stay in DINO's own output dtype -- **fp32**. `DINOv3FeatureExtractor.forward`
        runs the ViT under `autocast(fp16)`, but autocast keeps LayerNorm in fp32, so
        `last_hidden_state` comes back fp32 and that is what f_enc was trained on. Caching them as
        fp16 to halve memory shifts z by ~0.6% (max |d| = 6.1e-3 on a |z|max = 4.7 batch), which the
        self-check catches. ~220 MB per trajectory at N=140: the price of matching training exactly.

        LIBERO's foreground masks are a constant 255, so the trainer's `rgbs * (mask > 0)` is the
        identity and is skipped here; `selfcheck` re-proves that numerically on every run by going
        through `inference_batch`, which does apply the mask.

        **Every call is padded to exactly `batch_frames`.** A ViT treats images independently, so
        padding cannot change the rows we keep -- mathematically. It changes them in practice,
        because the kernels cuBLAS/SDPA select depend on the batch shape: measured on one 110-frame
        episode, frames 0..95 (full batches of 32) are bit-identical across `batch_frames` 8 and 32,
        while the 14 frames in the partial tail batch differ by 2.8e-2 on tokens of scale 3.33.
        Padding pins the shape, and two different `batch_frames` then agree to 0.00e+00 over the
        whole episode. Costs one partial batch of wasted compute per episode (~7% at the default 16)
        and buys a sidecar whose contents do not depend on how it was chunked.
        """
        num_frames = rgbs.shape[0]
        chunks: List[torch.Tensor] = []
        patch_hw: Tuple[int, int] = (0, 0)
        for lo in range(0, num_frames, batch_frames):
            block = rgbs[lo : lo + batch_frames]
            keep = block.shape[0]
            if keep < batch_frames:
                block = np.concatenate([block, np.repeat(block[-1:], batch_frames - keep, 0)], 0)
            batch = torch.from_numpy(block).to(self.device).float()
            tokens, patch_hw = self._extract_dino_tokens(batch)
            chunks.append(tokens[:keep])
        return torch.cat(chunks, dim=0), patch_hw

    @torch.no_grad()
    def episode_plucker(
        self,
        intrinsics: np.ndarray,
        cam2world: np.ndarray,
        patch_hw: Tuple[int, int],
        image_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """Per-frame Plücker rays via the trainer's `_plucker_rays` -> [N, Cam*Lp, 6].

        `_plucker_rays` indexes `data[...][:, 0]` (frame 0 of a sequence sample), so the singleton
        axis inserted here is the sequence axis, not a camera axis.
        """
        if not self._needs_plucker:
            return None
        data = {
            "intrinsics": torch.from_numpy(intrinsics)[:, None].to(self.device),  # [N,1,Cam,3,3]
            "w2c": torch.from_numpy(cam2world)[:, None].to(self.device),          # [N,1,Cam,4,4]
        }
        return self._plucker_rays(data, patch_hw=patch_hw, image_hw=image_hw)

    @torch.no_grad()
    def encode_pairs(
        self,
        flat_tokens: torch.Tensor,
        anchor_idx: torch.Tensor,
        future_idx: torch.Tensor,
        plucker: Optional[torch.Tensor],
        batch_pairs: int,
    ) -> torch.Tensor:
        """[P] index pairs into [N, Cam*Lp, C] tokens -> [P, Q, D] latent actions.

        Composition and the encoder call are the trainer's own `_compose_inverse_input` and
        `_forward_views`, so this differs from `inference_batch` only in reusing DINO tokens instead
        of recomputing them.
        """
        num_pairs = anchor_idx.numel()
        out = torch.empty(
            num_pairs, self.num_tokens, self.token_dim, device=self.device, dtype=torch.float32
        )
        for lo in range(0, num_pairs, batch_pairs):
            hi = min(lo + batch_pairs, num_pairs)
            anchors, futures = anchor_idx[lo:hi], future_idx[lo:hi]
            enc_input = self._compose_inverse_input(
                x_t_flat=flat_tokens[anchors], x_T_flat=flat_tokens[futures]
            )
            tokens, _ = self._forward_views(
                self.models["encoder"],
                enc_input,
                None,
                self.num_views,
                plucker=None if plucker is None else plucker[anchors],
            )
            if tokens.ndim != 3:
                raise ValueError(f"Encoder output must be rank-3 [B, Q, D], got {tokens.shape}")
            out[lo:hi] = tokens.float()
        return out

    @torch.no_grad()
    def encode_episode(
        self,
        tokens: torch.Tensor,
        plucker: Optional[torch.Tensor],
        chunk: int,
        batch_pairs: int,
        pairs_mode: str,
    ) -> torch.Tensor:
        """[N, Cam, Lp, C] -> [N, A, Q, D]."""
        num_frames, cams, lp, ch = tokens.shape
        if cams != self.num_views:
            raise ValueError(f"Expected {self.num_views} cameras, got {cams}")
        flat = tokens.reshape(num_frames, cams * lp, ch)

        if pairs_mode == "anchored":
            anchors, futures = anchored_pairs(num_frames, chunk, self.device)
            z = self.encode_pairs(flat, anchors.reshape(-1), futures.reshape(-1), plucker,
                                  batch_pairs)
            return z.reshape(num_frames, chunk, self.num_tokens, self.token_dim)

        # Consecutive latents are shareable across anchors (z_i(t) == z_0(t+i)), so only N-1 are
        # computed and then gathered into the same [N, A, Q, D] layout -- downstream cannot tell
        # the two modes apart from the file alone, which is why the manifest records `pairs`.
        num_base = max(num_frames - 1, 1)
        base_k = torch.arange(num_base, device=self.device)
        base = self.encode_pairs(
            flat, base_k, torch.clamp(base_k + 1, max=num_frames - 1), plucker, batch_pairs
        )
        t = torch.arange(num_frames, device=self.device)[:, None]
        i = torch.arange(chunk, device=self.device)[None, :]
        return base[torch.clamp(t + i, max=num_base - 1)]

    # -- the reference path -------------------------------------------------------------------

    def _reference_batch(
        self,
        rgbs: np.ndarray,
        intrinsics: np.ndarray,
        cam2world: np.ndarray,
        anchors: np.ndarray,
        futures: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """The `data` dict a TrajectoryDataset batch of (anchor, future) pairs would look like.

        Two frames per sample, in the order `inference_batch` reads them (`tokens_seq[:, 0]` is the
        anchor, `[:, 1]` the future). `foreground_masks` are the constant 255 the converter writes.
        """

        def pair(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.stack([array[anchors], array[futures]], axis=1))

        masks = np.full(
            (len(anchors), 2, rgbs.shape[1], rgbs.shape[3], rgbs.shape[4]), 255, dtype=np.uint8
        )
        return {
            "rgbs": pair(rgbs).float().to(self.device),
            "foreground_masks": torch.from_numpy(masks).to(self.device),
            "intrinsics": pair(intrinsics).to(self.device),
            "w2c": pair(cam2world).to(self.device),
        }

    @torch.no_grad()
    def selfcheck(
        self,
        rgbs: np.ndarray,
        intrinsics: np.ndarray,
        cam2world: np.ndarray,
        z_fast: torch.Tensor,
        chunk: int,
        num_pairs: int = 8,
    ) -> Tuple[float, float, float]:
        """(structural, end_to_end, |z|max) against the trainer's own `inference_batch`.

        Two differences, because they fail for different reasons and conflating them hides bugs:

        * **structural** feeds `inference_batch`'s *own* DINO tokens and Plücker rays through this
          class's encode call. It isolates everything this file is responsible for -- inverse-input
          composition, view count, and the (anchor, future) index order behind the chunk axis --
          and is exactly 0. This is the check that cannot be faked by a self-consistent bug, and
          the one that fails the run.
        * **end to end** compares the full fast path, which necessarily uses different batch shapes
          from an 8-pair reference call. It is *not* expected to be 0: measured on this tree,
          `f_enc` under `use_fp16: true` returns results that depend on the batch shape by up to
          8.0e-3 on |z|max 18.2 (0.04%) between pair-batches of 64 and 8. That is fp16 kernel
          selection, not an error -- it is smaller than the fp16 quantisation of `z` itself, which
          is one ulp ~ 7.8e-3 at that magnitude. So the bound below is relative and loose; its job
          is to catch a gross mistake such as encoding the wrong frames, not to police rounding.

        A single combined number would have reported 1.5e-1 on the first run and left it ambiguous
        whether the chunk index order was wrong or the ViT was merely shape-sensitive. It was the
        latter -- and chasing it down is what produced the DINO batch padding in `episode_tokens`.
        """
        num_frames = rgbs.shape[0]
        anchors_all, futures_all = anchored_pairs(num_frames, chunk, self.device)
        flat_a = anchors_all.reshape(-1).cpu().numpy()
        flat_f = futures_all.reshape(-1).cpu().numpy()
        # Spread the sampled pairs over the episode and over the chunk axis.
        picks = np.linspace(0, len(flat_a) - 1, min(num_pairs, len(flat_a))).astype(int)
        anchors, futures = flat_a[picks], flat_f[picks]

        data = self._reference_batch(rgbs, intrinsics, cam2world, anchors, futures)
        reference = self.inference_batch(self.models, data, training=False)["enc_tokens"].float()

        rgb_seq = self._get_masked_rgb_sequence(data)
        ref_tokens, ref_patch_hw = self._extract_dino_tokens_sequence(rgb_seq)
        ref_plucker = self._plucker_rays(
            data, patch_hw=ref_patch_hw, image_hw=rgb_seq.shape[-2:]
        )
        cams, lp, ch = ref_tokens.shape[2:]
        structural_z, _ = self._forward_views(
            self.models["encoder"],
            self._compose_inverse_input(
                x_t_flat=ref_tokens[:, 0].reshape(len(picks), cams * lp, ch),
                x_T_flat=ref_tokens[:, 1].reshape(len(picks), cams * lp, ch),
            ),
            None,
            self.num_views,
            plucker=ref_plucker,
        )

        mine = z_fast.reshape(-1, self.num_tokens, self.token_dim)[picks]
        return (
            float((structural_z.float() - reference).abs().max().item()),
            float((mine - reference).abs().max().item()),
            float(reference.abs().max().item()),
        )

    # -- visual verification ------------------------------------------------------------------

    @torch.no_grad()
    def filmstrip(
        self,
        rgbs: np.ndarray,
        intrinsics: np.ndarray,
        cam2world: np.ndarray,
        z_saved: np.ndarray,
        anchor: int,
        out_path: Path,
        note: str,
    ) -> str:
        """One panel: ground-truth frames t..t+A over the reconstruction decoded from saved `z`.

            GT     [  t  ][ t+1 ][ t+2 ] ... [ t+A ]      raw pixels, agentview | wrist per cell
            RECON  [ ---- ][ z_0 ][ z_1 ] ... [z_A-1]     image_decoder(f_dec(x_t, z_i))

        `z` is read back from disk, so a wrong chunk-axis order, a bad fp16 cast or a scrambled
        camera order all show up here. The bottom row should track the top row one step at a time;
        if it lags or leads, the chunk axis is off by one.

        Returns a one-line summary of the same question answered numerically rather than by eye:
        for each `i`, which ground-truth frame is `recon(z_i)` actually closest to? It should be
        `t+i+1` for every `i`. Eyeballing a filmstrip catches a gross scramble but not an off-by-one
        at the tail, where consecutive frames look nearly identical; the argmin does.
        """
        chunk = z_saved.shape[0]
        num_frames = rgbs.shape[0]
        futures = [min(anchor + i + 1, num_frames - 1) for i in range(chunk)]

        anchor_tokens, patch_hw = self.episode_tokens(rgbs[anchor : anchor + 1])
        cams, lp, ch = anchor_tokens.shape[1:]
        x0 = anchor_tokens.reshape(1, cams * lp, ch).repeat(chunk, 1, 1)

        plucker = self.episode_plucker(
            intrinsics[anchor : anchor + 1],
            cam2world[anchor : anchor + 1],
            patch_hw,
            rgbs.shape[-2:],
        )
        if plucker is not None:
            plucker = plucker.repeat(chunk, 1, 1)

        latents = torch.from_numpy(z_saved).float().to(self.device)  # [A, Q, D], as written
        _, pred_flat = self._forward_views(
            self.models["decoder"], x0, latents, self.num_views, plucker=plucker
        )
        recon = self._decode_tokens_to_vis(
            pred_flat.reshape(chunk, cams, lp, ch), patch_hw
        )  # [A, 3, H, Cam*W]

        gt_frames = torch.from_numpy(rgbs[[anchor] + futures]).to(self.device)  # [A+1, Cam, 3, S, S]
        gt = self._stitch_cameras(self._normalize_rgb_for_lpips(gt_frames))     # [A+1, 3, S, Cam*S]
        if gt.shape[-2:] != recon.shape[-2:]:
            raise RuntimeError(
                f"ground truth {tuple(gt.shape[-2:])} and reconstruction {tuple(recon.shape[-2:])} "
                "differ in size; --img-size must match the image decoder's patch upsampling"
            )

        top = torch.cat(list(gt), dim=-1)
        bottom = torch.cat([torch.zeros_like(recon[0])] + list(recon), dim=-1)
        panel = torch.cat([top, bottom], dim=-2).clamp(0, 1).cpu()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_with_notes(panel, str(out_path), notes=note)

        # Which real frame is each reconstruction actually closest to? Search a window either side
        # of the target so a systematic lead or lag is visible, not just a hit or miss.
        lags = list(range(-2, chunk + 4))
        candidates = self._stitch_cameras(
            self._normalize_rgb_for_lpips(
                torch.from_numpy(
                    rgbs[[min(max(anchor + lag, 0), num_frames - 1) for lag in lags]]
                ).to(self.device)
            )
        )
        best = [
            lags[int((recon[i : i + 1] - candidates).abs().mean(dim=(1, 2, 3)).argmin())]
            for i in range(chunk)
        ]
        offsets = [lag - (i + 1) for i, lag in enumerate(best)]
        hits = offsets.count(0)

        # Two quantities decide whether a non-zero offset means anything. `motion` is how much the
        # scene actually changes per step over this window; `resid` is how well f_dec reconstructs
        # its own target. When resid >= motion the argmin is picking between frames that differ by
        # less than the reconstruction error, so it carries no information either way.
        motion = float((gt[1:] - gt[:-1]).abs().mean().item())
        resid = float((recon - gt[1:]).abs().mean().item())

        # An off-by-one in the chunk axis is a *constant* offset -- every z_i would point at the
        # same wrong distance. Offsets that vary, and grow with i, are f_dec hedging toward the
        # anchor under fast motion: a reconstruction-quality property of the checkpoint, not an
        # indexing bug. Distinguishing these is the difference between "stop, the targets are
        # wrong" and "this f_enc/f_dec pair is still undertrained".
        constant = offsets[0] if len(set(offsets)) == 1 else None
        if constant not in (None, 0):
            verdict = f"CHUNK AXIS OFF BY {constant:+d} -- every step shifted alike"
        elif hits == chunk:
            verdict = "exact"
        elif resid >= motion:
            verdict = "unresolvable: recon error >= per-step motion, argmin is noise"
        else:
            verdict = "offsets vary -- f_dec undershoots fast motion, not an index shift"
        return (
            f"chunk alignment {hits}/{chunk}, offsets {offsets}, "
            f"recon err {resid:.3f} vs per-step motion {motion:.3f} -> {verdict}"
        )


# --------------------------------------------------------------------------------------------
# RLDS reading
# --------------------------------------------------------------------------------------------


def anchored_pairs(
    num_frames: int, chunk: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """([N, A] anchors, [N, A] futures) with future = min(t + i + 1, N - 1).

    The clamp matches `chunk_act_obs`'s own
    `floored_action_chunk_indices = min(max(idx, 0), traj_len - 1)`, so the targets run off the end
    of a trajectory exactly the way the action chunk does. (There are N frames but only N-1 real
    transitions: the last action leads nowhere.)
    """
    t = torch.arange(num_frames, device=device)[:, None].expand(num_frames, chunk)
    i = torch.arange(chunk, device=device)[None, :]
    return t.contiguous(), torch.clamp(t + i + 1, max=num_frames - 1)


def decode_episode(episode) -> Dict[str, np.ndarray]:
    """One RLDS episode -> stacked arrays. N frames <-> N actions <-> N states, verified."""
    frames: Dict[str, List[np.ndarray]] = {cam: [] for cam in CAMERA_ORDER}
    states: List[np.ndarray] = []
    instruction = ""
    for step in episode["steps"]:
        obs = step["observation"]
        for cam in CAMERA_ORDER:
            frames[cam].append(obs[RLDS_IMAGE_KEY[cam]].numpy())
        states.append(obs["state"].numpy())
        if not instruction:
            instruction = step["language_instruction"].numpy().decode("utf-8")

    out: Dict[str, np.ndarray] = {cam: np.stack(frames[cam]) for cam in CAMERA_ORDER}
    out["state"] = np.stack(states).astype(np.float32)
    out["instruction"] = instruction
    return out


def prepare_views(
    episode: Dict[str, np.ndarray], img_size: int, camera_params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize both views and build the matching per-frame K / camera-to-world.

    Resizing goes through `datalib.dataset.resize_image` (cv2 INTER_LINEAR) and `resize_intrinsics`
    because that is the exact path `read_frames` takes, i.e. what the autoencoder trained on. Do NOT
    substitute `tf.image.resize` (lanczos3), which is what VLA-Adapter's eval preprocessing uses.

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

    rgbs, intrinsics, cam2world = [], [], []
    for cam in CAMERA_ORDER:
        resized = np.stack([resize_image(frame, img_size) for frame in episode[cam]])
        rgbs.append(resized.transpose(0, 3, 1, 2))  # [N, 3, S, S]
        intrinsics.append(
            resize_intrinsics(cams[f"{cam}_intrinsics"][:, 0], orig_h, orig_w, img_size, img_size)
        )
        # TrajectoryDataset stores camera-to-world under the key `w2c` and never inverts it;
        # compute_plucker_rays expects camera-to-world, so pass it straight through.
        cam2world.append(convert_extrinsics_3x4_to_4x4(cams[f"{cam}_extrinsics"][:, 0]))

    return (
        np.stack(rgbs, axis=1),
        np.stack(intrinsics, axis=1).astype(np.float32),
        np.stack(cam2world, axis=1).astype(np.float32),
    )


def rank_slice(total: int, rank: int, world: int) -> Tuple[int, int]:
    """Contiguous [lo, hi) slice for this rank, so a worker never decodes what it does not own."""
    return total * rank // world, total * (rank + 1) // world


# --------------------------------------------------------------------------------------------
# Statistics and index
# --------------------------------------------------------------------------------------------


def write_json(path: Path, payload) -> None:
    """Atomic-ish write, so an interrupt cannot leave a half-written index behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def merge_keys(out_root: Path, suite: str) -> int:
    """Fold every rank's partial index into `<suite>/keys.json`.

    Shards from earlier runs are kept and merged rather than cleared, so a sidecar built up over
    several invocations (a different `--gpus` count, a raised `--limit`) still has a complete index.
    Entries whose file is no longer on disk are dropped: a stale shard pointing at a deleted episode
    would otherwise become a load error inside the tf.data graph at training time, where it is far
    harder to diagnose.
    """
    suite_dir = out_root / suite
    merged: Dict[str, dict] = {}
    dropped = 0
    for shard in sorted(suite_dir.glob("_keys_rank*.json")):
        for key, entry in json.loads(shard.read_text()).items():
            if not (suite_dir / entry["file"]).exists():
                dropped += 1
                continue
            existing = merged.get(key)
            if existing is not None and existing["file"] != entry["file"]:
                raise RuntimeError(
                    f"[{suite}] state hash collision: {key} maps to both {existing['file']} "
                    f"and {entry['file']}"
                )
            merged[key] = entry
    if dropped:
        print(f"  [{suite}] dropped {dropped} index entries with no file on disk")
    if merged:
        write_json(suite_dir / "keys.json", merged)
    return len(merged)


def episode_files(out_root: Path) -> List[Path]:
    return sorted([*out_root.glob("*/ep_*.npy"), *out_root.glob("*/ep_*.npz")])


def load_latents(path: Path) -> np.ndarray:
    """Read one episode's `(N, A, Q, D)` fp16 block, plain or DEFLATE-compressed."""
    if path.suffix == ".npz":
        with np.load(path) as handle:
            return handle["z"]
    return np.load(path)


def save_latents(path: Path, z: np.ndarray, compress: bool) -> None:
    """fp16 on disk either way; `compress` picks DEFLATE inside a zip container.

    `np.savez_compressed` names the array `z`, which is what `load_latents` and the sidecar reader
    look for. Written to a temp name first so an interrupt cannot leave a truncated episode that a
    later resume would happily skip over as "already done".
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    if compress:
        np.savez_compressed(tmp, z=z)
        tmp = tmp.with_suffix(tmp.suffix + ".npz")  # savez appends .npz to the given name
    else:
        np.save(tmp, z)
        tmp = tmp.with_suffix(tmp.suffix + ".npy")
    tmp.replace(path)


def compute_stats(out_root: Path) -> dict:
    """Per-(q, d) mean/std over every episode file under `out_root`, accumulated in float64.

    A final pass over the files rather than an accumulator threaded through extraction, because
    extraction skips episodes that already exist: an inline accumulator would silently compute the
    statistics from only the episodes a *resumed* run happened to write, and every target would then
    be normalised against the wrong mean and std.
    """
    files = episode_files(out_root)
    if not files:
        raise RuntimeError(f"no */ep_* files under {out_root}; nothing to compute statistics over")

    count = 0
    total = total_sq = None
    for idx, path in enumerate(files):
        array = load_latents(path)
        flat = array.reshape(-1, *array.shape[-2:]).astype(np.float64)
        if total is None:
            total = np.zeros(flat.shape[1:], dtype=np.float64)
            total_sq = np.zeros_like(total)
        count += flat.shape[0]
        total += flat.sum(axis=0)
        total_sq += (flat**2).sum(axis=0)
        if idx % 200 == 0:
            print(f"  stats {idx + 1}/{len(files)}", flush=True)

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    print(f"  stats over {len(files)} episodes, {count} rows")
    return {
        "count": int(count),
        "mean": mean.tolist(),
        "std": std.tolist(),
        # Floor the std so a dead latent slot cannot blow up the normalised target.
        "std_floor": 1e-3,
        "global_rms": float(np.sqrt(total_sq.sum() / (count * mean.size))),
    }


# --------------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------------


def extract_suite(
    suite: str,
    extractor: RlaSidecarExtractor,
    args: argparse.Namespace,
    camera_params: dict,
    tag: str,
) -> Tuple[int, int]:
    rlds_dir = REPO_ROOT / args.rlds_root / SUITES[suite] / "1.0.0"
    if not rlds_dir.is_dir():
        print(f"{tag} [skip] {suite}: {rlds_dir} does not exist")
        return 0, 0

    out_dir = Path(args.out) / suite
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(str(rlds_dir))
    total = builder.info.splits["train"].num_examples
    if args.limit > 0:
        total = min(total, args.limit)
    lo, hi = rank_slice(total, args._rank, args._world)
    if lo >= hi:
        print(f"{tag} [{suite}] nothing to do (0 of {total} episodes)")
        return 0, 0
    print(f"{tag} [{suite}] episodes [{lo}, {hi}) of {total} -> {out_dir}", flush=True)

    keys: Dict[str, dict] = {}
    keys_path = out_dir / f"_keys_rank{args._rank}.json"
    viz_done = 0
    written = 0
    started = time.time()

    for local_idx, raw in enumerate(
        builder.as_dataset(split=f"train[{lo}:{hi}]", shuffle_files=False)
    ):
        ep_idx = lo + local_idx
        rel = f"ep_{ep_idx:06d}.{'npz' if args.compress else 'npy'}"
        target = out_dir / rel

        episode = decode_episode(raw)
        num_frames = len(episode["state"])
        keys[episode_key(episode["state"])] = {
            "file": rel,
            "episode_index": ep_idx,
            "traj_len": num_frames,
        }

        want_viz = args.viz > 0 and args._rank == 0 and viz_done < args.viz
        if target.exists() and not args.overwrite and not want_viz:
            continue

        rgbs, intrinsics, cam2world = prepare_views(episode, args.img_size, camera_params)

        if target.exists() and not args.overwrite:
            z_np = load_latents(target)  # viz only: keep the artifact on disk authoritative
        else:
            tokens, patch_hw = extractor.episode_tokens(rgbs, args.dino_batch)
            plucker = extractor.episode_plucker(
                intrinsics, cam2world, patch_hw, (args.img_size, args.img_size)
            )
            z = extractor.encode_episode(
                tokens, plucker, args.chunk, args.batch_pairs, args.pairs
            )
            if args.selfcheck and written == 0 and args.pairs == "anchored":
                structural, end_to_end, scale = extractor.selfcheck(
                    rgbs, intrinsics, cam2world, z, args.chunk
                )
                # Only `structural` is exact. `end_to_end` also carries fp16 batch-shape
                # sensitivity, so its bound is relative and generous -- 2% of |z|, against a
                # measured ~0.1%. See RlaSidecarExtractor.selfcheck.
                e2e_limit = max(1e-3, 0.02 * scale)
                ok = structural < 1e-5 and end_to_end < e2e_limit
                print(
                    f"{tag} [{suite}] selfcheck vs inference_batch: structural {structural:.2e} "
                    f"(exact), end-to-end {end_to_end:.2e} = {100 * end_to_end / max(scale, 1e-9):.2f}% "
                    f"of |z|max {scale:.1f}  {'OK' if ok else 'FAIL'}",
                    flush=True,
                )
                if not ok:
                    raise RuntimeError(
                        "fast path disagrees with RlaAutoencoderMultiViewTrainer.inference_batch "
                        f"(structural {structural:.3e}, end-to-end {end_to_end:.3e}); a non-zero "
                        "structural term means this file composes or indexes the encoder inputs "
                        "differently from the trainer, and the targets would be wrong"
                    )
            z_np = z.detach().float().cpu().numpy().astype(np.float16)
            save_latents(target, z_np, args.compress)
            written += 1
            del tokens, plucker, z

        if want_viz:
            # Spread anchors over the trajectory rather than always sampling the same phase: the
            # approach, the contact and the retreat stress different things, and a bug that only
            # shows up while the gripper is closing would be invisible in three mid-episode frames.
            # Stay `chunk` frames clear of the end so the targets are not all the clamped last one.
            frac = (viz_done + 1) / (args.viz + 1)
            anchor = min(int(num_frames * frac), max(num_frames - 1 - args.chunk, 0))
            summary = extractor.filmstrip(
                rgbs,
                intrinsics,
                cam2world,
                z_np[anchor].astype(np.float32),
                anchor,
                Path(args.out) / "viz" / f"{suite}_ep{ep_idx:06d}_t{anchor:04d}.jpg",
                note=f"{suite} ep{ep_idx} t={anchor} | top: GT t..t+{args.chunk} | "
                f"bottom: f_dec(x_t, z_i) from {rel}",
            )
            print(f"{tag} [{suite}] ep{ep_idx} t={anchor}: {summary}", flush=True)
            viz_done += 1

        if local_idx % 25 == 0 or ep_idx == hi - 1:
            rate = (local_idx + 1) / max(time.time() - started, 1e-6)
            print(
                f"{tag} [{suite}] {ep_idx - lo + 1}/{hi - lo}  N={num_frames}  "
                f"z={tuple(z_np.shape)}  {rate:.2f} ep/s",
                flush=True,
            )
        if local_idx % 50 == 0:
            torch.cuda.empty_cache()
            # Flush the index as we go: a run interrupted before the end would otherwise leave
            # .npy files with no index to join them by -- unusable, and invisible to the reader.
            write_json(keys_path, keys)

    write_json(keys_path, keys)
    print(f"{tag} [{suite}] wrote {written} new episodes, indexed {len(keys)}", flush=True)
    return len(keys), written


def worker_main(args: argparse.Namespace) -> int:
    tag = f"[rank {args._rank}]" if args._world > 1 else ""
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("no CUDA device visible: DINOv3 + f_enc need a GPU")

    want_viz = args.viz > 0 and args._rank == 0
    extractor = RlaSidecarExtractor(
        config_path=args.rla_config,
        run_dir=args.rla_run,
        step=args.step,
        device=args.device,
        need_decoder=args.selfcheck or want_viz,
        need_image_decoder=want_viz,
        dino_compile=args.dino_compile,
    )
    camera_params = load_camera_params(args.camera_params)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for suite in args.suites:
        extract_suite(suite, extractor, args, camera_params, tag)
    return 0


def viz_only_main(args: argparse.Namespace) -> int:
    """Render filmstrips from an existing sidecar, without re-extracting anything."""
    root = Path(args.viz_only)
    manifest = json.loads((root / "manifest.json").read_text())
    args.chunk = int(manifest["chunk"])
    args.img_size = int(manifest["img_size"])
    print(f"[viz] {root}  chunk={args.chunk}  pairs={manifest['pairs']}  {manifest['checkpoint']}")

    extractor = RlaSidecarExtractor(
        config_path=args.rla_config,
        run_dir=args.rla_run,
        step=args.step,
        device=args.device,
        need_decoder=True,
        need_image_decoder=True,
        dino_compile=args.dino_compile,
    )
    camera_params = load_camera_params(args.camera_params)

    rendered = 0
    for suite in args.suites:
        keys_path = root / suite / "keys.json"
        if not keys_path.exists():
            continue
        wanted = {
            entry["episode_index"]: entry
            for entry in json.loads(keys_path.read_text()).values()
        }
        # `--viz N` is per suite here, matching what it means during extraction.
        picks = sorted(wanted)[:: max(len(wanted) // max(args.viz, 1), 1)][: args.viz]
        rlds_dir = REPO_ROOT / args.rlds_root / SUITES[suite] / "1.0.0"
        builder = tfds.builder_from_directory(str(rlds_dir))
        for slot, ep_idx in enumerate(picks):
            raw = next(iter(builder.as_dataset(split=f"train[{ep_idx}:{ep_idx + 1}]")))
            episode = decode_episode(raw)
            rgbs, intrinsics, cam2world = prepare_views(episode, args.img_size, camera_params)
            z_np = load_latents(root / suite / wanted[ep_idx]["file"])
            num_frames = len(episode["state"])
            frac = (slot + 1) / (args.viz + 1)                  # spread, as in extract_suite
            anchor = min(int(num_frames * frac), max(num_frames - 1 - args.chunk, 0))
            summary = extractor.filmstrip(
                rgbs,
                intrinsics,
                cam2world,
                z_np[anchor].astype(np.float32),
                anchor,
                root / "viz" / f"{suite}_ep{ep_idx:06d}_t{anchor:04d}.jpg",
                note=f"{suite} ep{ep_idx} t={anchor} | top: GT t..t+{args.chunk} | "
                f"bottom: f_dec(x_t, z_i) from {wanted[ep_idx]['file']}",
            )
            rendered += 1
            print(f"  {suite} ep{ep_idx} t={anchor}: {summary}", flush=True)
    print(f"[viz] {rendered} filmstrips -> {root / 'viz'}")
    return 0


# --------------------------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------------------------


def spawn_workers(args: argparse.Namespace, gpus: Sequence[int]) -> None:
    """One subprocess per GPU. No torch.distributed: extraction needs no collectives, and NCCL
    bootstrap is a liability on hosts whose NCCL_SOCKET_IFNAME does not exist."""
    passthrough = [
        "--rla-config", args.rla_config,
        "--rla-run", args.rla_run,
        "--out", args.out,
        "--rlds-root", args.rlds_root,
        "--camera-params", args.camera_params,
        "--suites", *args.suites,
        "--chunk", str(args.chunk),
        "--img-size", str(args.img_size),
        "--pairs", args.pairs,
        "--batch-pairs", str(args.batch_pairs),
        "--dino-batch", str(args.dino_batch),
        "--limit", str(args.limit),
        "--viz", str(args.viz),
        # Always explicit: main() resolved "latest" once, and a worker must not re-resolve it.
        "--step", str(args.step),
    ]
    if args.overwrite:
        passthrough.append("--overwrite")
    if not args.selfcheck:
        passthrough.append("--no-selfcheck")
    if args.compress:
        passthrough.append("--compress")
    if args.dino_compile:
        passthrough.append("--dino-compile")

    procs = []
    for rank, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, str(Path(__file__).resolve()), *passthrough,
               "--_rank", str(rank), "--_world", str(len(gpus))]
        print(f"[parent] rank {rank} -> GPU {gpu}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT)))

    failures = [rank for rank, proc in enumerate(procs) if proc.wait() != 0]
    if failures:
        raise SystemExit(f"[parent] ranks {failures} failed; not writing stats/manifest")


def finalize(args: argparse.Namespace, world: int, elapsed: float) -> None:
    out_root = Path(args.out)
    per_suite = {}
    for suite in args.suites:
        if (out_root / suite).is_dir():
            num_keys = merge_keys(out_root, suite)
            if num_keys:
                per_suite[suite] = {"num_episodes": num_keys}

    print("\ncomputing normalisation statistics")
    write_json(out_root / "stats.json", compute_stats(out_root))

    ckpt = RlaSidecarExtractor._ckpt_path("encoder", args.rla_run, args.step)
    # Hashed in main() before the workers started, not here: with `save_latest_only: true` a run
    # that is still training deletes this file the moment it saves the next step, which can happen
    # inside the half hour this extraction takes. The weights themselves are safe -- every rank
    # loaded them into memory at startup -- but hashing at the end would either crash on a missing
    # file or, worse, describe a different checkpoint than the one that produced the latents.
    cfg = edict(load_config(args.rla_config))
    write_json(
        out_root / "manifest.json",
        {
            "rla_config": args.rla_config,
            "rla_config_sha1": hashlib.sha1(Path(args.rla_config).read_bytes()).hexdigest(),
            "rla_run": args.rla_run,
            "checkpoint": str(ckpt),
            # save_latest_only=true means `checkpoint` is overwritten as training continues; the
            # hash is what actually identifies which f_enc produced this sidecar.
            "checkpoint_sha256": getattr(args, "ckpt_sha256", None) or sha256_file(ckpt),
            "random_encoder": False,
            "chunk": args.chunk,
            "num_tokens": int(cfg.models.encoder.args.num_tokens),
            "token_dim": int(cfg.models.encoder.args.out_channels),
            "img_size": args.img_size,
            "pairs": args.pairs,
            # "npz" holds the same fp16 (N, A, Q, D) block under the key `z`, DEFLATE-compressed.
            # A reader must branch on this -- see the "Reading a compressed sidecar" note above.
            "format": "npz" if args.compress else "npy",
            "cameras": list(CAMERA_ORDER),
            "inverse_input_mode": str(cfg.trainer.args.get("inverse_input_mode", "sub")).lower(),
            "dino_channels": int(cfg.trainer.args.get("dino_channels", 1024)),
            # DINO output depends on the batch shape, so these two pin reproducibility.
            "dino_batch": args.dino_batch,
            "dino_compile": args.dino_compile,
            "rlds_root": args.rlds_root,
            "world_size": world,
            "elapsed_sec": round(elapsed, 1),
            "suites": per_suite,
        },
    )
    files = episode_files(out_root)
    on_disk = sum(f.stat().st_size for f in files)
    ratio = ""
    if args.compress:
        sample = files[: min(len(files), 32)]
        raw = sum(int(np.prod(load_latents(f).shape)) * 2 for f in sample)
        ratio = f", {raw / max(sum(f.stat().st_size for f in sample), 1):.2f}x vs raw fp16"
    print(f"\ndone -> {out_root}  ({elapsed / 60:.1f} min)")
    print(f"{len(files)} episode files, {on_disk / 2**30:.2f} GiB{ratio}")
    if args.compress:
        print("format=npz: a reader must use np.load(path)['z'] -- see the module docstring")
    if args.viz > 0:
        print(f"inspect {out_root / 'viz'}: the bottom row should track the top row one step at "
              "a time; a lag or lead means the chunk axis is off by one")


def default_out_dir(args: argparse.Namespace) -> Path:
    """`data/libero_rla/<Q>x<D>-a<A>-step<step>` -- every knob that changes the *contents*.

    A sidecar is only meaningful next to the chunk size it was built for: `A` is the second axis of
    every `ep_*.npy`, and `RlaTargetStore` refuses to load one whose `manifest["chunk"]` disagrees
    with `NUM_ACTIONS_CHUNK`. Encoding A (and the latent shape, and the checkpoint step, which
    `save_latest_only` otherwise erases) in the directory name means the several sidecars that
    accumulate on disk stay tellable apart without opening a manifest. Non-default `--pairs` and
    `--img-size` are appended only when set, so the common name stays short.
    """
    cfg = edict(load_config(args.rla_config))
    number, suffix = parse_step_token(resolve_step_token(args))
    name = (f"{int(cfg.models.encoder.args.num_tokens)}x"
            f"{int(cfg.models.encoder.args.out_channels)}-a{args.chunk}-step{number:06d}")
    if suffix:                                  # ".snapshot" -> "-snapshot"
        name += "-" + suffix.strip(".").replace(".", "-")
    if args.pairs != "anchored":
        name += f"-{args.pairs}"
    if args.img_size != 224:
        name += f"-img{args.img_size}"
    return Path("data/libero_rla") / name


def resolve_step_token(args: argparse.Namespace) -> str:
    """The canonical token of the checkpoint this run will load, e.g. `0037500.snapshot`.

    Goes through `_ckpt_path`, so `--step 37500` comes back as the padded `0037500` the file
    actually carries, and `--step None` comes back as whichever checkpoint "latest" resolved to.
    Feeding *that* to the workers is what pins them all to one f_enc.
    """
    return ckpt_token(RlaSidecarExtractor._ckpt_path("encoder", args.rla_run, args.step), "encoder")


def check_compatible(args: argparse.Namespace) -> None:
    """Refuse to mix incompatible extractions into one sidecar directory."""
    manifest_path = Path(args.out) / "manifest.json"
    if not manifest_path.exists() or args.overwrite:
        return
    old = json.loads(manifest_path.read_text())
    for field, value in (("chunk", args.chunk), ("pairs", args.pairs),
                         ("img_size", args.img_size),
                         ("format", "npz" if args.compress else "npy")):
        if old.get(field) != value:
            raise SystemExit(
                f"{manifest_path} was written with {field}={old.get(field)!r}, but this run uses "
                f"{value!r}. Drop --out to get the auto-named directory for this configuration, "
                f"or pass --overwrite to rebuild this one."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rla-config",
                        default="rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml")
    parser.add_argument("--rla-run", default=None, help="RLA autoencoder run dir holding ckpts/")
    parser.add_argument("--step", default=None,
                        help="checkpoint step token: 37500, 0037500 or 37500.snapshot "
                             "(default: the highest-numbered one, milestone over snapshot)")
    parser.add_argument("--out", default=None,
                        help="sidecar root; default data/libero_rla/<Q>x<D>-a<chunk>-step<step>")
    parser.add_argument("--rlds-root", default="data/libero")
    parser.add_argument("--camera-params", default=CAMERA_PARAMS_DEFAULT)
    parser.add_argument("--suites", nargs="+", default=["all"], choices=["all", *SUITES])
    parser.add_argument("--chunk", type=int, default=8, help="A; must equal NUM_ACTIONS_CHUNK")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--pairs", choices=["anchored", "consecutive"], default="anchored")
    parser.add_argument("--batch-pairs", type=int, default=64, help="encoder pairs per forward")
    parser.add_argument("--dino-batch", type=int, default=16,
                        help="frames per DINO forward; batches are padded to exactly this, so it "
                             "trades wasted tail compute against peak memory, not accuracy")
    parser.add_argument("--compress", action="store_true",
                        help="DEFLATE-compress episodes into .npz. Measured 1.12x on real fp16 "
                             "latents, so off by default: not worth the CPU or the reader change.")
    parser.add_argument("--dino-compile", action="store_true",
                        help="torch.compile the ViT, as the trainer does. Makes DINO output "
                             "batch-shape dependent (~0.3%% on tokens) and the self-check "
                             "inexact; off by default for reproducible targets.")
    parser.add_argument("--limit", type=int, default=-1, help="only the first N episodes per suite")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viz", type=int, default=4, help="filmstrips to render (rank 0 only)")
    parser.add_argument("--viz-only", default=None, help="render filmstrips for an existing sidecar")
    parser.add_argument("--no-selfcheck", dest="selfcheck", action="store_false",
                        help="skip the inference_batch equality check (it costs ~5 s per suite)")
    parser.add_argument("--overwrite", action="store_true", help="re-extract existing episodes")
    parser.add_argument("--_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_world", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if "all" in args.suites:
        args.suites = list(SUITES)
    if not args.rla_run:
        parser.error("--rla-run is required")
    return args


def main() -> int:
    args = parse_args()
    if args.viz_only:
        return viz_only_main(args)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus must name at least one device")

    started = time.time()
    if args._world > 1:
        return worker_main(args)  # a spawned worker; the parent finalises once every rank is done

    # Pin the checkpoint here, in the parent, and pass the resolved token down explicitly. `--step
    # None` means "latest", and with `save_latest_only: true` the latest can change *while this
    # runs* -- two ranks starting either side of a save would otherwise encode with two different
    # f_enc, and the manifest would name only one of them.
    args.step = resolve_step_token(args)
    args.ckpt_sha256 = sha256_file(
        RlaSidecarExtractor._ckpt_path("encoder", args.rla_run, args.step)
    )  # now, while the file is certainly the one the workers are about to load -- see finalize()

    if not args.out:
        args.out = str(default_out_dir(args))
        print(f"[parent] --out not given, writing to {args.out}")
    elif f"a{args.chunk}" not in Path(args.out).name:
        print(f"[parent] note: --out {Path(args.out).name!r} does not name the chunk size; this "
              f"sidecar is only usable with NUM_ACTIONS_CHUNK={args.chunk}")

    check_compatible(args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if len(gpus) == 1:
        worker_main(args)  # single-GPU debug path -- same worker code, no subprocess
    else:
        spawn_workers(args, gpus)

    finalize(args, max(len(gpus), args._world), time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
