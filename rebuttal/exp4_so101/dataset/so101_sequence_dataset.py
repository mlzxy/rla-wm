"""`ManiSkillSequenceDataset` for the SO-101 policies. Three additions, all needed together.

1. A real observation **history** (`n_obs_steps > 1`)
--------------------------------------------------
`VLABCPolicyRLAUnified._forward_trunk` reads `nobs["image"][:, :n_obs_steps]` and
`nobs["state"][:, :n_obs_steps]`, and `predict_action` treats index `n_obs_steps - 1` as **now**.
The stock `_compute_in_horizon_image_positions` spreads extra frames evenly across the **horizon**::

    total = 1 + num_in_horizon_frames_after_first
    return np.linspace(0, self.horizon - 1, total)      # horizon=32, total=2 -> [0, 31]

so a naive `n_obs_steps=2` delivers ``image[1] = frame t + 31`` -- a frame 1.03 s in the *future*,
unobtainable on the robot. Here the slots are placed at ``now - k * obs_history_stride`` and may
reach back before the window start (down to the episode start), and the joint-state history is
re-gathered at the same instants. With ``obs_history_stride = n_action_steps``::

    slots:   [ t - 16 ,   t  ]        images AND states, same instants
              history    now          = the previous replan's observation, free at deploy time

Getting only the images right would be worse than doing nothing: the policy would fuse an image
from t-16 with a joint reading from t-1, an observation that never existed.

2. **Precomputed** RLA latent-action targets
--------------------------------------------
`rebuttal/exp4_so101/precompute_rla_latents.py` writes, for every frame, the latent action at each
precomputed gap. This dataset selects the ones `rla_gaps` asks for -- by gap VALUE, so dense and
sparse sidecars behave identically -- and returns them as a top-level `sample["rla_latents"]`, shaped
``(n_gaps, num_tokens, token_dim)`` for the *current* frame (slot `n_obs_steps - 1`).

That removes DINOv3 and the RLA encoder from the training loop entirely, which is what makes
augmentation safe: the target was computed once from clean frames and cannot be corrupted by
whatever we do to the policy's input pixels.

3. Colour / lighting **augmentation**
-------------------------------------
Applied to the policy's input images only, per (sample, camera) and shared across the observation
slots -- a camera's exposure and white balance are stable over the 0.53 s history window but differ
between the front and wrist cameras. Disabled automatically on the validation split.

**Augmentation is only sound when the latent targets are precomputed.** With online targets the
same augmented tensor feeds both the policy and the RLA encoder, so the target moves with the
augmentation; this class raises if you ask for that combination.
"""

import copy
import json
import os

import numpy as np
import torch

from policies.dataset.maniskill_sequence_dataset import ManiSkillSequenceDataset
from rebuttal.exp4_so101.ckpt_tokens import same_step


class PhotometricAugment:
    """Colour / lighting augmentation for `(..., 3, H, W)` images in [0, 1].

    Aimed at a workspace lit through a window, where the appearance drift between a recording
    session and an eval session is dominated by *daylight*, not by camera settings:

      * overall level swings with cloud cover and time of day  -> `brightness`, `contrast`
      * daylight colour temperature swings from ~5000 K (warm, low sun) to ~7000 K+ (cool,
        overcast)                                              -> `color_temp`
      * direct sun puts a smooth bright-to-dark ramp across the scene, and the direction of that
        ramp rotates through the day                           -> `illumination_gradient`
      * low light means more sensor gain                        -> `noise`

    `color_temp` and `illumination_gradient` are the two that matter for a window and that a stock
    `ColorJitter` does not model: a global hue rotation is not the same as a warm/cool shift, and
    nothing in ColorJitter is spatially varying.

    One transform is sampled per call and applied to the whole tensor, so passing all observation
    slots of one camera at once gives a temporally consistent result.
    """

    def __init__(
        self,
        brightness: float = 0.35,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.03,
        color_temp: float = 0.12,
        illumination_gradient: float = 0.25,
        noise: float = 0.01,
        gray_prob: float = 0.0,
        seed: int | None = None,
    ):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.color_temp = float(color_temp)
        self.illumination_gradient = float(illumination_gradient)
        self.noise = float(noise)
        self.gray_prob = float(gray_prob)
        self._rng = np.random.default_rng(seed)

    def _u(self, span: float) -> float:
        return float(self._rng.uniform(max(0.0, 1.0 - span), 1.0 + span))

    def _apply_color_temp(self, img: torch.Tensor) -> torch.Tensor:
        """Warm/cool shift: gain up on red and down on blue, or the reverse."""
        t = float(self._rng.uniform(-self.color_temp, self.color_temp))
        gain = torch.tensor([1.0 + t, 1.0, 1.0 - t], dtype=img.dtype, device=img.device)
        return img * gain.view(*([1] * (img.ndim - 3)), 3, 1, 1)

    def _apply_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Smooth directional brightness ramp: one side of the frame lit more than the other."""
        strength = float(self._rng.uniform(0.0, self.illumination_gradient))
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        h, w = img.shape[-2:]
        ys = torch.linspace(-1.0, 1.0, h, dtype=img.dtype, device=img.device).view(h, 1)
        xs = torch.linspace(-1.0, 1.0, w, dtype=img.dtype, device=img.device).view(1, w)
        ramp = np.cos(angle) * xs + np.sin(angle) * ys          # (h, w) in about [-1.4, 1.4]
        return img * (1.0 + strength * ramp / 1.4142)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        out = img
        # Random order, as torchvision's ColorJitter does: these ops do not commute.
        for op in self._rng.permutation(6):
            if op == 0 and self.brightness > 0:
                out = TF.adjust_brightness(out, self._u(self.brightness))
            elif op == 1 and self.contrast > 0:
                out = TF.adjust_contrast(out, self._u(self.contrast))
            elif op == 2 and self.saturation > 0:
                out = TF.adjust_saturation(out, self._u(self.saturation))
            elif op == 3 and self.hue > 0:
                out = TF.adjust_hue(out, float(self._rng.uniform(-self.hue, self.hue)))
            elif op == 4 and self.color_temp > 0:
                out = self._apply_color_temp(out)
            elif op == 5 and self.illumination_gradient > 0:
                out = self._apply_gradient(out)
        if self.gray_prob > 0 and self._rng.random() < self.gray_prob:
            out = TF.rgb_to_grayscale(out, num_output_channels=3)
        if self.noise > 0:
            sigma = float(self._rng.uniform(0.0, self.noise))
            if sigma > 0:
                out = out + torch.randn_like(out) * sigma
        return out.clamp_(0.0, 1.0)


class SO101SequenceDataset(ManiSkillSequenceDataset):
    """Sequence dataset with strided observation history, precomputed RLA targets and augmentation.

    Extra args on top of the parent:
        n_obs_steps:        observation slots the policy reads. Must equal the policy's value.
        obs_history_stride: frames between slots. Set it to `n_action_steps` so the history slot is
                            the previous replan's observation.
        rla_latent_dir:     directory written by precompute_rla_latents.py
                            (`<dataset>/rla_latents/<tag>`). None disables the targets.
        rla_latent_step:    RLA checkpoint the sidecar must have been built from, as a step TOKEN:
                            `40000`, `"0040000"` or `"40000.snapshot"` all name the same thing they
                            name on disk (see rebuttal/exp4_so101/ckpt_tokens.py). None accepts whatever
                            is there; setting it pins the exact RLA version this policy is trained
                            against, and the value is carried into the deploy bundle so the renderer
                            cannot silently be a different checkpoint.
        rla_gaps:           which gaps to expose, as frame counts. `[32]` is the single
                            end-of-chunk latent; `[4, 8, ..., 32]` is the 8-rung ladder.
                            TrainSO101Workspace fills this in from `latent_mode`.
        augment:            enable photometric augmentation (train split only).
        augment_kwargs:     PhotometricAugment settings.
    """

    def __init__(
        self,
        *args,
        n_obs_steps: int = 1,
        obs_history_stride: int = 1,
        rla_latent_dir: str | None = None,
        rla_gaps: list | None = None,
        rla_latent_step: int | str | None = None,
        augment: bool = False,
        augment_kwargs: dict | None = None,
        **kwargs,
    ):
        if n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
        if obs_history_stride < 1:
            raise ValueError(f"obs_history_stride must be >= 1, got {obs_history_stride}")
        # The parent counts in-horizon image frames this way; keep it consistent so its
        # base_num_frames / padding bookkeeping still adds up.
        kwargs["num_in_horizon_frames_after_first"] = n_obs_steps - 1
        super().__init__(*args, **kwargs)

        self.n_obs_steps = n_obs_steps
        self.obs_history_stride = obs_history_stride
        if n_obs_steps > 1 and not self.first_frame_only:
            raise ValueError(
                "SO101SequenceDataset needs first_frame_only=True; with first_frame_only=False the "
                "parent returns the whole horizon and the observation slots are the window's first "
                "frames, which is not a strided history."
            )

        self.augment = bool(augment)
        self._augment_kwargs = dict(augment_kwargs or {})
        self._augmenter: PhotometricAugment | None = None   # built lazily, per worker

        self.rla_latents = None
        self.rla_gaps = None
        self.rla_manifest = None
        if rla_latent_dir:
            self._load_rla_latents(rla_latent_dir, rla_gaps, rla_latent_step)

        if self.augment and self.rla_latents is None and self.load_extra_frame > 0:
            raise ValueError(
                "augment=True with online latent targets is unsound: the policy would compute the "
                "RLA target from the augmented pixels, but the RLA was trained on clean frames. "
                "Precompute the targets (rebuttal/exp4_so101/precompute_rla_latents.py) and set "
                "rla_latent_dir, or turn augmentation off."
            )

    # ------------------------------------------------------------------ #
    # Precomputed latent targets
    # ------------------------------------------------------------------ #

    def _load_rla_latents(
        self, latent_dir: str, gaps: list | None, want_step: int | str | None = None
    ) -> None:
        manifest_path = os.path.join(latent_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            root = os.path.dirname(os.path.normpath(latent_dir))
            available = sorted(os.listdir(root)) if os.path.isdir(root) else []
            raise FileNotFoundError(
                f"no RLA latent sidecar at {latent_dir}. "
                + (f"Available tags under {root}: {available}. " if available else "")
                + "Run rebuttal/exp4_so101/precompute_rla_latents.py first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        got_step = manifest.get("rla_step")
        # Compared as (number, suffix) tokens: `40000`, `"0040000"` and `"0040000"` from the
        # manifest are the same checkpoint, while `"0040000.snapshot"` is a different one.
        if want_step is not None and not same_step(got_step, want_step):
            root = os.path.dirname(os.path.normpath(latent_dir))
            siblings = []
            for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                mp = os.path.join(root, name, "manifest.json")
                if os.path.exists(mp):
                    with open(mp) as fh:
                        siblings.append((name, json.load(fh).get("rla_step")))
            raise ValueError(
                f"rla_latent_step={want_step!r} but {latent_dir} was built from RLA step "
                f"{got_step!r}. Sidecars present (tag -> step): {siblings}. Either point "
                f"rla_latent_tag at the right one, re-run precompute_rla_latents.py "
                f"--step {want_step}, or set rla_latent_step: null to accept whatever is there."
            )
        self.rla_step = got_step
        # Sidecars may be dense (1..max_gap) or sparse (only the gaps that were asked for), so
        # columns are looked up by gap VALUE rather than assumed to be index gap-1.
        available = [int(g) for g in manifest.get("gaps", range(1, int(manifest["max_gap"]) + 1))]
        column = {g: j for j, g in enumerate(available)}
        gaps = [int(g) for g in (gaps if gaps else [max(available)])]
        missing = [g for g in gaps if g not in column]
        if missing:
            raise ValueError(
                f"rla_gaps {missing} were not precomputed in {latent_dir} (it has {available}). "
                f"Re-run precompute_rla_latents.py with --max-gap >= {max(gaps)}, or "
                f"--gaps {','.join(str(g) for g in sorted(set(gaps) | set(available)))}."
            )
        cols = [column[g] for g in gaps]

        chunks = []
        for traj_id in self.traj_id_list:
            entry = manifest["trajectories"].get(traj_id)
            if entry is None:
                raise KeyError(
                    f"{latent_dir} has no latents for traj_{traj_id}. The sidecar was built from a "
                    "different dataset or an incomplete run; re-run precompute_rla_latents.py."
                )
            arr = np.load(os.path.join(latent_dir, entry["path"]), mmap_mode="r")
            chunks.append(np.asarray(arr[:, cols], dtype=np.float32))

        self.rla_latents = np.concatenate(chunks, axis=0)     # (N_frames, n_gaps, N, D)
        self.rla_gaps = gaps
        self.rla_manifest = manifest
        if self.rla_latents.shape[0] != self.qpos.shape[0]:
            raise ValueError(
                f"latent frame count {self.rla_latents.shape[0]} != dataset frame count "
                f"{self.qpos.shape[0]}; the sidecar is stale relative to the converted dataset"
            )
        print(
            f"  RLA latents: {latent_dir} tag={manifest['tag']} rla_step={self.rla_step} "
            f"gaps={gaps} -> {self.rla_latents.shape} "
            f"({self.rla_latents.nbytes / 2**30:.2f} GiB)"
        )

    # ------------------------------------------------------------------ #
    # Observation history
    # ------------------------------------------------------------------ #

    def _observation_offsets(self) -> np.ndarray:
        """Frame offsets of the observation slots relative to 'now', oldest first."""
        return np.arange(self.n_obs_steps - 1, -1, -1, dtype=np.int64) * self.obs_history_stride

    def _get_image_frame_indices(
        self,
        buffer_start: int,
        buffer_end: int,
        sample_start: int,
        ep_start: int,
        ep_len: int,
    ) -> np.ndarray:
        if self.n_obs_steps == 1:
            return super()._get_image_frame_indices(
                buffer_start=buffer_start, buffer_end=buffer_end,
                sample_start=sample_start, ep_start=ep_start, ep_len=ep_len,
            )

        first_rel = buffer_start - ep_start
        last_rel = buffer_end - ep_start - 1
        # "Now" is window position n_obs_steps - 1, matching predict_action's `start` index.
        now_rel = first_rel + (self.n_obs_steps - 1 - sample_start)
        # Lower bound 0, not first_rel: history frames legitimately precede the window.
        frame_indices = np.clip(now_rel - self._observation_offsets(), 0, ep_len - 1).astype(np.int64)

        if self.load_extra_frame > 0:
            future = np.minimum(
                last_rel + np.arange(1, self.load_extra_frame + 1, dtype=np.int64), ep_len - 1
            )
            frame_indices = np.concatenate([frame_indices, future], axis=0)
        return frame_indices

    # ------------------------------------------------------------------ #

    def _now_buffer_index(self, index: int) -> int:
        """Absolute buffer index of the 'now' observation for a sample."""
        buffer_start, _, sample_start, _ = self.indices[index]
        ep_idx, _ = self._buffer_idx_to_episode(buffer_start)
        ep_start = int(self.episode_starts[ep_idx])
        ep_end = int(self.episode_ends[ep_idx])
        now = buffer_start + (self.n_obs_steps - 1 - sample_start)
        return int(np.clip(now, ep_start, ep_end - 1))

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)

        if self.n_obs_steps > 1 and self.obs_history_stride > 1:
            # The parent filled obs["state"] from the contiguous window, so slots 0..n_obs-1 are
            # frames t0, t0+1, ... Re-gather them at the same instants as the image slots.
            buffer_start, _, sample_start, _ = self.indices[index]
            ep_idx, _ = self._buffer_idx_to_episode(buffer_start)
            ep_start = int(self.episode_starts[ep_idx])
            ep_end = int(self.episode_ends[ep_idx])
            now = buffer_start + (self.n_obs_steps - 1 - sample_start)
            idxs = np.clip(now - self._observation_offsets(), ep_start, ep_end - 1)
            sample["obs"]["state"][: self.n_obs_steps] = torch.from_numpy(self.qpos[idxs])

        if self.rla_latents is not None:
            # Target for the CURRENT frame -- the same frame the policy calls "now".
            # Deliberately a TOP-LEVEL key, not inside obs: LinearNormalizer.normalize() walks
            # every key of the dict it is given and raises KeyError on anything it was not fitted
            # on. `has_robot_data` sits at the top level for the same reason.
            sample["rla_latents"] = torch.from_numpy(
                self.rla_latents[self._now_buffer_index(index)]      # (n_gaps, N, D)
            )

        if self.augment:
            sample["obs"]["image"] = self._augment_images(sample["obs"]["image"])
        return sample

    # ------------------------------------------------------------------ #
    # Augmentation
    # ------------------------------------------------------------------ #

    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        """(N_img, Cam, 3, H, W) in [0,1] -> same, one photometric transform per camera.

        Shared across the observation slots on purpose: exposure and white balance do not flicker
        within the 0.53 s history window, but the two cameras are independent devices.
        """
        if self._augmenter is None:
            # Seeded per worker so workers do not draw identical augmentations.
            info = torch.utils.data.get_worker_info()
            seed = int(torch.initial_seed() % (2**31)) + (info.id if info else 0)
            self._augmenter = PhotometricAugment(seed=seed, **self._augment_kwargs)
        out = images.clone()
        for cam in range(images.shape[1]):
            out[:, cam] = self._augmenter(images[:, cam])
        return out

    # ------------------------------------------------------------------ #

    def get_validation_dataset(self):
        val = super().get_validation_dataset()
        # Never augment the held-out split: val_loss has to stay comparable across runs.
        val = copy.copy(val)
        val.augment = False
        val._augmenter = None
        return val
