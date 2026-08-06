"""BC + RLA policy that can consume a *multi-view* RLA encoder.

Why this file exists
--------------------
`policies/policy/vla_bc_policy_rla_unified.py::_load_encoder_from_work_dir` reads
`cfg.models.encoder.args` from the RLA run directory but ignores `cfg.models.encoder.name`, and
always builds a `SimpleTokenTransformer` with `load_state_dict(..., strict=True)`:

    encoder = SimpleTokenTransformer(...)            # vla_bc_policy_rla_unified.py:61
    encoder.load_state_dict(state_dict, strict=True) # vla_bc_policy_rla_unified.py:73

A 2-camera RLA trained with `MultiViewTokenTransformer` carries extra parameters
(`view_embed`, `pos_embed`), so that load raises

    RuntimeError: Error(s) in loading state_dict for SimpleTokenTransformer:
      Unexpected key(s) in state_dict: "view_embed", "pos_embed".

This subclass dispatches on `cfg.models.encoder.name` instead, and forwards `num_views` when the
encoder advertises `expects_num_views` -- mirroring how `RlaAutoencoderMultiViewTrainer._forward_views`
probes the same flag.

It also changes *where the latent target comes from* and *what the head predicts*.

`latent_source`
    **Always "precomputed"** -- the target is `batch["rla_latents"]`, written OFFLINE by
    `rebuttal/exp4_so101/precompute_rla_latents.py` from CLEAN frames. Neither DINOv3 nor the RLA
    encoder is constructed here at all, which removes the dominant cost of BC-RLA training and,
    critically, decouples the target from the policy's input pixels so the photometric augmentation
    is sound.

    The old "online" path (recompute the target in the loop from the policy's own image tensor) is
    **rejected**, not merely discouraged: with augmentation enabled it silently regresses against a
    target derived from augmented pixels, while the RLA that has to decode those latents was trained
    on clean frames. Nothing about the loss curve would reveal it.

`latent_gaps`
    Which t -> t+k transitions the head predicts, in frames.
        [32]                "final"  -- one latent for the whole predicted chunk  (B, 1, N, D)
        [4, 8, ..., 32]     "ladder" -- the chunk sampled evenly, 8 rungs         (B, 8, N, D)
    The gaps span the policy `horizon` (32 frames, 1.07 s) -- the chunk it PREDICTS -- not
    `n_action_steps` (16), which is only how much of it executes before the next replan.
    TrainSO101Workspace derives them from `latent_mode`; the ladder always ends at `horizon`.
    A ladder policy is strictly more informative at deploy time: it can render the predicted future
    at every gap, i.e. a short *video* of what the policy expects to happen, instead of a single
    frame. Train both and compare; `predict_latent_action(obs, gap=k)` selects one.

`latent_head_from_queries`
    Where the latent head reads from. Default `None` = automatic: **shared MLP for "final",
    per-query for "ladder"**.

    * "final" (1 gap) -- unchanged from the stock policy: one `Linear(shared_mlp_dim, N*D)` on the
      shared MLP output.
    * "ladder" (K gaps) -- the head reads the **output query tokens directly**, with the queries
      split into K contiguous groups and one small MLP shared across groups::

          output queries      (B, 16, H)
                |  split into K=8 contiguous groups of 16/8 = 2 queries
                v
          (B, 8, 2*H)  --  Linear(2H, hidden) - GELU - Linear(hidden, N*D)  -->  (B, 8, N, D)
                            ^ one small MLP, shared across the 8 groups

      So query pair 0 predicts the t+4 latent, pair 1 the t+8 latent, and so on. Each gap gets its
      own slice of attention capacity instead of every gap being squeezed through one 2048-wide
      bottleneck, and the head itself is ~40x smaller than the flat `Linear(2048, 8*N*D)` it
      replaces. `num_output_queries` must be divisible by the number of gaps.

`_decode_latent` always returns rank 4 `(B, n_gaps, num_tokens, token_dim)` in both modes, so
nothing downstream has to branch.

Used via hydra `_target_` in rebuttal/exp4_so101/configs/policy/bc_rla_so101_{100,25}.yaml:

    policy:
      _target_: rebuttal.exp4_so101.policy.rla_unified_mv.VLABCPolicyRLAUnifiedMV
      latent_encoder_work_dir: runs/so101/16x16_so101_dual/<timestamp>
      latent_num_views: 2
      latent_source: precomputed
      latent_gaps: [32]                # or [4, 8, ..., 32] for the 8-rung ladder
      latent_head_hidden: 256          # width of the small per-query MLP (ladder only)
"""

import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich import print

from policies.policy.vla_bc_policy_rla_unified import VLABCPolicyRLAUnified
from rebuttal.exp4_so101.ckpt_tokens import ckpt_token, find_ckpt, load_state_dict_file
from rebuttal.src.models.multiview_token_transformer import MultiViewTokenTransformer
from src.models.simple_token_transformer import SimpleTokenTransformer

_ENCODER_CLASSES = {
    "SimpleTokenTransformer": SimpleTokenTransformer,
    "MultiViewTokenTransformer": MultiViewTokenTransformer,
}


def _latent_shape_from_work_dir(work_dir: str) -> tuple[int, int, int]:
    """(num_tokens, token_dim, dino_channels) from an RLA run's config, without touching weights.

    Used by `latent_source="precomputed"`, where the encoder itself is never needed.
    """
    cfg = OmegaConf.load(os.path.join(work_dir, "config.yaml"))
    args = cfg.models.encoder.args
    return (
        int(args.num_tokens),
        int(args.out_channels),
        int(OmegaConf.select(cfg, "vars.dino_channels", default=1024)),
    )


def build_encoder_from_work_dir(
    work_dir: str,
    device: str = "cpu",
    step=None,
) -> tuple[nn.Module, int]:
    """Rebuild the RLA encoder exactly as its training config declared it, and load its weights.

    ``step`` is a checkpoint *token*, not necessarily a number -- ``40000``, ``"0040000"`` and
    ``"40000.snapshot"`` all name a real file (see ``rebuttal/exp4_so101/ckpt_tokens.py``). ``None``
    takes the newest, preferring the milestone over a snapshot at the same step. This is why the
    load does not go through ``utils.misc.fetch_state_dict``, which composes the filename as
    ``{name}_step{int(step):07d}.pt`` and cannot express a snapshot at all.

    Returns ``(encoder, dino_channels)``. The encoder is frozen and in eval mode.
    """
    cfg = OmegaConf.load(os.path.join(work_dir, "config.yaml"))
    dino_channels = int(OmegaConf.select(cfg, "vars.dino_channels", default=1024))

    name = str(OmegaConf.select(cfg, "models.encoder.name", default="SimpleTokenTransformer"))
    if name not in _ENCODER_CLASSES:
        raise ValueError(
            f"{work_dir}: models.encoder.name={name!r} is not supported here "
            f"(expected one of {sorted(_ENCODER_CLASSES)})"
        )
    args = cfg.models.encoder.args

    kwargs = dict(
        in_channels=int(args.in_channels),
        model_channels=int(args.model_channels),
        out_channels=int(args.out_channels),
        num_blocks=int(args.num_blocks),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.get("mlp_ratio", 4.0)),
        use_fp16=bool(args.get("use_fp16", False)),
        num_tokens=int(args.num_tokens),
        norm_output_tokens=bool(args.get("norm_output_tokens", False)),
    )
    if name == "MultiViewTokenTransformer":
        pos_embed_mode = str(args.get("pos_embed_mode", "plucker"))
        if pos_embed_mode == "plucker":
            # Plucker rays need per-frame camera intrinsics/extrinsics, which a policy observation
            # does not carry. Train the RLA with pos_embed_mode "learned" or "none" instead.
            raise ValueError(
                f"{work_dir}: the RLA encoder was trained with pos_embed_mode='plucker', which "
                "requires camera geometry the policy cannot supply at inference time. Retrain the "
                "RLA with pos_embed_mode: learned (or none)."
            )
        kwargs.update(
            num_views=int(args.get("num_views", 2)),
            use_view_embed=bool(args.get("use_view_embed", True)),
            pos_embed_mode=pos_embed_mode,
            max_tokens_per_view=int(args.get("max_tokens_per_view", 1024)),
        )

    encoder = _ENCODER_CLASSES[name](**kwargs)
    ckpt = find_ckpt(work_dir, "encoder", step)
    encoder.load_state_dict(load_state_dict_file(ckpt, device), strict=True)
    print(
        f"[green]Latent encoder ({name}) loaded from {work_dir} "
        f"@ step {ckpt_token(ckpt, 'encoder')}[/green]"
    )

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, dino_channels


class VLABCPolicyRLAUnifiedMV(VLABCPolicyRLAUnified):
    """`VLABCPolicyRLAUnified` whose frozen RLA encoder is built from the RLA run's own class name.

    Extra constructor args on top of the parent:
        latent_num_views:    number of camera view blocks in the DINO token sequence. Defaults to
                             ``num_cameras``; must match the RLA's ``vars.num_views``.
        latent_encoder_step: pin a specific RLA checkpoint step instead of the latest one.
    """

    def __init__(
        self,
        *args,
        use_latent_head: bool = True,
        latent_encoder_work_dir: str = "",
        latent_num_views: Optional[int] = None,
        latent_encoder_step: Optional[int] = None,
        latent_source: str = "precomputed",
        latent_gaps: Optional[Sequence[int]] = None,
        latent_head_from_queries: Optional[bool] = None,
        latent_head_hidden: int = 256,
        shared_mlp_dim: int = 2048,
        lambda_action: float = 1.0,
        lambda_latent_l1: float = 1.0,
        lambda_latent_mse: float = 1.0,
        **kwargs,
    ):
        # Build the trunk with the latent head disabled, so the parent never runs its
        # SimpleTokenTransformer-only loader, then attach the real encoder/head below.
        super().__init__(
            *args,
            use_latent_head=False,
            shared_mlp_dim=shared_mlp_dim,
            lambda_latent_l1=lambda_latent_l1,
            lambda_latent_mse=lambda_latent_mse,
            **kwargs,
        )

        # Raw-action loss weight. The BC objective is what actually drives the robot; the latent
        # action is an auxiliary signal, so `lambda_action : (lambda_latent_l1 + lambda_latent_mse)`
        # is kept at 2:1 in the shipped configs and the realised split is logged every step.
        self.lambda_action = float(lambda_action)

        num_cameras = int(kwargs.get("num_cameras", 1))
        self._latent_num_views = int(latent_num_views if latent_num_views else num_cameras)
        self.latent_source = str(latent_source)
        if self.latent_source != "precomputed":
            raise ValueError(
                f"latent_source must be 'precomputed', got {latent_source!r}. Latent-action targets "
                "are always computed OFFLINE by rebuttal/exp4_so101/precompute_rla_latents.py from "
                "clean frames: computing them in the loop would derive them from the AUGMENTED "
                "pixels the policy sees, while the RLA decoder that has to interpret those latents "
                "was trained on clean frames -- and the loss curve would look perfectly healthy."
            )
        # Which t -> t+k gaps the head predicts, in frames. One entry -> "final" mode; several ->
        # "ladder" mode, one latent per gap. Must match the dataset's `rla_gaps`, in order.
        self.latent_gaps = [int(g) for g in (latent_gaps or [])]

        if not use_latent_head:
            # Same behaviour as the parent with use_latent_head=False (vanilla BC).
            return

        if not latent_encoder_work_dir:
            raise ValueError(
                "latent_encoder_work_dir is required when use_latent_head=True "
                "- point it at an RLA training run directory."
            )

        # Targets come from the batch, so no DINO and no RLA encoder are needed at all -- the
        # single biggest cost of BC-RLA training disappears. Only the latent geometry is read from
        # the RLA run's config; its weights are never loaded here.
        num_tokens, token_dim, dino_channels = _latent_shape_from_work_dir(latent_encoder_work_dir)
        self.latent_encoder = None
        self.dino_extractor = None
        self.dino_channels = dino_channels
        if not self.latent_gaps:
            raise ValueError(
                "latent_gaps is required: it must list the same gaps, in the same order, as the "
                "dataset's rla_gaps. TrainSO101Workspace fills both from `latent_mode`."
            )

        self._latent_num_tokens = num_tokens
        self._latent_token_dim = token_dim
        self._latent_num_gaps = max(1, len(self.latent_gaps))

        # Default: per-query for a ladder, shared-MLP for a single gap (the stock behaviour).
        self._latent_from_queries = (
            self._latent_num_gaps > 1
            if latent_head_from_queries is None
            else bool(latent_head_from_queries)
        )
        self._query_tokens: Optional[torch.Tensor] = None

        if self._latent_from_queries:
            n_q = int(self._num_output_queries)
            if n_q % self._latent_num_gaps:
                raise ValueError(
                    f"num_output_queries={n_q} is not divisible by the {self._latent_num_gaps} "
                    f"latent gaps {self.latent_gaps}; each gap needs the same number of query "
                    "tokens. Pick num_output_queries as a multiple of the gap count (e.g. 16 "
                    "queries for 8 gaps -> 2 each)."
                )
            self._queries_per_gap = n_q // self._latent_num_gaps
            hidden_dim = int(self.output_norm.normalized_shape[0])   # spatial_hidden_dim
            self.latent_head = nn.Sequential(
                nn.Linear(self._queries_per_gap * hidden_dim, latent_head_hidden),
                nn.GELU(),
                nn.Linear(latent_head_hidden, num_tokens * token_dim),
            )
            # The trunk does not return its query tokens, and its signature is fixed by the parent's
            # predict_action. Rather than duplicate 50 lines of _forward_trunk, capture the ONE
            # tensor we need: shared_mlp's input is exactly `out_tokens.reshape(B, -1)`, the only
            # place the trunk flattens the normalised output queries. Forward hooks survive the
            # deepcopy that builds the EMA model (verified) and never enter state_dict.
            self.shared_mlp.register_forward_pre_hook(self._capture_query_tokens)
        else:
            self._queries_per_gap = 0
            # Stock behaviour: one flat Linear over all gaps, off the shared MLP.
            self.latent_head = nn.Linear(
                shared_mlp_dim, self._latent_num_gaps * num_tokens * token_dim
            )
        self.use_latent_head = True

    def _capture_query_tokens(self, module, inputs):
        self._query_tokens = inputs[0]

    # ------------------------------------------------------------------ #
    # Latent head
    # ------------------------------------------------------------------ #

    def _decode_latent(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """-> (B, n_gaps, num_tokens, token_dim). Rank 4 in both modes, so callers never branch.

        "final": from the shared MLP output, exactly as the stock policy does.
        "ladder": from the output query tokens, `num_output_queries / n_gaps` queries per gap
        through one small shared MLP.
        """
        G, N, D = self._latent_num_gaps, self._latent_num_tokens, self._latent_token_dim
        if not self._latent_from_queries:
            return self.latent_head(shared_feat).reshape(-1, G, N, D)

        flat = self._query_tokens
        if flat is None or flat.shape[0] != shared_feat.shape[0]:
            # The hook fires inside _forward_trunk, so this can only trip if the trunk stopped
            # routing the queries through shared_mlp -- fail loudly rather than use stale tokens.
            raise RuntimeError(
                "per-query latent head: the output query tokens were not captured during this "
                "forward pass. _forward_trunk must feed shared_mlp with the flattened, normalised "
                "output queries."
            )
        B = flat.shape[0]
        # (B, Nq*H) -> (B, G, queries_per_gap*H): queries are contiguous, so gap g owns queries
        # [g*queries_per_gap : (g+1)*queries_per_gap].
        grouped = flat.reshape(B, G, -1)
        return self.latent_head(grouped).reshape(B, G, N, D)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """BC loss + latent-action loss against the PRECOMPUTED targets in the batch."""
        if not self.use_latent_head:
            return super().compute_loss(batch)

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        has_robot = batch.get("has_robot_data", None)

        shared_feat = self._forward_trunk(nobs, has_robot_data=has_robot)
        pred_action = self._decode_action(shared_feat)

        # --- action loss (robot-data samples only) --- #
        loss_fn = {"l1": F.l1_loss, "smooth_l1": F.smooth_l1_loss}.get(self.loss_type, F.mse_loss)
        if has_robot is not None and not has_robot.all():
            mask = has_robot.bool()
            action_loss = (
                loss_fn(pred_action[mask], nactions[mask])
                if mask.any()
                else torch.zeros((), device=self.device)
            )
        else:
            action_loss = loss_fn(pred_action, nactions)

        # --- latent loss (all samples, including pixel-only ones) --- #
        gt = batch.get("rla_latents", None)
        if gt is None:
            raise KeyError(
                "latent_source='precomputed' but the batch has no 'rla_latents'. Set "
                "`rla_latent_dir` in the dataset config (and use SO101SequenceDataset via "
                "TrainSO101Workspace) -- see rebuttal/exp4_so101/precompute_rla_latents.py."
            )
        gt = gt.to(pred_action.dtype) / 10.0          # same /10 convention as the stock policy
        if gt.ndim == 3:                              # (B, N, D) -> (B, 1, N, D)
            gt = gt.unsqueeze(1)
        pred_latent = self._decode_latent(shared_feat)
        if gt.shape[1:] != pred_latent.shape[1:]:
            raise ValueError(
                f"precomputed latent target {tuple(gt.shape[1:])} does not match the head's "
                f"{tuple(pred_latent.shape[1:])}; dataset rla_gaps and policy latent_gaps must "
                "agree, in the same order"
            )

        latent_l1 = F.l1_loss(pred_latent, gt)
        latent_mse = F.mse_loss(pred_latent, gt)

        action_term = self.lambda_action * action_loss
        latent_term = self.lambda_latent_l1 * latent_l1 + self.lambda_latent_mse * latent_mse
        total = action_term + latent_term

        out = {
            "loss": total,
            "action_loss": action_loss,
            "latent_action_l1": latent_l1,
            "latent_action_mse": latent_mse,
        }
        with torch.no_grad():
            # The lambdas fix the NOMINAL ratio; this is the ratio that actually materialises, and
            # it moves as the two losses converge at different rates. Watch it: the raw-action term
            # must stay dominant. At the shipped 2:1 weighting `latent_loss_frac` should sit near
            # 1/3; if it climbs past ~0.5 the auxiliary signal has taken over and the lambdas want
            # lowering.
            denom = total.abs().clamp_min(1e-8)
            out["latent_loss_frac"] = latent_term.abs() / denom
            out["action_over_latent"] = action_term.abs() / latent_term.abs().clamp_min(1e-8)
        if self._latent_num_gaps > 1:
            # Per-gap breakdown: the near gaps should be much easier than the far ones.
            with torch.no_grad():
                for i, gap in enumerate(self.latent_gaps):
                    out[f"latent_l1_gap{gap}"] = F.l1_loss(pred_latent[:, i], gt[:, i])
        return out

    # ------------------------------------------------------------------ #
    # Deployment helper
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_latent_action(
        self, obs_dict: dict, gap: Optional[int] = None
    ) -> torch.Tensor:
        """Predicted latent action(s) for one observation, in *encoder* scale.

        `compute_loss` regresses `latent_head` against `encoder_tokens / 10.0`, so the raw head
        output has to be multiplied back by 10 before it is fed to the RLA decoder. This helper
        does that, so the deployment visualiser cannot get the scaling wrong.

        Args:
            gap: which t -> t+gap latent to return, in frames. Must be one of `latent_gaps`.
                None returns all of them.
        Returns:
            `(B, num_tokens, token_dim)` when `gap` is given (or the policy predicts a single gap),
            otherwise `(B, n_gaps, num_tokens, token_dim)`.
        """
        nobs = self.normalizer.normalize(obs_dict)
        z = self._decode_latent(self._forward_trunk(nobs)) * 10.0
        if gap is None:
            return z[:, 0] if z.shape[1] == 1 else z
        if gap not in self.latent_gaps:
            raise ValueError(
                f"gap={gap} was not trained; this policy predicts gaps {self.latent_gaps}"
            )
        return z[:, self.latent_gaps.index(gap)]
