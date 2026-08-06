from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rebuttal.src.models.plucker_embedding import PluckerEmbedding
from src.models.simple_token_transformer import SimpleTokenTransformer


class MultiViewTokenTransformer(SimpleTokenTransformer):
    """SimpleTokenTransformer with learned per-camera and per-patch embeddings.

    The RLA autoencoder flattens the patch tokens of every camera into one sequence
    ``[B, V * Lp, C]``. SimpleTokenTransformer has no positional or view encoding, so the blocks
    are permutation-equivariant over the sequence: with more than one camera they cannot tell
    which view a token came from, nor where inside a view it sits. This subclass adds, right
    after the input projection:

    * ``view_embed`` — one learned vector per camera, broadcast over that camera's patches;
    * a positional embedding selected by ``pos_embed_mode``:

      - ``"plucker"`` — the patch's camera ray encoded as Plücker coordinates (see
        :mod:`src.models.plucker_embedding`). Spatially grounded: two views looking at the same
        region get similar codes, which a slot index cannot express. Needs real camera
        intrinsics/extrinsics, which the caller passes as ``plucker`` rays.
      - ``"learned"`` — one learned vector per patch slot inside a view, shared across views.
      - ``"none"`` — no positional embedding, matching the parent exactly.

    Both are plain additive terms in ``model_channels`` space; nothing touches the attention
    computation, so the blocks stay stock ``AttentionBlock``s.

    The forward signature stays compatible with SimpleTokenTransformer, so a config that swaps one
    for the other only needs the extra ``num_views`` argument.
    """

    # Lets callers detect that forward() accepts num_views/plucker, mirroring the
    # `expects_per_token_input` convention RlaAutoencoderTrainer uses for VQ modules.
    expects_num_views: bool = True

    POS_EMBED_MODES = ("none", "learned", "plucker")

    def __init__(
        self,
        *args,
        num_views: int = 2,
        use_view_embed: bool = True,
        pos_embed_mode: str = "plucker",
        max_tokens_per_view: int = 1024,
        plucker_num_frequencies: int = 8,
        embed_scale: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        if pos_embed_mode not in self.POS_EMBED_MODES:
            raise ValueError(
                f"pos_embed_mode must be one of {self.POS_EMBED_MODES}, got {pos_embed_mode!r}"
            )
        self.num_views = num_views
        self.max_tokens_per_view = max_tokens_per_view
        self.pos_embed_mode = pos_embed_mode
        self.expects_plucker = pos_embed_mode == "plucker"

        self.view_embed = (
            nn.Parameter(torch.randn(num_views, self.model_channels) * embed_scale)
            if use_view_embed
            else None
        )
        # Sliced to the actual per-view token count at runtime, so one checkpoint covers any
        # image size up to max_tokens_per_view patches per view.
        self.pos_embed = (
            nn.Parameter(
                torch.randn(max_tokens_per_view, self.model_channels) * embed_scale
            )
            if pos_embed_mode == "learned"
            else None
        )
        self.plucker_embed = (
            PluckerEmbedding(
                out_channels=self.model_channels,
                num_frequencies=plucker_num_frequencies,
            )
            if pos_embed_mode == "plucker"
            else None
        )

    def _add_view_pos_embedding(
        self,
        h: torch.Tensor,
        num_views: Optional[int] = None,
        plucker: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # h: (B, V * Lp, Cm)
        if self.plucker_embed is not None:
            if plucker is None:
                raise ValueError(
                    "pos_embed_mode='plucker' requires `plucker` rays of shape [B, V * Lp, 6]"
                )
            if plucker.shape[:2] != h.shape[:2]:
                raise ValueError(
                    f"plucker rays {tuple(plucker.shape[:2])} do not match tokens "
                    f"{tuple(h.shape[:2])} on (B, L)"
                )
            h = h + self.plucker_embed(plucker.float())

        if self.view_embed is None and self.pos_embed is None:
            return h

        views = self.num_views if num_views is None else int(num_views)
        if views > self.num_views:
            raise ValueError(
                f"Requested {views} views but only {self.num_views} view embeddings exist"
            )
        bsz, seq_len, channels = h.shape
        if seq_len % views != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by num_views={views}; "
                "the per-view token blocks must be equally sized"
            )
        tokens_per_view = seq_len // views

        h = h.reshape(bsz, views, tokens_per_view, channels)
        if self.view_embed is not None:
            h = h + self.view_embed[:views].reshape(1, views, 1, channels)
        if self.pos_embed is not None:
            if tokens_per_view > self.max_tokens_per_view:
                raise ValueError(
                    f"{tokens_per_view} tokens per view exceeds max_tokens_per_view="
                    f"{self.max_tokens_per_view}"
                )
            h = h + self.pos_embed[:tokens_per_view].reshape(
                1, 1, tokens_per_view, channels
            )
        return h.reshape(bsz, seq_len, channels)

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        num_views: Optional[int] = None,
        plucker: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, V * Lp, C), plucker: (B, V * Lp, 6)
        h = self.input_layer(x.float())
        h = self._add_view_pos_embedding(h, num_views=num_views, plucker=plucker)

        if tokens is not None:
            tokens = tokens.float()
            if self.token_proj is not None:
                tokens = self.token_proj(tokens)
            if self.tokens is not None:
                tokens = tokens + self.tokens.unsqueeze(0)
        else:
            if self.num_tokens > 0:
                tokens = self.tokens.unsqueeze(0).repeat(h.shape[0], 1, 1)

        if tokens is not None:
            num_tokens = tokens.shape[1]
            h = torch.cat([tokens, h], dim=1)
        else:
            num_tokens = 0

        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h)

        h = h.float()
        out = self.out_layer(self.norm_out(h))
        tokens, visuals = out[:, :num_tokens], out[:, num_tokens:]
        if self.norm_output_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens, visuals
