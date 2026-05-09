"""Latent Action Flow Model — two-stage transformer for flow-matching on action latent codes.

Stage 1 (Conditioning):
    Query tokens + action-related tokens (task, horizon, qpos) + x_t DINO tokens
    → Transformer → conditioning query output

Stage 2 (Flow):
    [conditioning_queries | noisy_latent] + timestep modulation
    → Transformer → predicted velocity
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from rich import print

from torch import Tensor

from src.models.attention_block import AttentionBlock, ModCrossAttentionBlock
from src.models.sparse_structure_flow import TimestepEmbedder
from src.modules.utils import convert_module_to_f16, convert_module_to_f32


# ---------------------------------------------------------------------------
# Per-robot qpos encoder
# ---------------------------------------------------------------------------


class _RobotQposEncoder(nn.Module):
    def __init__(self, full_qpos_dim, model_channels, frame_channels=64, num_layers=1):
        super().__init__()
        self.per_frame_mlp = nn.Sequential(
            nn.Linear(full_qpos_dim, frame_channels),
            nn.GELU(),
            nn.Linear(frame_channels, frame_channels),
        )
        self.gru = nn.GRU(
            input_size=frame_channels,
            hidden_size=model_channels,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, qpos, mask):
        h = self.per_frame_mlp(qpos)  # (B, T, frame_channels)
        lengths = mask.sum(dim=1).long().cpu()  # true lengths per batch item
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)  # hidden: (num_layers, B, model_channels)
        return hidden[-1]  # (B, model_channels)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class RLAWM(nn.Module):
    """Two-stage transformer for flow-matching on action latent codes.

    Models dict key expected: ``flow_model``
    """

    def __init__(
        self,
        # core dims
        model_channels: int = 1024,
        flow_model_channels: int = -1,
        token_dim: int = 64,
        dino_channels: int = 1024,
        # conditioning stage
        num_cond_blocks: int = 6,
        # flow stage
        num_flow_blocks: int = 6,
        # attention
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        # task / horizon
        num_tasks: int = 8,
        # qpos encoding
        max_horizon: int = 100,
        robot_qpos_dims: Optional[Dict[str, int]] = None,
        qpos_frame_channels: int = 64,
        # centroid
        use_task_horizon_embedding: bool = False,
        no_horizon_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.no_horizon_embedding = no_horizon_embedding
        self.model_channels = model_channels
        self.flow_model_channels = (
            flow_model_channels if flow_model_channels > 0 else model_channels
        )
        self.token_dim = token_dim
        self.dino_channels = dino_channels
        self.num_cond_blocks = num_cond_blocks
        self.num_flow_blocks = num_flow_blocks
        self.max_horizon = max_horizon
        self.dtype = torch.float16 if use_fp16 else torch.float32

        flow_ch = self.flow_model_channels

        # ---- Task & horizon embedders ----
        if use_task_horizon_embedding:
            self.task_embedder = nn.Embedding(num_tasks, model_channels)
            if not no_horizon_embedding:
                self.horizon_embedder = TimestepEmbedder(model_channels)
        else:
            assert robot_qpos_dims is not None, (
                "Must provide robot_qpos_dims if not using task/horizon embedding"
            )
        self.use_task_horizon_embedding = use_task_horizon_embedding

        # ---- Per-robot qpos encoders ----
        self.robot_qpos_dims = {str(k): int(v) for k, v in robot_qpos_dims.items()}

        self.qpos_encoders = nn.ModuleDict()
        for rid_str, qdim in self.robot_qpos_dims.items():
            self.qpos_encoders[rid_str] = _RobotQposEncoder(
                full_qpos_dim=qdim,
                model_channels=model_channels,
                frame_channels=qpos_frame_channels,
            )

        # ---- Stage 1: Conditioning transformer ----
        self.xt_proj = (
            nn.Linear(dino_channels, model_channels)
            if dino_channels != model_channels
            else nn.Identity()
        )

        self.cond_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    channels=model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_fp16=use_fp16,
                    cond_channels=model_channels,
                    use_condition=True,
                )
                for _ in range(num_cond_blocks)
            ]
        )

        self.flow_latent_proj = nn.Linear(token_dim, flow_ch)
        self.flow_time_embedder = TimestepEmbedder(flow_ch)

        self.flow_blocks = nn.ModuleList(
            [
                ModCrossAttentionBlock(
                    channels=flow_ch,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_fp16=use_fp16,
                    cond_channels=model_channels,
                )
                for _ in range(num_flow_blocks)
            ]
        )
        self.flow_norm_out = nn.LayerNorm(flow_ch)
        self.flow_out_proj = nn.Linear(flow_ch, token_dim)

        self._initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def _initialize_weights(self) -> None:
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        # Zero-init the flow output projection for stable start
        nn.init.zeros_(self.flow_out_proj.weight)
        nn.init.zeros_(self.flow_out_proj.bias)

        nn.init.normal_(self.flow_time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.flow_time_embedder.mlp[2].weight, std=0.02)

        for blocks in [self.cond_blocks, self.flow_blocks]:
            for block in blocks:
                nn.init.constant_(block.modulation[-1].weight, 0)
                nn.init.constant_(block.modulation[-1].bias, 0)

    def convert_to_fp16(self) -> None:
        self.dtype = torch.float16
        self.cond_blocks.apply(convert_module_to_f16)
        self.flow_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.dtype = torch.float32
        self.cond_blocks.apply(convert_module_to_f32)
        self.flow_blocks.apply(convert_module_to_f32)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_action_tokens(
        self,
        task_inds: Tensor,  # (B,) int
        horizons: Tensor,  # (B,) int or float
        robot_ids: Tensor,  # (B,) int;  -1 = no robot
        full_qpos_list: List[
            Optional[Tensor]
        ],  # per-sample (horizon_i, full_qpos_dim) or None
    ) -> Tensor:
        """Build action-condition tokens per sample.

        Returns: (B, model_channels)
        """
        B = task_inds.shape[0]
        device = task_inds.device

        # task & horizon tokens  — (B, 1, C) each
        if self.use_task_horizon_embedding:
            task_tok = self.task_embedder(task_inds.long())  # (B,C)
            if self.no_horizon_embedding:
                action_tok = task_tok
            else:
                horizon_tok = self.horizon_embedder(horizons.float() * 100)  # (B, C)
                action_tok = task_tok + horizon_tok  # (B, C)
        else:
            action_tok = torch.zeros(
                B, self.model_channels, device=device, dtype=torch.float
            )

        dtype = action_tok.dtype

        # group by robot_id for efficient batched processing
        unique_rids = robot_ids.unique().tolist()
        for rid in unique_rids:
            rid_int = int(rid)
            if rid_int < 0:
                continue  # no robot — keep learnable defaults
            rid_str = str(rid_int)
            if rid_str not in self.qpos_encoders:
                continue

            batch_indices = (robot_ids == rid_int).nonzero(as_tuple=True)[0]
            qdim = self.robot_qpos_dims[rid_str]

            # Collect and pad qpos for this robot group
            qpos_padded = torch.zeros(
                len(batch_indices),
                self.max_horizon,
                qdim,
                device=device,
                dtype=dtype,
            )
            qpos_mask = torch.zeros(
                len(batch_indices),
                self.max_horizon,
                device=device,
                dtype=dtype,
            )

            for local_i, bi in enumerate(batch_indices.tolist()):
                q = full_qpos_list[bi]
                if q is None:
                    continue
                h_len = min(q.shape[0], self.max_horizon)
                q_dim_actual = min(q.shape[-1], qdim)
                qpos_padded[local_i, :h_len, :q_dim_actual] = q[
                    :h_len, :q_dim_actual
                ].to(
                    device=device,
                    dtype=dtype,
                )
                qpos_mask[local_i, :h_len] = 1.0

            encoded = self.qpos_encoders[rid_str](qpos_padded, qpos_mask)  # (n, C)
            action_tok[batch_indices] += encoded.to(action_tok.dtype)

        return action_tok

    # ------------------------------------------------------------------
    # forward — split into cond (one-time) and flow (per ODE step)
    # ------------------------------------------------------------------

    def forward_cond(
        self,
        xt_tokens: Tensor,  # (B, Lp, dino_channels) — DINO tokens of x_t
        task_inds: Tensor,  # (B,)
        horizons: Tensor,  # (B,)
        robot_ids: Tensor,  # (B,)
        full_qpos_list: List[Optional[Tensor]],  # per-sample variable-length qpos
    ) -> Tensor:
        """Stage 1: compute conditioning queries (one-time).

        Returns: cond_queries (B, Lp, model_channels)
        """
        B = xt_tokens.shape[0]

        # Project x_t tokens
        h = self.xt_proj(xt_tokens.float())  # (B, Lp, C)

        # Build action tokens
        action_h = self._build_action_tokens(
            task_inds,
            horizons,
            robot_ids,
            full_qpos_list,
        )  # (B, C)

        # Query tokens
        h = h.type(self.dtype)
        action_h = action_h.type(self.dtype)

        for block in self.cond_blocks:
            h = block(h, action_h)

        return h

    def forward_flow(
        self,
        cond_queries: Tensor,  # (B, lp, model_channels) — from forward_cond
        noisy_latent: Tensor,  # (B, num_latent_tokens, token_dim) — current flow state
        flow_t: Tensor,  # (B,) — flow timestep in [0, 1]
    ) -> Tensor:
        """Stage 2: predict velocity given pre-computed conditioning (called per ODE step).

        Returns: v_pred (B, num_latent_tokens, token_dim)
        """
        # Project noisy latent to flow_model_channels
        flow_h = self.flow_latent_proj(
            noisy_latent.float()
        )  # (B, num_latent_tokens, flow_ch)

        # Timestep embedding for modulation
        t_emb = self.flow_time_embedder(flow_t.float() * 1000)  # (B, flow_ch)

        flow_h = flow_h.type(self.dtype)
        for block in self.flow_blocks:
            flow_h = block(flow_h, mod=t_emb.type(self.dtype), cond=cond_queries)

        flow_h = flow_h.float()

        return self.flow_out_proj(self.flow_norm_out(flow_h))

    def forward(
        self,
        xt_tokens: Tensor,
        noisy_latent: Tensor,
        flow_t: Tensor,
        task_inds: Tensor,
        horizons: Tensor,
        robot_ids: Tensor,
        full_qpos_list: List[Optional[Tensor]],
    ) -> Tensor:
        """Convenience: compute cond + flow in one call (used during training)."""
        cond = self.forward_cond(
            xt_tokens, task_inds, horizons, robot_ids, full_qpos_list
        )
        return self.forward_flow(cond, noisy_latent, flow_t)
