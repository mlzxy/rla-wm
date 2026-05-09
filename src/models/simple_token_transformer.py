from typing import Optional, Literal
from dataclasses import dataclass
from jaxtyping import Float, Int

import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor
import math

from src.modules.utils import convert_module_to_f16, convert_module_to_f32
from src.modules.norm import LayerNorm32, BatchNorm32
from src.models.attention_block import AttentionBlock
from src.models.sparse_structure_flow import TimestepEmbedder



class SimpleTokenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        num_tokens: int = 16,
        zero_init: bool = False,
        token_channels: Optional[int] = None,
        norm_output_tokens: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_tokens = num_tokens
        self.zero_init = zero_init
        self.norm_output_tokens = norm_output_tokens

        if num_tokens > 0: # for internal tokens
            self.tokens = nn.Parameter(torch.randn(num_tokens, model_channels) * 0.2)
        else:
            self.tokens = None
        if token_channels is not None and token_channels != model_channels: # for input tokens
            self.token_proj = nn.Linear(token_channels, model_channels)
        else:
            self.token_proj = None
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    channels=model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_fp16=use_fp16,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_init)
        if self.zero_init:
            nn.init.zeros_(self.out_layer.weight)
            nn.init.zeros_(self.out_layer.bias)

    def convert_to_fp16(self) -> None:
        """Convert model torso to fp16."""
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert model torso to fp32."""
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, Lp, C)
        h = self.input_layer(x.float())
       
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


class FiLMSimpleTokenTransformer(nn.Module):
    """Simple token transformer with timestep FiLM and qpos token concat.

    - Visual stream: x (+ optional token_cond) concatenated on channel dim.
    - Qpos stream: qpos_tokens are projected to model width and prepended as tokens.
    - FiLM source: flow_t only (per-block modulation through AttentionBlock).
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        t_emb_channels: int = 256,
        qpos_token_channels: Optional[int] = None,
        use_fp16: bool = False,
        zero_init: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.out_channels = int(out_channels)
        self.t_emb_channels = int(t_emb_channels)
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Linear(self.in_channels, self.model_channels)
        qtok_ch = self.model_channels if qpos_token_channels is None else int(qpos_token_channels)
        self.qpos_token_proj = (
            nn.Identity()
            if qtok_ch == self.model_channels
            else nn.Linear(qtok_ch, self.model_channels)
        )
        self.t_embedder = TimestepEmbedder(self.t_emb_channels)

        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    channels=self.model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_fp16=use_fp16,
                    cond_channels=self.t_emb_channels,
                    use_condition=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(self.model_channels)
        self.out_layer = nn.Linear(self.model_channels, self.out_channels)
        self.zero_init = bool(zero_init)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        if self.zero_init:
            nn.init.zeros_(self.out_layer.weight)
            nn.init.zeros_(self.out_layer.bias)

    def convert_to_fp16(self) -> None:
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, Cx)
        flow_t: Optional[torch.Tensor] = None,  # (B,)
        token_cond: Optional[torch.Tensor] = None,  # (B, L, Ccond), concat on channel
        qpos_tokens: Optional[torch.Tensor] = None,  # (B, Lq, Cq), concat on sequence
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, L, C), got {tuple(x.shape)}")

        feats = x.float()
        if token_cond is not None:
            if token_cond.ndim != 3:
                raise ValueError(
                    f"Expected token_cond with shape (B, L, C), got {tuple(token_cond.shape)}"
                )
            if token_cond.shape[:2] != feats.shape[:2]:
                raise ValueError(
                    "token_cond and x must match on (B, L), got "
                    f"{tuple(token_cond.shape[:2])} vs {tuple(feats.shape[:2])}"
                )
            feats = torch.cat([feats, token_cond.float()], dim=-1)

        if feats.shape[-1] != self.in_channels:
            raise ValueError(
                f"Input feature dim mismatch, expected {self.in_channels}, got {feats.shape[-1]}"
            )

        h_vis = self.input_layer(feats)  # (B, L, Cm)
        qn = 0
        if qpos_tokens is not None:
            if qpos_tokens.ndim != 3:
                raise ValueError(
                    f"Expected qpos_tokens with shape (B, Lq, Cq), got {tuple(qpos_tokens.shape)}"
                )
            if qpos_tokens.shape[0] != h_vis.shape[0]:
                raise ValueError(
                    f"Batch mismatch between x and qpos_tokens: {h_vis.shape[0]} vs {qpos_tokens.shape[0]}"
                )
            q_h = self.qpos_token_proj(qpos_tokens.float())
            qn = q_h.shape[1]
            h = torch.cat([q_h, h_vis], dim=1)
        else:
            h = h_vis

        bsz = h.shape[0]
        if flow_t is None:
            flow_t = torch.zeros(bsz, device=h.device, dtype=h.dtype)
        t_mod = self.t_embedder(flow_t.float()).type(self.dtype)

        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h, cond=t_mod)

        h = h.float()
        h_vis_out = h[:, qn:]
        return self.out_layer(self.norm_out(h_vis_out))
