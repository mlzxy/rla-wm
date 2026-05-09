
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import math

from src.models.sparse_structure_flow import TimestepEmbedder
from src.modules.utils import convert_module_to_f16, convert_module_to_f32
from src.modules.norm import LayerNorm32, BatchNorm32
from src.modules.sparse import SparseTensor
from src.modules.sparse.attention import SparseMultiHeadAttention



class _DinoDynamicsBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        num_heads: int,
        mlp_ratio: float,
        attention_type: Literal["global", "local"],
        use_fp16: bool = False,
        use_condition: bool = True,
        cond_bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.attention_type = attention_type
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.norm1 = LayerNorm32(channels, elementwise_affine=False)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.use_condition = use_condition
        if use_condition:
            self.cond_proj = nn.Linear(cond_channels, channels, bias=cond_bias)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=cond_bias)
            )
        else:
            self.cond_proj = None
            self.modulation = None

        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def convert_to_fp16(self) -> None:
        """Convert attention/MLP torso to fp16."""
        self.dtype = torch.float16
        self.attn.apply(convert_module_to_f16)
        self.mlp.apply(convert_module_to_f16)
        if self.cond_proj is not None:
            self.cond_proj.apply(convert_module_to_f16)
        if self.modulation is not None:
            self.modulation.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert attention/MLP torso to fp32."""
        self.dtype = torch.float32
        self.attn.apply(convert_module_to_f32)
        self.mlp.apply(convert_module_to_f32)
        if self.cond_proj is not None:
            self.cond_proj.apply(convert_module_to_f32)
        if self.modulation is not None:
            self.modulation.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor | None=None,  # (B, x)
        cond: torch.Tensor | None=None,
    ) -> torch.Tensor:
        # x: (B, Cam, Lp, C)
        bsz, cams, lp, ch = x.shape

        if self.use_condition:
            assert self.cond_proj is not None
            assert self.modulation is not None
            if mod is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.modulation(self.cond_proj(mod)).chunk(6, dim=-1)
                )
                scale_msa = scale_msa[:, None, None, :]
                shift_msa = shift_msa[:, None, None, :]
                gate_msa = gate_msa[:, None, None, :]
                scale_mlp = scale_mlp[:, None, None, :]
                shift_mlp = shift_mlp[:, None, None, :]
                gate_mlp = gate_mlp[:, None, None, :]
            elif cond is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.modulation(self.cond_proj(cond)).chunk(6, dim=-1)
                )
                if shift_msa.shape[2] != lp:
                    diff = lp - shift_msa.shape[2]
                    assert diff > 0, f"Condition length {shift_msa.shape[2]} cannot be greater than input length {lp}"
                    # Pad missing patch tokens with identity modulation.
                    # need check
                    shift_msa = F.pad(shift_msa, (0, 0, diff, 0), value=0.0)
                    scale_msa = F.pad(scale_msa, (0, 0, diff, 0), value=0.0)
                    gate_msa = F.pad(gate_msa, (0, 0, diff, 0), value=1.0)
                    shift_mlp = F.pad(shift_mlp, (0, 0, diff, 0), value=0.0)
                    scale_mlp = F.pad(scale_mlp, (0, 0, diff, 0), value=0.0)
                    gate_mlp = F.pad(gate_mlp, (0, 0, diff, 0), value=1.0)
        else:
            shift_msa = 0.0
            scale_msa = 0.0
            gate_msa = 1.0
            shift_mlp = 0.0
            scale_mlp = 0.0
            gate_mlp = 1.0

        h = self.norm1(x)
        h = h * (1.0 + scale_msa) + shift_msa

        if self.attention_type == "global":
            # Global Attention: 全局视角和空间特征一起交互
            assert self.attn is not None
            h_seq = h.reshape(bsz, cams * lp, ch)
            h_seq, _ = self.attn(h_seq, h_seq, h_seq, need_weights=False)
            h = h_seq.reshape(bsz, cams, lp, ch)
            
        elif self.attention_type == "local":
            # Local Attention: 只在同一张图像 (单个 Camera) 的 Patch 内交互
            assert self.attn is not None
            h_seq = h.reshape(bsz * cams, lp, ch)
            h_seq, _ = self.attn(h_seq, h_seq, h_seq, need_weights=False)
            h = h_seq.reshape(bsz, cams, lp, ch)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
            
        x = x + h * gate_msa
        h2 = self.norm2(x)
        h2 = h2 * (1.0 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        x = x + h2 * gate_mlp
        
        return x
    
    
    

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        use_fp16: bool = False,
        cond_channels: int = -1,
        use_condition: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.norm1 = LayerNorm32(channels, elementwise_affine=False)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        if use_condition:
            self.cond_proj = nn.Linear(cond_channels, channels)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels)
            )
        else:
            self.cond_proj = None
            self.modulation = None
        self.use_condition = use_condition

        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def convert_to_fp16(self) -> None:
        """Convert attention/MLP torso to fp16."""
        self.dtype = torch.float16
        self.attn.apply(convert_module_to_f16)
        self.mlp.apply(convert_module_to_f16)
        if self.cond_proj is not None:
            self.cond_proj.apply(convert_module_to_f16)
        if self.modulation is not None:
            self.modulation.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert attention/MLP torso to fp32."""
        self.dtype = torch.float32
        self.attn.apply(convert_module_to_f32)
        self.mlp.apply(convert_module_to_f32)
        if self.cond_proj is not None:
            self.cond_proj.apply(convert_module_to_f32)
        if self.modulation is not None:
            self.modulation.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None=None,
    ) -> torch.Tensor:
        bsz, lp, ch = x.shape
        if self.use_condition:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation(self.cond_proj(cond)).chunk(6, dim=-1)
            )
            if scale_msa.ndim == 2:
                scale_msa = scale_msa[:,  None, :]
                shift_msa = shift_msa[:,  None, :]
                gate_msa = gate_msa[:,  None, :]
                scale_mlp = scale_mlp[:,  None, :]
                shift_mlp = shift_mlp[:,  None, :]
                gate_mlp = gate_mlp[:,  None, :]

            h = self.norm1(x)
            h = h * (1.0 + scale_msa) + shift_msa
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h * gate_msa
            h2 = self.norm2(x)
            h2 = h2 * (1.0 + scale_mlp) + shift_mlp
            h2 = self.mlp(h2)
            x = x + h2 * gate_mlp
        else:
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h 
            h2 = self.norm2(x)
            h2 = self.mlp(h2)
            x = x + h2 
        return x
    
    


class ModCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        cond_channels: int,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.norm1 = LayerNorm32(channels, elementwise_affine=False)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False)

        self.self_attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            kdim=cond_channels,
            vdim=cond_channels,
            batch_first=True,
        )
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 6 * channels)
        )


        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def convert_to_fp16(self) -> None:
        """Convert attention/MLP torso to fp16."""
        self.dtype = torch.float16
        self.self_attn.apply(convert_module_to_f16)
        self.cross_attn.apply(convert_module_to_f16)
        self.mlp.apply(convert_module_to_f16)
        self.modulation.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert attention/MLP torso to fp32."""
        self.dtype = torch.float32
        self.self_attn.apply(convert_module_to_f32)
        self.cross_attn.apply(convert_module_to_f32)
        self.mlp.apply(convert_module_to_f32)
        self.modulation.apply(convert_module_to_f32)


    def forward(
        self,
        x: torch.Tensor, # B, L, C
        mod: torch.Tensor, # B, C
        cond: torch.Tensor # B, Lcond, Ccond
    ) -> torch.Tensor:
        _bsz, _lp, _ch = x.shape
        _bsz_cond, _lp_cond, _ch_cond = cond.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation(mod).chunk(6, dim=-1)
            )
        scale_msa = scale_msa[:,  None, :]
        shift_msa = shift_msa[:,  None, :]
        gate_msa = gate_msa[:,  None, :]
        scale_mlp = scale_mlp[:,  None, :]
        shift_mlp = shift_mlp[:,  None, :]
        gate_mlp = gate_mlp[:,  None, :]

        h = self.norm1(x)
        h = h * (1.0 + scale_msa) + shift_msa
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h * gate_msa

        h = self.norm2(x)
        h, _ = self.cross_attn(h, cond, cond, need_weights=False)
        x = x + h

        h2 = self.norm3(x)
        h2 = h2 * (1.0 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        x = x + h2 * gate_mlp

        return x