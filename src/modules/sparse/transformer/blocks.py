from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from ...utils import zero_module


def align_sparse_tensor(cond: SparseTensor, target: SparseTensor, res = 4096) -> torch.Tensor:
    """Aligns cond features to target features by matching coordinates via hashing."""
    N_t = target.feats.shape[0]
    C_c = cond.feats.shape[1]
    device = target.feats.device
    dtype = target.feats.dtype

    if cond.feats.shape[0] == target.feats.shape[0] and torch.equal(cond.coords, target.coords):
        return cond.feats

    # safe large integer for unique hashing
    def get_hash(coords):
        return (coords[:, 0].long() * (res**3) +
                coords[:, 1].long() * (res**2) +
                coords[:, 2].long() * res +
                coords[:, 3].long())

    hash_c = get_hash(cond.coords)
    hash_t = get_hash(target.coords)

    sorted_hash_t, order_t = torch.sort(hash_t)
    idx_in_sorted_t = torch.searchsorted(sorted_hash_t, hash_c)
    idx_in_sorted_t = torch.clamp(idx_in_sorted_t, max=N_t - 1)
    
    valid_mask = sorted_hash_t[idx_in_sorted_t] == hash_c
    idx_in_t = order_t[idx_in_sorted_t[valid_mask]]

    out_feats = torch.zeros(N_t, C_c, device=device, dtype=dtype)
    out_feats[idx_in_t] = cond.feats[valid_mask]
    return out_feats


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        init_zero: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if init_zero:
            zero_module(self.attn.to_out)
            zero_module(self.mlp.mlp[-1])

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)


class SparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN).
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        init_zero: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if init_zero:
            zero_module(self.self_attn.to_out)
            zero_module(self.cross_attn.to_out)
            zero_module(self.mlp.mlp[-1])

    def _forward(self, x: SparseTensor, context: torch.Tensor):
        h = x.replace(self.norm1(x.feats))
        h = self.self_attn(h)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, context, use_reentrant=False
            )
        else:
            return self._forward(x, context)


class DoubleModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning from both global and local context.
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int,
        mod_channels: Optional[int] = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        init_zero: bool = False,
        cond_mode: str="local+global"
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.cond_mode = cond_mode
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        mod_channels = mod_channels if mod_channels is not None else channels

        self.self_modulate = 'self' in self.cond_mode
        channels_offset = channels if self.self_modulate else 0

        if self.self_modulate:
            self.self_norm = LayerNorm32(channels, eps=1e-6)

        if 'global' in self.cond_mode:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(mod_channels + channels_offset, 6 * channels, bias=True)
            )
        else:
            self.adaLN_modulation = None

        if 'local' in self.cond_mode:
            self.cond_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(cond_channels + channels_offset, 6 * channels, bias=True)
            )
        else:
            self.cond_modulation = None

        if init_zero:
            zero_module(self.attn.to_out)
            zero_module(self.mlp.mlp[-1])
            # By default, zero out the modulations so backward compatibility works
            # This is also good for training stability when fine-tuning
            if self.adaLN_modulation is not None:
                zero_module(self.adaLN_modulation[-1])
            if self.cond_modulation is not None:
                zero_module(self.cond_modulation[1])

    def _forward(
        self, x: SparseTensor, mod: Optional[torch.Tensor] = None, cond: Optional[SparseTensor] = None
    ) -> SparseTensor:
        # mod: (B, C)
        # cond: SparseTensor (N, C_cond)
        
        N, C = x.feats.shape
        device = x.feats.device
        dtype = x.feats.dtype
        channels = C

        # 1. Global Modulation
        if mod is not None:
            batch_indices = x.coords[:, 0].long()
            if self.self_modulate:
                mod = torch.cat([mod[batch_indices], self.self_norm(x.feats)], dim=1)

            shift_msa_g, scale_msa_g, gate_msa_g, shift_mlp_g, scale_mlp_g, gate_mlp_g = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

            if not self.self_modulate:
                shift_msa_g = shift_msa_g[batch_indices]
                scale_msa_g = scale_msa_g[batch_indices]
                gate_msa_g = gate_msa_g[batch_indices]
                shift_mlp_g = shift_mlp_g[batch_indices]
                scale_mlp_g = scale_mlp_g[batch_indices]
                gate_mlp_g = gate_mlp_g[batch_indices]
        else:
            shift_msa_g = scale_msa_g = gate_msa_g = shift_mlp_g = scale_mlp_g = gate_mlp_g = torch.zeros(
                N, channels, device=device, dtype=dtype
            )

        # 2. Local Modulation
        if cond is not None:
            # Align cond feats to x feats using spatial join, since x may contain extra learnable tokens
            cond_feats_aligned = align_sparse_tensor(cond, x)
            if self.self_modulate:
                cond_feats_aligned = torch.cat([cond_feats_aligned, self.self_norm(x.feats)], dim=1)

            shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = (
                self.cond_modulation(cond_feats_aligned).chunk(6, dim=-1)
            )
        else:
            shift_msa_c = scale_msa_c = gate_msa_c = shift_mlp_c = scale_mlp_c = gate_mlp_c = torch.zeros(
                N, channels, device=device, dtype=dtype
            )

        # Combine
        shift_msa = shift_msa_g + shift_msa_c
        scale_msa = scale_msa_g + scale_msa_c
        gate_msa = gate_msa_g + gate_msa_c
        shift_mlp = shift_mlp_g + shift_mlp_c
        scale_mlp = scale_mlp_g + scale_mlp_c
        gate_mlp = gate_mlp_g + gate_mlp_c

        # MSA
        h = x.replace(self.norm1(x.feats))
        h = h.replace(h.feats * (1 + scale_msa) + shift_msa)
        h = self.attn(h)
        h = h.replace(h.feats * (1 + gate_msa))
        x = x + h

        # MLP
        h = x.replace(self.norm2(x.feats))
        h = h.replace(h.feats * (1 + scale_mlp) + shift_mlp)
        h = self.mlp(h)
        h = h.replace(h.feats * (1 + gate_mlp))
        x = x + h

        return x

    def forward(
        self, x: SparseTensor, mod: Optional[torch.Tensor] = None, cond: Optional[SparseTensor] = None
    ) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, cond, use_reentrant=False
            )
        else:
            return self._forward(x, mod, cond)

