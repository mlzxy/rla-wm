import math
from typing import Literal, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.norm import LayerNorm32


class GlobalAttentionPooling(nn.Module):
    """
    Compresses variable length sequence (B, U, In) into a single global vector (B, 1, Out).
    Uses a single learnable query to attend to the most relevant parts of the input.
    """

    def __init__(
        self,
        input_dim: int,  # Text input dimension (I)
        output_dim: int,  # Cond channels (C_cond)
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Project Input Text to shared dimension
        self.input_proj = nn.Linear(input_dim, output_dim)

        # The "Summary Token" - A single learnable vector
        self.summary_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

        # Attention Mechanism (batch_first=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Post-Attention MLP (Capacity expansion)
        self.norm = LayerNorm32(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )
        self.norm_out = LayerNorm32(output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Text embeddings (B, U, input_dim)
            mask: Boolean mask (B, U), True indicates padding (ignored positions).
        Returns:
            out: (B, 1, output_dim)
        """
        B, U, _ = x.shape

        # Project K, V
        val = self.input_proj(x)  # (B, U, C)

        # Expand Query for the batch
        query = self.summary_query.expand(B, -1, -1)  # (B, 1, C)

        # PyTorch MHA expects key_padding_mask where True = IGNORE
        attn_out, _ = self.attn(query=query, key=val, value=val, key_padding_mask=mask)

        # Residual + Norm + MLP
        h = query + attn_out
        h = self.norm(h)
        h = h + self.mlp(h)
        out = self.norm_out(h)

        return out
