from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from ..sparse_elastic_mixin import SparseTransformerElasticMixin


class SparseConvBlock(nn.Module):
    """
    A simple sparse convolution block: Conv -> Norm -> Activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        num_groups: Number of groups for group normalization.
        indice_key: Key for sparse convolution indices.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        indice_key: Optional[str] = None,
    ):
        super().__init__()
        self.conv = sp.SparseConv3d(
            in_channels, out_channels, kernel_size, indice_key=indice_key
        )
        self.norm = sp.SparseGroupNorm32(num_groups, out_channels)
        self.act = sp.SparseSiLU()

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        return h


class SLatAttrClassifier(nn.Module):
    """
    Attribute classifier for structured latent VAE.
    Classifies each voxel in the encoded latent space for multiple binary attributes.

    Args:
        latent_channels: Number of channels in the latent space (from encoder).
        attr_types: List of attribute type names. Each will have its own binary classifier.
        hidden_channels: Number of hidden channels in the classifier.
        num_blocks: Number of convolution blocks.
        num_groups: Number of groups for group normalization.
        use_fp16: Whether to use float16 precision.
    """

    def __init__(
        self,
        latent_channels: int,
        attr_types: List[str],
        hidden_channels: int = 128,
        num_blocks: int = 2,
        num_groups: int = 32,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.attr_types = attr_types
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if len(attr_types) == 0:
            raise ValueError("attr_types must be a non-empty list")

        # Input projection if needed
        if latent_channels != hidden_channels:
            self.input_proj = sp.SparseLinear(latent_channels, hidden_channels)
        else:
            self.input_proj = nn.Identity()

        # Shared convolution blocks for feature extraction
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = hidden_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            self.blocks.append(
                SparseConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    num_groups=num_groups,
                    indice_key=f"classifier_block_{i}"
                    if num_blocks > 1
                    else "classifier",
                )
            )

        # Separate binary classifier head for each attribute type
        self.classifiers = nn.ModuleDict()
        for attr_type in attr_types:
            self.classifiers[attr_type] = sp.SparseLinear(hidden_channels, 1)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """Initialize weights of the classifiers."""
        # Initialize output layers with small values
        for classifier in self.classifiers.values():
            nn.init.normal_(classifier.weight, std=0.01)
            nn.init.constant_(classifier.bias, 0)

    def convert_to_fp16(self) -> None:
        """Convert the model to float16."""
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert the model to float32."""
        self.blocks.apply(convert_module_to_f32)

    def forward(self, x: sp.SparseTensor) -> dict[str, sp.SparseTensor]:
        """
        Forward pass of the attribute classifier.

        Args:
            x: SparseTensor from the encoder with shape [N, latent_channels, ...]

        Returns:
            Dictionary mapping attribute type names to SparseTensors.
            Each tensor has shape [N, 1, ...] containing binary logits per voxel.
        """
        # Project input if needed
        h = self.input_proj(x)
        h = h.type(self.dtype)
        # Apply shared convolution blocks
        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)
        # Classify each attribute type
        logits_dict = {}
        for attr_type, classifier in self.classifiers.items():
            logits_dict[attr_type] = classifier(h)

        return logits_dict


class ElasticSLatAttrClassifier(SparseTransformerElasticMixin, SLatAttrClassifier):
    """
    SLat VAE attribute classifier with elastic memory management.
    Used for training with low VRAM.
    """

    pass
