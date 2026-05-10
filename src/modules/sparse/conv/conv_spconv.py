import torch
import torch.nn as nn
try:
    pass
    # import spconv.pytorch as spconv
except ImportError:
    print("[SPARSE][CONV] spconv not found, SparseConv3d and SparseMaxPool")
from .. import SparseTensor
from .. import DEBUG
from . import SPCONV_ALGO


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=None,
        bias=True,
        indice_key=None,
    ):
        super(SparseConv3d, self).__init__()
        algo = None
        if SPCONV_ALGO == "native":
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == "implicit_gemm":
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                bias=bias,
                indice_key=indice_key,
                algo=algo,
            )
        else:
            self.conv = spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                indice_key=indice_key,
                algo=algo,
            )
        self.stride = (
            tuple(stride)
            if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        new_data = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(
                0, fwd, torch.arange(fwd.shape[0], device=fwd.device)
            )
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(
                sorted_feats,
                sorted_coords,
                unsorted_data.spatial_shape,
                unsorted_data.batch_size,
            )  # type: ignore

        out = SparseTensor(
            new_data,
            shape=torch.Size(new_shape),
            layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(
                f"conv_{self.stride}_unsorted_data", unsorted_data
            )
            out.register_spatial_cache(f"conv_{self.stride}_sort_bwd", bwd)

        return out


class SparseMaxPool3d(nn.Module):
    """
    Sparse 3D Max Pooling using spconv backend.
    Optionally aligns output coordinates with a target SparseTensor.
    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        indice_key=None,
    ):
        super(SparseMaxPool3d, self).__init__()
        algo = None
        if SPCONV_ALGO == "native":
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == "implicit_gemm":
            algo = spconv.ConvAlgo.MaskImplicitGemm
        self.pool = spconv.SparseMaxPool3d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            indice_key=indice_key,
            algo=algo,
        )
        self.stride = (
            tuple(stride)
            if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )
        self.padding = padding

    def forward(self, x: SparseTensor, target: SparseTensor = None) -> SparseTensor:
        """
        Forward pass for sparse max pooling.

        Args:
            x: Input SparseTensor
            target: Optional target SparseTensor for coordinate alignment.
                    When provided, pools features and gathers them to match
                    target coordinates exactly (for residual additions).
        """
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding != 0)

        # Perform pooling
        pooled_data = self.pool(x.data)

        in_channels = x.feats.shape[-1]
        new_shape = [x.shape[0], in_channels]

        if target is not None:
            # Align pooled output to target coordinates
            # Build a hash map from pooled coordinates to their indices
            pooled_coords = pooled_data.indices  # [N_pooled, 4] - (batch, z, y, x)
            target_coords = target.coords  # [N_target, 4]

            # Use a spatial hash to find matching coordinates
            spatial_shape = pooled_data.spatial_shape
            # Hash: batch * prod(spatial_shape) + z * (y_max * x_max) + y * x_max + x
            multipliers = torch.tensor(
                [
                    spatial_shape[0] * spatial_shape[1] * spatial_shape[2],
                    spatial_shape[1] * spatial_shape[2],
                    spatial_shape[2],
                    1,
                ],
                device=pooled_coords.device,
                dtype=torch.long,
            )

            pooled_hash = (pooled_coords.long() * multipliers).sum(-1)
            target_hash = (target_coords.long() * multipliers).sum(-1)

            # Create lookup table from pooled hash -> pooled index
            # For each target coord, find corresponding pooled feature
            pooled_feats = pooled_data.features

            # Sort pooled by hash for efficient lookup
            sorted_idx = pooled_hash.argsort()
            sorted_hash = pooled_hash[sorted_idx]

            # Use searchsorted to find where target hashes would fit
            insert_positions = torch.searchsorted(sorted_hash, target_hash)

            # Clamp to valid range and verify matches
            insert_positions = insert_positions.clamp(0, len(sorted_hash) - 1)
            matched_hash = sorted_hash[insert_positions]

            # Where hash matches, gather features; else use zeros
            matches = matched_hash == target_hash
            gathered_indices = sorted_idx[insert_positions]

            # Gather features
            aligned_feats = torch.zeros(
                target_coords.shape[0],
                in_channels,
                device=pooled_feats.device,
                dtype=pooled_feats.dtype,
            )
            aligned_feats[matches] = pooled_feats[gathered_indices[matches]]

            # Build output SparseTensor with target's coordinates and layout
            out = SparseTensor(
                aligned_feats,
                target_coords,
                torch.Size(new_shape),
                target.layout,
            )
            out._scale = tuple([s * stride for s, stride in zip(x._scale, self.stride)])
            out._spatial_cache = x._spatial_cache
            return out

        new_data = pooled_data
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv with non-1 stride breaks contiguous output, sort by coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(
                0, fwd, torch.arange(fwd.shape[0], device=fwd.device)
            )
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(
                sorted_feats,
                sorted_coords,
                unsorted_data.spatial_shape,
                unsorted_data.batch_size,
            )

        out = SparseTensor(
            new_data,
            shape=torch.Size(new_shape),
            layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(
                f"maxpool_{self.stride}_unsorted_data", unsorted_data
            )
            out.register_spatial_cache(f"maxpool_{self.stride}_sort_bwd", bwd)

        return out


class SparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super(SparseInverseConv3d, self).__init__()
        if "spconv" not in globals():
            import spconv.pytorch as spconv
        self.conv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key
        )
        self.stride = (
            tuple(stride)
            if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride)
        if spatial_changed:
            # recover the original spconv order
            data = x.get_spatial_cache(f"conv_{self.stride}_unsorted_data")
            bwd = x.get_spatial_cache(f"conv_{self.stride}_sort_bwd")
            data = data.replace_feature(x.feats[bwd])
            if DEBUG:
                assert torch.equal(data.indices, x.coords[bwd]), (
                    "Recover the original order failed"
                )
        else:
            data = x.data

        new_data = self.conv(data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout
        out = SparseTensor(
            new_data,
            shape=torch.Size(new_shape),
            layout=new_layout,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out
