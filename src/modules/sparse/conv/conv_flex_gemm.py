import importlib
import torch
import torch.nn as nn
from .. import SparseTensor
import math
import torch
import torch.nn as nn
import flex_gemm
from .flex_gemm_patch import sparse_submanifold_conv3d

FLEX_GEMM_ALGO = "masked_implicit_gemm_splitk"  # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0


def sparse_conv3d_init(
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
    assert stride == 1 and (padding is None), (
        "Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)"
    )

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (
        tuple(kernel_size)
        if isinstance(kernel_size, (list, tuple))
        else (kernel_size,) * 3
    )
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = (
        tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    )

    self.weight = nn.Parameter(
        torch.empty((out_channels, in_channels, *self.kernel_size))
    )
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    flex_gemm.ops.spconv.set_algorithm(FLEX_GEMM_ALGO)
    flex_gemm.ops.spconv.set_hashmap_ratio(FLEX_GEMM_HASHMAP_RATIO)

    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = (
        f"SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}"
    )
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation,
    )

    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)

    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError(
        "SparseInverseConv3d with flex_gemm is not implemented yet"
    )


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError(
        "SparseInverseConv3d with flex_gemm is not implemented yet"
    )


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
        self.conv = nn.Module()
        sparse_conv3d_init(
            self.conv,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            padding,
            bias,
            indice_key,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_conv3d_forward(self.conv, x)


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
        self.conv = nn.Module()
        sparse_inverse_conv3d_init(
            self.conv,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            indice_key,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_inverse_conv3d_forward(self.conv, x)
