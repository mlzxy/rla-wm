from typing import *
import torch
import torch.nn as nn
from jaxtyping import Float, Int, UInt64
from torch import Tensor
from . import BACKEND, DEBUG
from .basic import SparseTensor as SparseTensorBase

try:
    from torch_scatter import segment_csr
except ImportError:
    pass
SparseTensorData = None  # Lazy import


__all__ = [
    "SparseTensor",
    "sparse_batch_broadcast",
    "sparse_batch_op",
    "sparse_cat",
    "sparse_unbind",
]


class SparseTensor(SparseTensorBase):
    """
    Sparse tensor with support for both torchsparse and spconv backends.

    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor (int or float).
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion
    - resolution (Tuple[int, int, int] | int): Resolution of the voxel grid (D, H, W).

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """

    @overload
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: Optional[torch.Size] = None,
        layout: Optional[List[slice]] = None,
        resolution: Optional[Union[int, Tuple[int, int, int]]] = None,
        **kwargs,
    ): ...

    @overload
    def __init__(
        self,
        data,
        shape: Optional[torch.Size] = None,
        layout: Optional[List[slice]] = None,
        **kwargs,
    ): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        global SparseTensorData
        if SparseTensorData is None:
            import importlib

            if BACKEND == "torchsparse":
                SparseTensorData = importlib.import_module("torchsparse").SparseTensor
            elif BACKEND == "spconv":
                SparseTensorData = importlib.import_module(
                    "spconv.pytorch"
                ).SparseConvTensor

        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if "data" in kwargs else 0

        # Pop custom args from kwargs early to avoid backend error
        resolution_arg = kwargs.pop("resolution", None)
        float_coords_arg = kwargs.pop("float_coords", None)
        norm_coords_arg = kwargs.pop("norm_coords", None)

        if norm_coords_arg is not None:
            if resolution_arg is None:
                raise ValueError(
                    "resolution must be provided when initializing with norm_coords"
                )

            # Convert norm_coords to float_coords
            # float = (norm + 1) * (res - 1) / 2
            if isinstance(resolution_arg, int):
                res_t = torch.tensor(
                    [resolution_arg, resolution_arg, resolution_arg],
                    device=norm_coords_arg.device,
                    dtype=norm_coords_arg.dtype,
                )
            else:
                res_t = torch.tensor(
                    resolution_arg,
                    device=norm_coords_arg.device,
                    dtype=norm_coords_arg.dtype,
                )

            # Clone to avoid modifying input
            computed_float_coords = norm_coords_arg.clone()
            # Inverse normalize spatial dims
            computed_float_coords[:, 1:] = (
                (norm_coords_arg[:, 1:] + 1) * (res_t - 1) / 2
            )

            # If coords not provided, use these computed float coords
            # args[1] corresponds to coords if len(args) > 1
            if "coords" not in kwargs and len(args) <= 1:
                kwargs["coords"] = computed_float_coords

            # Also ensure float_coords_arg is set if not explicitly passed,
            # so we preserve the float precision explicitly if logic down ref uses float_coords_arg
            if float_coords_arg is None:
                float_coords_arg = computed_float_coords

        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args))
            if "feats" in kwargs:
                feats = kwargs["feats"]
                del kwargs["feats"]
            if "coords" in kwargs:
                coords = kwargs["coords"]
                del kwargs["coords"]
            if "shape" in kwargs:
                shape = kwargs["shape"]
                del kwargs["shape"]
            if "layout" in kwargs:
                layout = kwargs["layout"]
                del kwargs["layout"]

            if coords is None and float_coords_arg is not None:
                coords = float_coords_arg

            _float_coords_init = None
            # Per user request: if dtype is not int, treat as float coords to preserve precision
            if coords is not None and coords.dtype not in [
                torch.int32,
                torch.int64,
                torch.int16,
                torch.int8,
                torch.uint8,
            ]:
                _float_coords_init = coords
                coords = torch.round(coords).int()

            if shape is None:
                shape = self.__cal_shape(feats, coords)
            if layout is None:
                layout = self.__cal_layout(coords, shape[0])
            if BACKEND == "torchsparse":
                self.data = SparseTensorData(feats, coords, **kwargs)
            elif BACKEND == "spconv":
                spatial_shape = list(coords.max(0)[0] + 1)[1:]
                # Note: spconv may not like duplicates. If we assume lazy sync handles it, fine.
                # Otherwise, this might be a point of failure if user passes duplicates and spconv rejects/merges in weird ways.
                # However, user said "duplicate voxels... shall be FINE".
                # We will rely on SparseTensorData to handle it or crash, but we maintain _float_coords.
                self.data = SparseTensorData(
                    feats.reshape(feats.shape[0], -1),
                    coords,
                    spatial_shape,
                    shape[0],
                    **kwargs,
                )
                self.data._features = feats
        elif method_id == 1:
            data, shape, layout = args + (None,) * (3 - len(args))
            if "data" in kwargs:
                data = kwargs["data"]
                del kwargs["data"]
            if "shape" in kwargs:
                shape = kwargs["shape"]
                del kwargs["shape"]
            if "layout" in kwargs:
                layout = kwargs["layout"]
                del kwargs["layout"]

            self.data = data
            if shape is None:
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None:
                layout = self.__cal_layout(self.coords, shape[0])

        self._shape = shape
        self._layout = layout
        self._scale = kwargs.pop("scale", (1, 1, 1))
        self._spatial_cache = kwargs.pop("spatial_cache", {})
        self._resolution = resolution_arg
        if hasattr(self, "_resolution") and isinstance(self._resolution, int):
            self._resolution = (self._resolution, self._resolution, self._resolution)

        # Float coords from kwargs if present (e.g. from replace)
        # float_coords_arg already popped

        if method_id == 0:
            if _float_coords_init is not None:
                self._float_coords = _float_coords_init
            else:
                # Coords were passed as int (or inferred)
                self._float_coords = (
                    self.coords.float() if hasattr(self, "coords") else None
                )
        else:
            # method_id == 1 (initialized from data)
            self._float_coords = (
                self.coords.float() if hasattr(self, "coords") else None
            )

        # If float_coords was passed in kwargs (e.g. via replace), use it overrides everything
        if float_coords_arg is not None:
            self._float_coords = float_coords_arg
            # We don't necessarily want to _sync() here if we want to preserve exact float coords
            # and they are already close to the rounded ones in data.Indices.
            # But SparseTensorExt._sync() rounds and sets self.coords.
            # If we are initializing from existing data, self.coords is already set.
            # If we call _sync(), we might regenerate int coords.
            # However, SparseTensorExt.__init__ for method_id=0 already did it.
            # For method_id=1, we might need it if float_coords_arg is new.
            if method_id == 1:
                self._sync()
            pass

        if DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], (
                    f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                )
                assert self.shape == self.__cal_shape(self.feats, self.coords), (
                    f"Invalid shape: {self.shape}"
                )
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), (
                    f"Invalid layout: {self.layout}"
                )
                for i in range(self.shape[0]):
                    assert torch.all(self.coords[self.layout[i], 0] == i), (
                        f"The data of batch {i} is not contiguous"
                    )
            except Exception as e:
                print("Debugging information:")
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords: {self.coords}")
                raise e

    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)

    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [
            slice((offset[i] - seq_len[i]).item(), offset[i].item())
            for i in range(batch_size)
        ]
        return layout

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def dim(self) -> int:
        return len(self.shape)

    @property
    def layout(self) -> List[slice]:
        return self._layout

    @property
    def feats(self) -> Float[Tensor, "N C"]:
        if BACKEND == "torchsparse":
            return self.data.F
        elif BACKEND == "spconv":
            return self.data.features

    @feats.setter
    def feats(self, value: torch.Tensor):
        if BACKEND == "torchsparse":
            self.data.F = value
        elif BACKEND == "spconv":
            self.data.features = value

    @property
    def float_coords(self) -> Float[Tensor, "N 4"]:
        return self._float_coords

    @float_coords.setter
    def float_coords(self, value: torch.Tensor):
        self._float_coords = value
        self._sync()

    def _set_float_coords_without_sync(self, value: torch.Tensor):
        self._float_coords = value

    def _sync(self):
        """Sync backend int coords with internal float coords"""
        if self._float_coords is None:
            return

        # Round to nearest integer
        rounded_coords = torch.round(self._float_coords).int()

        # Update backend data
        # Note: We do NOT update self.data directly if possible, we use the setter
        self.coords = rounded_coords

        # Update shape if extents changed
        # We assume layout (batch grouping) doesn't change as per user "voxels DO not move across samples"
        # But we do need to update shape if max coord increases
        if self._shape is not None:
            # Re-calculate shape dim 0 (spatial extent)
            # Basic implementation:
            new_max = rounded_coords[:, 0].max().item() + 1
            new_shape_list = list(self._shape)
            new_shape_list[0] = new_max
            self._shape = torch.Size(new_shape_list)

    @property
    def norm_coords(self) -> Float[Tensor, "N 4"]:
        """Normalized coordinates in [-1, 1] based on resolution."""
        if self._resolution is None:
            raise ValueError("resolution must be set to use norm_coords")

        # Map [0, res-1] to [-1, 1]? Or [0, res] to [-1, 1]?
        # Usually standard is:
        # range = res - 1 (if 0..7)
        # val / range * 2 - 1

        # Batch dim is at index 0, we don't normalize it usually?
        # Or do we? User said "coordinates ranging from {0, ... ,7}... read the coordinates in [-1.0, 1.0]"
        # Typically batch index shouldn't be normalized to [-1, 1] spatial range.
        # So we normalize dims 1, 2, 3.

        res_tensor = torch.tensor(
            self._resolution, device=self.device, dtype=self.dtype
        )

        # Assuming coords is [B, 3] or [N, 4] (batch + 3)
        # basic_ext indicates coords is [N, 4] usually (batch idx at 0) based on `__cal_shape`

        normalized = self._float_coords.clone()
        # Skip batch dim
        coord_vals = normalized[:, 1:]

        # Normalize: 2 * x / (res - 1) - 1
        # If res=8, max idx=7. 7/7 * 2 - 1 = 1. 0 -> -1.
        scale = 2.0 / (res_tensor - 1.0)
        coord_vals = coord_vals * scale - 1.0

        normalized[:, 1:] = coord_vals
        return normalized

    @property
    def coords(self) -> Int[Tensor, "N 4"]:
        if BACKEND == "torchsparse":
            return self.data.C
        elif BACKEND == "spconv":
            return self.data.indices

    @property
    def full_feats(self) -> Float[Tensor, "N 3+C"]:
        return torch.cat([self.norm_coords[:, 1:], self.feats], dim=1)

    @coords.setter
    def coords(self, value: torch.Tensor):
        if BACKEND == "torchsparse":
            self.data.C = value
        elif BACKEND == "spconv":
            self.data.indices = value

    def dedup(self) -> "SparseTensor":
        """
        Deduplicate voxels by averaging features and float_coords for duplicates.
        Returns a new SparseTensor.
        """
        # 1. Get discrete coords (which we want to be unique)
        # self.coords is already synced and rounded
        discrete_coords = self.coords

        # 2. Find unique coords and inverse indices
        # We need unique ROWS.
        # torch.unique with dim=0
        unique_coords, inverse_indices = torch.unique(
            discrete_coords, dim=0, return_inverse=True, sorted=True
        )

        # 3. Aggregate features and float_coords
        # inverse_indices maps original -> unique
        # We want to sum/mean over these groups.
        # Use torch_scatter.segment_csr (fast) or scatter_mean

        # segment_csr expects sorted indices usually?
        # torch.unique with sorted=True returns unique_coords sorted.
        # inverse_indices are consistent with unique_coords.
        # But segment_csr expects the data to be sorted by cluster?
        # No, segment_csr expects `indptr` (ptr to start/end).
        # scatter_mean takes `index` (inverse_indices).

        # If we don't have scatter_mean, we can use user suggested approach:
        # sort by inverse_indices, then segment_csr.

        perm = torch.argsort(inverse_indices)
        feats_sorted = self.feats[perm]
        float_coords_sorted = self._float_coords[perm]

        # Calculate counts
        unique_counts = torch.bincount(inverse_indices)

        # Construct ptr
        # ptr = cumsum of counts
        # [0, count0, count0+count1, ...]
        ptr = torch.cat(
            [unique_counts.new_zeros(1), torch.cumsum(unique_counts, dim=0)]
        )

        new_feats = segment_csr(feats_sorted, ptr, reduce="mean")
        new_float_coords = segment_csr(float_coords_sorted, ptr, reduce="mean")

        # 4. Create new SparseTensor
        # We have new_float_coords (averaged) and matches unique_coords (discrete)
        # unique_coords should match look of round(new_float_coords).

        return SparseTensor(
            feats=new_feats,
            coords=new_float_coords,
            resolution=self._resolution,
        )

    def sample_to(self, num: int) -> tuple["SparseTensor", UInt64[Tensor, "N"]]:
        keep_indices = []
        for i in range(self.shape[0]):
            sl = self.layout[i]
            n_points = sl.stop - sl.start
            if n_points > num:
                perm = torch.randperm(n_points, device=self.device)[:num]
                keep_indices.append(sl.start + perm)
            elif n_points < num:
                # Take all existing points
                base_indices = torch.arange(sl.start, sl.stop, device=self.device)
                # Sample remaining with replacement
                needed = num - n_points
                if n_points > 0:
                    rand_indices = torch.randint(
                        0, n_points, (needed,), device=self.device
                    )
                    extra_indices = sl.start + rand_indices
                    keep_indices.append(torch.cat([base_indices, extra_indices]))
                else:
                    # If n_points is 0? Rare for sparse tensor but possible if empty batch?
                    # If empty, we can't really sample.
                    # User said "for each sample... if it does not have required num".
                    # If it has 0, we can't sample from it.
                    # We might warn or just return empty?
                    # But if we want exactly num points, we need to pad?
                    # SparseTensor usually implies valid coordinates.
                    # If empty, we probably shouldn't invent coordinates.
                    # But let's assume non-empty for now or handle gracefully.
                    # If empty, we just append nothing? But then it won't be mismatching `num`.
                    # Let's assume n_points > 0.
                    pass
            else:
                keep_indices.append(torch.arange(sl.start, sl.stop, device=self.device))

        if len(keep_indices) == 0:
            return self, torch.tensor([], dtype=torch.long, device=self.device)

        keep_indices = torch.cat(keep_indices)

        if self._float_coords is not None:
            final_coords_arg = self._float_coords[keep_indices]
        else:
            final_coords_arg = self.coords[keep_indices]

        return SparseTensor(
            feats=self.feats[keep_indices],
            coords=final_coords_arg,
            resolution=self._resolution,
            # scale=self._scale,
            # spatial_cache=self._spatial_cache,
        ), keep_indices

    def shrink_to(self, num: int) -> tuple["SparseTensor", UInt64[Tensor, "N"]]:
        keep_indices = []
        for i in range(self.shape[0]):
            sl = self.layout[i]
            n_points = sl.stop - sl.start
            if n_points > num:
                perm = torch.randperm(n_points, device=self.device)[:num]
                keep_indices.append(sl.start + perm)
            else:
                keep_indices.append(torch.arange(sl.start, sl.stop, device=self.device))

        if len(keep_indices) == 0:
            return self

        keep_indices = torch.cat(keep_indices)

        if self._float_coords is not None:
            final_coords_arg = self._float_coords[keep_indices]
        else:
            final_coords_arg = self.coords[keep_indices]

        return SparseTensor(
            feats=self.feats[keep_indices],
            coords=final_coords_arg,
            resolution=self._resolution,
            # scale=self._scale,
            # spatial_cache=self._spatial_cache,
        ), keep_indices

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @overload
    def to(self, dtype: torch.dtype) -> "SparseTensor": ...

    @overload
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SparseTensor": ...

    def to(self, *args, **kwargs) -> "SparseTensor":
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if "dtype" in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs["dtype"]
        if "device" in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs["device"]

        new_feats = self.feats.to(device=device, dtype=dtype)
        # We MUST move float_coords too to preserve gradient flow
        new_float_coords = None
        if self._float_coords is not None:
            new_float_coords = self._float_coords.to(device=device)

        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords, float_coords=new_float_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> "SparseTensor":
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        new_float_coords = (
            self._float_coords.cpu() if self._float_coords is not None else None
        )
        return self.replace(new_feats, new_coords, float_coords=new_float_coords)

    def cuda(self) -> "SparseTensor":
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        new_float_coords = (
            self._float_coords.cuda() if self._float_coords is not None else None
        )
        return self.replace(new_feats, new_coords, float_coords=new_float_coords)

    def half(self) -> "SparseTensor":
        new_feats = self.feats.half()
        return self.replace(new_feats)

    def float(self) -> "SparseTensor":
        new_feats = self.feats.float()
        return self.replace(new_feats)

    def detach(self) -> "SparseTensor":
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        new_float_coords = (
            self._float_coords.detach() if self._float_coords is not None else None
        )
        return self.replace(new_feats, new_coords, float_coords=new_float_coords)

    def dense(self) -> torch.Tensor:
        if BACKEND == "torchsparse":
            return self.data.dense()
        elif BACKEND == "spconv":
            return self.data.dense()

    def reshape(self, *shape) -> "SparseTensor":
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)

    def unbind(self, dim: int) -> List["SparseTensor"]:
        return sparse_unbind(self, dim)

    def replace(
        self,
        feats: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        float_coords: Optional[torch.Tensor] = None,
    ) -> "SparseTensor":
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        if BACKEND == "torchsparse":
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        elif BACKEND == "spconv":
            new_data = SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict,
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
            if coords is not None:
                new_data.indices = coords
        new_tensor = SparseTensor(
            new_data,
            shape=torch.Size(new_shape),
            layout=self.layout,
            scale=self._scale,
            spatial_cache=self._spatial_cache,
            resolution=self._resolution,
            float_coords=float_coords,  # Pass explicitly
        )
        # If float_coords was NOT passed, we reuse existing coords. We should preserve existing float_coords
        if coords is None and float_coords is None:
            new_tensor._float_coords = self._float_coords
        elif float_coords is not None:
            # already set in __init__? Let's be sure.
            new_tensor._float_coords = float_coords
        else:
            # coords was passed, but not float_coords.
            if coords.is_floating_point():
                new_tensor._float_coords = coords
                new_tensor._sync()
            else:
                new_tensor._float_coords = coords.float()

        return new_tensor

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> "SparseTensor":
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        coords = torch.cat(
            [
                torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
                coords.repeat(N, 1),
            ],
            dim=1,
        ).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: "SparseTensor") -> dict:
        new_cache = {}
        for k in set(
            list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())
        ):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache

    def __neg__(self) -> "SparseTensor":
        return self.replace(-self.feats)

    def __elemwise__(
        self, other: Union[torch.Tensor, "SparseTensor"], op: callable
    ) -> "SparseTensor":
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __add__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)

    def __radd__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)

    def __sub__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.sub)

    def __rsub__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)

    def __rmul__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)

    def __truediv__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(
        self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), (
                    f"Invalid index shape: {idx.shape}"
                )
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            coords.append(self.coords[self.layout[old_idx]].clone())
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        coords = torch.cat(coords, dim=0).contiguous()
        feats = torch.cat(feats, dim=0).contiguous()
        return SparseTensor(feats=feats, coords=coords)

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)

    def visualize(self, save_path: str):
        try:
            # Attempt relative import first, then absolute if needed (or assume utils is in path)
            # The user snippet used "....utils.pv".
            # basic_ext is in src.modules.sparse.
            # src.modules.sparse -> .. -> modules -> ... -> src -> .... -> root?
            # If utils is in root, then ....utils is correct relative to src.modules.sparse.
            # But let's try standard import if root is in pythonpath
            try:
                from utils.pv import render_pcds
            except ImportError:
                from ....utils.pv import render_pcds
        except (ImportError, ModuleNotFoundError):
            print("Visualization code cannot be imported, skip...")
            return

        # Visualize both discrete (synced) and float coords
        # Discrete coords are int, convert to float for viz

        # We need to unbatch? render_pcds usually takes a dict of name -> tensor [N, 3] or [N, 4].
        # If we just viz the whole batch, points might overlap if they are in same spatial domain but different batch indices.
        # But for simple debug, we can just show them all in one scene, maybe coloring by batch index or just spatial.

        # Coords: [N, 4] (batch, z, y, x) or (batch, x, y, z)?
        # basic_ext indicates batch at index 0. `coords` property returns backend coords.
        # spconv/torchsparse: (batch, z, y, x) or (batch, x, y, z)?
        # Usually (batch, x, y, z) or similar.
        # `render_pcds` expects [N, 3] usually.

        # Let's take the first sample in the batch for simplicity if batch > 0, or just plot all ignoring batch dim if they are spatially distinct?
        # Better: use batch index to shift them? Or just plot raw spatial coords.

        discrete_spatial = self.coords[:, 1:].float()
        float_spatial = self._float_coords[:, 1:].float()

        # If we have resolution, we can show bounds?

        pcds = {
            "Discrete (Rounded)": discrete_spatial.cpu().numpy(),
            "Float (Continuous)": float_spatial.cpu().numpy(),
        }

        render_pcds(pcds, save_path)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.

    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(
    input: SparseTensor, other: torch.Tensor, op: callable = torch.add
) -> SparseTensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.

    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.

    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.

    Args:
        input (SparseTensor): Sparse tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
