from typing import Optional, Any, TypedDict
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from ...modules.sparse.basic_ext import SparseTensor as SparseTensorExt
from ...modules.sparse.basic import SparseTensor as VanillaSparseTensor
from torch_cluster import knn
from jaxtyping import Float, UInt32, Int64, Bool
from torch_linear_assignment import batch_linear_assignment
from torch import Tensor

SparseTensor = SparseTensorExt | VanillaSparseTensor


class _SparseSampleOnceOutput(TypedDict):
    pred_x_prev: SparseTensorExt  # [P, 1+3+zdim]


@torch.no_grad()
def sample_x0_xT(
    st: SparseTensorExt,  # [P, 1+3+zdim]
    st_plus_1: SparseTensorExt,  # [P, 1+3+zdim]
    sample_to: int = 16384,
    std: float = 10.0,
    **kwargs,
) -> tuple[
    SparseTensorExt, SparseTensorExt, Tensor[UInt32, " sample_to "]
]:  # [N_src, 1+3+zdim]
    """
    it returns the x0 and xT, if st_plus_1 is None, x0 is None
    it also returns the original indices of the resampled, reordered points
    """
    B = st.coords.shape[0]
    st, st_indices = st.sample_to(sample_to)
    st.float_coords = torch.clamp(
        st.float_coords + std * torch.randn_like(st.float_coords),
        torch.zeros(1, 3, device=st.float_coords.device).float(),
        torch.as_tensor(st._resolution, device=st.float_coords.device)
        .reshape(1, 3)
        .float()
        - 1,
    )

    if st_plus_1 is None:
        return None, st, st_indices, None
    st_plus_1, st_plus_1_indices = st_plus_1.sample_to(sample_to)

    st_x = st.float_coords.reshape(B, sample_to, 3)
    st_1_x = st_plus_1.float_coords.reshape(B, sample_to, 3)
    cost = torch.cdist(st_x, st_1_x, p=2)
    assignment = batch_linear_assignment(cost)
    assignment = assignment + (torch.arange(B, device=st.device).view(B, 1) * sample_to)
    assignment = assignment.flatten()

    st_indices = st_indices[assignment]
    st = SparseTensorExt(
        feats=st.feats[assignment],
        float_coords=st.float_coords[assignment],
        resolution=st.resolution,
    )
    return st_plus_1, st, st_plus_1_indices, st_indices


class LagrangianVoxelFlowSampler(Sampler):
    def __init__(
        self,
        sigma_min: float,
        coord_sigma_min: float,
        resolution: int | tuple[int, int, int],
    ):
        """
        Initialize the Lagrangian Voxel Flow Sampler.

        Args:
            sigma_min: The minimum standard deviation for the flow matching noise.
            coord_sigma_min: The minimum standard deviation for the coordinate noise.
            resolution: The resolution of the voxel grid (D, H, W).
        """

        self.sigma_min = sigma_min
        self.coord_sigma_min = coord_sigma_min  # e.g., 1e-3
        self.resolution = resolution

    def _v_to_xstart_eps(
        self,
        xt: SparseTensorExt,  # [N_src, 1+3+zdim]
        t: Float[Tensor, " batch "],
        v_full: Float[Tensor, "N_src 3+zdim"],
    ) -> tuple[
        SparseTensorExt, SparseTensorExt
    ]:  # [N_tgt, 1+3+zdim], [N_src, 1+3+zdim]
        """
        Convert the flow + xt to x0 and xT

        Args:
            xt: [N_src, 1+3+zdim] float sparse tensor of coordinates + noise at time t.
            t: [batch] float tensor of diffusion steps.
            v_full: [N_src, 3+zdim] float tensor of flow, result of `get_v`

        Returns:
            x0: [N_tgt, 1+3+zdim] float sparse tensor of target coordinates + noise.
            xT: [N_src, 1+3+zdim] float sparse tensor of source coordinates + noise.
        """
        batch_idx = xt.coords[:, 0].long()
        flat_t = t[batch_idx].view(-1, 1)

        v_coords = v_full[:, :3]
        v_feats = v_full[:, 3:]

        xt_coords = xt.norm_coords[:, 1:]
        xt_feats = xt.feats

        # Coordinates (Linear Interpolation)
        xT_coords = (1 - flat_t) * v_coords + xt_coords
        x0_coords = xt_coords - flat_t * v_coords

        # Features (Sigma Min Interpolation)
        xT_feats = (1 - flat_t) * v_feats + xt_feats
        x0_feats = (1 - self.sigma_min) * xt_feats - (
            self.sigma_min + (1 - self.sigma_min) * flat_t
        ) * v_feats

        x0 = SparseTensorExt(
            feats=x0_feats,
            norm_coords=torch.cat([batch_idx.reshape(-1, 1), x0_coords], dim=1),
            resolution=self.resolution,
        )
        xT = SparseTensorExt(
            feats=xT_feats,
            norm_coords=torch.cat([batch_idx.reshape(-1, 1), xT_coords], dim=1),
            resolution=self.resolution,
        )
        return x0, xT

    def diffuse(
        self,
        x0: SparseTensorExt,  # [N_tgt, 1+3+zdim]
        xT: SparseTensorExt,  # [N_src, 1+3+zdim]
        t: Float[Tensor, " batch "],
    ) -> tuple[SparseTensorExt, Float[Tensor, "N_src 3"]]:  # [P, 1+3+zdim]
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
             x0: [N_tgt, 1+3+zdim] float sparse tensor of target coordinates + target features.
            xT: [N_src, 1+3+zdim] float sparse tensor of source coordinates + noise
            t: [batch] float tensor of diffusion steps.

        Returns:
            x_t (SparseTensorExt): The noisy sparse tensor constructed from diffused
                coordinates and features.
        """
        assert len(x0.coords) == len(xT.coords), (
            "N_src and N_tgt must be equal in the new version"
        )
        batch_idx = xT.coords[:, 0].long()
        flat_t = t[batch_idx].view(-1, 1)

        x_t_feats = (1 - flat_t) * x0.feats + (
            self.sigma_min + (1 - self.sigma_min) * flat_t
        ) * xT.feats

        voxel_noise = self.coord_sigma_min * torch.randn(
            xT.coords.shape[0], 3, device=xT.feats.device
        )

        x_t_norm_coords = (1 - flat_t) * (
            x0.norm_coords[:, 1:] + voxel_noise
        ) + flat_t * xT.norm_coords[:, 1:]

        x_t = SparseTensorExt(
            feats=x_t_feats,
            norm_coords=torch.cat([batch_idx.reshape(-1, 1), x_t_norm_coords], dim=1),
            resolution=self.resolution,
        )
        return x_t, voxel_noise

    def get_v(  # derivative of xt
        self,
        x0: SparseTensorExt,  # [N_tgt, 1+3+zdim]
        xT: SparseTensorExt,  # [N_src, 1+3+zdim]
        t: Float[Tensor, " B "],
        voxel_noise: Float[Tensor, "N_src 3"],
    ) -> Float[Tensor, "N_src 3+zdim"]:
        """
        Compute the velocity of the diffusion process at time t.

        Args:
            x0: [N_tgt, 1+3+zdim] float sparse tensor of target coordinates + target features.
            xT: [N_src, 1+3+zdim] float sparse tensor of source coordinates + noise
            t: [batch] float tensor of diffusion steps.
            matching: [N_src] long tensor of indices mapping source to target.
            voxel_noise: [N_src*2, 3] float tensor of voxel noise.

        Returns:
            v: [N_src, 3+zdim] float tensor combining coordinate velocity and feature velocity.
        """
        feats_v = (1 - self.sigma_min) * xT.feats - x0.feats
        v_coords = xT.norm_coords[:, 1:] - (x0.norm_coords[:, 1:] + voxel_noise)
        return torch.cat([v_coords, feats_v], dim=1)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        assert cond is not None, "cond must be provided"
        assert isinstance(cond, dict), "cond must be a dict"
        t = torch.tensor(
            [1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32
        )
        cfg_strength = kwargs.pop("cfg_strength", 1.0)
        if "text" in cond and "neg_cond" in kwargs:
            pred = model(x_t, t, cond, **kwargs)
            neg_pred = model(
                x_t, t, {"xt": cond["xt"], "text": kwargs["neg_cond"]}, **kwargs
            )
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return model(x_t, t, cond, **kwargs)

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t: SparseTensor,  # [P, 1+3+zdim]
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs,
    ) -> _SparseSampleOnceOutput:
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            model: The model to sample from.
            x_t: The input sparse tensor at time t with shape [P, 1+3+zdim].
            t: The current timestep.
            t_prev: The previous timestep.
            cond: Optional dictionary containing conditional information (e.g., text, images).
            **kwargs: Additional arguments for model inference (e.g., cfg_strength).

        Returns:
            dict containing:
            - 'pred_x_prev': The predicted sparse tensor at time t-1.
        """
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        batch_idx = x_t.coords[:, :1]
        pred_x_prev = x_t.full_feats - (t - t_prev) * pred_v.feats
        pred_x_prev = SparseTensorExt(
            feats=pred_x_prev[:, 3:],
            norm_coords=torch.cat([batch_idx, pred_x_prev[:, :3]], dim=1),
            resolution=x_t._resolution,
        )
        return edict({"pred_x_prev": pred_x_prev})

    @torch.no_grad()
    def sample(
        self,
        model,
        sparse_noise: SparseTensor,  # [P, 1+3+zdim]
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        return_history: bool = False,
        **kwargs,
    ) -> SparseTensorExt:
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            sparse_noise: The initial noisy sparse tensor.
            cond: Optional dictionary containing conditional information.
            steps: The number of sampling steps.
            rescale_t: The rescale factor for time scheduling.
            verbose: If True, show a progress bar.
            return_history: If True, return the list of intermediate samples.
            **kwargs: Additional arguments for model inference.

        Returns:
            sample (SparseTensor): The final generated sample.
            If return_history is True, might return list of intermediate samples.
        """
        sample = sparse_noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = []
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            if return_history:
                ret.append(out.pred_x_prev)
        return sample


sample_xT_from_x0 = sample_xT

if __name__ == "__main__":
    print("Running verification for LagrangianVoxelFlowSampler...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # CPU for simple test

    # 1. Initialize Sampler
    resolution = (32, 32, 32)
    sampler = LagrangianVoxelFlowSampler(
        sigma_min=1e-5, coord_sigma_min=1e-3, resolution=resolution
    )

    # 2. Fake Data
    N_src = 100
    N_tgt = 80
    zdim = 16
    batch = 2

    # Init coords: [b, x, y, z] -> but here [N, 4] with batch idx at 0
    init_coords = torch.rand(N_src, 4, device=device) * 31
    init_coords[:, 0] = torch.randint(0, batch, (N_src,), device=device).float()

    target_coords = torch.rand(N_tgt, 4, device=device) * 31
    target_coords[:, 0] = torch.randint(0, batch, (N_tgt,), device=device).float()

    target_feats = torch.randn(N_tgt, zdim, device=device)
    noise = torch.randn(N_src, zdim, device=device)
    t = torch.tensor([0.5, 0.8], device=device)  # [batch]

    # Wrap in SparseTensorExt
    x0 = SparseTensorExt(
        feats=target_feats, coords=target_coords, resolution=resolution
    )
    xT = SparseTensorExt(feats=noise, coords=init_coords, resolution=resolution)

    # 3. Test get_matching
    # print("Testing get_matching...")
    # matching = sampler.get_matching(target_coords, init_coords)
    # assert matching.shape == (N_src,)
    # print("  get_matching passed.")
    # Mock matching for subsequent tests since corr.py is missing
    matching = torch.arange(N_src, device=device)

    # 4. Test diffuse
    # print("Testing diffuse...")
    # x_t, voxel_noise = sampler.diffuse(x0, xT, t, matching)
    # print(f"  x_t type: {type(x_t)}")
    # print(f"  x_t feats shape: {x_t.feats.shape}")
    # print(f"  x_t coords shape: {x_t.coords.shape}")
    # assert x_t.feats.shape == (N_src, zdim)
    # assert x_t.coords.shape == (N_src, 4)
    # print("  diffuse passed.")

    # 5. Test get_v
    # print("Testing get_v...")
    # v = sampler.get_v(x0, xT, t, matching, voxel_noise)
    # print(f"  v shape: {v.shape}")
    # assert v.shape == (N_src, 3 + zdim)
    # print("  get_v passed.")

    # 6. Test _v_to_xstart_eps
    # print("Testing _v_to_xstart_eps...")
    # x0_pred, xT_pred = sampler._v_to_xstart_eps(x_t, t, v)

    # Calculate Expected Values (Effective inputs used in diffuse/get_v)
    # Note: diffuse() adds voxel_noise to coords
    # expected_x0_coords = x0.norm_coords[matching, 1:] + voxel_noise[:N_src]
    # expected_xT_coords = xT.norm_coords[:, 1:] + voxel_noise[N_src:]

    # expected_x0_feats = x0.feats[matching]
    # expected_xT_feats = xT.feats

    # Verify Coords
    # diff_x0_coords = torch.abs(x0_pred.norm_coords[:, 1:] - expected_x0_coords).max()
    # diff_xT_coords = torch.abs(xT_pred.norm_coords[:, 1:] - expected_xT_coords).max()
    # print(f"  Max diff x0 coords: {diff_x0_coords.item()}")
    # print(f"  Max diff xT coords: {diff_xT_coords.item()}")

    # Verify Feats
    # diff_x0_feats = torch.abs(x0_pred.feats - expected_x0_feats).max()
    # diff_xT_feats = torch.abs(xT_pred.feats - expected_xT_feats).max()
    # print(f"  Max diff x0 feats: {diff_x0_feats.item()}")
    # print(f"  Max diff xT feats: {diff_xT_feats.item()}")

    # assert diff_x0_coords < 1e-5, "x0 coords reconstruction failed"
    # assert diff_xT_coords < 1e-5, "xT coords reconstruction failed"
    # Note: Feats reconstruction might have strictly 0 error if math is perfect, but use tolerance for float.
    # assert diff_x0_feats < 1e-5, "x0 feats reconstruction failed"
    # assert diff_xT_feats < 1e-5, "xT feats reconstruction failed"

    # print("  _v_to_xstart_eps passed.")

    # 7. Test sample_xT
    print("Testing sample_xT...")
    n_queries = 2
    kernel_size = 4
    st_sample = x0  # Use x0 as input

    # Test without static_mask
    xT_sample, matching_sample = sample_xT(st_sample, n_queries, kernel_size)
    print(f"  xT_sample feats shape: {xT_sample.feats.shape}")
    print(f"  xT_sample coords shape: {xT_sample.coords.shape}")

    assert xT_sample.feats.shape[0] == st_sample.feats.shape[0] * n_queries, (
        "Incorrect number of points sampled"
    )
    assert xT_sample.feats.shape[1] == zdim, "Incorrect feature dimension"

    # Verify batch indices preserved
    orig_batch = torch.repeat_interleave(st_sample.coords[:, 0], n_queries)
    assert torch.all(xT_sample.coords[:, 0] == orig_batch), "Batch indices mismatch"

    # Verify matching
    expected_matching = torch.arange(
        st_sample.feats.shape[0], device=device
    ).repeat_interleave(n_queries)
    assert torch.all(matching_sample == expected_matching), "Matching indices mismatch"

    # Test with static_mask
    print("Testing sample_xT with static_mask...")
    static_mask = torch.zeros(st_sample.feats.shape[0], dtype=torch.bool, device=device)
    static_mask[0] = True  # Make first point static
    n_queries_static = 5
    static_kernel_size = 0.1

    xT_static, matching_static = sample_xT(
        st_sample,
        n_queries,
        kernel_size,
        static_mask,
        n_queries_static,
        static_kernel_size,
    )

    expected_points = (len(st_sample.feats) - 1) * n_queries + 1 * n_queries_static
    assert xT_static.feats.shape[0] == expected_points, (
        f"Incorrect number of points with static mask. Got {xT_static.feats.shape[0]}, expected {expected_points}"
    )

    # Verify matching with static mask
    repeats = torch.where(
        static_mask,
        torch.tensor(n_queries_static, device=device),
        torch.tensor(n_queries, device=device),
    )
    expected_matching_static = torch.arange(
        st_sample.feats.shape[0], device=device
    ).repeat_interleave(repeats)
    assert torch.all(matching_static == expected_matching_static), (
        "Matching indices mismatch with static mask"
    )

    print("  sample_xT passed.")

    # 8. Test sample_xT_from_x0
    print("Testing sample_xT_from_x0...")
    # Alias for testing
    sample_xT_from_x0 = sample_xT

    xT_from_x0, matching_from_x0 = sample_xT_from_x0(
        x0, n_queries=n_queries, kernel_size=kernel_size
    )
    print(f"  xT_from_x0 feats shape: {xT_from_x0.feats.shape}")
    print(f"  matching_from_x0 shape: {matching_from_x0.shape}")

    assert xT_from_x0.feats.shape[0] == N_tgt * n_queries
    assert matching_from_x0.shape[0] == N_tgt * n_queries

    # Check if matching is correct (0,0, 1,1, etc.)
    expected_matching = torch.arange(N_tgt, device=device).repeat_interleave(n_queries)
    assert torch.all(matching_from_x0 == expected_matching)
    print("  sample_xT_from_x0 passed.")

    print("All verification tests passed!")
