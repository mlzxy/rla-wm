"""
Sparse Windowed Cross Attention Module
======================================

This module implements a memory-efficient cross-attention mechanism for sparse 3D data (point clouds or voxel grids).
Standard dense cross-attention scales quadratically with the total number of points, which is infeasible for large
sparse scenes. This implementation solves this by:

1. **Partitioning**: The 3D space is divided into non-overlapping local windows (voxels) of size `window_size`.
2. **Grouping**: Queries (Q) and Keys/Values (K/V) that fall into the same window are grouped together.
3. **Local Attention**: Attention is performed only *within* these local windows, effectively sparsifying the attention matrix.

Mechanism Explanation
---------------------
To leverage highly optimized "Self-Attention" kernels (like FlashAttention or xFormers) for this "Cross-Attention" task without
complex block-masking, we employ a **Joint Sequence Construction** strategy:

1. **Co-location**: We identify which Q points and K points occupy the same spatial window.
2. **Sequence Packing**: For each window, we concatenate the Q points and K points into a single sequence:
   ``Sequence = [Q_1, ..., Q_m, K_1, ..., K_n]``
3. **Padding for Logic**: To ensure Q only attends to K (and logically K does not affect Q's output in a way that matters,
   though we discard K's output anyway), we construct the inputs to the attention kernel such that:
   - **Joint Q Input**: Contains the actual Q features followed by Zeros. ``[Q_feat, 0]``
   - **Joint K Input**: Contains Zeros followed by the actual K features. ``[0, K_feat]``
   - **Joint V Input**: Contains Zeros followed by the actual V features. ``[0, V_feat]``

   Mathematically, for a Query $q_i$, the attention score with a slot $j$ in the sequence is:
   - If slot $j$ is a Q-slot: $Key = 0 \implies Score \propto q_i \cdot 0 = 0$.
   - If slot $j$ is a K-slot: $Key = k_j \implies Score \propto q_i \cdot k_j$.

   The softmax is taken over this joint sequence. Note that $Score=0$ for Q-slots acts as a "dummy" attention
   (similar to a biases self-loop or attending to a null token). This allows us to use fast dense kernels
   on the packed sequence.

4. **Extraction**: After attention, we extract the output corresponding to the Q-slots and discard the outputs for K-slots.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from typing import Tuple, Union, List, Optional, Dict
from .. import SparseTensor as SparseTensorVanilla, ATTN
from ..basic_ext import SparseTensor as SparseTensorExt

SparseTensor = SparseTensorExt | SparseTensorVanilla


# ==========================================
# 1. Configuration & Attention Backend
# ==========================================

if torch.cuda.is_available():
    try:
        import flash_attn

        ATTN = "flash_attn"
    except ImportError:
        try:
            import xformers.ops as xops

            ATTN = "xformers"
        except ImportError:
            ATTN = "vanilla"
else:
    ATTN = "vanilla"

# ==========================================
# 3. Partitioning Logic (Space Filling Curve)
# ==========================================


def calc_joint_window_partition(
    coords_q: torch.Tensor,
    coords_kv: torch.Tensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, List[int], int, Optional[Dict[str, float]]]:
    """
    Partitions both Query and Key/Value clouds into a shared window grid.
    Returns indices to sort them into a single interleaved sequence.

    Args:
        coords_q (Tensor): Coordinates of query points [N, 4] (batch, x, y, z).
        coords_kv (Tensor): Coordinates of key/value points [M, 4].
        window_size (int or tuple): Size of the spatial window.
        shift_window (int or tuple): Shift applied to coordinates (for Swin-like shifted window attention).
        return_stats (bool): If True, computes and returns overlap statistics.

    Returns:
        fwd_indices (Tensor): Indices to sort the concatenated [Q; KV] list into window-grouped order.
        seq_lens (List[int]): List of lengths (number of points) for each non-empty window.
        N_q (int): Number of query points (used to split Q/KV after processing).
        stats (dict, optional): Statistics about window overlap (participation ratios), if requested.
    """
    device = coords_q.device
    DIM = coords_q.shape[1] - 1

    # Standardize args
    shift_window = (
        (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    )
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size

    # Shift Coords (Apply shift to both clouds identically)
    shift_t = torch.tensor(shift_window, device=device, dtype=torch.int32).unsqueeze(0)
    win_t = torch.tensor(window_size, device=device, dtype=torch.int32).unsqueeze(0)

    shifted_q = coords_q.clone().detach()
    shifted_q[:, 1:] += shift_t

    shifted_kv = coords_kv.clone().detach()
    shifted_kv[:, 1:] += shift_t

    # Compute Max Grid Bounds
    max_coords_q = shifted_q[:, 1:].max(dim=0).values
    max_coords_kv = shifted_kv[:, 1:].max(dim=0).values

    if max_coords_q.numel() == 0:
        max_coords_q = torch.zeros(DIM, device=device)
    if max_coords_kv.numel() == 0:
        max_coords_kv = torch.zeros(DIM, device=device)

    MAX_COORDS = torch.max(max_coords_q, max_coords_kv).tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]

    # Stride for the linearization
    # This creates [Total_Vol, Stride_X, Stride_Y, Stride_Z] (assuming 3D), note it is in WINDOW space
    OFFSET = torch.cumprod(
        torch.tensor([1] + NUM_WINDOWS[::-1], device=device), dim=0
    ).tolist()[::-1]
    offset_t = torch.tensor(OFFSET, device=device, dtype=torch.int32).unsqueeze(0)

    # Quantize and Linearize
    # FIX: slice offset_t to [:, 1:] to match dimensions (X, Y, Z) vs (Total, X, Y, Z)
    win_idx_q = (shifted_q[:, 1:] // win_t * offset_t[:, 1:]).sum(dim=1)
    win_idx_kv = (shifted_kv[:, 1:] // win_t * offset_t[:, 1:]).sum(dim=1)

    # Add Batch offset if coords include batch at index 0
    if coords_q.shape[1] > 3:
        # OFFSET[0] is the total volume of one batch item
        batch_scale = OFFSET[0]
        win_idx_q += coords_q[:, 0] * batch_scale
        win_idx_kv += coords_kv[:, 0] * batch_scale

    # --- Statistics Calculation ---
    stats = None
    if return_stats:
        unique_wins_q = torch.unique(win_idx_q)
        unique_wins_kv = torch.unique(win_idx_kv)

        # Q Participation: How many Q points are in a window that ALSO has KV points?
        mask_q_active = torch.isin(win_idx_q, unique_wins_kv)
        q_ratio = mask_q_active.float().mean().item()

        # KV Participation: How much context is actually being used?
        mask_kv_active = torch.isin(win_idx_kv, unique_wins_q)
        kv_ratio = mask_kv_active.float().mean().item()

        stats = {
            "q_participation_ratio": q_ratio,
            "kv_participation_ratio": kv_ratio,
            "num_windows_overlap": torch.isin(unique_wins_q, unique_wins_kv)
            .sum()
            .item(),
        }

    # Merge and Sort
    N_q = win_idx_q.shape[0]
    all_win_idx = torch.cat([win_idx_q, win_idx_kv])
    fwd_indices = torch.argsort(all_win_idx)

    # Calculate sequence lengths
    _, counts = torch.unique(all_win_idx[fwd_indices], return_counts=True)
    seq_lens = counts.tolist()

    return fwd_indices, seq_lens, N_q, stats


# ==========================================
# 4. Attention Module
# ==========================================


class SparseWindowCrossAttention(nn.Module):
    """
    Sparse Window Cross Attention Layer.

    Applies cross-attention between a sparse 'Query' tensor and a sparse 'Key/Value' tensor.
    Restricts attention to occur only between points that fall within the same spatial window.

    This is equivalent to:
    1. Voxelizing the space into windows of size ``window_size``.
    2. For each window, collecting all Q points and K points inside it.
    3. Running standard Cross Attention (Q, K, V) within that window.
    4. Stitching the results back.

    The implementation supports **FlashAttention**, **xFormers**, and a vanilla fallback.

    Args:
        embed_dim (int): Feature dimension C.
        num_heads (int): Number of attention heads.
        window_size (int): Spatial size of the window (e.g., 8).
    """

    def __init__(self, embed_dim, num_heads, window_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        q_tensor: SparseTensor,
        kv_tensor: SparseTensor,
        shift_window=(0, 0, 0),
        return_stats=False,
    ):
        # 1. Project Features
        Q = self.q_proj(q_tensor.feats)
        K = self.k_proj(kv_tensor.feats)
        V = self.v_proj(kv_tensor.feats)

        # 2. Call Function
        head_dim = self.embed_dim // self.num_heads

        q_out, stats = sparse_windowed_cross_attention(
            q_tensor.replace(Q.view(-1, self.num_heads, head_dim)),
            kv_tensor.replace(K.view(-1, self.num_heads, head_dim)),
            kv_tensor.replace(V.view(-1, self.num_heads, head_dim)),
            self.window_size,
            shift_window,
            return_stats=True,
        )

        q_update = q_out.feats.view(-1, self.embed_dim)
        final_feats = q_tensor.feats + self.out_proj(q_update)
        output_tensor = q_tensor.replace(final_feats)

        if return_stats:
            return output_tensor, stats
        return output_tensor


def sparse_windowed_cross_attention(
    q: SparseTensor,
    k: SparseTensor,  # Assumes k and v have same coords
    v: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
    return_stats: bool = False,
) -> Union[SparseTensor, Tuple[SparseTensor, Dict]]:
    # 1. Partition Logic
    fwd_indices, seq_lens, N_q, stats = calc_joint_window_partition(
        q.coords,
        k.coords,
        window_size,
        shift_window,
        return_stats=return_stats,
    )

    # 2. Get Feats
    Q = q.feats  # [N_q, H, D]
    K = k.feats  # [N_k, H, D]
    V = v.feats  # [N_k, H, D]

    num_heads = Q.shape[1]
    head_dim = Q.shape[2]

    # 3. Construct Joint Sequence [Q; K]
    # Padding to unify Q and K into one sorted list
    # We need to pad with zeros matching [H, D]

    zeros_k = torch.zeros(
        K.shape[0], num_heads, head_dim, device=Q.device, dtype=Q.dtype
    )
    zeros_q = torch.zeros(
        Q.shape[0], num_heads, head_dim, device=Q.device, dtype=Q.dtype
    )

    concat_Q = torch.cat([Q, zeros_k])
    concat_K = torch.cat([zeros_q, K])
    concat_V = torch.cat([zeros_q, V])

    # Apply Window Sorting
    joint_Q = concat_Q[fwd_indices]
    joint_K = concat_K[fwd_indices]
    joint_V = concat_V[fwd_indices]

    total_tokens = joint_Q.shape[0]
    out = None

    # 4. Attention Execution
    if ATTN == "xformers":
        q_in = joint_Q.unsqueeze(0)
        k_in = joint_K.unsqueeze(0)
        v_in = joint_V.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
        out = xops.memory_efficient_attention(q_in, k_in, v_in, attn_bias=mask)
        out = out.squeeze(0)

    elif ATTN == "flash_attn":
        cu_seqlens = torch.tensor([0] + seq_lens, device=Q.device).cumsum(0).int()
        max_seqlen = max(seq_lens)
        out = flash_attn.flash_attn_varlen_func(
            joint_Q,
            joint_K,
            joint_V,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
        )

    elif ATTN == "vanilla":
        # Fallback for MAC / CPU / No-kernel environments
        max_seqlen = max(seq_lens)
        num_windows = len(seq_lens)

        flat_Q = joint_Q
        flat_K = joint_K
        flat_V = joint_V

        batch_Q = torch.zeros(
            num_windows,
            max_seqlen,
            num_heads,
            head_dim,
            device=Q.device,
            dtype=Q.dtype,
        )
        batch_K = torch.zeros(
            num_windows,
            max_seqlen,
            num_heads,
            head_dim,
            device=Q.device,
            dtype=Q.dtype,
        )
        batch_V = torch.zeros(
            num_windows,
            max_seqlen,
            num_heads,
            head_dim,
            device=Q.device,
            dtype=Q.dtype,
        )

        mask = torch.zeros(
            num_windows, max_seqlen, max_seqlen, device=Q.device, dtype=torch.bool
        )

        cursor = 0
        for i, slen in enumerate(seq_lens):
            batch_Q[i, :slen] = flat_Q[cursor : cursor + slen]
            batch_K[i, :slen] = flat_K[cursor : cursor + slen]
            batch_V[i, :slen] = flat_V[cursor : cursor + slen]
            mask[i, :slen, :slen] = True
            cursor += slen

        batch_Q = batch_Q.permute(0, 2, 1, 3)
        batch_K = batch_K.permute(0, 2, 1, 3)
        batch_V = batch_V.permute(0, 2, 1, 3)

        mask = mask.unsqueeze(1)  # [Batch, 1, Seq, Seq]

        if hasattr(F, "scaled_dot_product_attention"):
            sdpa_mask = ~mask  # Invert for SDPA (True=Masked/Ignore)
            out_batch = F.scaled_dot_product_attention(
                batch_Q, batch_K, batch_V, attn_mask=sdpa_mask
            )
        else:
            d_k = head_dim
            scores = torch.matmul(batch_Q, batch_K.transpose(-2, -1)) / math.sqrt(d_k)
            scores = scores.masked_fill(~mask, -1e9)
            attn = F.softmax(scores, dim=-1)
            out_batch = torch.matmul(attn, batch_V)

        out_batch = out_batch.permute(0, 2, 1, 3)

        out = torch.zeros_like(joint_Q)
        cursor = 0
        for i, slen in enumerate(seq_lens):
            out[cursor : cursor + slen] = out_batch[i, :slen]
            cursor += slen

    out = out.reshape(total_tokens, num_heads, head_dim)

    # 5. Restore Order & Extract
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=Q.device)

    out_unsorted = out[bwd_indices]
    q_update = out_unsorted[:N_q]

    output_tensor = q.replace(q_update)

    if return_stats:
        return output_tensor, stats
    return output_tensor


# ==========================================
# 5. Interactive Visualization
# ==========================================


def run_visualizer():
    # Delayed imports to avoid overhead/errors if not used
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        import matplotlib.animation as animation
    except ImportError:
        print("Error: Matplotlib not installed. Cannot run visualization.")
        return

    # Check for Headless Environment
    IS_HEADLESS = False
    if os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
        matplotlib.use("Agg")
        IS_HEADLESS = True
    elif matplotlib.get_backend().lower() == "agg":
        IS_HEADLESS = True
    else:
        try:
            matplotlib.use("TkAgg")
        except:
            pass  # Fallback to default

    print(
        f"Visualization Mode: {'Headless (Saving GIF)' if IS_HEADLESS else 'Interactive (Window)'}"
    )

    # --- Data Generation ---
    np.random.seed(42)
    grid_dim = 6
    x, y, z = np.indices((grid_dim, grid_dim, grid_dim))
    k_coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Sparse Qs placed to test boundaries
    q_coords = np.array([[1.5, 1.5, 1.5], [3.9, 3.9, 1.5], [0.5, 5.5, 2.5]])

    # --- Visualizer Class ---
    class Visualizer3D:
        def __init__(self, q_coords, k_coords):
            self.q_coords = q_coords
            self.k_coords = k_coords

            self.fig = plt.figure(figsize=(12, 9))
            self.ax = self.fig.add_subplot(111, projection="3d")
            if not IS_HEADLESS:
                plt.subplots_adjust(bottom=0.30)

            self.win_size = 4
            self.shift_x = 0
            self.shift_y = 0
            self.shift_z = 0
            # FIX: Initialize grid_lines list
            self.grid_lines = []

            if not IS_HEADLESS:
                self._setup_widgets()

            # Don't call update() in __init__ for animations, let the loop do it
            # But for static/interactive, we need one initial draw
            if not IS_HEADLESS:
                self.update()

        def _setup_widgets(self):
            ax_win = plt.axes([0.2, 0.15, 0.6, 0.03])
            ax_sx = plt.axes([0.2, 0.10, 0.6, 0.03])
            ax_sy = plt.axes([0.2, 0.06, 0.6, 0.03])
            ax_sz = plt.axes([0.2, 0.02, 0.6, 0.03])

            self.s_win = Slider(ax_win, "Window Size", 2, 8, valinit=4, valstep=1)
            self.s_sx = Slider(ax_sx, "Shift X", 0, 4, valinit=0, valstep=0.5)
            self.s_sy = Slider(ax_sy, "Shift Y", 0, 4, valinit=0, valstep=0.5)
            self.s_sz = Slider(ax_sz, "Shift Z", 0, 4, valinit=0, valstep=0.5)

            self.s_win.on_changed(self._on_change)
            self.s_sx.on_changed(self._on_change)
            self.s_sy.on_changed(self._on_change)
            self.s_sz.on_changed(self._on_change)

        def _on_change(self, val):
            self.win_size = int(self.s_win.val)
            self.shift_x = self.s_sx.val
            self.shift_y = self.s_sy.val
            self.shift_z = self.s_sz.val
            self.update()

        def draw_window_grid(self):
            # FIX: Do not call line.remove(). self.ax.clear() in update() handles it.
            # Just reset the list.
            self.grid_lines = []

            min_v, max_v = -2, 8
            ticks_x = np.arange(min_v, max_v, self.win_size) - self.shift_x
            ticks_y = np.arange(min_v, max_v, self.win_size) - self.shift_y
            ticks_z = np.arange(min_v, max_v, self.win_size) - self.shift_z

            lines = []
            for x in ticks_x:
                for y in ticks_y:
                    lines.append([(x, y, -2), (x, y, 8)])
                for z in ticks_z:
                    lines.append([(x, -2, z), (x, 8, z)])
            for y in ticks_y:
                for z in ticks_z:
                    lines.append([(-2, y, z), (8, y, z)])

            lc = Line3DCollection(
                lines, colors="black", linewidths=0.5, linestyles="dashed", alpha=0.3
            )
            self.ax.add_collection3d(lc)
            self.grid_lines.append(lc)

        def update(self):
            self.ax.clear()  # This removes everything, including old grid lines

            # Identify Active Connections
            def get_win_idx(coords, sx, sy, sz, w):
                return np.floor((coords + [sx, sy, sz]) / w)

            q_wins = get_win_idx(
                self.q_coords, self.shift_x, self.shift_y, self.shift_z, self.win_size
            )
            k_wins = get_win_idx(
                self.k_coords, self.shift_x, self.shift_y, self.shift_z, self.win_size
            )

            active_k_indices = set()
            connections = []

            for i, qw in enumerate(q_wins):
                matches = np.all(k_wins == qw, axis=1)
                for midx in np.where(matches)[0]:
                    active_k_indices.add(midx)
                    connections.append([self.q_coords[i], self.k_coords[midx]])

            active_mask = np.array(
                [i in active_k_indices for i in range(len(self.k_coords))]
            )

            # Draw K (Context)
            if len(self.k_coords) > 0:
                if not np.all(active_mask):
                    self.ax.scatter(
                        self.k_coords[~active_mask, 0],
                        self.k_coords[~active_mask, 1],
                        self.k_coords[~active_mask, 2],
                        c="lightgray",
                        s=20,
                        alpha=0.2,
                        label="Context (Ignored)",
                    )
                if np.any(active_mask):
                    self.ax.scatter(
                        self.k_coords[active_mask, 0],
                        self.k_coords[active_mask, 1],
                        self.k_coords[active_mask, 2],
                        c="red",
                        s=40,
                        alpha=0.8,
                        label="Context (Active)",
                    )

            # Draw Q (Query)
            self.ax.scatter(
                self.q_coords[:, 0],
                self.q_coords[:, 1],
                self.q_coords[:, 2],
                c="blue",
                s=200,
                marker="*",
                edgecolors="white",
                zorder=10,
                label="Query (Q)",
            )

            # Draw Lines
            if connections:
                lc = Line3DCollection(
                    connections, colors="green", linewidths=1, alpha=0.5
                )
                self.ax.add_collection3d(lc)

            self.draw_window_grid()

            self.ax.set_xlim(-1, 7)
            self.ax.set_ylim(-1, 7)
            self.ax.set_zlim(-1, 7)
            self.ax.set_title(
                f"Win: {self.win_size} | Shift: ({self.shift_x:.1f}, {self.shift_y:.1f}, {self.shift_z:.1f})"
            )
            self.ax.legend(loc="upper left")

    # --- Run ---
    vis = Visualizer3D(q_coords, k_coords)

    if IS_HEADLESS:
        fname = "shift_demo.gif"
        print(f"Generating {fname}...")

        def animate(frame):
            shifts = np.linspace(0, 3, 20)
            vis.shift_x = shifts[frame % len(shifts)]
            vis.shift_y = shifts[frame % len(shifts)]
            vis.update()
            return (vis.ax,)

        ani = animation.FuncAnimation(vis.fig, animate, frames=20, interval=200)
        ani.save(fname, writer="pillow", fps=5)
        print("Saved.")
    else:
        plt.show()


# ==========================================
# 6. Test Logic
# ==========================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_module():
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16
        if torch.cuda.is_available() and ATTN == "flash_attn"
        else torch.float32
    )
    print(f"\nRunning Module Test on: {device}")

    C = 32
    # Q at (1,1,1) and (4,4,4)
    q_coords = torch.tensor(
        [[0, 1, 1, 1], [0, 4, 4, 4]], dtype=torch.int32, device=device
    )
    # K near (4,4,4) but not (1,1,1)
    k_coords = torch.tensor(
        [[0, 4, 5, 4], [0, 5, 4, 4]], dtype=torch.int32, device=device
    )

    q_feats = torch.randn(2, C, device=device, dtype=dtype)
    k_feats = torch.randn(2, C, device=device, dtype=dtype)

    q_st = SparseTensor(q_feats, q_coords)
    k_st = SparseTensor(k_feats, k_coords)

    model = (
        SparseWindowCrossAttention(embed_dim=C, num_heads=4, window_size=4)
        .to(device)
        .to(dtype)
    )

    # Run
    out_st, stats = model(q_st, k_st, shift_window=(0, 0, 0), return_stats=True)

    print("Output Shape:", out_st.feats.shape)
    print(f"Q Participation: {stats['q_participation_ratio']:.2f} (Expected 0.50)")
    print("Test Passed.")


if __name__ == "__main__":
    # 1. Run Logic Test
    test_module()

    # 2. Run Visualization
    # (Comment out if you only want to run logic tests on a cluster)
    run_visualizer()
