import numpy as np
import torch
from torch_geometric.nn.pool import voxel_grid, global_mean_pool
from torch_scatter import segment_csr

############################################
# Utility
############################################

def to_dense_batch(x, batch, return_length=False, input_offset=False):
    """
    Converts a flat (stacked) batch of variable-size data into a dense (padded) batch.

    Args:
        x (torch.Tensor): Input features of shape [N_total, C] or [N_total].
        batch (torch.Tensor): Batch indices of shape [N_total] (e.g. [0, 0, 1, 1, 1]).
            If input_offset is True, this is expected to be an offset tensor of shape [B].
        return_length (bool): If True, returns the number of valid elements per batch.
        input_offset (bool): If True, interprets the 'batch' argument as 'offset'.

    Returns:
        dense_x (torch.Tensor): Padded features of shape [B, N_max, C].
        mask (torch.BoolTensor): Mask of valid elements of shape [B, N_max].
        length (torch.LongTensor, optional): Length of each batch of shape [B].
    """
    input_1d = False
    if len(x.shape) == 1:
        input_1d = True
        x = x[:, None]

    if len(x) == 0:
        a, b = x.reshape(0, *x.shape), torch.zeros([0, 0], dtype=torch.bool, device=x.device)
        if return_length:
            return a, b, torch.zeros([0,], dtype=torch.long, device=x.device)
        else:
            return a, b
    
    if input_offset:
        assert batch[-1].item() == len(x)
        offset = batch
        batch = offset2batch(offset)
    else:
        offset = batch2offset(batch)
    length = offset2length(offset)
    max_n = length.max()
    mask = torch.arange(max_n, device=x.device).reshape(1, -1).repeat(len(offset), 1) < length.view(-1, 1)
    dense_x = torch.zeros((len(offset), max_n, x.shape[-1]), dtype=x.dtype, device=x.device)
    dense_x.view(-1, dense_x.shape[-1])[mask.flatten(), :] = x
    if input_1d:
        dense_x = dense_x[:, :, 0]

    if return_length:
        length = mask.long().sum(1)
        return dense_x, mask, length
    else:
        return dense_x, mask

def to_flat_batch(dense_x, mask):
    """
    Converts a dense (padded) batch back into a flat (stacked) representation.
    

    Args:
        dense_x (torch.Tensor): Padded features of shape [B, N_max, C].
        mask (torch.BoolTensor): Boolean mask of shape [B, N_max] where True indicates valid data.

    Returns:
        x (torch.Tensor): Flat features of shape [N_total, C].
        offset (torch.Tensor): Offset tensor indicating the end index of each batch, shape [B].
    """
    if len(dense_x.shape) ==  3:
        x = dense_x.reshape(-1, dense_x.shape[-1])[mask.flatten()]
    else:
        x = dense_x.flatten()[mask.flatten()]
    return x, mask2offset(mask)
    

def batch2mask(batch):
    """
    Creates a boolean mask from a batch index vector.
    effectively acts as a wrapper to extract just the mask from to_dense_batch.

    Args:
        batch (torch.Tensor): Batch indices of shape [N_total].

    Returns:
        mask (torch.BoolTensor): Mask of shape [B, N_max].
    """
    return to_dense_batch(torch.zeros([len(batch)], device=batch.device), batch)[1]

def offset2mask(offset):
    """
    Creates a boolean mask from an offset vector.

    Args:
        offset (torch.Tensor): Offset indices of shape [B].

    Returns:
        mask (torch.BoolTensor): Mask of shape [B, N_max].
    """
    return batch2mask(offset2batch(offset))

def mask2offset(mask):
    """
    Converts a boolean mask to an offset vector.

    Args:
        mask (torch.BoolTensor): Mask of shape [B, N_max].

    Returns:
        offset (torch.Tensor): Cumulative sum of lengths, shape [B].
    """
    length = mask.sum(dim=1).flatten()
    return length2offset(length)

def offset2batch(offset):
    """
    Converts an offset vector to a batch index vector.
    Example: offset=[2, 5] -> batch=[0, 0, 1, 1, 1]

    Args:
        offset (torch.Tensor): Offset indices of shape [B].

    Returns:
        batch (torch.Tensor): Batch indices of shape [N_total].
    """
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else torch.tensor([i] * o) for i, o in enumerate(offset)], dim=0).long().to(offset.device)

def batch2offset(batch):
    """
    Converts a batch index vector to an offset vector.
    Example: batch=[0, 0, 1, 1, 1] -> offset=[2, 5]

    Args:
        batch (torch.Tensor): Batch indices of shape [N_total].

    Returns:
        offset (torch.Tensor): Cumulative counts, shape [B].
    """
    return torch.cumsum(batch.bincount(), dim=0).long()

def offset2length(offset):
    """
    Converts an offset vector to a length vector (number of items per batch).

    Args:
        offset (torch.Tensor): Offset indices of shape [B].

    Returns:
        length (torch.Tensor): Length of each batch item, shape [B].
    """
    length = offset.clone()
    length[1:] = offset[1:] - offset[:-1]
    return length

def length2offset(length):
    """
    Converts a length vector to a cumulative offset vector.

    Args:
        length (torch.Tensor): Length of each batch item, shape [B].

    Returns:
        offset (torch.Tensor): Offset indices of shape [B].
    """
    return torch.cumsum(length, dim=0).long()

def padoffset(offset, L):
    """
    Pads the offset vector to a fixed size L by repeating the last element.
    Useful for distributed training with fixed tensor sizes.

    Args:
        offset (torch.Tensor): Original offset vector.
        L (int): Target length.

    Returns:
        torch.Tensor: Padded offset vector of shape [L].
    """
    if L > len(offset):
        return torch.cat([offset, torch.full([L - len(offset)], fill_value=offset[-1].item(), device=offset.device)])
    else:
        return offset

o2b = offset2batch
b2o = batch2offset

def split_list_into_groups(lst, n):
    """
    Splits a list into chunks of size n.
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def fallback(*args):
    """
    Returns the first non-None argument.
    """
    for a in args:
        if a is not None:
            return a

############################################
# Numpy Utilities
############################################

def order_preserved_unique_np(array, return_inverse=False):
    """
    Finds unique elements in a numpy array while preserving their original order of appearance.
    Standard np.unique sorts the output.

    Args:
        array (np.ndarray): Input array.
        return_inverse (bool): If True, also return indices to reconstruct the original array.

    Returns:
        unique_elements (np.ndarray): Unique elements in order of appearance.
        inverse_indices (np.ndarray, optional): Indices s.t. unique[inverse] == original.
    """
    u, ind, inverse = np.unique(array, return_index=True, return_inverse=True)
    ind = np.argsort(ind)
    u = u[ind]
    if not return_inverse:
        return u
    else:
        for index, value in enumerate(u):
            inverse[array == value] = index
        assert np.all(u[inverse] == array)
        return u, inverse
    

def truncate_top_k_np(x, k, inplace=False):
    """
    Keeps only the top-k values per row in a 2D array, setting others to 0.

    Args:
        x (np.ndarray): Input array of shape [M, N].
        k (int): Number of top elements to keep per row.
        inplace (bool): Whether to modify x in place.

    Returns:
        np.ndarray: Array with non-top-k values set to 0.
    """
    m, n = x.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = x[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = x < kth_vals[:, None]
    # replace mask by 0
    if not inplace:
        return np.where(is_smaller_than_kth, 0, x)
    x[is_smaller_than_kth] = 0
    return x    
    
###########################################
# Reduce
###########################################

def expand(x, index, dim=0):
    """
    Expands a tensor based on an index vector (inverse of scatter/segment_reduce).
    Useful for broadcasting global/pooled features back to individual nodes/points.

    Args:
        x (torch.Tensor): Input tensor to expand (e.g., global features).
        index (torch.Tensor): Index tensor dictating how many times to repeat each element.
        dim (int): Dimension along which to expand.

    Returns:
        expanded_x (torch.Tensor): Expanded tensor.

    Example: 
        x = [[1,2], [3,4]] (Batch size 2, feature dim 2)
        index = [0, 0, 0, 1, 1] (Batch indices for 5 points)
        output -> [[1,2], [1,2], [1,2], [3,4], [3,4]]
    """
    expanded_x = torch.index_select(x, dim, index)
    return expanded_x


###########################################
# Voxel Pooling and Sampling
###########################################

"""
class VoxelPooling(nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(VoxelPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, skip_fc=False, **point_attributes):
        coord, feat, offset = points
        batch = offset2batch(offset) # [0000...1111]
        if not skip_fc: feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]), # [     0,  76806, 156806]
                reduce="min") # [2, 3]
        cluster = voxel_grid( # torch_geometric
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0 # grid_size = 0.1
        )
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster, stable=True)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)]) # [     0,      3,      9,  ..., 156796, 156800, 156806]
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean") # pooling
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max") # the segment csr and voxel grid is the key operation for pooling
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        # segment_csr(flow[indices], idx_ptr, reduce='mean')
        out_point_attributes = {}
        if len(point_attributes) > 0:
            for k, v in point_attributes.items():
                if k in ['cluster', 'indices', 'idx_ptr']: continue
                reduce = 'max'
                if isinstance(v, (list, tuple)):
                    reduce, v = v
                prev_bool = False
                if v.dtype == torch.bool:
                    prev_bool = True
                    v = v.long()
                elif v.dtype in [torch.float32, torch.float64]:
                    reduce = 'mean'
                out_point_attributes[k] = segment_csr(v[sorted_cluster_indices], idx_ptr, reduce=reduce)
                if prev_bool: v = v.bool()
        
        # to transform coordinate, just `cluster[corr[:, 0]]`
        return [coord, feat, offset], {'cluster': cluster, 'indices': sorted_cluster_indices, 'idx_ptr': idx_ptr, **out_point_attributes}
"""
 