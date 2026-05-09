from typing import *
import torch
import torch.nn as nn
from src.modules.sparse.basic import SparseTensor


class SparseSpatial2Channel(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from spatial to channel.
    """

    def __init__(self, factor: int = 2):
        super(SparseSpatial2Channel, self).__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1

        coord = list(x.coords.unbind(dim=-1))
        for i in range(DIM):
            coord[i + 1] = coord[i + 1] // self.factor
        subidx = x.coords[:, 1:] % self.factor
        subidx = sum([subidx[..., i] * self.factor**i for i in range(DIM)])

        MAX = [(s + self.factor - 1) // self.factor for s in x.spatial_shape]
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        code, idx = code.unique(return_inverse=True)

        new_coords = torch.stack(
            [code // OFFSET[0]]
            + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
            dim=-1,
        )

        new_feats = torch.zeros(
            new_coords.shape[0] * self.factor**DIM,
            x.feats.shape[1],
            device=x.feats.device,
            dtype=x.feats.dtype,
        )
        new_feats[idx * self.factor**DIM + subidx] = x.feats

        out = SparseTensor(
            new_feats.reshape(new_coords.shape[0], -1),
            new_coords,
            torch.Size([x._shape[0], x._shape[1] * self.factor**DIM]),
            spatial_shape=[s // self.factor for s in x.spatial_shape],
        )
        return out
