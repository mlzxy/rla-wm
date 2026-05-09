import torch
from sklearn.neighbors import NearestNeighbors
from gsplat.rendering import rasterization
from jaxtyping import Float32
from torch import Tensor
from typing import TypedDict


class SplatsDict(TypedDict):
    means: Float32[Tensor, "g 3"] | Float32[Tensor, "t g 3"]
    quats: Float32[Tensor, "g 4"] | Float32[Tensor, "t g 4"]

    colors: Float32[Tensor, "g 3"]
    scales: Float32[Tensor, "g 3"]
    opacities: Float32[Tensor, " g "]

    masks: Float32[Tensor, "g class_num"] 


class RenderOutput(TypedDict):
    color: Float32[Tensor, "t 3 h w"]
    alpha: Float32[Tensor, "t 1 h w"]
    depth: Float32[Tensor, "t 1 h w"]
    bg_color: Float32[Tensor, "t 3"]
    masks: Float32[Tensor, "t class_num h w"]  
    info: any


def knn(x, K: int = 4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def init_splat_dict_from_pcd(pts, rgbs) -> SplatsDict:
    if not isinstance(pts, torch.Tensor):
        pts, rgbs = torch.from_numpy(pts).float(), torch.from_numpy(rgbs).float()
    device = pts.device
    assert rgbs.max() <= 1.0 and rgbs.min() >= 0.0

    rotations = torch.zeros((len(pts), 4))  # quaternion
    rotations[:, 0] = 1

    dist2_avg = (knn(pts, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg).unsqueeze(-1).repeat(1, 3)

    return dict(
        means=pts,
        colors=rgbs,
        scales=dist_avg.to(device),
        quats=rotations.to(device),
        opacities=0.1 * torch.ones((len(pts),), dtype=torch.float).to(device),
    )


def render(
    splats: SplatsDict,
    w2c: Float32[Tensor, "t 4 4"] | Float32[Tensor, "4 4"],
    K: Float32[Tensor, "t 3 3"] | Float32[Tensor, "3 3"],
    img_wh: tuple[int, int],
    bg_color: Float32[Tensor, "t 3"] | str | None = None,
    bg_mask: Float32[Tensor, "t num_classes"] | None = None,
    depth: bool = True,
) -> RenderOutput:
    if bg_color is None:
        bg_color = torch.ones(3).to(w2c.device)
    if isinstance(bg_color, (tuple, list)):
        bg_color = torch.as_tensor(bg_color, dtype=torch.float).to(w2c.device)
    if bg_color == "random":
        bg_color = torch.rand((w2c.shape[0], 3), device=w2c.device)
    if bg_color.dim() == 1:
        bg_color = bg_color.unsqueeze(0).repeat(w2c.shape[0], 1)

    has_masks = "masks" in splats
    if has_masks:
        masks = splats["masks"]
        class_num = masks.shape[-1]
        if bg_mask is None:
            bg_color_pad = torch.zeros((bg_color.shape[0], class_num), device=bg_color.device, dtype=bg_color.dtype)
        else:
            bg_color_pad = bg_mask
        bg_color_padded = torch.cat([bg_color, bg_color_pad], dim=-1)
    else:
        bg_color_padded = bg_color

    if w2c.dim() == 2:
        w2c = w2c.unsqueeze(0)
    if K.dim() == 2:
        K = K.unsqueeze(0)
    W, H = img_wh

    render_kwargs = {k: v for k, v in splats.items() if k not in ["masks", "colors"]}
    if has_masks:
        render_kwargs["colors"] = torch.cat([splats["colors"], masks], dim=-1)
    else:
        render_kwargs["colors"] = splats["colors"]

    render_arrs, alphas, info = (
        rasterization(  # render_arrs: [t, h, w, channels (+1 if depth is True)]
            **render_kwargs,
            backgrounds=bg_color_padded.type(w2c.dtype),
            viewmats=w2c,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            render_mode="RGB+ED" if depth else "RGB",
        )
    )
    if depth:
        render_colors = render_arrs[..., :3]
        if has_masks:
            render_masks = render_arrs[..., 3:-1]
            render_depth = render_arrs[..., -1:]
        else:
            render_depth = render_arrs[..., 3:]
    else:
        render_colors = render_arrs[..., :3]
        if has_masks:
            render_masks = render_arrs[..., 3:]
        render_depth = torch.zeros_like(alphas)

    out = {
        "color": render_colors.permute(0, 3, 1, 2),
        "alpha": alphas.permute(0, 3, 1, 2),
        "depth": render_depth.permute(0, 3, 1, 2),
        "bg_color": bg_color,
        "info": info,
    }
    if has_masks:
        out["masks"] = render_masks.permute(0, 3, 1, 2)
        
    return out
