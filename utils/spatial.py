import torch
import torch.nn.functional as F

def generate_plucker_rays(w2cs: torch.Tensor, intrinsics: torch.Tensor, image_wh: tuple) -> torch.Tensor:
    """
    Args:
        w2cs: World-to-Camera 矩阵，形状为 (B, 4, 4)
        intrinsics: Camera 内参矩阵，形状为 (B, 3, 3)
        image_wh: 图像宽高 (W, H)
    Returns:
        plucker_rays: 普吕克射线张量，形状为 (B, 6, H, W)
    """
    B = w2cs.shape[0]
    W, H = image_wh
    device, dtype = w2cs.device, w2cs.dtype

    # 1. 构建 2D 像素网格 (加上 0.5 获取像素中心对齐)
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    x = x + 0.5
    y = y + 0.5
    
    # 构建齐次像素坐标 [u, v, 1]^T，并铺平为 (B, 3, H*W)
    pixels = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0)
    pixels = pixels.unsqueeze(0).expand(B, -1, -1)

    # 2. 计算 c2w (Camera-to-World) 并提取旋转和平移
    c2ws = torch.inverse(w2cs)
    R_c2w = c2ws[:, :3, :3]  # (B, 3, 3)
    t_c2w = c2ws[:, :3, 3:]  # (B, 3, 1)，即射线原点 o

    # 3. 计算世界坐标系下的射线方向 d
    K_inv = torch.inverse(intrinsics)  # (B, 3, 3)
    
    # 像素转相机坐标系方向: K^-1 @ [u, v, 1]^T -> (B, 3, H*W)
    d_cam = torch.bmm(K_inv, pixels)   
    
    # 相机坐标系转世界坐标系方向: R @ d_cam -> (B, 3, H*W)
    d_world = torch.bmm(R_c2w, d_cam)  
    
    # 归一化方向向量并 reshape
    d_world = F.normalize(d_world, p=2, dim=1) # (B, 3, H*W)
    d_world = d_world.view(B, 3, H, W)

    # 4. 扩展原点 o 的形状以匹配图像尺寸
    o = t_c2w.view(B, 3, 1, 1).expand(B, 3, H, W)

    # 5. 计算力矩 (Moment) m = o x d
    m = torch.cross(o, d_world, dim=1) # (B, 3, H, W)

    # 6. 拼接方向 d 和力矩 m 组成 6D 普吕克坐标
    plucker_rays = torch.cat([d_world, m], dim=1) # (B, 6, H, W)

    return plucker_rays


def generate_pointmaps(
    w2cs: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_map: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    background_mode: str = 'per_image_far',
    far_depth: float = 100.0,
    far_scale: float = 1.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        w2cs: World-to-Camera 矩阵，形状为 (B, 4, 4)
        intrinsics: Camera 内参矩阵，形状为 (B, 3, 3)
        depth_map: 深度图，形状为 (B, H, W)
        foreground_mask: 前景掩码，形状为 (B, H, W)，True/1 表示有效前景深度
        background_mode: 背景深度填充策略，可选 ['far_plane', 'per_image_far', 'zero']
        far_depth: 当 background_mode='far_plane' 时使用的固定远平面深度
        far_scale: 当 background_mode='per_image_far' 时，背景深度 = far_scale * 每张图有效前景最大深度
        eps: 数值稳定性项，避免除零
    Returns:
        point_map: 世界坐标系下的 3D 点图，形状为 (B, 3, H, W)
    """
    B, H, W = depth_map.shape
    device, dtype = w2cs.device, w2cs.dtype

    if foreground_mask is not None:
        if foreground_mask.shape != depth_map.shape:
            raise ValueError(
                f"foreground_mask shape {foreground_mask.shape} must match depth_map shape {depth_map.shape}."
            )

        fg_mask = foreground_mask.to(dtype=torch.bool, device=device)

        if background_mode == 'far_plane':
            bg_depth = torch.full((B, 1, 1), far_depth, dtype=dtype, device=device)
        elif background_mode == 'per_image_far':
            valid_depth = torch.where(fg_mask, depth_map, torch.zeros_like(depth_map))
            max_fg_depth = valid_depth.view(B, -1).amax(dim=1, keepdim=True)
            fallback_far = torch.full_like(max_fg_depth, far_depth)
            max_fg_depth = torch.where(max_fg_depth > eps, max_fg_depth, fallback_far)
            bg_depth = (max_fg_depth * far_scale).view(B, 1, 1)
        elif background_mode == 'zero':
            bg_depth = torch.zeros((B, 1, 1), dtype=dtype, device=device)
        else:
            raise ValueError(
                f"Unsupported background_mode: {background_mode}. "
                "Use one of ['far_plane', 'per_image_far', 'zero']."
            )

        depth_map = torch.where(fg_mask, depth_map, bg_depth)

    # 1. 构建 2D 像素网格
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    x = x + 0.5
    y = y + 0.5
    
    pixels = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0)
    pixels = pixels.unsqueeze(0).expand(B, -1, -1) # (B, 3, H*W)

    # 2. 计算 c2w 并提取旋转和平移
    c2ws = torch.inverse(w2cs)
    R_c2w = c2ws[:, :3, :3]  # (B, 3, 3)
    t_c2w = c2ws[:, :3, 3:]  # (B, 3, 1)

    # 3. 投影到相机坐标系
    K_inv = torch.inverse(intrinsics)  # (B, 3, 3)
    
    # 计算相机坐标系下的归一化平面坐标 (X/Z, Y/Z, 1) -> (B, 3, H*W)
    d_cam = torch.bmm(K_inv, pixels)   

    # 铺平深度图 (B, 1, H*W)，并将深度 (Z) 乘回去得到实际的相机系 3D 坐标 (X, Y, Z)
    depth_flat = depth_map.view(B, 1, H * W)
    P_cam = d_cam * depth_flat         # (B, 3, H*W)

    # 4. 转换到世界坐标系: P_world = R @ P_cam + t
    P_world = torch.bmm(R_c2w, P_cam) + t_c2w # (B, 3, H*W)

    # 5. 还原空间维度
    point_map = P_world.view(B, 3, H, W)

    return point_map