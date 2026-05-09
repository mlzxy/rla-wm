import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from typing import Tuple, List, Union, Dict, Any, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
from lpips import LPIPS
from utils.vis import to_pil


def create_spatial_weight_mask(
    robot_masks: Float[Tensor, "... H W"],
    static_masks: Float[Tensor, "... H W"],
    objects_masks: Float[Tensor, "... H W"],
    weight_static: float = 1.0,
    weight_robot: float = 2.0,
    weight_object: float = 4.0,
    weight_other: float = 1.0,
) -> Float[Tensor, "... H W"]:
    """
    Create a spatial weight mask based on semantic regions.

    Args:
        robot_masks: Binary mask for robot regions (1 for robot, 0 otherwise).
        static_masks: Binary mask for static (background) regions (1 for static, 0 otherwise).
        objects_masks: Optional binary mask for object regions.
        weight_static: Weight for static/background regions.
        weight_robot: Weight for robot regions.
        weight_object: Weight for object (foreground) regions.
        weight_other: Weight for other regions (invalid regions).

    Returns:
        A spatial weight mask tensor of the same shape as the input masks.
    """
    other_masks = (1.0 - robot_masks) * (1.0 - static_masks) * (1.0 - objects_masks)
    weight_mask = (
        static_masks * weight_static
        + robot_masks * weight_robot
        + objects_masks * weight_object
        + other_masks * weight_other
    )
    return weight_mask


def smooth_l1_loss(
    pred: Float[Tensor, "..."], target: Float[Tensor, "..."], beta: float = 1.0
) -> Float[Tensor, ""]:
    """
    Compute smooth L1 loss.

    Args:
        pred: Predicted tensor of any shape.
        target: Target tensor of same shape as pred.
        beta: Threshold for switching between L1 and L2 loss.

    Returns:
        Scalar mean smooth L1 loss.
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss.mean()


def l1_loss(
    network_output: Float[Tensor, "..."],
    gt: Float[Tensor, "..."],
    weight_mask: Optional[Float[Tensor, "..."]] = None,
) -> Float[Tensor, ""]:
    """
    Compute mean L1 loss, optionally with spatial weights.

    Args:
        network_output: Predicted tensor of any shape.
        gt: Ground truth tensor of same shape as network_output.
        weight_mask: Optional spatial weight mask of same shape.

    Returns:
        Scalar mean L1 loss.
    """
    loss = torch.abs(network_output - gt)
    if weight_mask is not None:
        loss = loss * weight_mask
    return loss.mean()


def l2_loss(
    network_output: Float[Tensor, "..."],
    gt: Float[Tensor, "..."],
    weight_mask: Optional[Float[Tensor, "..."]] = None,
) -> Float[Tensor, ""]:
    """
    Compute mean L2 (MSE) loss, optionally with spatial weights.

    Args:
        network_output: Predicted tensor of any shape.
        gt: Ground truth tensor of same shape as network_output.
        weight_mask: Optional spatial weight mask of same shape.

    Returns:
        Scalar mean L2 loss.
    """
    loss = (network_output - gt) ** 2
    if weight_mask is not None:
        loss = loss * weight_mask
    return loss.mean()


def gaussian(window_size: int, sigma: float) -> Float[Tensor, "window_size"]:
    """
    Create a 1D Gaussian kernel.

    Args:
        window_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        1D tensor representing the Gaussian kernel.
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(
    window_size: int, channel: int
) -> Float[Variable, "channel 1 window_size window_size"]:
    """
    Create a 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the 2D window.
        channel: Number of input channels.

    Returns:
        4D tensor (Variable) representing the Gaussian window.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def psnr(
    img1: Float[Tensor, "B C H W"], img2: Float[Tensor, "B C H W"], max_val: float = 1.0
) -> Float[Tensor, ""]:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image batch.
        img2: Second image batch.
        max_val: Maximum possible pixel value.

    Returns:
        Scalar PSNR value.
    """
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def image_gradients(
    image: Float[Tensor, "B C H W"]
) -> Tuple[Float[Tensor, "B C H W-1"], Float[Tensor, "B C H-1 W"]]:
    """
    Compute image gradients along x and y directions.

    Args:
        image: Input image batch.

    Returns:
        A tuple of (grad_x, grad_y).
    """
    # image shape: (B, C, H, W)
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    return grad_x, grad_y

def gradient_l1_loss(
    pred: Float[Tensor, "B C H W"],
    gt: Float[Tensor, "B C H W"],
    weight_mask: Optional[Float[Tensor, "B C H W"]] = None,
) -> Float[Tensor, ""]:
    """
    Compute L1 loss between image gradients, optionally with spatial weights.

    Args:
        pred: Predicted image batch.
        gt: Ground truth image batch.
        weight_mask: Optional spatial weight mask of shape [B, C, H, W] or broadcastable.

    Returns:
        Scalar gradient L1 loss.
    """
    pred_grad_x, pred_grad_y = image_gradients(pred)
    gt_grad_x, gt_grad_y = image_gradients(gt)

    loss_x = torch.abs(pred_grad_x - gt_grad_x)
    loss_y = torch.abs(pred_grad_y - gt_grad_y)

    if weight_mask is not None:
        # Match dimensions of gradients for broadcasting (Hx(W-1) and (H-1)xW)
        weight_mask_x = weight_mask[..., :, :-1]
        weight_mask_y = weight_mask[..., :-1, :]
        loss_x = loss_x * weight_mask_x
        loss_y = loss_y * weight_mask_y

    return loss_x.mean() + loss_y.mean()



def ssim(
    img1: Float[Tensor, "B C H W"],
    img2: Float[Tensor, "B C H W"],
    window_size: int = 11,
    size_average: bool = True,
    return_loss: bool = False,
    weight_mask: Optional[Float[Tensor, "B C H W"]] = None,
) -> Union[Float[Tensor, ""], Float[Tensor, "B"]]:
    """
    Compute Structural Similarity Index Measure (SSIM).

    Args:
        img1: First image batch.
        img2: Second image batch.
        window_size: Size of the Gaussian window.
        size_average: Whether to average the SSIM map.
        return_loss: If True, returns 1 - SSIM.
        weight_mask: Optional spatial weight mask. Only applied if return_loss=True.

    Returns:
        Scalar SSIM or a batch of SSIM values.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # When applying a pixel-wise weight mask, we need the un-averaged map first
    apply_weights = (return_loss and weight_mask is not None)
    compute_average = size_average and not apply_weights

    ssim_map = _ssim(img1, img2, window, window_size, channel, size_average=compute_average)
    
    if return_loss:
        loss_map = 1 - ssim_map
        if apply_weights:
            loss_map = loss_map * weight_mask
            if size_average:
                return loss_map.mean()
            else:
                return loss_map.mean(1).mean(1).mean(1)
        return loss_map
    else:
        return ssim_map


def _ssim(
    img1: Float[Tensor, "B C H W"],
    img2: Float[Tensor, "B C H W"],
    window: Float[Tensor, "C 1 WS WS"],
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> Union[Float[Tensor, ""], Float[Tensor, "B C H W"]]:
    """
    Internal helper for SSIM computation.

    Args:
        img1: First image batch.
        img2: Second image batch.
        window: Gaussian window tensor.
        window_size: Size of the window.
        channel: Number of channels.
        size_average: Whether to average the result.

    Returns:
        SSIM result.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


loss_fn_vgg = None


def lpips(
    img1: Float[Tensor, "B C H W"],
    img2: Float[Tensor, "B C H W"],
    value_range: Tuple[float, float] = (0, 1),
) -> Float[Tensor, ""]:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Args:
        img1: First image batch.
        img2: Second image batch.
        value_range: Range of pixel values in the input images.

    Returns:
        Scalar mean LPIPS loss.
    """
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net="vgg").cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()


def normal_angle(
    pred: Float[Tensor, "... 3"], gt: Float[Tensor, "... 3"]
) -> Union[Float[Tensor, ""], int]:
    """
    Compute the mean angle (in degrees) between two normal vector fields.

    Args:
        pred: Predicted normal vectors.
        gt: Ground truth normal vectors.

    Returns:
        Mean angle in degrees, or -1 if the result is NaN.
    """
    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    norms = pred.norm(dim=-1) * gt.norm(dim=-1)
    cos_sim = (pred * gt).sum(-1) / (norms + 1e-9)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos_sim[norms > 1e-9])).mean()
    if ang.isnan():
        return -1
    return ang


def compute_zoom_in_loss(
    rec_image: Float[Tensor, "B 3 H W"],
    gt_image: Float[Tensor, "B 3 H W"],
    key_mask: Union[Float[Tensor, "B 1 H W"], Float[Tensor, "B H W"]],
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.2,
    crop_size: int = 128,
    max_crop_size: int = 160,
    min_crop_size: int = 32,
    num_crops: int = 8,
    min_pixels: int = 100,
    weight_mask: Optional[Union[Float[Tensor, "B 1 H W"], Float[Tensor, "B H W"]]] = None,
    debug: bool = False,
) -> Dict[str, Union[Float[Tensor, ""], Any]]:
    """
    Compute zoom-in loss on key regions by randomly sampling multiple crops centered on active mask pixels.

    Args:
        rec_image: Reconstructed image batch.
        gt_image: Ground truth image batch.
        key_mask: Binary mask for key regions.
        lambda_ssim: Weight for SSIM loss.
        lambda_lpips: Weight for LPIPS loss.
        crop_size: Target size for cropped regions after resizing.
        min_crop_size: Minimum bounding box size before resizing.
        num_crops: Number of random crops to extract per image.
        min_pixels: Minimum active mask pixels for a valid key region.

    Returns:
        Dict with 'focus_loss', 'focus_l1', 'focus_ssim', 'focus_lpips' keys.
    """
    B, C, H, W = rec_image.shape

    # Ensure key_mask is [B, 1, H, W]
    if key_mask.dim() == 3:
        key_mask = key_mask.unsqueeze(1)
        
    if weight_mask is not None and weight_mask.dim() == 3:
        weight_mask = weight_mask.unsqueeze(1)

    rec_crops = []
    gt_crops = []
    weight_crops = [] if weight_mask is not None else None

    for i in range(B):
        mask_i = key_mask[i, 0]  # [H, W]
        ys, xs = torch.where(mask_i > 0)
        n_pixels = len(ys)

        if n_pixels < min_pixels:
            rec_crops.append(
                torch.zeros(num_crops, C, crop_size, crop_size, device=rec_image.device)
            )
            gt_crops.append(
                torch.zeros(num_crops, C, crop_size, crop_size, device=gt_image.device)
            )
            if weight_crops is not None:
                weight_crops.append(
                    torch.zeros(num_crops, 1, crop_size, crop_size, device=weight_mask.device)
                )
            continue

        # Randomly pick `num_crops` center points from the active pixels
        rand_indices = torch.randint(0, n_pixels, (num_crops,), device=mask_i.device)
        center_ys = ys[rand_indices]
        center_xs = xs[rand_indices]

        rec_crops_i = []
        gt_crops_i = []
        weight_crops_i = [] if weight_mask is not None else None

        for c_idx in range(num_crops):
            cy, cx = center_ys[c_idx].item(), center_xs[c_idx].item()
            
            # Randomly sample crop context size between min_crop_size and half the image size
            max_possible_crop = max(min_crop_size, max_crop_size)
            if max_possible_crop == min_crop_size:
                c_side = min_crop_size
            else:
                c_side = torch.randint(min_crop_size, max_possible_crop + 1, (1,)).item()
            
            half_side = c_side // 2
            
            # Initial boundaries
            y1 = cy - half_side
            y2 = cy + half_side + (c_side % 2) # ensure exact size
            x1 = cx - half_side
            x2 = cx + half_side + (c_side % 2)
            
            # Shift boundaries if they go outside the image
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if y2 > H:
                y1 -= (y2 - H)
                y2 = H
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if x2 > W:
                x1 -= (x2 - W)
                x2 = W
                
            y1, y2 = max(0, y1), min(H, y2)
            x1, x2 = max(0, x1), min(W, x2)

            # Crop and resize
            rec_crop = rec_image[i : i + 1, :, y1:y2, x1:x2]
            gt_crop = gt_image[i : i + 1, :, y1:y2, x1:x2]

            rec_crop = F.interpolate(
                rec_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False
            )
            gt_crop = F.interpolate(
                gt_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False
            )

            rec_crops_i.append(rec_crop)
            gt_crops_i.append(gt_crop)
            
            if weight_crops is not None:
                w_crop = weight_mask[i : i + 1, :, y1:y2, x1:x2]
                w_crop = F.interpolate(
                    w_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False
                )
                weight_crops_i.append(w_crop)
                
        rec_crops.append(torch.cat(rec_crops_i, dim=0))
        gt_crops.append(torch.cat(gt_crops_i, dim=0))
        if weight_crops is not None:
            weight_crops.append(torch.cat(weight_crops_i, dim=0))

    if len(rec_crops) == 0:
        return {}

    rec_crops = torch.cat(rec_crops, dim=0)
    gt_crops = torch.cat(gt_crops, dim=0)
    
    if weight_crops is not None:
        weight_crops = torch.cat(weight_crops, dim=0)

    if debug:
        import os
        from torchvision.utils import save_image
        os.makedirs("runs/debug_crops", exist_ok=True)
        # Create a grid of images for visualization
        # shape: [B*num_crops, C, H, W]
        
        # Save reconstructed crops
        save_image(
            rec_crops, 
            os.path.join("runs/debug_crops", f"rec_crops_bs{B}.png"), 
            nrow=num_crops, 
            normalize=False, 
            value_range=(0, 1) # assuming images are usually in [-1, 1] for VAEs
        )
        
        # Save ground truth crops
        save_image(
            gt_crops, 
            os.path.join("runs/debug_crops", f"gt_crops_bs{B}.png"), 
            nrow=num_crops, 
            normalize=False, 
            value_range=(0, 1)
        )
        
        # Save weights if available
        if weight_crops is not None:
            save_image(
                weight_crops, 
                os.path.join("runs/debug_crops", f"weight_crops_bs{B}.png"), 
                nrow=num_crops,
                normalize=True,
                value_range=(0, weight_crops.max().clamp(min=1e-5).item())
            )

    terms = {}
    terms["focus_l1"] = l1_loss(rec_crops, gt_crops, weight_mask=weight_crops)
    loss = terms["focus_l1"]

    if lambda_ssim > 0:
        terms["focus_ssim"] = ssim(rec_crops, gt_crops, return_loss=True, weight_mask=weight_crops)
        loss = loss + lambda_ssim * terms["focus_ssim"]

    if lambda_lpips > 0:
        terms["focus_lpips"] = lpips(rec_crops, gt_crops)
        loss = loss + lambda_lpips * terms["focus_lpips"]

    terms["focus_loss"] = loss
    return terms


def apply_gradient_hack(modules_list: List[torch.nn.Module]) -> Union[Float[Tensor, ""], int]:
    """
    Sums all parameters in a list of modules and multiplies by 0
    to force gradient synchronization in distributed training.

    Args:
        modules_list: List of PyTorch modules.

    Returns:
        A scalar dummy loss (zero).
    """
    dummy_loss = 0
    for module in modules_list:
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                # We sum the parameters and multiply by 0.
                # .sum() handles tensors of any shape.
                dummy_loss += (param.float() * 0.0).sum()

    return dummy_loss


def multiclass_focal_loss(
    inputs: Float[Tensor, "B C ..."],
    targets: Union[Int[Tensor, "B ..."], Float[Tensor, "B C ..."]],
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    """
    Compute multiclass focal loss.
    
    Args:
        inputs: Logits from the model, shape (B, C, ...)
        targets: Class indices, shape (B, ...) OR probabilities/one-hot, shape (B, C, ...)
        gamma: Focal loss focusing parameter
        reduction: String specifying the reduction to apply ('none', 'mean', 'sum')
        
    Returns:
        Focal loss of shape (B, ...) if reduction is 'none'
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    
    probs = F.softmax(inputs, dim=1)
    
    if targets.dim() == inputs.dim():
        # Soft targets / one-hot
        pt = (probs * targets).sum(dim=1)
    else:
        # Class indices. We need to gather the probability of the true class.
        pt = probs.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        
    focal_weight = (1.0 - pt) ** gamma
    loss = focal_weight * ce_loss
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
