import torch
import torch.nn.functional as F


def dice_loss(pred_logits, target, smooth=1e-5):
    """
    Compute the Dice Loss.

    Args:
        pred_logits (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth labels (binary 0/1).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Dice loss.
    """
    pred_probs = torch.sigmoid(pred_logits)
    intersection = (pred_probs * target).sum()
    union = pred_probs.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def focal_loss(pred_logits, target, gamma=2.0, alpha=0.25, reduction="mean"):
    """
    Compute the Focal Loss.

    Args:
        pred_logits (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth labels (binary 0/1).
        gamma (float): Focusing parameter.
        alpha (float): Balancing parameter.

    Returns:
        torch.Tensor: Focal loss.
    """
    pred_probs = torch.sigmoid(pred_logits)
    pt = torch.where(target == 1, pred_probs, 1 - pred_probs)

    # Calculate alpha factor
    alpha_factor = torch.ones_like(target) * alpha
    alpha_factor = torch.where(target == 1, alpha_factor, 1 - alpha_factor)

    focal_weight = alpha_factor * (1 - pt).pow(gamma)

    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    loss = focal_weight * bce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
