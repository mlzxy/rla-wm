from typing import *
import torch
import math
from . import DEBUG, BACKEND

if BACKEND == "xformers":
    import xformers.ops as xops
elif BACKEND == "flash_attn":
    pass
    # import flash_attn
elif BACKEND == "sdpa":
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == "naive":
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = [
    "scaled_dot_product_attention",
]


def _naive_sdpa(q, k, v, attn_mask=None):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)  # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)  # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)  # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor

    if attn_mask is not None:
        # attn_mask: (B, L_kv) or (B, L_q, L_kv)
        # expand to (B, H, L_q, L_kv)
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        attn_weight = attn_weight.masked_fill(attn_mask, float("-inf"))

    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    out = out.permute(0, 2, 1, 3)  # [N, L, H, C]
    return out


@overload
def scaled_dot_product_attention(
    qkv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
        attn_mask (torch.Tensor, optional): Mask for attention.
    """
    ...


@overload
def scaled_dot_product_attention(
    q: torch.Tensor, kv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
        attn_mask (torch.Tensor, optional): Mask for attention.
    """
    ...


@overload
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.
        attn_mask (torch.Tensor, optional): Mask for attention.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...


def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}

    # Check if attn_mask is passed as kwarg or positional (not supported positional here for simplicity in logic)
    attn_mask = kwargs.pop("attn_mask", None)

    num_all_args = len(args) + len(kwargs)

    assert num_all_args in arg_names_dict, (
        f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    )

    # Validate args
    # ... (skipping some detailed validation for brevity, rely on usage)

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, (
            f"Invalid shape for qkv, got {qkv.shape} (expected 5 dims)"
        )
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        device = q.device

    # Determine backend to use
    backend_to_use = BACKEND
    if attn_mask is not None:
        if BACKEND in ["xformers", "flash_attn"]:
            # Fallback to sdpa if mask is present, as they might not support arbitrary masks easily or require different API
            # For simplicity in this task:
            backend_to_use = "sdpa"

    if backend_to_use == "xformers":
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        # xformers specific mask handling if needed, but we fallbacked above
        out = xops.memory_efficient_attention(q, k, v)

    elif backend_to_use == "flash_attn":
        if num_all_args == 1:
            out = flash_attn.flash_attn_qkvpacked_func(qkv)
        elif num_all_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
        elif num_all_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)

    elif backend_to_use == "sdpa":
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)  # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)  # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)  # [N, H, L, C]

        # Prepare mask for sdpa
        # SDPA expects attn_mask to be broadcastable
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )  # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)  # [N, L, H, C]

    elif backend_to_use == "naive":
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = _naive_sdpa(q, k, v, attn_mask=attn_mask)
    else:
        raise ValueError(f"Unknown attention module: {backend_to_use}")

    return out
