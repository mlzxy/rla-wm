import torch
from torch import Tensor
import numpy as np
from typing import List, Union, Dict, Any, Optional

from ....utils.general_utils import dict_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(
        self,
        cond: Union[Tensor, List[Any], Dict[str, Any]],
        neg_cond: Union[Tensor, List[Any], Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Tensor, List[Any], Dict[str, Any]]:
        """
        Get the conditioning data with random unconditional replacement (CFG training).

        During training, this method randomly replaces the conditioning signal `cond`
        with the negative/unconditional signal `neg_cond` with probability `p_uncond`.
        This enables the model to learn both conditional and unconditional generation,
        required for Classifier-Free Guidance.

        Args:
            cond: The conditioning signal. Can be a Tensor, List, or Dict of them.
            neg_cond: The negative/unconditional signal. Must have the same structure/shape as `cond`.
            **kwargs: Additional arguments.

        Returns:
            The potentially modified conditioning signal (either `cond` or `neg_cond` for each batch element).
        """
        assert neg_cond is not None, (
            "neg_cond must be provided for classifier-free guidance"
        )

        if self.p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            ref_cond = (
                cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            )
            B = get_batch_size(ref_cond)

            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(
                        -1, *[1] * (cond.ndim - 1)
                    )
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            mask = list(np.random.rand(B) < self.p_uncond)
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach(
                    [cond, neg_cond], lambda x: select(x[0], x[1], mask)
                )

        return cond

    def get_inference_cond(
        self,
        cond: Union[Tensor, List[Any], Dict[str, Any]],
        neg_cond: Union[Tensor, List[Any], Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get the conditioning data for inference.

        It bundles the conditional and unconditional signals into a dictionary
        that the sampler can use for CFG sampling.

        Args:
            cond: The conditioning signal.
            neg_cond: The negative/unconditional signal.
            **kwargs: Additional arguments.

        Returns:
            A dictionary containing:
                - 'cond': The conditioning signal.
                - 'neg_cond': The negative/unconditional signal.
                - **kwargs: Other arguments.
        """
        assert neg_cond is not None, (
            "neg_cond must be provided for classifier-free guidance"
        )
        return {"cond": cond, "neg_cond": neg_cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.

        Returns:
            A `FlowEulerCfgSampler` instance initialized with `sigma_min`.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
