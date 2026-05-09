import torch.nn as nn
from rich import print


# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params, model_param_names=[]):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_param = master_params[0]
    if master_param.grad is None:
        master_param.grad = master_param.new_empty(master_param.size())

    grad_flat = master_param.grad
    offset = 0
    for pi, param in enumerate(model_params):
        if param.grad is None and model_param_names:
            print(f"[red]{model_param_names[pi]} grad is NONE[/red]")
        grad = param.grad.data
        numel = grad.numel()
        grad_flat[offset : offset + numel].copy_(grad.flatten())
        offset += numel


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0
