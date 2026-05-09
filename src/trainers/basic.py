import os
import sys
from utils.misc import attach_debugger
import os.path as osp
import re
import copy
from functools import partial
from rich import print
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from .utils import *
from .base import Trainer
from ..utils.general_utils import *
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils


class BasicTrainer(Trainer):
    """
    Trainer for basic training loop.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Models:")
        for name, model in self.models.items():
            lines.append(f"    - {name}: {model.__class__.__name__}")
        lines.append(f"  - Dataset: {indent(str(self.dataset), 2)}")
        lines.append(f"  - Dataloader:")
        if getattr(self, "dataloader", None) is not None:
            lines.append(f"    - Sampler: {self.dataloader.sampler.__class__.__name__}")
            lines.append(f"    - Num workers: {self.dataloader.num_workers}")
        lines.append(f"  - Number of steps: {self.max_steps}")
        lines.append(f"  - Number of GPUs: {self.world_size}")
        lines.append(f"  - Batch size: {self.batch_size}")
        lines.append(f"  - Batch size per GPU: {self.batch_size_per_gpu}")
        lines.append(f"  - Batch split: {self.batch_split}")
        if getattr(self, "inference_only", False):
            lines.append("  - Mode: Inference Only")
        else:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                lines.append(f"  - Optimizer: {self.optimizer.__class__.__name__}")
                lines.append(
                    f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}"
                )
            if self.lr_scheduler_config is not None:
                lines.append(
                    f"  - LR scheduler: {self.lr_scheduler.__class__.__name__}"
                )
        if self.elastic_controller_config is not None:
            lines.append(
                f"  - Elastic memory: {indent(str(self.elastic_controller), 2)}"
            )
        if self.grad_clip is not None:
            lines.append(f"  - Gradient clip: {indent(str(self.grad_clip), 2)}")
        lines.append(f"  - EMA rate: {self.ema_rate}")
        lines.append(f"  - FP16 mode: {self.fp16_mode}")
        return "\n".join(lines)

    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        for name, model in self.models.items():
            if name in self.freeze_models:
                print(f"[blue]Freezing model: {name}[/blue]")
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()

        if getattr(self, "inference_only", False):
            self.training_models = self.models
            self.model_params = []
            self.master_params = []
            self.ema_params = []
            self.optimizer = None
            self.scaler = None

            # Initialize elastic memory controller if configured
            if getattr(self, "elastic_controller_config", None) is not None:
                self.elastic_controller = getattr(
                    elastic_utils, self.elastic_controller_config["name"]
                )(**self.elastic_controller_config["args"])
                for model in self.models.values():
                    if isinstance(
                        model,
                        (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin),
                    ):
                        model.register_memory_controller(self.elastic_controller)
            return

        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: (
                    DDP(
                        model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        bucket_cap_mb=128,
                        find_unused_parameters=False,  # NOTE: set to True to debug DDP issues
                    )
                    if name not in self.freeze_models
                    and any(p.requires_grad for p in model.parameters())
                    else model
                )
                for name, model in self.models.items()
            }
            # if self.fp16_mode == "amp":
            #     for name, model in self.training_models.items():
            #         if isinstance(model, DDP):
            #             model._set_static_graph()
        else:
            self.training_models = self.models

        # Build master params (only from non-frozen models)
        self.model_param_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
                if name not in self.freeze_models
            ],
            [],
        )
        self.model_params = sum(
            [
                [p for p in model.parameters() if p.requires_grad]
                for name, model in self.models.items()
                if name not in self.freeze_models
            ],
            [],
        )
        if self.fp16_mode == "amp":
            self.master_params = self.model_params
            self.scaler = torch.GradScaler() if self.fp16_mode == "amp" else None
        elif self.fp16_mode == "inflat_all":
            self.master_params = make_master_params(self.model_params)
            self.fp16_scale_growth = self.fp16_scale_growth
            if self.log_scale <= 0:
                self.log_scale = 20.0
        elif self.fp16_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f"FP16 mode {self.fp16_mode} is not implemented.")

        # Build EMA params
        if self.is_master and not getattr(self, "inference_only", False):
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if getattr(self, "optimizer_config", None):
            if hasattr(torch.optim, self.optimizer_config["name"]):
                self.optimizer = getattr(torch.optim, self.optimizer_config["name"])(
                    self.master_params, **self.optimizer_config["args"]
                )
            else:
                self.optimizer = globals()[self.optimizer_config["name"]](
                    self.master_params, **self.optimizer_config["args"]
                )
        else:
            self.optimizer = None

        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"]):
                self.lr_scheduler = getattr(
                    torch.optim.lr_scheduler, self.lr_scheduler_config["name"]
                )(self.optimizer, **self.lr_scheduler_config["args"])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config["name"]](
                    self.optimizer, **self.lr_scheduler_config["args"]
                )

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any(
                [
                    isinstance(
                        model,
                        (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin),
                    )
                    for model in self.models.values()
                ]
            ), (
                "No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin"
            )
            self.elastic_controller = getattr(
                elastic_utils, self.elastic_controller_config["name"]
            )(**self.elastic_controller_config["args"])
            for model in self.models.values():
                if isinstance(
                    model,
                    (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin),
                ):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip["name"])(
                    **self.grad_clip["args"]
                )

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        Only includes non-frozen models.
        """
        # TODO: debug why a small model also saves a big file
        if self.fp16_mode == "inflat_all":
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {
            name: model.state_dict()
            for name, model in self.models.items()
            if name not in self.freeze_models
        }
        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
                if name not in self.freeze_models
            ],
            [],
        )
        for i, (model_name, param_name) in enumerate(master_params_names):
            if model_name in state_dicts:
                state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        if not master_params:
            return

        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
                if name not in self.freeze_models
            ],
            [],
        )
        params = [
            state_dicts[name][param_name] for name, param_name in master_params_names
        ]
        if self.fp16_mode == "inflat_all":
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def _load_state_dict_with_mismatch_check(self, model, model_ckpt, name=""):
        if not self.load_strict:
            model_state_dict = model.state_dict()
            mismatched_keys = []
            for k in list(model_ckpt.keys()):
                if (
                    k in model_state_dict
                    and model_ckpt[k].shape != model_state_dict[k].shape
                ):
                    ckpt_shape = model_ckpt[k].shape
                    model_shape = model_state_dict[k].shape
                    if len(ckpt_shape) == len(model_shape):
                        if self.is_master:
                            print(
                                f"    [yellow]Warning: {name} parameter {k} shape mismatch, {ckpt_shape} vs {model_shape}. Copying matched parts...[/yellow]"
                            )
                        mismatched_keys.append(k)
                        new_param = model_state_dict[k].clone()
                        slices = tuple(
                            slice(0, min(c, m)) for c, m in zip(ckpt_shape, model_shape)
                        )
                        new_param[slices] = model_ckpt[k][slices]
                        model_ckpt[k] = new_param
                    else:
                        if self.is_master:
                            print(
                                f"    [yellow]Warning: {name} parameter {k} dim mismatch, {len(ckpt_shape)} vs {len(model_shape)}. Skipping...[/yellow]"
                            )
                        mismatched_keys.append(k)
                        del model_ckpt[k]

            load_result = model.load_state_dict(model_ckpt, strict=False)

            if load_result is not None:
                for k in mismatched_keys:
                    if k in load_result.missing_keys:
                        load_result.missing_keys.remove(k)
                load_result.unexpected_keys.extend(mismatched_keys)
            return load_result
        else:
            return model.load_state_dict(model_ckpt, strict=True)

    def load(self, load_dir, step=None, ema=True, load_misc=True):
        """
        Load a checkpoint.
        Should be called by all processes.

        Args:
            load_misc (bool): Whether to restore misc states (optimizer/sampler/scaler/etc.).
        """
        if step is None:
            ckpt_dir = os.path.join(load_dir, "ckpts")
            if os.path.isdir(ckpt_dir):
                files = os.listdir(ckpt_dir)
                # Matches both regular and EMA checkpoints: name_step123.pt, name_ema_step123.pt, misc_step123.pt
                pattern = re.compile(r".*_step(\d+)\.pt$")
                steps = []
                for f in files:
                    if not ema and "ema" in osp.basename(f):
                        continue
                    match = pattern.match(f)
                    if match:
                        steps.append(int(match.group(1)))
                if steps:
                    step = max(steps)
        if step is None:
            print("[blue]No checkpoint found.[/blue]")
            return

        if self.is_master:
            print(f"\nLoading checkpoint from step {step}...", end="")

        # Prepare step_str for filename formatting
        if isinstance(step, int):
            step_str = f"{step:07d}"
        else:
            try:
                # If it's a string that represents an integer, still pad it
                step_str = f"{int(step):07d}"
            except (ValueError, TypeError):
                # Use as a literal string suffix
                step_str = str(step)

        has_bad_keys = False
        nothing_trainable_loaded = True
        model_ckpts = {}
        for name, model in self.models.items():
            # Strategy: Fallback to EMA if regular checkpoint is not available
            ckpt_path = os.path.join(load_dir, "ckpts", f"{name}_step{step_str}.pt")

            if not os.path.exists(ckpt_path) and self.ema_rate:
                # Try EMA files in descending order of rate
                for ema_rate in sorted(self.ema_rate, reverse=True):
                    test_path = os.path.join(
                        load_dir, "ckpts", f"{name}_ema{ema_rate}_step{step_str}.pt"
                    )
                    if os.path.exists(test_path):
                        ckpt_path = test_path
                        if self.is_master:
                            print(
                                f"\n  [yellow]Regular checkpoint for {name} not found. Using EMA checkpoint with rate {ema_rate}.[/yellow]"
                            )
                        break

            if name in self.freeze_models:
                # For frozen models, loading is optional
                if os.path.exists(ckpt_path):
                    model_ckpt = torch.load(
                        read_file_dist(ckpt_path),
                        map_location=self.device,
                        weights_only=True,
                    )
                    load_result = self._load_state_dict_with_mismatch_check(
                        model, model_ckpt, name
                    )
                    if not self.load_strict and load_result is not None:
                        if load_result.missing_keys or load_result.unexpected_keys:
                            has_bad_keys = True
                    if self.is_master:
                        print(
                            f"\n  Loaded frozen model {name} from {os.path.basename(ckpt_path)}."
                        )
                        if not self.load_strict and load_result is not None:
                            if load_result.missing_keys:
                                print(
                                    f"[red]    Missing keys: {load_result.missing_keys}[/red]"
                                )
                            if load_result.unexpected_keys:
                                print(
                                    f"[red]    Unexpected keys: {load_result.unexpected_keys}[/red]"
                                )
                else:
                    if self.is_master:
                        print(f"\n  Skipped frozen model {name} (no checkpoint file).")
            else:
                if os.path.exists(ckpt_path):
                    nothing_trainable_loaded = False
                    if self.is_master:
                        print(
                            f"\n  Loading model {name} from {os.path.basename(ckpt_path)}..."
                        )
                    model_ckpt = torch.load(
                        read_file_dist(ckpt_path),
                        map_location=self.device,
                        weights_only=True,
                    )
                    load_result = self._load_state_dict_with_mismatch_check(
                        model, model_ckpt, name
                    )
                    model_ckpts[name] = dict(model.state_dict())
                    if not self.load_strict and load_result is not None:
                        if load_result.missing_keys or load_result.unexpected_keys:
                            has_bad_keys = True
                    if (
                        self.is_master
                        and not self.load_strict
                        and load_result is not None
                    ):
                        if load_result.missing_keys:
                            print(f"    Missing keys: {load_result.missing_keys}")
                        if load_result.unexpected_keys:
                            print(f"    Unexpected keys: {load_result.unexpected_keys}")
                else:
                    if self.is_master:
                        print(
                            f"\n[bold red]Warning: {name} checkpoint not found at {ckpt_path}. Skipping model load.[/bold red]"
                        )
                    model_ckpts[name] = dict(model.state_dict())
                if self.fp16_mode == "inflat_all":
                    if hasattr(model, "convert_to_fp16"):
                        model.convert_to_fp16()

        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    if name in self.freeze_models:
                        # Skip frozen models for EMA loading
                        continue

                    # Look for EMA checkpoint, fallback to standard if missing
                    ema_path = os.path.join(
                        load_dir, "ckpts", f"{name}_ema{ema_rate}_step{step_str}.pt"
                    )
                    if not os.path.exists(ema_path):
                        ema_path = os.path.join(
                            load_dir, "ckpts", f"{name}_step{step_str}.pt"
                        )

                    if os.path.exists(ema_path):
                        ema_ckpt = torch.load(
                            ema_path,
                            map_location=self.device,
                            weights_only=True,
                        )
                        merged_ema_ckpt = dict(model.state_dict())
                        for k, v in ema_ckpt.items():
                            if (
                                k in merged_ema_ckpt
                                and merged_ema_ckpt[k].shape == v.shape
                            ):
                                merged_ema_ckpt[k] = v
                        ema_ckpts[name] = merged_ema_ckpt
                    else:
                        print(
                            f"\n  [yellow]Warning: EMA parameters for {name} rate {ema_rate} could not be restored (no file).[/yellow]"
                        )

                if ema_ckpts:
                    self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts

        # Load misc details if they exist
        misc_path = os.path.join(load_dir, "ckpts", f"misc_step{step_str}.pt")
        should_load_misc = load_misc and not self.skip_load_misc
        if (
            should_load_misc
            and
            os.path.exists(misc_path)
            and not has_bad_keys
            and not getattr(self, "inference_only", False)
            and not nothing_trainable_loaded
        ):
            misc_ckpt = torch.load(
                read_file_dist(misc_path),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            self.optimizer.load_state_dict(misc_ckpt["optimizer"])
            self.step = int(misc_ckpt["step"])
            self.data_sampler.load_state_dict(misc_ckpt["data_sampler"])
            if self.fp16_mode == "amp":
                self.scaler.load_state_dict(misc_ckpt["scaler"])
            elif self.fp16_mode == "inflat_all":
                self.log_scale = misc_ckpt["log_scale"]
            if self.lr_scheduler_config is not None:
                self.lr_scheduler.load_state_dict(misc_ckpt["lr_scheduler"])
            if self.elastic_controller_config is not None:
                self.elastic_controller.load_state_dict(misc_ckpt["elastic_controller"])
            if self.grad_clip is not None and not isinstance(self.grad_clip, float):
                self.grad_clip.load_state_dict(misc_ckpt["grad_clip"])
            del misc_ckpt
        else:
            if self.is_master:
                if not should_load_misc:
                    print(
                        "\n  [yellow]Warning: Skipping misc checkpoint restore because skip_load_misc is enabled.[/yellow]"
                    )
                elif has_bad_keys:
                    print(
                        f"\n  [yellow]Warning: Skipping misc checkpoint {misc_path} because of bad keys during model loading. Optimizer state etc. will not be restored.[/yellow]"
                    )
                else:
                    print(
                        f"\n  [yellow]Warning: Misc checkpoint not found at {misc_path}. Optimizer state etc. will not be restored.[/yellow]"
                    )
            if not nothing_trainable_loaded:
                self.step = int(step)
                if self.skip_load_misc:
                    self.step = 0

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(" Done.")

        if self.world_size > 1:
            self.check_ddp()

    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, "save() should be called only by the rank 0 process."
        print(f"\nSaving checkpoint at step {self.step}...", end="")

        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            # Move to CPU and clone to trim shared storage
            model_ckpt = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in model_ckpt.items()
            }
            torch.save(
                model_ckpt,
                os.path.join(
                    self.output_dir, "ckpts", f"{name}_step{self.step:07d}.pt"
                ),
            )

        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                # Move to CPU and clone to trim shared storage
                ema_ckpt = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in ema_ckpt.items()
                }
                torch.save(
                    ema_ckpt,
                    os.path.join(
                        self.output_dir,
                        "ckpts",
                        f"{name}_ema{ema_rate}_step{self.step:07d}.pt",
                    ),
                )

        misc_ckpt = {
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "data_sampler": self.data_sampler.state_dict(),
        }
        if self.fp16_mode == "amp":
            misc_ckpt["scaler"] = self.scaler.state_dict()
        elif self.fp16_mode == "inflat_all":
            misc_ckpt["log_scale"] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt["elastic_controller"] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt["grad_clip"] = self.grad_clip.state_dict()
        torch.save(
            misc_ckpt,
            os.path.join(self.output_dir, "ckpts", f"misc_step{self.step:07d}.pt"),
        )

        if self.save_latest_only:
            ckpt_dir = os.path.join(self.output_dir, "ckpts")
            for f in os.listdir(ckpt_dir):
                match = re.match(r".*_step(\d+)\.pt$", f)
                if match:
                    step = int(match.group(1))
                    if step != self.step:
                        os.remove(os.path.join(ckpt_dir, f))
        print(" Done.")

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print("\nFinetuning from:")
            for name, path in finetune_ckpt.items():
                print(f"  - {name}: {path}")

        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                print(f"[red]Loading {name} from {finetune_ckpt[name]}[/red]")
                model_ckpt = torch.load(
                    read_file_dist(finetune_ckpt[name]),
                    map_location=self.device,
                    weights_only=True,
                )
                for k, v in model_ckpt.items():
                    if model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(
                                f"Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped."
                            )
                        model_ckpt[k] = model_state_dict[k]
                model.load_state_dict(model_ckpt)
                if self.fp16_mode == "inflat_all":
                    model.convert_to_fp16()
                # Only collect for master params if not frozen
                if name not in self.freeze_models:
                    model_ckpts[name] = model_ckpt
            else:
                if self.is_master:
                    print(f"Warning: {name} not found in finetune_ckpt, skipped.")
                # Only collect state dict for non-frozen models
                if name not in self.freeze_models:
                    model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print("Done.")

        if self.world_size > 1:
            self.check_ddp()

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, (
            "update_ema() should be called only by the rank 0 process."
        )
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(
                    master_param, alpha=1.0 - ema_rate
                )

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print("\nPerforming DDP check...")

        if self.is_master:
            print("Checking if parameters are consistent across processes...")
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i : i + sub_size]
                    # gather from all processes
                    sub_p_gather = [
                        torch.empty_like(sub_p) for _ in range(self.world_size)
                    ]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all(
                        [
                            torch.equal(sub_p, sub_p_gather[i])
                            for i in range(self.world_size)
                        ]
                    ), "parameters are not consistent across processes"
        except AssertionError as e:
            if self.is_master:
                print(f"\n\033[91mError: {e}\033[0m")
                print("DDP check failed.")
            raise e

        dist.barrier()
        if self.is_master:
            print("Done.")

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {"loss": {}, "status": {}}
        amp_context = (
            partial(torch.autocast, device_type="cuda")
            if self.fp16_mode == "amp"
            else nullcontext
        )
        elastic_controller_context = (
            self.elastic_controller.record
            if self.elastic_controller_config is not None
            else nullcontext
        )

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = (
                [
                    self.training_models[name].no_sync
                    for name in self.training_models
                    if name not in self.freeze_models
                    and hasattr(self.training_models[name], "no_sync")
                ]
                if i != len(data_list) - 1 and self.world_size > 1
                else [nullcontext]
            )
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    if self.profile and self.step > 0:
                        from scalene.scalene_profiler import enable_profiling

                        with enable_profiling():
                            loss, status = self.training_losses(**mb_data)
                        if self.debug:
                            print("[green]Exit on debug mode[/green]")
                            sys.exit(0)
                    else:
                        loss, status = self.training_losses(**mb_data)
                    # if self.is_master and self.step % 10 == 0 and i == 0:
                    #     print(f"Step {self.step}: loss={loss['loss'].item():.4f}")
                    l = loss["loss"] / len(data_list)
                ## backward
                if self.fp16_mode == "amp":
                    self.scaler.scale(l).backward()
                elif self.fp16_mode == "inflat_all":
                    scaled_l = l * (2**self.log_scale)
                    scaled_l.backward()
                    # Check for NaN gradients
                    for name, model in self.training_models.items():
                        for pname, param in model.named_parameters():
                            if (
                                param.grad is not None
                                and not param.grad.isfinite().all()
                            ):
                                print(
                                    f"[Step {self.step} RANK {self.rank}] NaN grad in {name}.{pname}"
                                )
                                # attach_debugger(f"NaN grad in {name}.{pname}")
                else:
                    l.backward()
            ## log
            losses.append(
                dict_foreach(
                    loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x
                )
            )
            statuses.append(
                dict_foreach(
                    status, lambda x: x.item() if isinstance(x, torch.Tensor) else x
                )
            )
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.fp16_mode == "amp":
                self.scaler.unscale_(self.optimizer)
            elif self.fp16_mode == "inflat_all":
                model_grads_to_master_grads(
                    self.model_params, self.master_params, self.model_param_names
                )
                self.master_params[0].grad.mul_(1.0 / (2**self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.master_params, self.grad_clip
                )
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]["grad_norm"] = grad_norm.item()
        ## step
        # with master_first():
        #     print("Master first")

        if self.fp16_mode == "amp":
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.fp16_mode == "inflat_all":
            prev_scale = 2**self.log_scale
            for (
                pname,
                p,
            ) in zip(self.model_param_names, self.model_params):
                if p.grad is None:
                    print(f"[red]Missing grad for {pname}[/red]")
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                    self.master_params[0].grad.mul_(1.0 / (2**self.log_scale))
                self.optimizer.step()
                master_params_to_model_params(self.model_params, self.master_params)
                self.log_scale += self.fp16_scale_growth
            else:
                self.log_scale -= 1
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print(
                    "\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m"
                )
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]["lr"] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log["loss"] = dict_reduce(losses, lambda x: np.mean(x))
        step_log["status"] = dict_reduce(
            statuses,
            lambda x: np.mean(x),
            special_func={"min": lambda x: np.min(x), "max": lambda x: np.max(x)},
        )
        if self.elastic_controller_config is not None:
            step_log["elastic"] = dict_reduce(
                elastic_controller_logs, lambda x: np.mean(x)
            )
        if self.grad_clip is not None:
            step_log["grad_clip"] = (
                self.grad_clip
                if isinstance(self.grad_clip, float)
                else self.grad_clip.log()
            )

        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param_norms[name] = param.norm().item()
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        param_grads[name] = param.grad.norm().item() / prev_scale
            step_log["param_norms"] = param_norms
            step_log["param_grads"] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log
