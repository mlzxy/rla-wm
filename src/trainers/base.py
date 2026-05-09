from abc import abstractmethod
import sys
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from rich import print

try:
    import wandb
except ImportError:
    wandb = None
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from utils.misc import pretty_print_log, split_batch


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Trainer:
    """
    Base class for training.

    Args:
        models: Models to train.
        dataset: Dataset for training.
        output_dir: Output directory for checkpoints and logs.
        load_dir: Directory to load checkpoints from.
        step: Step to resume from.
        max_steps: Maximum number of training steps.
        batch_size: Batch size (ignored if batch_size_per_gpu is specified).
        batch_size_per_gpu: Batch size per GPU.
        batch_split: Split batch with gradient accumulation.
        optimizer: Optimizer configuration.
        lr_scheduler: Learning rate scheduler configuration.
        elastic: Elastic memory management configuration.
        grad_clip: Gradient clipping configuration.
        ema_rate: Exponential moving average rate(s).
        fp16_mode: FP16 mode ('inflat_all', 'amp', or None).
        fp16_scale_growth: Scale growth for FP16 gradient backpropagation.
        finetune_ckpt: Finetune checkpoint configuration.
        log_param_stats: Whether to log parameter statistics.
        prefetch_data: Whether to prefetch data.
        i_print: Print interval.
        i_log: Log interval.
        i_sample: Sample interval.
        i_save: Save checkpoint interval.
        i_ddpcheck: DDP check interval.
        skip_load_misc: Whether to skip restoring optimizer/sampler/scaler state from misc checkpoint.
        wandb_run: wandb.Run object for logging. If None, wandb logging is disabled.
            To enable wandb logging, add a 'wandb' section to your config file:

            ```json
            {
              "wandb": {
                "enabled": true,
                "project": "your-project-name",
                "entity": "your-entity",
                "name": "experiment-name",
                "tags": ["tag1", "tag2"],
                "notes": "experiment notes"
              }
            }
            ```

            The wandb run is automatically initialized in train.py on rank 0 when enabled.
            All training metrics (losses, status, time, scale, etc.) are logged to wandb
            at the same interval as tensorboard logging (i_log).
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode="inflat_all",
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        freeze_models=[],
        log_param_stats=False,
        prefetch_data=True,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        save_latest_only=True,
        i_ddpcheck=10000,
        wandb_run=None,
        num_samples=32,
        log_scale=20.0,
        load_strict=True,
        profile=None,
        debug: bool = False,
        val_dataset=None,
        log_everystep: bool = False,
        inference_only: bool = False,
        force_load_suffix: str = "",
        skip_load_misc: bool = False,
        **kwargs,
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, (
            "Either batch_size or batch_size_per_gpu must be specified."
        )
        self.log_everystep = log_everystep
        self.log_scale = log_scale
        self.load_strict = load_strict
        self.profile = profile
        self.debug = debug
        self.skip_initial_snapshot = kwargs.get("skip_initial_snapshot", False)
        self.save_latest_only = save_latest_only
        self.skip_load_misc = skip_load_misc

        self.models = models
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.freeze_models = freeze_models
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        
        self.inference_only = inference_only
        if self.inference_only:
            self.freeze_models = list(self.models.keys())

        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.num_samples = num_samples

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = (
            batch_size
            if batch_size_per_gpu is None
            else batch_size_per_gpu * self.world_size
        )
        self.batch_size_per_gpu = (
            batch_size_per_gpu
            if batch_size_per_gpu is not None
            else batch_size // self.world_size
        )
        assert self.batch_size % self.world_size == 0, (
            "Batch size must be divisible by the number of GPUs."
        )
        assert self.batch_size_per_gpu % self.batch_split == 0, (
            "Batch size per GPU must be divisible by batch split."
        )

        # Initialize wandb_run for all processes (will be None for non-master processes)
        self.wandb_run = wandb_run if self.is_master else None

        self.init_models_and_more(**kwargs)
        if self.dataset is not None:
            self.prepare_dataloader(**kwargs)
        else:
            print(f"[yellow]{self.__class__.__name__}: No dataset provided, skipping dataloader preparation.[/yellow]")

        # Load checkpoint
        self.step = 0
        if (
            load_dir is not None
        ):  # NOTE: provide load_dir to finetune from a checkpoint!
            self.load(load_dir, step if not force_load_suffix else force_load_suffix)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)

        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "ckpts"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, "tb_logs"))

        if self.world_size > 1:
            self.check_ddp()

        if self.is_master:
            print("\n\nTrainer initialized.")
            print(self)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, "device"):
                return model.device
        return next(list(self.models.values())[0].parameters()).device

    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        num_workers = kwargs.get("num_workers", 0)
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,  # int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=self.dataset.collate_fn
            if hasattr(self.dataset, "collate_fn")
            else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass

    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn
            if hasattr(self.dataset, "collate_fn")
            else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f"dataset_{k}", v) for k, v in vis.items()]
        else:
            save_cfg = [("dataset", vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=-1, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if num_samples == -1:
            num_samples = self.num_samples
        if self.is_master:
            print(f"\nSampling {num_samples} images...", end="")

        if suffix is None:
            suffix = f"step{self.step:07d}"

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(
            num_samples_per_process, batch_size=batch_size, verbose=verbose
        )

        # log non-images
        scalars = {}
        for key in list(samples.keys()):
            if samples[key]["type"] == "scalar":
                scalars[key] = samples[key]["value"]
                del samples[key]

        # Gather scalars
        if self.world_size > 1:
            for key in list(scalars.keys()):
                val = scalars[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                # We just average scalars across processes for simplicity, or just take master's
                # For accuracy, averaging is roughly correct if batch sizes are equal
                val_tensor = torch.tensor(val, device=self.device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                scalars[key] = val_tensor.item() / self.world_size

        if self.is_master:
            # Log scalars to wandb/tb
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"val/{k}": v for k, v in scalars.items()}, step=self.step
                )
            for k, v in scalars.items():
                self.writer.add_scalar(f"val/{k}", v, self.step)
                if verbose:
                    print(f"{k}: {v:.4f}")

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]["type"] == "sample":
                vis = self.visualize_sample(samples[key]["value"])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f"{key}_{k}"] = {"value": v, "type": "image"}
                    del samples[key]
                else:
                    samples[key] = {"value": vis, "type": "image"}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                samples[key]["value"] = samples[key]["value"].contiguous()
                if self.is_master:
                    all_images = [
                        torch.empty_like(samples[key]["value"])
                        for _ in range(self.world_size)
                    ]
                else:
                    all_images = []
                dist.gather(samples[key]["value"], all_images, dst=0)
                if self.is_master:
                    samples[key]["value"] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "samples", suffix), exist_ok=True)
            wandb_images = {}
            for key in samples.keys():
                if samples[key]["type"] == "image":
                    # Save to disk
                    utils.save_image(
                        samples[key]["value"],
                        os.path.join(
                            self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"
                        ),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                    # Log to wandb if available
                    if self.wandb_run is not None and wandb is not None:
                        # Create grid for wandb
                        grid = utils.make_grid(
                            samples[key]["value"],
                            nrow=int(np.sqrt(num_samples)),
                            normalize=True,
                            value_range=self.dataset.value_range,
                        )
                        # Convert to numpy and transpose for wandb (C, H, W) -> (H, W, C)
                        grid_np = grid.permute(1, 2, 0).cpu().numpy()
                        # Clip to [0, 1] and convert to uint8
                        grid_np = np.clip(grid_np, 0, 1)
                        grid_np = (grid_np * 255).astype(np.uint8)
                        wandb_images[f"samples/{key}"] = wandb.Image(grid_np)
                elif samples[key]["type"] == "number":
                    min = samples[key]["value"].min()
                    max = samples[key]["value"].max()
                    images = (samples[key]["value"] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    # Save to disk
                    save_image_with_notes(
                        images,
                        os.path.join(
                            self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"
                        ),
                        notes=f"{key} min: {min}, max: {max}",
                    )
                    # Log to wandb if available
                    if self.wandb_run is not None and wandb is not None:
                        # Convert to numpy and transpose for wandb
                        images_np = images.permute(1, 2, 0).cpu().numpy()
                        images_np = np.clip(images_np, 0, 1)
                        images_np = (images_np * 255).astype(np.uint8)
                        wandb_images[f"samples/{key}"] = wandb.Image(
                            images_np, caption=f"{key} min: {min:.4f}, max: {max:.4f}"
                        )
                elif samples[key]["type"] == "video":
                    videos = samples[key]["value"]
                    if not isinstance(videos, torch.Tensor):
                        raise ValueError(
                            f"Video samples for key '{key}' must be a torch.Tensor, got {type(videos)}"
                        )
                    if videos.ndim != 5:
                        raise ValueError(
                            f"Video samples for key '{key}' must have shape (N, T, C, H, W), got {tuple(videos.shape)}"
                        )

                    if imageio is None:
                        print(
                            "[yellow]imageio is not available; skipping local/wandb video logging.[/yellow]"
                        )
                        continue

                    nrow = int(np.sqrt(num_samples))
                    fps = 8
                    video_path = os.path.join(
                        self.output_dir, "samples", suffix, f"{key}_{suffix}.mp4"
                    )
                    grids = []
                    for t in range(videos.shape[1]):
                        grid = utils.make_grid(
                            videos[:, t],
                            nrow=nrow,
                            normalize=True,
                            value_range=self.dataset.value_range,
                        )
                        grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
                        grid_np = np.clip(grid_np, 0, 1)
                        grids.append((grid_np * 255).astype(np.uint8))
                    imageio.mimwrite(video_path, grids, fps=fps)

                    if self.wandb_run is not None and wandb is not None:
                        wandb_images[f"samples/{key}"] = wandb.Video(
                            video_path, fps=fps, format="mp4"
                        )

            # Log all images to wandb at once
            if self.wandb_run is not None and wandb is not None and wandb_images:
                self.wandb_run.log(wandb_images, step=self.step)

        if self.is_master:
            print(" Done.")

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass

    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(
                    next(self.data_iterator), self.device, non_blocking=True
                )
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(
                next(self.data_iterator), self.device, non_blocking=True
            )
        else:
            data = recursive_to_device(
                next(self.data_iterator), self.device, non_blocking=True
            )

        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            data_list = split_batch(
                data, 
                self.batch_split, 
                keys_as_list=['structure', 'unstructure']
            )
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError("Data must be a dict or a list of dicts.")

        return data_list
    
    @property
    def snapshot_dataset(self):
        return self.dataset if self.val_dataset is None else self.val_dataset

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        # if self.is_master:
        #     print("\nStarting training...")
        #     self.snapshot_dataset()
        if self.step == 0:
            if not self.skip_initial_snapshot:
                self.snapshot(suffix="init")
        else:  # resume
            self.snapshot(suffix=f"resume_step{self.step:07d}")

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f"Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)",
                    f"Elapsed: {time_elapsed / 3600:.2f} h",
                    f"Speed: {speed:.2f} steps/h",
                    f"ETA: {(self.max_steps - self.step) / speed:.2f} h",
                ]
                print(" | ".join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if (
                self.world_size > 1
                and self.i_ddpcheck is not None
                and self.step % self.i_ddpcheck == 0
            ):
                # if self.is_master:
                #     print(
                #         f"\n[{self.step:07d}] Performing DDP check...  ",
                #         flush=True,
                #         end="",
                #     )
                self.check_ddp()
                # if self.is_master:
                #     print("DDP check done.", flush=True)

            if self.is_master:
                log.append((self.step, {}))

                # Log time
                log[-1][1]["time"] = {
                    "step": time_end - time_start,
                    "elapsed": time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if self.fp16_mode == "amp":
                    log[-1][1]["scale"] = self.scaler.get_scale()
                elif self.fp16_mode == "inflat_all":
                    log[-1][1]["log_scale"] = self.log_scale
                
                if self.log_everystep or self.step == 1:
                    log_show = dict_flatten(log[-1][1], sep="/")
                    pretty_print_log(log_show, self.step)

                # Save log
                if self.step % self.i_log == 0 or self.step == 0:
                    ## save to log file
                    log_str = "\n".join(
                        [
                            f"{step}: {json.dumps(log, cls=NpEncoder)}"
                            for step, log in log
                        ]
                    )
                    with open(
                        os.path.join(self.output_dir, "log.jsonl"), "a"
                    ) as log_file:
                        log_file.write(log_str + "\n")

                    # show with tensorboard
                    log_show = [
                        l for _, l in log if not dict_any(l, lambda x: np.isnan(x))
                    ]
                    if len(log_show) == 0:
                        print("No valid log to show, all loss are NAN!")
                        sys.exit(1)
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep="/")
                    
                    # Pretty print in one line
                    if self.log_everystep:
                        print('============================================================')
                    pretty_print_log(log_show, self.step)
                    for key, value in log_show.items():
                        self.writer.add_scalar(key, value, self.step)
                    if self.wandb_run is not None:
                        self.wandb_run.log(log_show, step=self.step)
                    log = []

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()

            # Sample images
            if self.step % self.i_sample == 0:
                if self.step != 0:
                    self.snapshot()

        if self.is_master:
            self.snapshot(suffix="final")
            self.writer.close()
            if self.wandb_run is not None:
                self.wandb_run.finish()
            print("Training finished.")

    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        if self.is_master:
            self.snapshot(suffix="profile_init")
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(self.output_dir, "profile")
            ),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                data_list = self.load_data()
                self.run_step(data_list)
                prof.step()
