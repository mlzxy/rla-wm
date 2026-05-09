import os
import re
import os.path as osp
import sys
from datetime import datetime
import glob
import argparse
from pathlib import Path
from easydict import EasyDict as edict
import jstyleson
from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp
import numpy as np
import random
import wandb
from rich.console import Console
from rich import print
from rich.syntax import Syntax
from rich.table import Table
from rich.box import ROUNDED

from src import models, datasets, trainers
from src.utils.dist_utils import setup_dist, master_first, find_free_port
from utils.misc import load_config


def find_ckpt(cfg):
    # Load checkpoint
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            ckpt_dir = os.path.join(cfg.load_dir, "ckpts")
            if os.path.isdir(ckpt_dir):
                files = os.listdir(ckpt_dir)
                pattern = re.compile(r".*_step(\d+)\.pt$")
                steps = []
                for f in files:
                    match = pattern.match(f)
                    if match:
                        steps.append(int(match.group(1)))
                if steps:
                    cfg.load_ckpt = max(steps)
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def ensure_compiled():
    from gsplat.cuda import _backend  # ensure gsplat is compiled
    # from geomloss import SamplesLoss  # ensure geomloss is compiled


def get_model_summary(model):
    model_summary = "Parameters:\n"
    model_summary += "=" * 128 + "\n"
    model_summary += f"{'Name':<{72}}{'Shape':<{32}}{'Type':<{16}}{'Grad'}\n"
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f"{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n"
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += "\n"
    model_summary += f"Number of parameters: {num_params}\n"
    model_summary += f"Number of trainable parameters: {num_trainable_params}\n"
    return model_summary


def init_wandb(rank, cfg):
    """
    Initialize wandb (only on rank 0).

    To enable wandb logging, add a 'wandb' section to your config file:
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
    The full training config will be logged to wandb automatically.

    Args:
        rank: Process rank (only rank 0 will initialize wandb)
        cfg: Configuration object

    Returns:
        wandb.Run object if initialized, None otherwise
    """
    wandb_run = None
    if rank == 0:
        wandb_config = getattr(cfg, "wandb", {})
        if isinstance(wandb_config, dict) and wandb_config.get("enabled", False):
            wandb_config = wandb_config.copy()
            enabled = wandb_config.pop("enabled", True)
            if enabled:
                # Convert config to dict for wandb
                config_dict = dict(cfg)
                # Remove wandb config from main config to avoid recursion
                if "wandb" in config_dict:
                    del config_dict["wandb"]
                wandb_run = wandb.init(
                    project=wandb_config.get("project", "contrastive-world"),
                    name=wandb_config.get("name", None),
                    tags=wandb_config.get("tags", None),
                    notes=wandb_config.get("notes", None),
                    config=config_dict,
                    dir=cfg.output_dir,
                    **{
                        k: v
                        for k, v in wandb_config.items()
                        if k
                        not in ["project", "entity", "name", "tags", "notes", "enabled"]
                    },
                )
    return wandb_run


def summarize_nccl(console):
    """Summarize NCCL setup."""
    nccl_vars = [
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_TC",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_CROSS_NIC",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_IFNAME",
        "NCCL_PROTO",
        "NCCL_ALGO",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NRINGS",
        "NCCL_TREE_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_LL128_THRESHOLD",
        "NCCL_BUFFSIZE",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_CUMEM_DISABLE",
        "NCCL_GRAPH_DISABLE",
        "NCCL_SET_STACK_SIZE",
    ]

    table = Table(
        title="NCCL Configuration",
        box=ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
        show_header=True,
    )
    table.add_column("Variable", style="dim")
    table.add_column("Value", style="green")

    # Add NCCL version
    try:
        nccl_version = torch.cuda.nccl.version()
        table.add_row("NCCL Version", str(nccl_version))
    except Exception:
        table.add_row("NCCL Version", "Unknown")

    # Add relevant env vars
    any_set = False
    for var in sorted(nccl_vars):
        if var in os.environ:
            table.add_row(var, os.environ[var])
            any_set = True

    if not any_set:
        table.add_row("Env Variables", "No specific NCCL env vars detected")

    console.print(table)


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Initialize wandb (only on rank 0)
    wandb_run = init_wandb(rank, cfg)
    print("OUTPUT_DIR: ", cfg.output_dir)

    # Seed rngs
    setup_rng(rank)
    dataset_name, dataset_args = cfg.dataset.name, cfg.dataset.args
    # if cfg.dataset.name == "SimpleTransitionDataset":
    # if cfg.vars.debug or getattr(cfg.trainer.args, "process_data_locally", False):
    # dataset_name = cfg.dataset.full_args.name
    # dataset_args = cfg.dataset.full_args.args
    # print("[yellow]DEBUG MODE: Using TrajectoryDataset for training[/yellow]")

    # Load data
    with master_first():
        ensure_compiled()
        dataset = getattr(datasets, dataset_name)(cfg.data_dir, **dataset_args)
        if hasattr(cfg, "val_dataset") and cfg.val_dataset:
            print("Loading validation dataset... ", end="")
            val_dataset = getattr(datasets, cfg.val_dataset.name)(
                cfg.data_dir, **cfg.val_dataset.args
            )
            print("Done")
        else:
            val_dataset = None

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f"\n\nBackbone: {name}\n" + model_summary)
            with open(
                os.path.join(cfg.output_dir, f"{name}_model_summary.txt"), "w"
            ) as fp:
                print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        # load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
        wandb_run=wandb_run,
        val_dataset=val_dataset,
        cfg=cfg,
    )

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == "__main__":
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument(
        "--config", type=str, required=True, help="Experiment config file"
    )
    ## io and resume
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--load_dir", type=str, default="", help="Load directory, default to output_dir"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="latest",
        help="Checkpoint step to resume training, default to latest",
    )
    parser.add_argument(
        "--data_dir", type=str, default="to be set (optional)", help="Data directory"
    )
    parser.add_argument(
        "--auto_retry", type=int, default=0, help="Number of retries on error"
    )
    parser.add_argument(
        "--batch_size",
        default=None,
    )
    ## dubug
    parser.add_argument(
        "--tryrun", action="store_true", help="Try run without training"
    )
    parser.add_argument("--profile", action="store_true", help="Profile training")
    ## multi-node and multi-gpu
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs per node, default to all",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master address for distributed training",
    )
    parser.add_argument(
        "--master_port", type=str, default="12345", help="Port for distributed training"
    )
    opt = parser.parse_args()


    opt.output_dir = os.path.join(
        opt.output_dir,
        osp.basename(opt.config).split(".")[0],
        datetime.now().strftime("%Y%m%d_%H-%M-%S"),
    )
    opt.load_dir = opt.load_dir if opt.load_dir != "" else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    # Support both JSON (with comments via jstyleson) and YAML (via OmegaConf)
    config = load_config(opt.config, debug=opt.debug)

    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        cfg.vars.debug = True
        cfg.trainer.args.debug = True
    if cfg.vars.debug:
        cfg.wandb.enabled = False
        cfg.trainer.args.num_workers = 0
        cfg.trainer.args.i_ddpcheck = 5
        # cfg.trainer.args.i_sample = 5
        # cfg.trainer.args.i_save = 5
        cfg.trainer.args.num_samples = 2 # cfg.trainer.args.batch_size_per_gpu
    if opt.batch_size is not None:
        cfg.trainer.args.batch_size = int(opt.batch_size)

    cfg.wandb.name = (
        cfg.wandb.name if cfg.wandb.name else "/".join(opt.output_dir.split("/")[-2:])
    )

    print("\n\nConfig:")
    print("=" * 80)
    yaml_cfg = OmegaConf.to_yaml(OmegaConf.create(config))
    console = Console()
    syntax = Syntax(yaml_cfg, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)
    summarize_nccl(console)

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
            print(" ".join(["python"] + sys.argv), file=fp)
        config_save_path = os.path.join(cfg.output_dir, "config.yaml")
        OmegaConf.save(OmegaConf.create(config), config_save_path)

    # Resolve port before spawning to avoid races
    port = int(cfg.master_port)
    resolved_port = find_free_port(cfg.master_addr, port)
    if resolved_port != str(port):
        print(f"[yellow]Port {port} occupied, using {resolved_port} instead.[/yellow]")
    cfg.master_port = resolved_port

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({rty + 1}/{cfg.auto_retry})...")
