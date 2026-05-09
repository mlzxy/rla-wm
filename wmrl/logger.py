"""Lightweight logger with print / TensorBoard / W&B backends."""

from __future__ import annotations

import time
from rich import print
from typing import Dict, Optional


class Logger:
    """Unified logger that always prints and optionally writes to
    TensorBoard and/or Weights & Biases.

    Usage::

        logger = Logger(log_dir="runs/my_run", use_tb=True, use_wandb=False)
        logger.scalar("losses/pg", 0.12, step=1000)
        logger.log("Starting eval ...")
        logger.scalars({"eval/return": 3.2, "eval/success": 0.8}, step=1000)
        logger.close()
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_tb: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "wmrl",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        self._tb = None
        self._use_wandb = use_wandb

        if use_tb and log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir)

        if use_wandb:
            import wandb
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_config.get("name") if wandb_config is not None else None,
                config=wandb_config or {},
                dir=log_dir,
            )

    # -- public API --

    def log(self, msg: str, console=print) -> None:
        """Print a timestamped message."""
        ts = time.strftime("[%H:%M:%S]")
        if console is not None:
            console(f"{ts} {msg}")

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar."""
        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)
        if self._use_wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def scalars(self, kv: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once."""
        for tag, value in kv.items():
            self.scalar(tag, value, step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._use_wandb:
            import wandb
            wandb.finish()
