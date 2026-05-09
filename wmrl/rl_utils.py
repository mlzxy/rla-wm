"""Utility helpers for wmrl training scripts."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import fields
from typing import Any, Type, TypeVar

import numpy as np
import torch
import tyro
import yaml
from utils.misc import load_config

T = TypeVar("T")


def import_cls(dotted_path: str) -> type:
    """Import a class from a dotted path like ``wmrl.vec_env.SimVecEnv``."""
    module_path, _, cls_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path (no module): {dotted_path!r}")
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)



def parse_dataclass_with_optional_yaml(cls: Type[T]) -> T:
    """Parse CLI with optional ``--config-file`` YAML pre-loading.

    Strategy: if ``--config-file <path>`` appears in argv, remove it, build a
    dataclass instance from YAML values, then let tyro override with remaining
    CLI flags.
    """
    argv = list(sys.argv[1:])
    yaml_defaults: dict = {}
    yaml_path: str | None = None

    if "--config-file" in argv:
        idx = argv.index("--config-file")
        if idx + 1 >= len(argv):
            raise ValueError("--config-file requires a path argument")
        yaml_path = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2 :]
        yaml_defaults = load_config(yaml_path)

    field_names = {f.name for f in fields(cls)}
    base_kwargs = {k: v for k, v in yaml_defaults.items() if k in field_names}
    if yaml_path is not None and "config_file" in field_names:
        base_kwargs["config_file"] = yaml_path

    if base_kwargs:
        base = cls(**base_kwargs)
        return tyro.cli(cls, args=argv, default=base)

    return tyro.cli(cls, args=argv)


def obs_to_video_frame(obs: torch.Tensor, env_index: int = 0) -> np.ndarray:
    """Convert a batch observation tensor ``(N, H, W, C)`` to one uint8 frame."""
    frame = obs[env_index].detach().cpu().float().numpy()
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}")
    if frame.max() <= 1.5:
        frame = frame * 255.0
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def save_video(frames: list[np.ndarray], output_path: str, fps: int = 20) -> None:
    """Write ``frames`` to an mp4 file at ``output_path``."""
    if not frames:
        raise ValueError("Cannot save video: frames is empty")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(output_path, np.stack(frames, axis=0), fps=fps)
