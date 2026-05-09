"""
Robust checkpoint save / resume / prune utilities for v4world policy training.
Follows the same conventions as atomic_policy's checkpoint_util.
"""

import copy
import pathlib
import re
import threading

import dill
import torch


def copy_to_cpu(x):
    """Recursively copy tensors to CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        return {k: copy_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


def prune_checkpoints(checkpoint_dir, keep=3):
    """Keep only the last *keep* epoch checkpoints in the folder."""
    if keep is None:
        return
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    epoch_files = []
    for file in checkpoint_dir.glob("*.ckpt"):
        match = re.search(r"_epoch(\d+)\.ckpt$", file.name)
        if match:
            epoch_files.append((int(match.group(1)), file))
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    if len(epoch_files) > keep:
        for _, file in epoch_files[keep:]:
            try:
                file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {e}")


def save_checkpoint_with_epoch(workspace, path=None, tag="latest", epoch=0, use_thread=True, keep_last=None):
    if path is None:
        path = pathlib.Path(workspace.output_dir).joinpath("checkpoints", f"{tag}_epoch{epoch}.ckpt")
    else:
        path = pathlib.Path(path)
        path = path.with_name(f"{path.stem}_epoch{epoch}.ckpt")

    exclude_keys = tuple(workspace.exclude_keys)
    include_keys = tuple(workspace.include_keys) + ("_output_dir",)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"cfg": workspace.cfg, "state_dicts": dict(), "pickles": dict()}
    for key, value in workspace.__dict__.items():
        if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
            if key not in exclude_keys:
                payload["state_dicts"][key] = copy_to_cpu(value.state_dict()) if use_thread else value.state_dict()
        elif key in include_keys:
            payload["pickles"][key] = dill.dumps(value)

    if use_thread:
        workspace._saving_thread = threading.Thread(
            target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
        )
        workspace._saving_thread.start()
    else:
        torch.save(payload, path.open("wb"), pickle_module=dill)

    if keep_last is None:
        checkpoint_cfg = getattr(getattr(workspace, "cfg", None), "checkpoint", None)
        if checkpoint_cfg is not None:
            keep_last = checkpoint_cfg.get("keep_last", None)

    prune_checkpoints(path.parent, keep=keep_last)
    return str(path.absolute())


def get_latest_checkpoint_path(output_dir):
    checkpoint_dir = pathlib.Path(output_dir).joinpath("checkpoints")
    if not checkpoint_dir.exists():
        return None
    epoch_files = []
    for file in checkpoint_dir.glob("*.ckpt"):
        match = re.search(r"_epoch(\d+)\.ckpt$", file.name)
        if match:
            epoch_files.append((int(match.group(1)), file))
    if epoch_files:
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        return epoch_files[0][1]
    return None


def get_previous_checkpoint_path(output_dir, current_path):
    checkpoint_dir = pathlib.Path(output_dir).joinpath("checkpoints")
    epoch_files = []
    for file in checkpoint_dir.glob("*.ckpt"):
        match = re.search(r"_epoch(\d+)\.ckpt$", file.name)
        if match:
            epoch_files.append((int(match.group(1)), file))
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    current_match = re.search(r"_epoch(\d+)\.ckpt$", pathlib.Path(current_path).name)
    if current_match:
        current_epoch = int(current_match.group(1))
        for epoch, file in epoch_files:
            if epoch < current_epoch:
                return file
    return None


def resume_training(workspace, cfg):
    if not cfg.training.resume:
        return
    print("Resuming training...")
    latest_path = get_latest_checkpoint_path(workspace.output_dir)
    if latest_path is None:
        print("No checkpoints found. Starting from scratch.")
        return
    path = latest_path
    while path:
        print(f"Attempting to resume from checkpoint {path}")
        try:
            workspace.load_checkpoint(path=path)
            print(f"Successfully resumed from checkpoint {path}")
            return
        except Exception as e:
            print(f"Failed to load checkpoint {path}: {e}")
            path = get_previous_checkpoint_path(workspace.output_dir, path)
    print("No valid checkpoints found. Starting from scratch.")
