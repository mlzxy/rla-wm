import os
import socket
import random
from rich import print
import sys
import functools
import inspect
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import torch
import numpy as np
import debugpy
import jstyleson
import torch.distributed as dist
from omegaconf import OmegaConf
from rich.console import Console
from easydict import EasyDict as edict
from rich.syntax import Syntax


def attach_debugger(message="", target_rank=-1, port=None, tui=False):
    """
    Pauses execution and attaches a debugger (debugpy or pudb).
    Can be used as a direct call for immediate attachment OR as a decorator to catch exceptions.

    Args:
        message (str or function): Message to display or function to decorate.
        target_rank (int): Rank to attach to. -1 means ALL ranks will attach.
        port (int): Base port. If None, random ports are chosen.
        tui (bool): If True, use PUDB remote TUI. Otherwise use debugpy.
    """

    def _do_attach(msg=None):
        # 1. Get current rank safely
        try:
            if dist.is_initialized():
                my_rank = dist.get_rank()
            else:
                my_rank = int(os.environ.get("RANK", "0"))
        except (ValueError, KeyError):
            my_rank = 0

        # 2. Determine if this rank should activate the debugger
        should_attach = (target_rank == -1) or (my_rank == target_rank)

        if should_attach:
            # 3. Port Allocation Logic
            if port is None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    assigned_port = s.getsockname()[1]
            else:
                assigned_port = port + my_rank

            disp_msg = msg if msg else (message if isinstance(message, str) else "")
            print(f"\n{'=' * 60}")
            print(f"🐛 RANK {my_rank} WAITING FOR DEBUGGER: {disp_msg}")
            print(f"    Host: {socket.gethostname()}")
            print(f"    Port: {assigned_port}")
            print(f"{'=' * 60}\n")
            sys.stdout.flush()

            # 4. Start Listening (Blocks until you attach)
            try:
                if tui:
                    # Use PUDB for TUI debugging
                    import pudb.remote

                    pudb.remote.set_trace(
                        term_size=(160, 40), host="0.0.0.0", port=assigned_port
                    )
                else:
                    # Use debugpy for VSCode/PyCharm
                    import debugpy

                    debugpy.listen(("0.0.0.0", assigned_port))
                    debugpy.wait_for_client()

                print(f"✅ Rank {my_rank} Attached!")
                return
            except Exception as e:
                print(f"❌ Rank {my_rank} failed to attach debugger: {e}")

    # Decorator logic
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import traceback

                error_msg = (
                    f"Exception in {func.__name__}: {e}\n{traceback.format_exc()}"
                )
                _do_attach(error_msg)
                raise e

        return wrapper

    # Case 1: Used as @attach_debugger (no parens)
    if callable(message):
        return decorator(message)

    # Case 2: Distinguish @attach_debugger(...) factory from attach_debugger(...) direct call
    # We check the stack to see if the caller line starts with '@' (decorator factory)
    frame = inspect.stack()[1]
    context = frame.code_context
    is_decorator = context and context[0].strip().startswith("@")

    if is_decorator:
        # Used as @attach_debugger(...) -> return the decorator
        return decorator
    else:
        # Used as attach_debugger(...) -> attach immediately
        _do_attach()
        return None


def load_config(config_path, debug=None) -> dict:
    def _resolve_path(path, *, _root_):
        missing = object()
        value = OmegaConf.select(_root_, path, default=missing)
        if value is missing:
            raise KeyError(f"Cannot resolve path '{path}' from config root.")
        return value

    def _apply_expand(value):
        if isinstance(value, dict):
            expanded = {}
            expand_items = value.get("__expand__")
            if expand_items is not None:
                if not isinstance(expand_items, list):
                    expand_items = [expand_items]
                for item in expand_items:
                    item = _apply_expand(item)
                    if not isinstance(item, dict):
                        raise TypeError(
                            "__expand__ items must resolve to dict values, "
                            f"got {type(item).__name__}."
                        )
                    expanded.update(deepcopy(item))

            for key, item in value.items():
                if key == "__expand__":
                    continue
                expanded[key] = _apply_expand(item)
            return expanded

        if isinstance(value, list):
            return [_apply_expand(item) for item in value]

        return value

    config_path = Path(config_path)
    if config_path.suffix.lower() in [".yaml", ".yml"]:
        config = OmegaConf.load(config_path)
        # Register resolvers for expression eval, basename, and root path lookup.
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver("at", _resolve_path, replace=True)
        OmegaConf.register_new_resolver(
            "basename", lambda: config_path.stem, replace=True
        )
        if debug is not None:
            config.vars.debug = debug
        config = OmegaConf.to_container(config, resolve=True)
        config = _apply_expand(config)
    elif config_path.suffix.lower() == ".json":
        with open(config_path, "r") as f:
            config = jstyleson.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    return config


class TimerContext:
    """Context manager for timing execution with CUDA event support."""

    def __init__(self, title: str = "", enable: bool = True):
        self.title = title
        self.enable = enable
        self.elapsed_time = 0.0

    def __enter__(self):
        import time

        if self.enable:
            if torch.cuda.is_available():
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
            else:
                self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import time

        if self.enable:
            if torch.cuda.is_available():
                self.end_event.record()
                self.end_event.synchronize()
                self.elapsed_time = (
                    self.start_event.elapsed_time(self.end_event) / 1000.0
                )
            else:
                self.elapsed_time = time.time() - self.start_time

            prefix = f"[{self.title}] " if self.title else ""
            print(f"{prefix}Elapsed time: {self.elapsed_time:.4f}s")


def load_model_from_dir(
    model_class,
    model_dir: str,
    device: str = "cpu",
    model_name: str = "decoder",
    ckpt_prefix: str = "decoder_step",
):
    import os
    import os.path as osp
    import re

    import torch
    from easydict import EasyDict as edict

    # Load config
    config_path = osp.join(model_dir, "config.yaml")
    if not osp.exists(config_path):
        # Try finding any yaml in the dir
        yamls = [f for f in os.listdir(model_dir) if f.endswith(".yaml")]
        if yamls:
            config_path = osp.join(model_dir, yamls[0])
        else:
            raise FileNotFoundError(f"No config.yaml found in {model_dir}")

    cfg = edict(load_config(config_path))

    if model_name in cfg.models:
        model_args = cfg.models[model_name].args
    else:
        raise ValueError(
            f"Could not find '{model_name}' in models config from {config_path}"
        )

    print(f"Loading {model_class.__name__} from {model_dir}...")
    model_instance = model_class(**model_args).to(device)

    # Find latest checkpoint
    ckpt_dir = osp.join(model_dir, "ckpts")
    step = None
    if osp.isdir(ckpt_dir):
        files = os.listdir(ckpt_dir)
        pattern = re.compile(rf"{ckpt_prefix}(\d+)\.pt$")
        steps = []
        for f in files:
            match = pattern.match(f)
            if match:
                steps.append(int(match.group(1)))
        if steps:
            step = max(steps)

    if step is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt_path = osp.join(ckpt_dir, f"{ckpt_prefix}{step:07d}.pt")
    print(f"Loading {model_name} checkpoint from step {step}: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model_instance.load_state_dict(state_dict)
    model_instance.eval()
    model_instance.requires_grad_(False)
    return model_instance


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).to(device)
    elif hasattr(batch, "to"):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def pretty_print_log(log_show, step, prefix="loss/"):
    import numpy as np
    from rich import print as rprint

    # Pretty print in one line
    items = [f"[bold yellow]Step {step:07d}[/bold yellow]"]
    for key in sorted(log_show.keys()):
        if prefix and not key.startswith(prefix):
            continue

        display_key = key[len(prefix) :] if prefix else key
        value = log_show[key]
        if isinstance(value, (float, np.floating)):
            items.append(f"[cyan]{display_key}[/cyan]: [magenta]{value:.6f}[/magenta]")
        else:
            items.append(f"[cyan]{display_key}[/cyan]: [magenta]{value}[/magenta]")

    if len(items) > 1:
        rprint(" | ".join(items))


def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)


def split_batch(data, batch_split, keys_as_list=None):
    if keys_as_list is None:
        keys_as_list = []

    if batch_split == 1:
        return [data]

    batch_size = len([v for v in data.values() if isinstance(v, torch.Tensor)][0])

    data_list = []
    for i in range(batch_split):
        start_idx = i * batch_size // batch_split
        end_idx = (i + 1) * batch_size // batch_split

        split_dict = {}
        for k, v in data.items():
            if k in keys_as_list and isinstance(v, list):
                split_dict[k] = [item[start_idx:end_idx] for item in v]
            else:
                split_dict[k] = v[start_idx:end_idx]

        data_list.append(split_dict)

    return data_list


def dataclasses_to_dict(dataset_cfg) -> dict:
    """Convert DatasetConfig dataclass to kwargs dict for TrajectoryDataset."""
    if isinstance(dataset_cfg, dict):
        return dataset_cfg
    import dataclasses

    d = dataclasses.asdict(dataset_cfg)
    # Remove None values so TrajectoryDataset uses its own defaults
    return {k: v for k, v in d.items() if v is not None}


def fetch_state_dict(model_name, work_dir, device, step=None):
    ckpt_dir = os.path.join(work_dir, "ckpts")
    ckpt_files = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith(f"{model_name}_step") and f.endswith(".pt")
    ]

    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoints found for model '{model_name}' in {ckpt_dir}"
        )

    if step is None:
        ckpt_files = sorted(ckpt_files)
        ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
    else:
        ckpt_name = f"{model_name}_step{int(step):07d}.pt"
        ckpt = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint for model '{model_name}' at step {step} not found: {ckpt}"
            )

    print(f"[yellow]Loading {model_name} from {ckpt}[/yellow]")
    ckpt = torch.load(ckpt, map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    return state_dict


def apply_cli_overrides(cfg: dict, overrides: list):
    """
    Apply CLI overrides to a raw config dict.

    Overrides are parsed from unknown args as --key value pairs.
    Dotted keys (e.g. --policy.checkpoint) set nested values.
    Values are auto-cast to int/float/bool/null where possible.

    Example:
        args, unknown = parser.parse_known_args()
        raw_cfg = load_config(args.config)
        apply_cli_overrides(raw_cfg, unknown)
    """
    import json
    from rich import print

    def _cast(v: str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        if v.lower() in ("null", "none"):
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        try:
            parsed = json.loads(v)
            if isinstance(parsed, (list, dict)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return v

    i = 0
    while i < len(overrides):
        key = overrides[i]
        if not key.startswith("--"):
            i += 1
            continue
        key = key[2:]
        if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
            value = _cast(overrides[i + 1])
            i += 2
        else:
            value = True
            i += 1

        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
        print(f"  [dim]Override: {key} = {value!r}[/dim]")


def dict_to_dataclass(cls, d: dict):
    """Recursively convert a dict to a dataclass, ignoring extra keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return d
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in d:
            val = d[f.name]
            ftype = f.type
            if isinstance(ftype, str):
                ftype = eval(ftype)
            if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
                kwargs[f.name] = dict_to_dataclass(ftype, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def pretty_print_config(config):
    yaml_cfg = OmegaConf.to_yaml(OmegaConf.create(config))
    console = Console()
    in_colab = "google.colab" in sys.modules
    theme = "github" if in_colab else "monokai"
    syntax = Syntax(yaml_cfg, "yaml", theme=theme, line_numbers=True)
    console.print(syntax)


def print_slurm_environment_summary(prefix: str = "") -> None:
    env = os.environ
    if not any(env.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_PROCID")):
        return

    summary_keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_STEP_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_JOB_NODELIST",
        "SLURM_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
    ]
    summary_parts = []
    for key in summary_keys:
        value = env.get(key)
        if value:
            summary_parts.append(f"{key}={value}")

    launcher = "srun" if env.get("SLURM_STEP_ID") else "sbatch"
    prefix_text = f"{prefix} " if prefix else ""
    print(f"{prefix_text}Detected SLURM {launcher} environment: {' '.join(summary_parts)}")


@contextmanager
def local_seed_scope(seed: int):
    """Temporarily set numpy/python/torch RNG seeds and restore previous states on exit."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    bounded_seed = int(seed)
    bounded_seed_32 = bounded_seed % (2**32)

    np.random.seed(bounded_seed_32)
    random.seed(bounded_seed_32)
    torch.manual_seed(bounded_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bounded_seed)

    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def make_worker_seed_init_fn(base_seed: int):
    """Create a DataLoader worker_init_fn that seeds numpy/python/torch per worker."""

    def _seed_worker(worker_id: int):
        worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def edict_to_dict(obj):
    # Check if the object is an EasyDict instance
    if isinstance(obj, edict):
        # Convert it to a regular dict and recursively process its values
        return {k: edict_to_dict(v) for k, v in obj.items()}
    # Check if the object is a standard dict (useful if the input can also be a mix)
    elif isinstance(obj, dict):
        return {k: edict_to_dict(v) for k, v in obj.items()}
    # Check if the object is a list and process its elements
    elif isinstance(obj, list):
        return [edict_to_dict(elem) for elem in obj]
    # Otherwise, return the object as is (base case)
    else:
        return obj


def unwrap(module):
    return module.module if hasattr(module, "module") else module
