"""Worker process for multi-GPU world-model environment.

Each worker owns a :class:`FlowWorldModelVecEnv` on a single GPU and
processes commands from the parent process via a ``multiprocessing.Connection``
(pipe).  All tensors crossing the pipe are on CPU to avoid CUDA IPC issues.

Protocol
--------
Parent sends ``(command: str, payload: dict)`` tuples.
Worker sends back ``(result: dict)`` where tensor values are on CPU.

Commands:
    ``"init"``      — construct the env (payload = env_kwargs).
    ``"reset"``     — call ``env.reset()``.
    ``"step"``      — call ``env.step_chunked(action_chunks)``.
    ``"state"``     — call ``env.get_state_history()``.
    ``"sample"``    — call ``env.sample_actions_from_dataset()``.
    ``"render"``    — call ``env.render(**kwargs)``.
    ``"getattr"``   — return a scalar attribute (obs_shape, action_dim, …).
    ``"close"``     — shut down the env and exit the loop.
"""

from __future__ import annotations

import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers: move tensors to CPU recursively
# ---------------------------------------------------------------------------


def _to_cpu(obj: Any) -> Any:
    """Recursively move tensors to CPU.  Leaves non-tensors unchanged."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj  # already on CPU
    return obj


def _step_result_to_cpu(sr: Any) -> Dict[str, Any]:
    """Serialise a :class:`StepResult` into a plain dict of CPU tensors."""
    return {
        "obs": _to_cpu(sr.obs),
        "reward": _to_cpu(sr.reward),
        "done": _to_cpu(sr.done),
        "truncated": _to_cpu(sr.truncated),
        "success": _to_cpu(sr.success),
        "info": _to_cpu(sr.info),
    }


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def worker_fn(pipe: Connection, gpu_id: int, env_kwargs: Dict[str, Any]) -> None:
    """Entry point for a world-model worker process.

    Args:
        pipe: Parent-side ``Connection`` for bidirectional IPC.
        gpu_id: CUDA device index this worker should own.
        env_kwargs: Keyword arguments forwarded to
            :class:`FlowWorldModelVecEnv`.
    """
    env: Optional[Any] = None
    device = torch.device(f"cuda:{gpu_id}")

    try:
        # Set default CUDA device for this process.
        torch.cuda.set_device(device)

        # Build the env inside the worker so all CUDA state stays local.
        from wmrl.world_model_env import FlowWorldModelVecEnv

        env_kwargs = dict(env_kwargs)
        env_kwargs["device"] = device
        env = FlowWorldModelVecEnv(**env_kwargs)

        # Signal that init succeeded — send back metadata.
        pipe.send({
            "status": "ready",
            "obs_shape": env.obs_shape,
            "action_dim": env.action_dim,
            "state_dim": env.state_dim,
            "chunk_size": env.chunk_size,
            "num_envs": env.num_envs,
        })

        # Command loop.
        while True:
            try:
                cmd, payload = pipe.recv()
            except EOFError:
                break

            if cmd == "reset":
                obs = env.reset(**payload)
                pipe.send({"obs": _to_cpu(obs)})

            elif cmd == "step":
                action_chunks = payload["action_chunks"].to(device)
                sr = env.step_chunked(action_chunks)
                pipe.send(_step_result_to_cpu(sr))

            elif cmd == "state":
                state = env.get_state_history()
                pipe.send({"state": _to_cpu(state)})

            elif cmd == "sample":
                actions = env.sample_actions_from_dataset()
                pipe.send({"actions": _to_cpu(actions)})

            elif cmd == "render":
                frame = env.render(**payload)
                pipe.send({"frame": frame})  # numpy array, already on CPU

            elif cmd == "getattr":
                name = payload["name"]
                val = getattr(env, name)
                pipe.send({"value": val})

            elif cmd == "close":
                pipe.send({"status": "closed"})
                break

            else:
                pipe.send({"error": f"Unknown command: {cmd}"})

    except Exception:
        tb = traceback.format_exc()
        try:
            pipe.send({"status": "error", "traceback": tb})
        except Exception:
            pass
    finally:
        if env is not None:
            env.close()
        pipe.close()
