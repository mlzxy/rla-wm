"""Multi-process multi-GPU world-model VecEnv.

Spawns one worker process per GPU, each hosting a
:class:`~wmrl.world_model_env.FlowWorldModelVecEnv` with a subset of the
total envs.  Exposes the same public interface so ``train.py`` can swap
``env_cls`` with zero code changes.

All tensors cross process boundaries as CPU tensors.  The proxy moves
results to ``policy_device`` before returning them to the caller.

Example usage::

    env = MultiProcessWorldModelVecEnv(
        num_envs=32,
        device="cuda:0",
        world_model_gpu_ids=[0, 1, 2, 3],
        **env_kwargs,
    )
    obs = env.reset()              # (32, *obs_shape) on cuda:0
    sr  = env.step_chunked(acts)   # StepResult on cuda:0
    env.close()
"""

from __future__ import annotations

import atexit
import math
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from wmrl.rl_types import StepResult
from wmrl.world_model_env_worker import worker_fn as _flow_worker_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_env_splits(num_envs: int, num_workers: int) -> List[int]:
    """Divide *num_envs* as evenly as possible across *num_workers*.

    Returns a list of per-worker env counts summing to *num_envs*.
    """
    base = num_envs // num_workers
    remainder = num_envs % num_workers
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


def _parse_gpu_ids(
    gpu_ids: Optional[Sequence[int] | str],
) -> List[int]:
    """Normalise the user-provided ``world_model_gpu_ids`` value."""
    if gpu_ids is None:
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("No CUDA devices available")
        return list(range(n))
    if isinstance(gpu_ids, str):
        gpu_ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
    ids = [int(g) for g in gpu_ids]
    n = torch.cuda.device_count()
    for g in ids:
        if g < 0 or g >= n:
            raise ValueError(f"Invalid GPU id {g}; available: 0..{n - 1}")
    return ids


# ---------------------------------------------------------------------------
# Multi-process proxy
# ---------------------------------------------------------------------------


class MultiProcessWorldModelVecEnv:
    """Multi-GPU world-model VecEnv backed by per-GPU worker processes.

    Public interface matches :class:`FlowWorldModelVecEnv` so it is a
    transparent drop-in replacement.

    Subclasses can override :attr:`_worker_fn` to use a different worker
    entry point (e.g. for DINO world-model env).

    Args:
        num_envs: Total number of parallel environments.
        device: Policy device — results are moved here before returning.
        world_model_gpu_ids: CUDA device indices to distribute work across.
            ``None`` uses all available GPUs.
        **env_kwargs: Forwarded to each worker's env constructor.
            Each worker receives a copy with ``num_envs`` adjusted to its
            share and ``device`` set to its GPU.
    """

    _worker_fn = staticmethod(_flow_worker_fn)

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cuda:0",
        world_model_gpu_ids: Optional[Sequence[int] | str] = None,
        **env_kwargs: Any,
    ) -> None:
        self.policy_device = torch.device(device)
        if self.policy_device.type == "cuda" and self.policy_device.index is None:
            self.policy_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
            )

        self._gpu_ids = _parse_gpu_ids(world_model_gpu_ids)
        self._num_workers = len(self._gpu_ids)
        self.num_envs = int(num_envs)
        self._env_splits = _compute_env_splits(self.num_envs, self._num_workers)

        # Keep a copy of env_kwargs (minus things we override per-worker).
        self._env_kwargs = dict(env_kwargs)
        # Remove keys that the proxy manages.
        self._env_kwargs.pop("device", None)
        self._env_kwargs.pop("num_envs", None)
        self._env_kwargs.pop("world_model_gpu_ids", None)

        # Spawn workers.
        ctx = mp.get_context("spawn")
        self._pipes: List[Connection] = []
        self._procs: List[BaseProcess] = []

        base_seed = int(self._env_kwargs.get("seed", 0))
        base_flow_seed = int(self._env_kwargs.get("flow_seed", base_seed))
        env_offset = 0

        for worker_idx in range(self._num_workers):
            parent_conn, child_conn = ctx.Pipe()
            wk_kwargs = dict(self._env_kwargs)
            wk_kwargs["num_envs"] = self._env_splits[worker_idx]
            wk_kwargs["device"] = f"cuda:{self._gpu_ids[worker_idx]}"
            # Keep dataset selection and reset streams anchored to the same
            # base seed; individual env reset streams are derived from global
            # env ids inside the worker env.
            wk_kwargs["seed"] = base_seed
            wk_kwargs["flow_seed"] = base_flow_seed
            wk_kwargs["global_env_offset"] = env_offset
            wk_kwargs["worker_id"] = worker_idx
            env_offset += self._env_splits[worker_idx]

            p = ctx.Process(
                target=self._worker_fn,
                args=(child_conn, self._gpu_ids[worker_idx], wk_kwargs),
                daemon=True,
                name=f"wm-worker-gpu{self._gpu_ids[worker_idx]}",
            )
            p.start()
            child_conn.close()  # parent doesn't need the child end
            self._pipes.append(parent_conn)
            self._procs.append(p)

        # Wait for all workers to initialise & collect metadata.
        self._worker_metadata: List[Dict[str, Any]] = []
        for i, pipe in enumerate(self._pipes):
            meta = pipe.recv()
            if meta.get("status") == "error":
                raise RuntimeError(
                    f"Worker {i} (GPU {self._gpu_ids[i]}) failed during init:\n"
                    f"{meta.get('traceback', 'unknown error')}"
                )
            if meta.get("status") != "ready":
                raise RuntimeError(
                    f"Worker {i} sent unexpected init response: {meta}"
                )
            self._worker_metadata.append(meta)

        # Copy VecEnv protocol attributes from the first worker.
        m0 = self._worker_metadata[0]
        self.obs_shape: Tuple[int, ...] = tuple(m0["obs_shape"])
        self.action_dim: int = int(m0["action_dim"])
        self.state_dim: int = int(m0["state_dim"])
        self.chunk_size: int = int(m0["chunk_size"])
        self.device = self.policy_device 

        # Pre-compute cumulative offsets for splitting / gathering.
        self._offsets: List[int] = []
        offset = 0
        for n in self._env_splits:
            self._offsets.append(offset)
            offset += n

        self._closed = False
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Internal IPC helpers
    # ------------------------------------------------------------------

    def _send_all(self, cmd: str, payloads: List[Dict[str, Any]]) -> None:
        """Send a command to every worker."""
        for pipe, payload in zip(self._pipes, payloads):
            pipe.send((cmd, payload))

    def _recv_all(self) -> List[Dict[str, Any]]:
        """Receive one response from every worker."""
        results = []
        for i, pipe in enumerate(self._pipes):
            resp = pipe.recv()
            if "error" in resp:
                raise RuntimeError(
                    f"Worker {i} (GPU {self._gpu_ids[i]}) error: {resp['error']}"
                )
            results.append(resp)
        return results

    def _broadcast(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Send the same command+payload to all workers and collect replies."""
        payloads = [payload or {} for _ in range(self._num_workers)]
        self._send_all(cmd, payloads)
        return self._recv_all()

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor to ``policy_device``."""
        return t.to(self.policy_device, non_blocking=True)

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self, sync_all_envs: bool = False) -> torch.Tensor:
        """Reset all envs and return initial observations.

        Returns:
            ``(num_envs, *obs_shape)`` tensor on ``policy_device``.
        """
        responses = self._broadcast("reset", {"sync_all_envs": sync_all_envs})
        obs_parts = [resp["obs"] for resp in responses]
        return self._to_device(torch.cat(obs_parts, dim=0))

    @torch.no_grad()
    def get_state_history(self) -> torch.Tensor:
        """Return state history for all envs.

        Returns:
            ``(num_envs, n_obs_steps, state_dim)`` on ``policy_device``.
        """
        responses = self._broadcast("state")
        parts = [resp["state"] for resp in responses]
        return self._to_device(torch.cat(parts, dim=0))

    def step(self, actions: torch.Tensor) -> StepResult:
        raise NotImplementedError("Use step_chunked()")

    def step_fast(self, actions: torch.Tensor) -> StepResult:
        raise NotImplementedError("Use step_chunked()")

    @torch.no_grad()
    def step_chunked(self, action_chunks: torch.Tensor) -> StepResult:
        """Run one chunked decision step distributed across GPU workers.

        Args:
            action_chunks: ``(num_envs, chunk_size, action_dim)`` on any device.

        Returns:
            :class:`StepResult` with all tensors on ``policy_device``.
        """
        action_chunks = action_chunks.detach().cpu()
        payloads: List[Dict[str, Any]] = []
        for w in range(self._num_workers):
            start = self._offsets[w]
            end = start + self._env_splits[w]
            payloads.append({"action_chunks": action_chunks[start:end]})
        self._send_all("step", payloads)
        responses = self._recv_all()
        return self._merge_step_results(responses)

    def _merge_step_results(self, responses: List[Dict[str, Any]]) -> StepResult:
        """Concatenate per-worker StepResult dicts into a single StepResult."""
        dev = self.policy_device

        obs = torch.cat([r["obs"] for r in responses], dim=0).to(dev, non_blocking=True)
        reward = torch.cat([r["reward"] for r in responses], dim=0).to(dev, non_blocking=True)
        done = torch.cat([r["done"] for r in responses], dim=0).to(dev, non_blocking=True)
        truncated = torch.cat([r["truncated"] for r in responses], dim=0).to(dev, non_blocking=True)
        success = torch.cat([r["success"] for r in responses], dim=0).to(dev, non_blocking=True)

        # Merge info dicts — concat tensor values, handle bootstrap fields.
        merged_info = self._merge_info_dicts(
            [r["info"] for r in responses],
        )

        return StepResult(
            obs=obs, reward=reward, done=done,
            truncated=truncated, success=success, info=merged_info,
        )

    def _merge_info_dicts(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge per-worker info dicts into a single dict.

        Simple tensor fields (state_history, token_err, etc.) are concatenated
        along dim 0.  The bootstrap fields require careful offset-aware merging.
        """
        dev = self.policy_device
        merged: Dict[str, Any] = {}

        # --- Simple tensor concat fields ---
        for key in ("state_history", "chunk_return_sum", "token_err", "goal_err"):
            parts = [info[key] for info in infos if key in info]
            if parts:
                merged[key] = torch.cat(parts, dim=0).to(dev, non_blocking=True)

        # --- Bootstrap fields (only present when some envs are truncated) ---
        # chunk_bootstrap_mask: (N,) bool per worker → concat to (total_N,)
        masks = []
        for info in infos:
            if "chunk_bootstrap_mask" in info:
                masks.append(info["chunk_bootstrap_mask"])
            else:
                # Worker had no truncated envs — fill with False.
                n = info["state_history"].shape[0] if "state_history" in info else 0
                if n > 0:
                    masks.append(torch.zeros(n, dtype=torch.bool))

        if masks:
            full_mask = torch.cat(masks, dim=0).to(dev, non_blocking=True)
            merged["chunk_bootstrap_mask"] = full_mask

            # Gather the actual bootstrap obs/state from workers that had them.
            obs_parts = []
            state_parts = []
            for info in infos:
                if "chunk_final_obs_tensor" in info and info["chunk_final_obs_tensor"] is not None:
                    obs_parts.append(info["chunk_final_obs_tensor"])
                if "chunk_final_state_obs" in info and info["chunk_final_state_obs"] is not None:
                    state_parts.append(info["chunk_final_state_obs"])

            if obs_parts:
                merged["chunk_final_obs_tensor"] = (
                    torch.cat(obs_parts, dim=0).to(dev, non_blocking=True)
                )
            if state_parts:
                merged["chunk_final_state_obs"] = (
                    torch.cat(state_parts, dim=0).to(dev, non_blocking=True)
                )

        # --- Pre-reset obs for visualization ---
        pre_reset_masks = []
        pre_reset_obs_parts = []
        for info in infos:
            if "pre_reset_done_mask" in info:
                pre_reset_masks.append(info["pre_reset_done_mask"])
                if "pre_reset_obs" in info and info["pre_reset_obs"] is not None:
                    pre_reset_obs_parts.append(info["pre_reset_obs"])
            else:
                n = info["state_history"].shape[0] if "state_history" in info else 0
                if n > 0:
                    pre_reset_masks.append(torch.zeros(n, dtype=torch.bool))

        if pre_reset_masks:
            full_pre_reset_mask = torch.cat(pre_reset_masks, dim=0).to(dev, non_blocking=True)
            if full_pre_reset_mask.any():
                merged["pre_reset_done_mask"] = full_pre_reset_mask
                if pre_reset_obs_parts:
                    merged["pre_reset_obs"] = (
                        torch.cat(pre_reset_obs_parts, dim=0).to(dev, non_blocking=True)
                    )

        return merged

    @torch.no_grad()
    def sample_actions_from_dataset(self) -> torch.Tensor:
        """Return ground-truth action chunks from dataset for all envs.

        Returns:
            ``(num_envs, chunk_size, action_dim)`` on ``policy_device``.
        """
        responses = self._broadcast("sample")
        parts = [resp["actions"] for resp in responses]
        return self._to_device(torch.cat(parts, dim=0))

    def render(
        self,
        env_index: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Render observations as a uint8 HWC numpy array.

        When ``env_index`` is ``None``, gathers per-env frames from every
        worker and stitches them into a single grid — matching the
        single-GPU ``FlowWorldModelVecEnv.render()`` behaviour.

        When ``env_index`` is given, renders only that env from its worker.
        """
        if env_index is not None:
            worker_idx, local_idx = self._global_to_local(env_index)
            payload = dict(kwargs, env_index=local_idx)
            self._pipes[worker_idx].send(("render", payload))
            resp = self._pipes[worker_idx].recv()
            return resp["frame"]

        # Render all envs: ask each worker to render each of its envs
        # individually, then stitch into a unified grid.
        mode = kwargs.get("mode", "side_by_side")
        cells: List[np.ndarray] = []
        for w in range(self._num_workers):
            for local_i in range(self._env_splits[w]):
                payload = dict(kwargs, env_index=local_i)
                self._pipes[w].send(("render", payload))
                resp = self._pipes[w].recv()
                cells.append(resp["frame"])

        if not cells:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        # Arrange into a grid.
        import math as _math
        cell_h, cell_w = cells[0].shape[:2]
        cols = int(_math.ceil(_math.sqrt(len(cells))))
        rows = int(_math.ceil(len(cells) / cols))
        pad = 2
        canvas_h = rows * cell_h + (rows - 1) * pad
        canvas_w = cols * cell_w + (cols - 1) * pad
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for idx, cell in enumerate(cells):
            r, c = divmod(idx, cols)
            y0 = r * (cell_h + pad)
            x0 = c * (cell_w + pad)
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell
        return canvas

    def _global_to_local(self, env_index: int) -> Tuple[int, int]:
        """Map a global env index to ``(worker_idx, local_env_idx)``."""
        for w in range(self._num_workers):
            if env_index < self._offsets[w] + self._env_splits[w]:
                return w, env_index - self._offsets[w]
        raise IndexError(f"env_index {env_index} out of range [0, {self.num_envs})")

    def close(self) -> None:
        """Shut down all workers and join processes."""
        if self._closed:
            return
        self._closed = True
        for pipe in self._pipes:
            try:
                pipe.send(("close", {}))
            except Exception:
                pass
        for pipe in self._pipes:
            try:
                pipe.recv()
            except Exception:
                pass
        for pipe in self._pipes:
            try:
                pipe.close()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

