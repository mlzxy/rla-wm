"""Shared dataclass interfaces for wmrl PPO training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from rich import print
import torch
from PIL import Image, ImageDraw, ImageFont


def _load_pil_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size,
        )
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            return ImageFont.load_default()


def _obs_tensor_to_rgb(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 6:
        if obs.shape[2] != 1 or obs.shape[3] != 3:
            raise ValueError(
                "Expected obs with shape (T, N, 1, 3, H, W) when obs.ndim == 6, "
                f"got {tuple(obs.shape)}"
            )
        obs = obs[:, :, 0]
    elif obs.ndim != 5:
        raise ValueError(
            "Expected obs with shape (T, N, 3, H, W) or (T, N, 1, 3, H, W), "
            f"got {tuple(obs.shape)}"
        )

    if obs.shape[2] != 3:
        raise ValueError(f"Expected RGB observations, got {tuple(obs.shape)}")

    obs = obs.detach().cpu().float()
    if torch.isfinite(obs).any():
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        obs = torch.zeros_like(obs)
    if obs.max().item() > 1.0:
        obs = obs / 255.0
    return obs.clamp(0.0, 1.0)


def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
    rgb = (frame.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return Image.fromarray(rgb.permute(1, 2, 0).contiguous().numpy(), mode="RGB")


@dataclass
class StepResult:
    """Returned by VecEnv.step(). All tensors live on the VecEnv's device.
    """

    obs: torch.Tensor  # (num_envs, *obs_shape)
    reward: torch.Tensor  # (num_envs,)
    done: torch.Tensor  # (num_envs,) bool — terminated | truncated
    truncated: torch.Tensor  # (num_envs,) bool
    success: torch.Tensor  # (num_envs,) bool — per-env success flag
    info: Dict[str, Any] = field(default_factory=dict)  # raw env info dict


@dataclass
class RolloutBatch:
    """Collected rollout data fed to PolicyAgent.update().

    All tensors are on the agent's device, N is num of envs
    """

    obs: torch.Tensor  # (T, N, *obs_shape)
    actions: torch.Tensor  # (T, N, chunk_size, action_dim)
    logprobs: torch.Tensor  # (T, N)
    rewards: torch.Tensor  # (T, N)
    dones: torch.Tensor  # (T, N)
    next_obs: torch.Tensor  # (N, *obs_shape)  — obs after last step
    next_done: torch.Tensor  # (N,)
    # Optional side-channel obs (e.g. per-env state history for policies that
    # consume more than a single image tensor). Agents that don't need it
    # can leave these as None.
    state_obs: Optional[torch.Tensor] = None  # (T, N, *state_obs_shape)
    next_state_obs: Optional[torch.Tensor] = None  # (N, *state_obs_shape)
    # Pre-reset (terminal) obs for visualization.  Same shape as obs but only
    # filled for steps where a done occurred; other entries are zero/unused.
    terminal_obs: Optional[torch.Tensor] = None  # (T, N, *obs_shape)
    # Critic baseline: predicted values and bootstrap value.
    values: Optional[torch.Tensor] = None  # (T, N)
    next_value: Optional[torch.Tensor] = None  # (N,)

    def render(
        self,
        output_path: str,
        *,
        img_size: Optional[int] = None,
        max_envs: Optional[int] = None,
    ) -> str:
        """Render rollout as ``s[t] --r[t]--> s[t+1]`` transition grid.

        Renders an HTML file with full-resolution images encoded as base64
        data URIs.  Layout per row (one env):

          [s0] --r0--> [s1] --r1--> ... --r(T-1)--> [s_next]

        When ``terminal_obs`` is available and a reset occurs at step t,
        a full-size terminal frame (orange border) is shown between the
        arrow source and destination.

        ``next_obs`` is appended so there are T+1 images and T reward arrows.

        Returns the *output_path* that was written.
        """
        import base64
        from io import BytesIO

        def _pil_to_data_uri(img: Image.Image) -> str:
            buf = BytesIO()
            img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        # ------------------------------------------------------------------
        # Prepare tensors
        # ------------------------------------------------------------------
        obs_rgb = _obs_tensor_to_rgb(self.obs)
        next_obs_rgb = _obs_tensor_to_rgb(self.next_obs.unsqueeze(0))
        all_frames = torch.cat([obs_rgb, next_obs_rgb], dim=0)  # (T+1, N, 3, H, W)

        has_terminal = self.terminal_obs is not None
        if has_terminal:
            assert self.terminal_obs is not None
            term_rgb = _obs_tensor_to_rgb(self.terminal_obs)

        num_steps, num_envs = self.rewards.shape
        num_frames = num_steps + 1

        render_envs = min(num_envs, max_envs) if max_envs is not None else num_envs
        if render_envs <= 0:
            raise ValueError("render_envs must be > 0")
        rewards = self.rewards.detach().cpu().float()[:, :render_envs]
        dones = self.dones.detach().cpu().bool()[:, :render_envs]
        next_done = self.next_done.detach().cpu().bool()[:render_envs]
        all_frames = all_frames[:, :render_envs]
        if has_terminal:
            term_rgb = term_rgb[:, :render_envs]

        # Arrow t crosses reset if the *destination* frame (t+1) is post-reset.
        arrow_resets = torch.zeros(num_steps, render_envs, dtype=torch.bool)
        if num_steps > 1:
            arrow_resets[:num_steps - 1] = dones[1:]
        arrow_resets[num_steps - 1] = next_done

        # Optional CSS display size.
        css_size = f"width:{img_size}px;height:{img_size}px;" if img_size else ""

        # ------------------------------------------------------------------
        # Build HTML
        # ------------------------------------------------------------------
        parts: list[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        parts.append("<style>")
        parts.append("""
body { background:#121212; color:#ddd; font-family:monospace; margin:8px; }
.env-row { display:flex; align-items:center; margin-bottom:12px; gap:0; }
.env-label { writing-mode:horizontal-tb; min-width:60px; font-size:14px;
             color:#dcdcdc; text-align:right; padding-right:8px; flex-shrink:0; }
.frame-cell { display:flex; flex-direction:column; align-items:center; }
.frame-cell img { display:block; IMGSIZE border-radius:2px; }
.frame-label { font-size:11px; margin-bottom:2px; }
.arrow-cell { display:flex; flex-direction:column; align-items:center;
              justify-content:center; min-width:54px; padding:0 2px; }
.arrow-sym { font-size:22px; line-height:1; }
.reward-text { font-size:11px; white-space:nowrap; }
.border-normal { border:3px solid #5a5a5a; }
.border-reset  { border:3px solid #30d66e; }
.border-terminal { border:3px solid #ffa500; }
.border-next   { border:3px solid #64a0ff; }
.color-normal  { color:#b4b4b4; }
.color-reset   { color:#dc5050; }
.color-reward  { color:#ffe678; }
.color-step    { color:#8c8c8c; }
.color-terminal { color:#ffa500; }
""".replace("IMGSIZE", css_size))
        parts.append("</style></head><body>")

        for env_idx in range(render_envs):
            parts.append(f'<div class="env-row"><div class="env-label">env {env_idx}</div>')

            for frame_idx in range(num_frames):
                is_post_reset = frame_idx < num_steps and bool(dones[frame_idx, env_idx].item())
                is_next = frame_idx == num_steps

                if is_next:
                    label = "s_next"
                    bclass = "border-next"
                elif is_post_reset:
                    label = f"s{frame_idx}*"
                    bclass = "border-reset"
                else:
                    label = f"s{frame_idx}"
                    bclass = "border-normal"

                uri = _pil_to_data_uri(_frame_to_pil(all_frames[frame_idx, env_idx]))
                parts.append(
                    f'<div class="frame-cell">'
                    f'<span class="frame-label color-step">{label}</span>'
                    f'<img class="{bclass}" src="{uri}"></div>'
                )

                # Arrow + reward between frames.
                if frame_idx < num_steps:
                    step_idx = frame_idx
                    is_reset_arrow = bool(arrow_resets[step_idx, env_idx].item())
                    r_val = rewards[step_idx, env_idx].item()

                    show_terminal = (
                        is_reset_arrow
                        and has_terminal
                        and term_rgb[step_idx, env_idx].abs().sum().item() > 0.0
                    )

                    acls = "color-reset" if is_reset_arrow else "color-normal"
                    arrow_char = "&#x21E2;" if is_reset_arrow else "&#x2192;"

                    if show_terminal:
                        t_uri = _pil_to_data_uri(_frame_to_pil(term_rgb[step_idx, env_idx]))
                        parts.append(
                            f'<div class="arrow-cell">'
                            f'<span class="reward-text {acls}">r={r_val:+.3f}</span>'
                            f'<span class="arrow-sym {acls}">{arrow_char}</span>'
                            f'</div>'
                            f'<div class="frame-cell">'
                            f'<span class="frame-label color-terminal">s{step_idx}\'</span>'
                            f'<img class="border-terminal" src="{t_uri}"></div>'
                            f'<div class="arrow-cell">'
                            f'<span class="arrow-sym {acls}">{arrow_char}</span>'
                            f'</div>'
                        )
                    else:
                        r_str = f"r={r_val:+.3f}"
                        if is_reset_arrow:
                            r_str = f"RESET {r_str}"
                        rcls = acls if is_reset_arrow else "color-reward"
                        parts.append(
                            f'<div class="arrow-cell">'
                            f'<span class="reward-text {rcls}">{r_str}</span>'
                            f'<span class="arrow-sym {acls}">{arrow_char}</span>'
                            f'</div>'
                        )

            parts.append("</div>")  # close env-row

        parts.append("</body></html>")

        html = "\n".join(parts)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        print(f"Saved rollout visualization to {output_path}")
        return output_path


@dataclass
class UpdateMetrics:
    """Returned by PolicyAgent.update()."""

    policy_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    bc_loss: float = 0.0
    value_loss: float = 0.0
