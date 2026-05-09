"""
Time-warp resampling for speed-diverse trajectory execution.

Resamples a trajectory according to a smooth, randomly varying speed profile
along arc-length so that slow segments get more waypoints and fast segments
fewer. Control-frequency agnostic: produces a list of waypoints; the execution
loop feeds them to the controller at its native rate.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from .primitives import PrimitiveStep


# Internal defaults (not part of public API)
_NUM_SPEED_CONTROL_POINTS = 5
_STEP_DT = 0.05  # nominal path-time between output waypoints (seconds)
_V_MIN_CLIP = 1e-6
_POS_TOL = 1e-6
_QUAT_TOL = 1e-6


def _arc_lengths(positions: np.ndarray) -> np.ndarray:
    """Compute cumulative arc-length s[0..n] with s[0]=0."""
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(positions), dtype=np.float64)
    s[1:] = np.cumsum(segment_lengths)
    return s


def _pose_same(a: PrimitiveStep, b: PrimitiveStep) -> bool:
    """Return True if two steps have (nearly) identical pose."""
    pa = np.asarray(a.position, dtype=np.float64)
    pb = np.asarray(b.position, dtype=np.float64)
    if np.linalg.norm(pa - pb) > _POS_TOL:
        return False
    qa = np.asarray(a.quaternion, dtype=np.float64)
    qb = np.asarray(b.quaternion, dtype=np.float64)
    # quaternion sign ambiguity; use |dot|
    dot = float(np.abs(np.dot(qa, qb)))
    return (1.0 - dot) <= _QUAT_TOL


def _smooth_speed_profile(
    s: np.ndarray,
    speed_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> CubicSpline:
    """Build v(s) from piecewise-constant control points smoothed by cubic spline."""
    n = len(s)
    if n < 2:
        return None
    s_total = s[-1]
    # Control points at uniform arc-length (internal default count)
    num_ctrl = min(_NUM_SPEED_CONTROL_POINTS, max(2, n))
    s_ctrl = np.linspace(0, s_total, num_ctrl)
    v_ctrl = rng.uniform(speed_bounds[0], speed_bounds[1], size=num_ctrl)
    v_ctrl = np.maximum(v_ctrl, _V_MIN_CLIP)
    spline = CubicSpline(s_ctrl, v_ctrl)
    return spline


def _time_mapping(s: np.ndarray, v_spline: CubicSpline, n_fine: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute T(s) = integral_0^s (1/v(s')) ds' on a fine grid.
    Returns (s_fine, T_fine) with T_fine[0]=0.
    """
    s_fine = np.linspace(s[0], s[-1], n_fine)
    v_fine = np.maximum(v_spline(s_fine), _V_MIN_CLIP)
    # Integrate 1/v ds: approximate by (delta_s) / v at each segment
    inv_v = 1.0 / v_fine
    dT = np.diff(s_fine) * 0.5 * (inv_v[:-1] + inv_v[1:])
    T_fine = np.zeros_like(s_fine)
    T_fine[1:] = np.cumsum(dT)
    return s_fine, T_fine


def _interpolate_step_at_alpha(
    steps: List[PrimitiveStep],
    segment_i: int,
    alpha: float,
) -> PrimitiveStep:
    """Interpolate a single PrimitiveStep between steps[segment_i] and steps[segment_i+1] at alpha in [0,1]."""
    a = steps[segment_i]
    b = steps[segment_i + 1]
    pos = (1 - alpha) * np.asarray(a.position, dtype=np.float64) + alpha * np.asarray(b.position, dtype=np.float64)
    # Slerp for quaternion
    rots = R.from_quat([a.quaternion, b.quaternion])
    slerp = Slerp([0.0, 1.0], rots)
    quat = slerp(alpha).as_quat()
    gripper = (1 - alpha) * a.gripper + alpha * b.gripper
    phase = a.phase if alpha < 0.5 else b.phase
    is_interaction = a.is_interaction if alpha < 0.5 else b.is_interaction
    return PrimitiveStep(
        position=pos.astype(np.float32),
        quaternion=quat.astype(np.float32),
        gripper=float(gripper),
        phase=phase,
        is_interaction=is_interaction,
        joints=None,
        metadata=a.metadata if alpha < 0.5 else b.metadata,
    )


def _warp_motion_block(
    steps: List[PrimitiveStep],
    speed_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> List[PrimitiveStep]:
    """
    Time-warp resample a motion-only block (poses change between steps).
    Endpoints are preserved exactly.
    """
    if len(steps) < 2:
        return steps

    positions = np.array([s.position for s in steps], dtype=np.float64)
    s = _arc_lengths(positions)
    s_total = s[-1]
    if s_total <= 0:
        # No motion (numerically); keep as-is
        return steps

    v_spline = _smooth_speed_profile(s, speed_bounds, rng)
    s_fine, T_fine = _time_mapping(s, v_spline)
    T_total = float(T_fine[-1])
    if T_total <= 0:
        return steps

    # Input-density-aware step period: at neutral speed (v=1), N = len(steps)
    step_dt = s_total / len(steps)
    N = max(2, int(round(T_total / step_dt)))
    N_max = max(2, 10 * len(steps))
    N = min(N, N_max)
    T_k = np.linspace(0, T_total, N, endpoint=True)
    s_k = np.interp(T_k, T_fine, s_fine)

    out_steps: List[PrimitiveStep] = []
    n_seg = len(s) - 1
    for sk in s_k:
        sk = float(np.clip(sk, s[0], s[-1]))
        if sk >= s[-1]:
            seg_i = n_seg - 1
            alpha = 1.0
        else:
            seg_i = int(np.searchsorted(s, sk, side="right") - 1)
            seg_i = max(0, min(seg_i, n_seg - 1))
            ds = float(s[seg_i + 1] - s[seg_i])
            alpha = (sk - float(s[seg_i])) / ds if ds > 0 else 0.0
            alpha = float(np.clip(alpha, 0.0, 1.0))
        out_steps.append(_interpolate_step_at_alpha(steps, seg_i, alpha))

    # Hard guardrails: preserve endpoints exactly (gripper/phase/metadata too)
    out_steps[0] = steps[0]
    out_steps[-1] = steps[-1]
    return out_steps


def resample_trajectory_with_speed_profile(
    steps: List[PrimitiveStep],
    speed_bounds: Tuple[float, float] = (0.5, 2.0),
    random_state: Optional[int] = None,
) -> List[PrimitiveStep]:
    """
    Resample a trajectory with a smooth, randomly varying speed profile.

    Slow segments get more waypoints, fast segments fewer. Speed transitions
    are smooth (cubic spline over piecewise-constant control points). The
    resulting list is consumed by the execution loop one waypoint per env step
    at the controller's native frequency.

    Args:
        steps: Input trajectory waypoints (Cartesian; joints are ignored and set to None in output).
        speed_bounds: (v_min, v_max) speed factor range along the path (e.g. 0.5 = half speed, 2.0 = double).
        random_state: Optional seed for reproducible speed profiles.

    Returns:
        New list of PrimitiveStep waypoints with variable density. If speed_bounds is None or len(steps) < 2,
        returns the input list unchanged.
    """
    if speed_bounds is None or len(steps) < 2:
        return steps

    rng = np.random.default_rng(random_state)

    # Safeguard for grasp trajectories (and other dwell-heavy segments):
    # preserve in-place "dwell" runs (pose doesn't change) exactly, and only
    # time-warp resample true motion blocks (pose changes each step). This keeps
    # pregrasp/grasp holds and gripper transitions at the intended pose.
    out: List[PrimitiveStep] = []
    n = len(steps)
    if n == 0:
        return []

    # Classify each step (i>=1) by whether pose changed from previous step
    step_type = ["motion"] * n
    step_type[0] = "motion"
    for i in range(1, n):
        step_type[i] = "dwell" if _pose_same(steps[i - 1], steps[i]) else "motion"

    # Build blocks of consecutive step_type
    block_start = 0
    current = step_type[0]
    for i in range(1, n):
        if step_type[i] != current:
            block = steps[block_start:i]
            if current == "motion":
                out.extend(_warp_motion_block(block, speed_bounds, rng))
            else:
                out.extend(block)
            block_start = i
            current = step_type[i]

    # Final block
    block = steps[block_start:n]
    if current == "motion":
        out.extend(_warp_motion_block(block, speed_bounds, rng))
    else:
        out.extend(block)

    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random
    from pathlib import Path

    # Synthetic trajectory: 15 waypoints along a simple path
    n_way = 15
    # n_way = 40
    t_way = np.linspace(0, 1, n_way)
    positions = np.column_stack([
        0.3 * np.cos(2 * np.pi * t_way),
        0.2 * np.sin(2 * np.pi * t_way),
        0.1 + 0.05 * t_way,
    ])
    quat = np.array([0, 0, 0, 1.0], dtype=np.float64)
    quats = np.tile(quat, (n_way, 1))
    steps = [
        PrimitiveStep(position=positions[i], quaternion=quats[i], gripper=1.0, phase="move", is_interaction=False)
        for i in range(n_way)
    ]

    # Single call: only resample_trajectory_with_speed_profile produces the output
    speed_bounds_demo = (0.2, 3.0)  # wider range so non-uniform density is obvious
    # speed_bounds_demo = (0.99, 1.0)  # wider range so non-uniform density is obvious
    random_state_demo = random.randint(0, 1000000)
    out = resample_trajectory_with_speed_profile(
        steps, speed_bounds=speed_bounds_demo, random_state=random_state_demo
    )

    # Derive plot data from output and from same params (for reference curves only)
    pos_out = np.array([st.position for st in out], dtype=np.float64)
    s_out = _arc_lengths(pos_out)
    ds_out = np.diff(s_out)

    s = _arc_lengths(positions)
    rng = np.random.default_rng(random_state_demo)
    v_spline = _smooth_speed_profile(s, speed_bounds_demo, rng)
    s_fine = np.linspace(s[0], s[-1], 200)
    v_fine = np.maximum(v_spline(s_fine), _V_MIN_CLIP)
    s_fine_T, T_fine = _time_mapping(s, v_spline, n_fine=200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1) Speed v(s) vs arc-length s (same profile used for resample)
    ax = axes[0, 0]
    ax.plot(s_fine, v_fine, "b-", label="v(s)")
    ax.set_xlabel("Arc-length s")
    ax.set_ylabel("Speed v(s)")
    ax.set_title("Smooth speed profile v(s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Arc-length s vs time T (slow where v low -> flat; fast -> steep)
    ax = axes[0, 1]
    ax.plot(T_fine, s_fine_T, "g-", label="s(T)")
    ax.set_xlabel("Path-time T")
    ax.set_ylabel("Arc-length s")
    ax.set_title("Arc-length vs path-time (slow where v low)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Output: step index vs arc-length (flat slope = dense, steep = sparse)
    ax = axes[1, 0]
    ax.scatter(np.arange(len(s_out)), s_out, s=8, alpha=0.8)
    ax.set_xlabel("Step index k")
    ax.set_ylabel("Arc-length s")
    ax.set_title("Output waypoints (flat slope = dense, steep = sparse)")
    ax.grid(True, alpha=0.3)

    # 4) Segment length ds per step: small = dense (slow), large = sparse (fast)
    ax = axes[1, 1]
    ax.bar(np.arange(len(ds_out)), ds_out, width=0.8, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Step index k")
    ax.set_ylabel("Segment length ds")
    ax.set_title("Output segment lengths (small = slow/dense, large = fast/sparse)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / ".." / ".." / ".." / "runs" / "time_warp_demo.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"time_warp __main__: saved {out_path}")
    print(f"Input waypoints: {len(steps)}, output waypoints: {len(out)}")
