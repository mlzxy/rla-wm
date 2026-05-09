"""
Trajectory interpolation utilities using scipy.

Provides smooth interpolation between waypoints using:
- CubicSpline for position (per-dimension)
- Slerp for rotation
"""

import numpy as np
from typing import List
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

from .primitives import PrimitiveStep


def interpolate_trajectory(
    steps: List[PrimitiveStep],
    num_interp: int = 5,
) -> List[PrimitiveStep]:
    """
    Interpolate a trajectory to generate smoother motion.
    
    Uses scipy CubicSpline for position and Slerp for rotation.
    
    Args:
        steps: List of PrimitiveStep waypoints.
        num_interp: Number of interpolated points between each pair of waypoints.
        
    Returns:
        List of interpolated PrimitiveStep objects.
    """
    if len(steps) < 2:
        return steps
    
    n = len(steps)
    
    # Extract positions and quaternions
    positions = np.array([s.position for s in steps])  # (n, 3)
    quaternions = np.array([s.quaternion for s in steps])  # (n, 4) xyzw
    grippers = np.array([s.gripper for s in steps])  # (n,)
    
    # Create time points (uniform spacing)
    t_waypoints = np.linspace(0, 1, n)
    
    # Create query times
    total_interp_points = (n - 1) * num_interp + 1
    t_query = np.linspace(0, 1, total_interp_points)
    
    # Interpolate position (per-dimension cubic spline)
    interp_positions = np.zeros((total_interp_points, 3))
    for dim in range(3):
        spline = CubicSpline(t_waypoints, positions[:, dim])
        interp_positions[:, dim] = spline(t_query)
    
    # Interpolate rotation (scipy Slerp)
    # Convert xyzw to scipy Rotation (xyzw is scipy's native format)
    rotations = R.from_quat(quaternions)  # xyzw format
    slerp = Slerp(t_waypoints, rotations)
    interp_rotations = slerp(t_query)
    interp_quaternions = interp_rotations.as_quat()  # (n, 4) xyzw
    
    # Interpolate gripper (linear)
    interp_grippers = np.interp(t_query, t_waypoints, grippers)
    
    # Build output steps
    # For metadata, we use the phase from the nearest original waypoint
    output_steps = []
    for i, t in enumerate(t_query):
        # Find the nearest original waypoint index
        orig_idx = min(int(t * (n - 1) + 0.5), n - 1)
        orig_step = steps[orig_idx]
        
        # Mark only the original waypoint positions as is_interaction
        is_original = any(np.isclose(t, tw) for tw in t_waypoints)
        
        output_steps.append(PrimitiveStep(
            position=interp_positions[i],
            quaternion=interp_quaternions[i],
            gripper=float(interp_grippers[i]),
            phase=orig_step.phase,
            is_interaction=orig_step.is_interaction if is_original else False,
        ))
    
    return output_steps

