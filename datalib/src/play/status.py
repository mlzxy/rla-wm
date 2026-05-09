"""
Status Monitor for detecting failures during trajectory execution.

Provides checks for grasp success, safety violations, and other failure conditions.
Returns "Red Flags" (strings) when issues are detected.
"""

import numpy as np
from typing import List, Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class StatusCheck:
    """Result of a status check."""
    passed: bool
    message: str
    severity: str = "warning"  # "warning", "error", "info"


class StatusMonitor:
    """
    Monitors robot status and detects failures during trajectory execution.
    
    Checks include:
    - Grasp verification (object held above threshold)
    - Safety limits (force/velocity bounds)
    - Collision detection (excessive forces)
    """
    
    # Default thresholds
    DEFAULT_GRASP_HEIGHT_THRESHOLD = 0.05  # 5cm above table
    DEFAULT_MAX_QFORCE = 100.0  # Max joint force (Nm)
    DEFAULT_MAX_QVEL = 2.0  # Max joint velocity (rad/s)
    DEFAULT_MAX_EE_FORCE = 50.0  # Max end-effector force (N)
    
    def __init__(
        self,
        grasp_height_threshold: float = DEFAULT_GRASP_HEIGHT_THRESHOLD,
        max_qforce: float = DEFAULT_MAX_QFORCE,
        max_qvel: float = DEFAULT_MAX_QVEL,
        max_ee_force: float = DEFAULT_MAX_EE_FORCE,
    ):
        """
        Initialize the status monitor.
        
        Args:
            grasp_height_threshold: Minimum height for object to be considered grasped
            max_qforce: Maximum joint torque before flagging
            max_qvel: Maximum joint velocity before flagging
            max_ee_force: Maximum EE force before flagging
        """
        self.grasp_height_threshold = grasp_height_threshold
        self.max_qforce = max_qforce
        self.max_qvel = max_qvel
        self.max_ee_force = max_ee_force
        
        # Track red flags
        self._red_flags: List[str] = []
    
    def check_grasp(
        self,
        robot,
        obj,
        height_threshold: Optional[float] = None
    ) -> StatusCheck:
        """
        Check if an object is successfully grasped (held above table).
        
        Args:
            robot: The robot agent (not used directly, for future contact checks)
            obj: The object actor to check
            height_threshold: Override the default height threshold
            
        Returns:
            StatusCheck with pass/fail and message
        """
        threshold = height_threshold or self.grasp_height_threshold
        
        try:
            # Get object position
            if hasattr(obj, 'pose'):
                pose = obj.pose
                if hasattr(pose, 'p'):
                    z = float(pose.p[2])
                else:
                    z = float(pose[2])
            else:
                return StatusCheck(
                    passed=False,
                    message="Cannot get object pose",
                    severity="error"
                )
            
            if z >= threshold:
                return StatusCheck(
                    passed=True,
                    message=f"Object at z={z:.3f} (threshold: {threshold:.3f})"
                )
            else:
                return StatusCheck(
                    passed=False,
                    message=f"🚩 GRASP FAILURE: Object at z={z:.3f} < {threshold:.3f}",
                    severity="error"
                )
        except Exception as e:
            return StatusCheck(
                passed=False,
                message=f"Error checking grasp: {e}",
                severity="error"
            )
    
    def check_safety(
        self,
        robot,
        qforce: Optional[np.ndarray] = None,
        qvel: Optional[np.ndarray] = None,
    ) -> StatusCheck:
        """
        Check robot safety limits (force and velocity).
        
        Args:
            robot: The robot agent
            qforce: Joint forces (uses robot.qf if None)
            qvel: Joint velocities (uses robot.qvel if None)
            
        Returns:
            StatusCheck with pass/fail and message
        """
        warnings = []
        
        try:
            # Get joint velocities
            if qvel is None and hasattr(robot, 'robot'):
                if hasattr(robot.robot, 'qvel'):
                    qvel = robot.robot.qvel
                    if hasattr(qvel, 'cpu'):
                        qvel = qvel.cpu().numpy()
                    qvel = np.array(qvel).flatten()
            
            if qvel is not None:
                max_vel = np.max(np.abs(qvel))
                if max_vel > self.max_qvel:
                    warnings.append(
                        f"🚩 HIGH VELOCITY: max={max_vel:.2f} > {self.max_qvel:.2f} rad/s"
                    )
            
            # Get joint forces/torques
            if qforce is None and hasattr(robot, 'robot'):
                if hasattr(robot.robot, 'qf'):
                    qforce = robot.robot.qf
                    if hasattr(qforce, 'cpu'):
                        qforce = qforce.cpu().numpy()
                    qforce = np.array(qforce).flatten()
            
            if qforce is not None:
                max_force = np.max(np.abs(qforce))
                if max_force > self.max_qforce:
                    warnings.append(
                        f"🚩 HIGH FORCE: max={max_force:.1f} > {self.max_qforce:.1f} Nm"
                    )
            
            if warnings:
                return StatusCheck(
                    passed=False,
                    message="; ".join(warnings),
                    severity="warning"
                )
            else:
                return StatusCheck(
                    passed=True,
                    message="Safety limits OK"
                )
                
        except Exception as e:
            return StatusCheck(
                passed=False,
                message=f"Error checking safety: {e}",
                severity="error"
            )
    
    def check_all(
        self,
        robot,
        grasped_object: Optional[Any] = None,
    ) -> List[str]:
        """
        Run all status checks and return list of red flags.
        
        Args:
            robot: The robot agent
            grasped_object: Object being held (if any)
            
        Returns:
            List of red flag messages (empty if all checks pass)
        """
        red_flags = []
        
        # Check safety
        safety = self.check_safety(robot)
        if not safety.passed:
            red_flags.append(safety.message)
        
        # Check grasp if object provided
        if grasped_object is not None:
            grasp = self.check_grasp(robot, grasped_object)
            if not grasp.passed:
                red_flags.append(grasp.message)
        
        # Update internal tracking
        self._red_flags = red_flags
        
        return red_flags
    
    @property
    def has_red_flags(self) -> bool:
        """Check if any red flags are active."""
        return len(self._red_flags) > 0
    
    @property
    def red_flags(self) -> List[str]:
        """Get current red flags."""
        return self._red_flags.copy()
    
    def clear(self):
        """Clear red flag history."""
        self._red_flags = []
    
    def format_status(self, include_ok: bool = False) -> str:
        """
        Format current status as a human-readable string.
        
        Args:
            include_ok: Include "OK" messages when no issues
            
        Returns:
            Formatted status string
        """
        if self._red_flags:
            return "\n".join(self._red_flags)
        elif include_ok:
            return "✓ All checks passed"
        else:
            return ""


def check_grasp_simple(obj, threshold: float = 0.05) -> bool:
    """
    Simple function to check if object is grasped (z > threshold).
    
    Args:
        obj: Object actor with pose attribute
        threshold: Height threshold (default 5cm)
        
    Returns:
        True if object z-position >= threshold
    """
    try:
        if hasattr(obj, 'pose'):
            pose = obj.pose
            if hasattr(pose, 'p'):
                z = float(pose.p[2])
            else:
                z = float(pose[2])
            return z >= threshold
    except Exception:
        pass
    return False


def check_safety_simple(
    qvel: Optional[np.ndarray] = None,
    qforce: Optional[np.ndarray] = None,
    max_vel: float = 2.0,
    max_force: float = 100.0
) -> List[str]:
    """
    Simple function to check safety limits.
    
    Args:
        qvel: Joint velocities
        qforce: Joint forces/torques
        max_vel: Maximum velocity threshold
        max_force: Maximum force threshold
        
    Returns:
        List of warning messages (empty if safe)
    """
    warnings = []
    
    if qvel is not None:
        qvel = np.array(qvel).flatten()
        peak_vel = np.max(np.abs(qvel))
        if peak_vel > max_vel:
            warnings.append(f"🚩 HIGH VELOCITY: {peak_vel:.2f} rad/s")
    
    if qforce is not None:
        qforce = np.array(qforce).flatten()
        peak_force = np.max(np.abs(qforce))
        if peak_force > max_force:
            warnings.append(f"🚩 HIGH FORCE: {peak_force:.1f} Nm")
    
    return warnings
