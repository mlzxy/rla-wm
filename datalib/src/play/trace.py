"""
Trace class for encapsulating trajectory data.

Provides serialization, chunking, and factory methods for working with
generated manipulation trajectories.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Iterator, Generator
from dataclasses import dataclass, field, asdict
from enum import Enum

# Forward import for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import TrajectoryStep


@dataclass
class Waypoint:
    """Single waypoint in a trace with position and action info."""
    position: np.ndarray  # 3D position
    orientation: Optional[np.ndarray] = None  # Optional quaternion
    action_type: str = "move"
    gripper_state: Optional[str] = None  # "open", "closed", None
    metadata: Dict[str, Any] = field(default_factory=dict)
    action: Optional[np.ndarray] = None  # Low-level action for replay
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist() if self.orientation is not None else None,
            "action_type": self.action_type,
            "gripper_state": self.gripper_state,
            "metadata": self.metadata,
            "action": self.action.tolist() if self.action is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Waypoint":
        """Create from dict."""
        return cls(
            position=np.array(data["position"]),
            orientation=np.array(data["orientation"]) if data.get("orientation") else None,
            action_type=data.get("action_type", "move"),
            gripper_state=data.get("gripper_state"),
            metadata=data.get("metadata", {}),
            action=np.array(data["action"]) if data.get("action") is not None else None
        )


class Trace:
    """
    Encapsulates a sequence of waypoints representing a trajectory.
    
    Supports serialization, chunking for visualization, and factory methods
    for creating traces from TrajectoryEngine output.
    """
    
    def __init__(
        self,
        waypoints: Optional[List[Waypoint]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Trace.
        
        Args:
            waypoints: List of Waypoint objects
            metadata: Optional metadata about the trace (e.g., action type, target)
        """
        self.waypoints: List[Waypoint] = waypoints or []
        self.metadata: Dict[str, Any] = metadata or {}
    
    def __len__(self) -> int:
        return len(self.waypoints)
    
    def __iter__(self) -> Iterator[Waypoint]:
        return iter(self.waypoints)
    
    def __getitem__(self, idx: int) -> Waypoint:
        return self.waypoints[idx]
    
    def append(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the trace."""
        self.waypoints.append(waypoint)
    
    def extend(self, waypoints: List[Waypoint]) -> None:
        """Add multiple waypoints."""
        self.waypoints.extend(waypoints)
    
    def chunk(self, size: int) -> Generator["Trace", None, None]:
        """
        Split trace into smaller sub-traces of given size.
        
        Args:
            size: Maximum number of waypoints per chunk
            
        Yields:
            Trace objects containing up to `size` waypoints
        """
        for i in range(0, len(self.waypoints), size):
            chunk_waypoints = self.waypoints[i:i + size]
            chunk_metadata = {
                **self.metadata,
                "chunk_index": i // size,
                "chunk_start": i,
                "chunk_end": min(i + size, len(self.waypoints)),
                "total_waypoints": len(self.waypoints)
            }
            yield Trace(waypoints=chunk_waypoints, metadata=chunk_metadata)
    
    def serialize(self) -> str:
        """
        Serialize trace to JSON string.
        
        Returns:
            JSON string representation
        """
        data = {
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "metadata": self.metadata
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def deserialize(cls, json_str: str) -> "Trace":
        """
        Create Trace from JSON string.
        
        Args:
            json_str: JSON string from serialize()
            
        Returns:
            Trace object
        """
        data = json.loads(json_str)
        waypoints = [Waypoint.from_dict(wp) for wp in data.get("waypoints", [])]
        return cls(waypoints=waypoints, metadata=data.get("metadata", {}))
    
    @classmethod
    def from_trajectory_steps(
        cls,
        steps: List["TrajectoryStep"],
        include_results: bool = True
    ) -> "Trace":
        """
        Create Trace from TrajectoryEngine output.
        
        Extracts waypoints from the result poses in each step.
        
        Args:
            steps: List of TrajectoryStep from TrajectoryEngine
            include_results: Whether to include result data in metadata
            
        Returns:
            Trace with waypoints extracted from steps
        """
        waypoints = []
        for step in steps:
            if step.result is None:
                continue
            
            # Extract waypoints from result poses and actions if available
            has_poses = hasattr(step.result, 'poses') and step.result.poses
            has_actions = hasattr(step.result, 'actions') and step.result.actions
            
            if has_poses:
                for idx, pose in enumerate(step.result.poses):
                    # pose is typically (position, quaternion)
                    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                        pos, quat = pose[0], pose[1]
                    else:
                        # Assume it's just position
                        pos = pose
                        quat = None
                    
                    # Match action to pose if possible
                    action = step.result.actions[idx] if has_actions and idx < len(step.result.actions) else None
                    
                    wp = Waypoint(
                        position=np.array(pos) if not isinstance(pos, np.ndarray) else pos,
                        orientation=np.array(quat) if quat is not None else None,
                        action_type=step.action_type.value if hasattr(step.action_type, 'value') else str(step.action_type),
                        gripper_state="open", # Default for trajectory steps if unknown
                        metadata={"step_index": len(waypoints), "pose_index": idx},
                        action=action
                    )
                    waypoints.append(wp)
            elif step.target_position is not None:
                # Fallback: use target position
                wp = Waypoint(
                    position=np.array(step.target_position),
                    action_type=step.action_type.value if hasattr(step.action_type, 'value') else str(step.action_type),
                    metadata={"step_index": len(waypoints)}
                )
                waypoints.append(wp)
        
        metadata = {
            "source": "trajectory_engine",
            "total_steps": len(steps),
            "success_count": sum(1 for s in steps if s.result and s.result.success)
        }
        
        return cls(waypoints=waypoints, metadata=metadata)
    
    @classmethod
    def from_positions(
        cls,
        positions: List[np.ndarray],
        action_type: str = "move",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Trace":
        """
        Create Trace from a simple list of positions.
        
        Args:
            positions: List of 3D position arrays
            action_type: Action type to assign to all waypoints
            metadata: Optional trace metadata
            
        Returns:
            Trace with waypoints at each position
        """
        waypoints = [
            Waypoint(position=np.array(pos), action_type=action_type)
            for pos in positions
        ]
        return cls(waypoints=waypoints, metadata=metadata or {})
    
    @classmethod
    def from_primitive_steps(
        cls,
        steps: List,  # List of PrimitiveStep from primitives.py
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Trace":
        """
        Create Trace from dense PrimitiveStep objects.
        
        Extracts position, orientation, and metadata (is_interaction, phase)
        from each step for visualization with alpha dimming support.
        
        Args:
            steps: List of PrimitiveStep from trajectory generation
            metadata: Optional trace metadata
            
        Returns:
            Trace with waypoints including is_interaction/phase metadata
        """
        waypoints = []
        for i, step in enumerate(steps):
            gripper_state = "closed" if step.gripper < 0 else "open"
            wp = Waypoint(
                position=np.array(step.position),
                orientation=np.array(step.quaternion),
                action_type=step.phase,
                gripper_state=gripper_state,
                metadata={
                    "is_interaction": step.is_interaction,
                    "phase": step.phase,
                    "step_index": i,
                    **step.metadata,
                },
            )
            waypoints.append(wp)
        
        return cls(waypoints=waypoints, metadata=metadata or {"source": "primitive_steps"})
