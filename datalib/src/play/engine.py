"""
Trajectory Engine for generating diverse manipulation sequences.

Orchestrates atomic primitives to create continuous, randomized
trajectories for data collection.
"""

import numpy as np
from typing import List, Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

from .primitives import AtomicPrimitives, PrimitiveResult, PrimitiveStep
from .utils import get_actor_world_pose
from .kinematics_helper import KinematicsHelper
from .trace import Trace, Waypoint


class ActionType(Enum):
    """Types of actions the engine can execute."""

    PICK = "pick"
    PLACE = "place"
    PUSH = "push"
    TOOL_PUSH = "tool_push"
    HOME = "home"


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""

    action_type: ActionType
    target_actor: Optional[Any] = None  # Actor for pick/push
    target_position: Optional[np.ndarray] = None  # For place
    direction: Optional[np.ndarray] = None  # For push
    result: Optional[PrimitiveResult] = None


@dataclass
class PlannedAction:
    """
    A planned action that can be visualized before execution.

    This separates the "planning" phase (sampling parameters) from
    the "execution" phase (actually running the physics simulation).
    """

    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_actor: Optional[Any] = None
    target_position: Optional[np.ndarray] = None
    trajectory_steps: Optional[List[PrimitiveStep]] = (
        None  # Pre-generated dense trajectory
    )

    def to_trace(self) -> Trace:
        """
        Convert this planned action to a Trace for visualization.

        Generates waypoints from the parameters so the user can see
        where the robot will move before executing.
        """
        waypoints = []

        if self.trajectory_steps is not None:
            # Use pre-generated dense trajectory
            for step in self.trajectory_steps:
                waypoints.append(
                    Waypoint(
                        position=step.position,
                        orientation=step.quaternion,
                        action_type=self.action_type.value,
                        gripper_state="closed" if step.gripper < 0 else "open",
                        metadata={
                            "is_interaction": step.is_interaction,
                            "label": step.phase,
                            "gripper": step.gripper,
                        },
                    )
                )
            return Trace(
                waypoints=waypoints, metadata={"action_type": self.action_type.value}
            )

        # Legacy fallback (parameter-based)
        if self.action_type == ActionType.PICK:
            # Waypoints: approach -> pregrasp -> lift
            approach_pos = self.parameters.get("approach_pos")
            pregrasp_pos = self.parameters.get("pregrasp_pos")
            obj_pos = self.parameters.get("obj_pos")

            if approach_pos is not None:
                waypoints.append(
                    Waypoint(
                        position=approach_pos.copy(),
                        action_type="pick",
                        metadata={"label": "approach", "is_interaction": False},
                    )
                )
            if pregrasp_pos is not None:
                waypoints.append(
                    Waypoint(
                        position=pregrasp_pos.copy(),
                        action_type="pick",
                        metadata={"label": "grasp", "is_interaction": True},
                    )
                )
            # Add lift waypoint
            if pregrasp_pos is not None:
                lift_pos = pregrasp_pos.copy()
                lift_pos[2] += 0.10  # Default lift height
                waypoints.append(
                    Waypoint(
                        position=lift_pos,
                        action_type="pick",
                        metadata={"label": "lift", "is_interaction": False},
                    )
                )

        elif self.action_type == ActionType.PLACE:
            target_pos = self.parameters.get("target_pos")
            if target_pos is not None:
                # Above target
                above_pos = target_pos.copy()
                above_pos[2] += 0.15
                waypoints.append(
                    Waypoint(
                        position=above_pos,
                        action_type="place",
                        metadata={"label": "above", "is_interaction": False},
                    )
                )
                # Place position
                waypoints.append(
                    Waypoint(
                        position=target_pos.copy(),
                        action_type="place",
                        metadata={"label": "place", "is_interaction": True},
                    )
                )
                # Retract
                retract_pos = target_pos.copy()
                retract_pos[2] += 0.10
                waypoints.append(
                    Waypoint(
                        position=retract_pos,
                        action_type="place",
                        metadata={"label": "retract", "is_interaction": False},
                    )
                )

        elif self.action_type == ActionType.PUSH:
            obj_pos = self.parameters.get("obj_pos")
            direction = self.parameters.get("direction")
            distance = self.parameters.get("distance", 0.1)

            if obj_pos is not None and direction is not None:
                # Approach (behind object)
                approach_pos = obj_pos.copy()
                approach_pos[0] -= direction[0] * 0.08
                approach_pos[1] -= direction[1] * 0.08
                approach_pos[2] = 0.05
                waypoints.append(
                    Waypoint(
                        position=approach_pos,
                        action_type="push",
                        metadata={"label": "approach", "is_interaction": False},
                    )
                )
                # Contact
                contact_pos = obj_pos.copy()
                contact_pos[2] = 0.05
                waypoints.append(
                    Waypoint(
                        position=contact_pos,
                        action_type="push",
                        metadata={"label": "contact", "is_interaction": True},
                    )
                )
                # Push through
                push_pos = contact_pos.copy()
                push_pos[0] += direction[0] * distance
                push_pos[1] += direction[1] * distance
                waypoints.append(
                    Waypoint(
                        position=push_pos,
                        action_type="push",
                        metadata={"label": "push", "is_interaction": True},
                    )
                )
                # Retract
                retract_pos = push_pos.copy()
                retract_pos[2] += 0.1
                waypoints.append(
                    Waypoint(
                        position=retract_pos,
                        action_type="push",
                        metadata={"label": "retract", "is_interaction": False},
                    )
                )

        elif self.action_type == ActionType.HOME:
            home_pos = self.parameters.get("home_pos")
            if home_pos is not None:
                waypoints.append(
                    Waypoint(
                        position=home_pos.copy(),
                        action_type="home",
                        metadata={"label": "home", "is_interaction": False},
                    )
                )

        return Trace(
            waypoints=waypoints, metadata={"action_type": self.action_type.value}
        )


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""

    # Action distribution
    pick_weight: float = 0.3
    place_weight: float = 0.2
    push_weight: float = 0.4
    tool_push_weight: float = 0.1
    home_weight: float = 0.1

    # Placement bounds (x_min, x_max, y_min, y_max)
    place_bounds: Tuple[float, float, float, float] = (-0.4, 0.4, -0.6, 0.6)
    min_place_height: float = 0.02
    max_place_height: float = 0.06

    # Push parameters
    min_push_distance: float = 0.05
    max_push_distance: float = 0.15

    # Episode parameters
    max_steps_per_episode: int = 10
    home_after_failed: bool = True

    # Waypoint generation
    waypoint_interval: float = 0.03  # 3cm between waypoints

    # Speed-diverse execution: (v_min, v_max) to time-warp resample trajectories; None to disable
    time_warp_speed_bounds: Optional[Tuple[float, float]] = (0.5, 2.0)


class TrajectoryEngine:
    """
    Generates diverse manipulation trajectories by orchestrating primitives.

    Randomly selects and executes pick, place, and push actions to create
    varied interaction data for learning.
    """

    def __init__(
        self,
        env,
        primitives: Optional[AtomicPrimitives] = None,
        config: Optional[TrajectoryConfig] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the trajectory engine.

        Args:
            env: ManiSkill environment
            primitives: Custom primitives instance (creates default if None)
            config: Trajectory generation config
            random_seed: Random seed for reproducibility
        """
        self.env = env
        self.primitives = primitives or AtomicPrimitives(env)
        self.config = config or TrajectoryConfig()

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Track trajectory history
        self.trajectory: List[TrajectoryStep] = []
        self._actors_cache: List = []

        # Initialize kinematics helper for reachability validation
        self.kinematics = KinematicsHelper(env=env)

    @property
    def available_actors(self) -> List:
        """Get list of actors available for interaction."""
        actors = []
        if hasattr(self.env, "distractors"):
            actors.extend(self.env.distractors)

        if hasattr(self.env, "unwrapped"):
            unwrapped = self.env.unwrapped
            # Common ManiSkill task object attributes
            for attr in ["obj", "cube", "tool", "peg", "goal", "box"]:
                if hasattr(unwrapped, attr):
                    val = getattr(unwrapped, attr)
                    if isinstance(val, (list, tuple)):
                        actors.extend(val)
                    else:
                        actors.append(val)
        return actors

    def _validate_trajectory(
        self,
        trajectory: List[PrimitiveStep],
        sample_rate: int = 3,
    ) -> Tuple[bool, Optional[int]]:
        """
        Validate that all waypoints in a dense trajectory are reachable.

        Uses KinematicsHelper for workspace-based validation.

        Args:
            trajectory: List of PrimitiveStep to validate
            sample_rate: Check every N steps (default 3 for efficiency)

        Returns:
            Tuple of (is_valid, first_invalid_index or None)
        """
        return self.kinematics.validate_trajectory(trajectory, sample_rate)

    def _normalize_weights(self) -> list:
        """Normalize action weights for sampling."""
        weights = [
            self.config.pick_weight,
            self.config.place_weight,
            self.config.push_weight,
            self.config.tool_push_weight,
            self.config.home_weight,
        ]
        total = sum(weights)
        return [w / total for w in weights]

    def _sample_action_type(self) -> ActionType:
        """Sample an action type based on current state and weights."""
        is_holding = self.primitives.is_holding

        if is_holding:
            # Can only place if holding
            return ActionType.PLACE
        else:
            # Sample from available actions
            weights = self._normalize_weights()
            action_types = [
                ActionType.PICK,
                ActionType.PLACE,  # Will be skipped
                ActionType.PUSH,
                ActionType.TOOL_PUSH,
                ActionType.HOME,
            ]

            # Remove place since not holding
            weights[1] = 0
            total = sum(weights)
            weights = [w / total for w in weights]

            sampled_type = np.random.choice(action_types, p=weights)

            # UR10eStick Refinement: Force Push-Only
            agent = self.env.unwrapped.agent
            if "stick" in agent.uid or "close" in agent.uid:
                # Reset holding if it was somehow set
                self.primitives._held_object = None

                # Allow HOME occasionally even for stick
                stick_action_types = [ActionType.PUSH, ActionType.HOME]
                stick_weights = [0.9, 0.1]
                return np.random.choice(stick_action_types, p=stick_weights)

            return sampled_type

    def _sample_target_actor(self, exclude: List = None) -> Optional[Any]:
        """Sample a random target actor."""
        actors = self.available_actors
        if exclude:
            actors = [a for a in actors if a not in exclude]

        if not actors:
            return None

        return random.choice(actors)

    def _sample_place_position(self) -> np.ndarray:
        """Sample a random placement position within bounds."""
        x_min, x_max, y_min, y_max = self.config.place_bounds
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(
            self.config.min_place_height, self.config.max_place_height
        )
        return np.array([x, y, z], dtype=np.float32)

    def _sample_push_direction(self) -> np.ndarray:
        """Sample a random push direction."""
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def _sample_push_distance(self) -> float:
        """Sample a random push distance."""
        return np.random.uniform(
            self.config.min_push_distance, self.config.max_push_distance
        )

    def reset(self):
        """Reset the engine state."""
        self.trajectory = []
        self.primitives._held_object = None
        self.env.reset()

    def plan_episode(
        self, steps: int = 1, max_retries: int = 20
    ) -> List[PlannedAction]:
        """
        Plan actions WITHOUT executing them.

        This allows visualization of planned actions before committing
        to the physics simulation.

        Args:
            steps: Number of actions to plan
            max_retries: Maximum attempts to sample a valid action per step

        Returns:
            List of PlannedAction objects ready for visualization/execution
        """
        planned_actions = []

        # Get start state from actual robot
        current_pos, current_quat = self.primitives.get_tcp_pose()

        # Track simulated state through the plan
        sim_pos = current_pos.copy()
        sim_quat = current_quat.copy()

        for _ in range(steps):
            # Attempt to sample a valid action
            for attempt in range(max_retries):
                # Update primitives internal state for sampling (hacky but needed for is_holding check)
                # self.primitives._held_object state is used by _sample_action_type
                # We need to ensure logic is consistent if we plan multiple steps.
                # For now, we assume 1 step planning or that is_holding state is tracked externally?
                # Actually, _sample_action_type checks self.primitives.is_holding.
                # If we plan multiple steps, we can't easily update that without executing.
                # LIMITATION: Multi-step planning might be inaccurate regarding is_holding
                # if we don't track it virtually.
                # For Phase 13, we focus on 1-step or handle loose tracking.

                action_type = self._sample_action_type()
                planned_action = None
                trajectory_steps = None

                if action_type == ActionType.PICK:
                    actor = self._sample_target_actor()
                    if actor is not None:
                        params = self.primitives.sample_pick_parameters(actor)
                        if params is not None:
                            # Generate dense trajectory from SIMULATED start pose
                            trajectory_steps = self.primitives.generate_pick_trajectory(
                                params,
                                start_pos=sim_pos,
                                start_quat=sim_quat,
                                resolution=self.config.waypoint_interval,
                            )
                            planned_action = PlannedAction(
                                action_type=action_type,
                                parameters=params,
                                target_actor=actor,
                                trajectory_steps=trajectory_steps,
                            )

                elif action_type == ActionType.PLACE:
                    params = self.primitives.sample_place_parameters(
                        bounds=self.config.place_bounds
                    )
                    if params is not None:
                        trajectory_steps = self.primitives.generate_place_trajectory(
                            params,
                            start_pos=sim_pos,
                            start_quat=sim_quat,
                            resolution=self.config.waypoint_interval,
                        )
                        planned_action = PlannedAction(
                            action_type=action_type,
                            parameters=params,
                            trajectory_steps=trajectory_steps,
                        )

                elif action_type == ActionType.PUSH:
                    actor = self._sample_target_actor()
                    if actor is not None:
                        params = self.primitives.sample_push_parameters(actor)
                        if params is not None:
                            trajectory_steps = self.primitives.generate_push_trajectory(
                                params,
                                start_pos=sim_pos,
                                start_quat=sim_quat,
                                resolution=self.config.waypoint_interval,
                            )
                            planned_action = PlannedAction(
                                action_type=action_type,
                                parameters=params,
                                target_actor=actor,
                                trajectory_steps=trajectory_steps,
                            )

                elif action_type == ActionType.HOME:
                    params = self.primitives.sample_home_parameters()
                    if params is not None:
                        trajectory_steps = self.primitives.generate_home_trajectory(
                            params,
                            start_pos=sim_pos,
                            start_quat=sim_quat,
                            resolution=self.config.waypoint_interval,
                        )
                        planned_action = PlannedAction(
                            action_type=action_type,
                            parameters=params,
                            trajectory_steps=trajectory_steps,
                        )

                # If we successfully planned an action, validate reachability
                if planned_action is not None and planned_action.trajectory_steps:
                    is_valid, invalid_idx = self._validate_trajectory(
                        planned_action.trajectory_steps,
                        sample_rate=1,  # Check ALL waypoints for safety
                    )

                    if is_valid:
                        planned_actions.append(planned_action)

                        # Update simulated state for next step
                        last_step = planned_action.trajectory_steps[-1]
                        sim_pos = last_step.position
                        sim_quat = last_step.quaternion

                        # Update holding state if needed (virtual tracking)
                        # This is tricky because primitives._held_object is real state.
                        # For now, just break and assume 1 step or consistent state.
                        break
                    # else: reachability failed, retry
            else:
                # Loop completed without break = failed to plan action
                # This is common if the workspace is empty or crowded
                print(
                    f"[TrajectoryEngine] Failed to plan valid action after {max_retries} attempts"
                )
                pass

        return planned_actions

    def execute_action(self, planned_action: PlannedAction) -> PrimitiveResult:
        """
        Execute a previously planned action.

        If the action has pre-generated trajectory_steps, executes those directly.
        Otherwise falls back to legacy per-action execution.

        Args:
            planned_action: PlannedAction from plan_episode()

        Returns:
            PrimitiveResult from the execution
        """
        result = self.primitives.execute_trajectory(
            planned_action.trajectory_steps,
            self.kinematics,
            time_warp_speed_bounds=self.config.time_warp_speed_bounds,
            target_actor=planned_action.target_actor
            if planned_action.action_type == ActionType.PICK
            else None,
            action_name=planned_action.action_type.value,
        )

        if result.success:
            if planned_action.action_type == ActionType.PICK:
                self.primitives._held_object = planned_action.target_actor

        if planned_action.action_type == ActionType.PLACE:
            self.primitives._held_object = None

        return result
