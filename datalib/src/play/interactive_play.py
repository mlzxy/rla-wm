"""
Interactive Play Script for Human-in-the-Loop Trajectory Verification.

Allows users to:
- Preview generated trajectories with waypoint visualization
- Step through trajectory in chunks (SPACE to advance)
- Reset environment (R)
- Quit (Q)
- See real-time failure detection (Red Flags)

Usage:
    .venv/bin/python tests/interactive_play.py [--robot panda] [--num-distractors 5]
"""

import sys
import traceback
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import tyro
from rich import print as rprint
from rich.console import Console


import gymnasium as gym

# Import to register environment
from datalib.src import unified_workspace
from datalib.src.play.trace import Trace, Waypoint
from datalib.src.play.visualizer import WaypointVisualizer
from datalib.src.play.status import StatusMonitor
from datalib.src.play.engine import TrajectoryEngine, TrajectoryConfig
from datalib.src.play.utils import sapien_pose_to_numpy, just_run_env


@dataclass
class Args:
    robot: str = "panda"
    """Robot type"""
    num_distractors: int = 20
    """Number of objects"""
    chunk_size: int = 10
    """Waypoints per chunk"""
    scale_min: float = 1.0
    """Min object scale"""
    scale_max: float = 1.0
    """Max object scale"""
    robot_init_high: bool = True
    """Init robot upright"""
    random_rotation: bool = True
    """Randomize object rotation"""
    x_bounds: Tuple[float, float] = (-0.4, 0.4)
    """X workspace bounds"""
    y_bounds: Tuple[float, float] = (-0.6, 0.6)
    """Y workspace bounds"""
    waypoint_interval: float = 0.03
    """Distance between waypoints (meters)"""


class InteractivePlayer:
    """
    Interactive trajectory player with visualization and failure detection.
    """
    
    def __init__(
        self,
        env,
        chunk_size: int = 10,
        engine_config: TrajectoryConfig = None,
        robot_name: str = "panda"
    ):
        """
        Initialize the interactive player.
        
        Args:
            env: ManiSkill environment
            chunk_size: Number of steps to preview/execute at a time
            engine_config: Configuration for trajectory generation
        """
        self.env = env
        self.chunk_size = chunk_size
        self.robot_name = robot_name
        controllers = self.env.agent.controller.controllers
        assert not controllers['arm']._normalize_action, "does not support normalized action"
        if 'gripper' in controllers:
            assert controllers['gripper']._normalize_action
        
        # Create components
        self.visualizer = WaypointVisualizer(env, pool_size=100)
        self.monitor = StatusMonitor()
        
        # Trajectory engine (will be created after reset)
        self._engine = None
        self._engine_config = engine_config or TrajectoryConfig(
            pick_weight=0.3,
            place_weight=0.2,
            push_weight=0.4,
            tool_push_weight=0.1,
            max_steps_per_episode=20,
            waypoint_interval=0.03, # Default, will be overridden by args if passed
        )
        
        # Current trajectory state
        self._current_trace: Trace = None
        self._current_chunks = []
        self._chunk_index = 0
        self._executing = False
        
        # Stats
        self._total_steps = 0
        self._red_flag_count = 0
    
    def _create_engine(self, initial_qpos: Optional[np.ndarray] = None):
        """Create or recreate the trajectory engine."""
        try:
            from datalib.src.play.primitives import AtomicPrimitives
            # Pass render callback to visualize execution steps
            primitives = AtomicPrimitives(
                self.env,
                robot_name=self.robot_name,
                initial_qpos=initial_qpos,
                render_callback=self.env.render
            )
            self._engine = TrajectoryEngine(
                self.env,
                primitives=primitives,
                config=self._engine_config,
            )
        except Exception as e:
            console = Console()
            rprint(f"[red bold]ERROR: Could not create trajectory engine[/red bold]")
            console.print_exception(show_locals=False)
            self._engine = None
            raise  # Fail fast instead of silently falling back
    
    def reset(self):
        """Reset environment and prepare for new interactions."""
        print("\n[Reset] Resetting environment...")
        self.env.reset()
        
        # Get initial qpos for Home action
        initial_qpos = self.env.unwrapped.agent.robot.get_qpos().cpu().numpy()
        
        # Recreate engine with fresh environment state and home pose
        self._create_engine(initial_qpos=initial_qpos)
        
        # Clear visualizer
        self.visualizer.clear()
        self.monitor.clear()
        
        # Plan first action
        self._current_plan = None
        self._plan_next_action()
        
        print("  Ready for interaction!")
    
    def _plan_next_action(self):
        """Plan the next action and visualize it."""
        if self._engine is None:
            print("  [Error] No engine available")
            return
        
        # Plan a single action
        plans = self._engine.plan_episode(steps=1)
        
        if plans:
            self._current_plan = plans[0]
            # Visualize the planned action
            trace = self._current_plan.to_trace()
            self.visualizer.clear()
            self.visualizer.visualize_trace(trace)
            print(f"  [Planned] {self._current_plan.action_type.value} action")
            print(f"    Waypoints: {len(trace)}")
        else:
            self._current_plan = None
            print("  [Warning] No action could be planned")
    
    def _execute_current_plan(self):
        """Execute the current planned action."""
        if self._current_plan is None:
            print("  [Error] No plan to execute")
            return
        
        print(f"\n[Executing] {self._current_plan.action_type.value}")
        
        # Execute the action
        result = self._engine.execute_action(self._current_plan)
        self._total_steps += result.steps_taken
        
        # Show result
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.message}")
        print(f"  Steps: {result.steps_taken}")
        
        # Clear current plan
        self._current_plan = None
    
    def _get_current_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current robot TCP pose."""
        agent = self.env.unwrapped.agent
        tcp_pose = agent.tcp.pose
        pos, quat = sapien_pose_to_numpy(tcp_pose)
        return pos.astype(np.float32), quat.astype(np.float32)
    
    def run(self):
        """
        Main interaction loop with infinite plan-visualize-execute cycle.
        
        Flow:
        1. Plan next action → show waypoints
        2. Wait for SPACE key
        3. Execute the action
        4. Repeat forever (until Q pressed)
        """
        print("\n" + "=" * 60)
        print(" INTERACTIVE PLAY - Phase 9 Lazy Planning")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE - Execute current action and plan next")
        print("  R     - Reset environment")
        print("  Q     - Quit")
        print("=" * 60 + "\n")
        
        # Initial reset
        self.reset()
        
        try:
            while True:
                # Update TCP visualization
                tcp_pos, tcp_quat = self._get_current_tcp_pose()
                self.visualizer.update_tcp_pose(tcp_pos, tcp_quat)

                # Render
                self.env.render()
                
                # Check viewer
                viewer = self.env.unwrapped.viewer
                if viewer is None or viewer.window is None:
                    break
                
                if viewer.window.should_close:
                    break
                
                # Handle input
                if viewer.window.key_press("q"):
                    print("\n[Quit]")
                    break
                
                if viewer.window.key_press("r"):
                    self.reset()
                
                if viewer.window.key_press(" "):
                    # Execute current plan, then plan next
                    self._execute_current_plan()
                    self._plan_next_action()
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        
        finally:
            # Print summary
            print("\n" + "=" * 60)
            print(" Session Summary")
            print("=" * 60)
            print(f"  Total steps: {self._total_steps}")
            print(f"  Red flags: {self._red_flag_count}")
            print("=" * 60)
            
            # Cleanup
            self.visualizer.destroy()
            self.env.close()


def main():
    from datalib.src.play.unified_play import UnifiedArgs, UnifiedPlay
    args = tyro.cli(UnifiedArgs)
    args.mode = "auto"  # This script defaults to auto (engine-driven) play
    print(f"Creating environment with {args.robot} robot and {args.num_distractors} objects...")
    UnifiedPlay(args).run()



if __name__ == "__main__":
    main()
