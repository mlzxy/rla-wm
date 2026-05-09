"""
Unified Play and Tune Action Tool.

Supports two modes:
- auto: Engine plans one action, SPACE executes and plans next; R reset, Q quit.
- manual: TAB object, G/T/H/Y action type, SPACE sample, ARROWS tune, ENTER execute.

Runtime switch: A = auto, M = manual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import sapien.core as sapien
import tyro
from rich import print

import mani_skill.envs  # noqa: F401
from datalib.src import unified_workspace  # noqa: F401
from datalib.src.play.engine import ActionType, PlannedAction, TrajectoryConfig, TrajectoryEngine
from datalib.src.play.kinematics_helper import KinematicsHelper
from datalib.src.play.primitives import AtomicPrimitives, PrimitiveStep
from datalib.src.play.trace import Trace, Waypoint
from datalib.src.play.utils import get_actor_world_pose, sapien_pose_to_numpy
from datalib.src.play.visualizer import HIDDEN_POS, WaypointVisualizer


@dataclass
class UnifiedArgs:
    """Unified arguments for play (auto) and tune (manual) modes."""
    robot: str = "panda" # panda, xarm6_robotiq, ur10e_stick
    """Robot name (e.g. panda, xarm6)"""
    scene: str = "TableOnly-v2"
    """ManiSkill environment id"""
    shader: str = "default"
    """Viewer shader"""
    num_distractors: int = 20
    """Number of objects"""
    scale_min: float = 1.0
    """Min object scale"""
    scale_max: float = 1.0
    """Max object scale"""
    robot_init_high: bool = True
    """Init robot upright"""
    random_rotation: bool = True
    """Randomize object rotation"""
    x_bounds: Tuple[float, float] = (-0.4, 0.2)
    """X workspace bounds"""
    y_bounds: Tuple[float, float] = (-0.5, 0.5)
    """Y workspace bounds"""
    chunk_size: int = 10
    """Waypoints per chunk (auto mode)"""
    waypoint_interval: float = 0.03
    """Distance between waypoints (meters)"""
    max_actions: int = 50
    """Maximum number of actions to execute in auto mode"""
    headless: bool = False
    """Run without viewer (always True if not darwin, unless forced)"""
    behavior: Literal["all", "pick_place", "push"] = "pick_place"
    """Auto mode behavior: all, pick_place (only pick/place), or push (only push). Ignored if action weights are set."""
    pick_weight: Optional[float] = 0.4 
    """Action weight for pick (auto mode). If set with place_weight/push_weight/tool_push_weight, overrides behavior."""
    place_weight: Optional[float] = 0.2
    """Action weight for place (auto mode)."""
    push_weight: Optional[float] = 0.4
    """Action weight for push (auto mode)."""
    tool_push_weight: Optional[float] = 0.4
    """Action weight for tool_push (auto mode)."""
    mode: Literal["auto", "manual"] = "manual"
    """auto = engine plans, SPACE to execute; manual = tune object/action/params"""
    time_warp_speed_bounds: Optional[Tuple[float, float]] = (0.5, 2.0)
    """(v_min, v_max) to time-warp resample trajectories for speed diversity; None = no warp"""

    reconfiguration_freq: int = 1


class UnifiedPlay:
    """
    Single tool: auto mode (engine plan + execute) or manual mode (tune action).
    """

    def __init__(self, args: UnifiedArgs):
        self.args = args
        render_mode = "human"
        if args.headless:
            render_mode = None
        
        self.env = gym.make(
            args.scene,
            obs_mode="none",
            control_mode="pd_joint_pos_6d" if args.robot == "ur10e_stick" else "pd_joint_pos",
            render_mode=render_mode,
            robot_uids=args.robot,
            num_distractors=args.num_distractors,
            distractor_types=["cube", "sphere", "box", "stick", "triangle", "polyhedron", "number"],
            distractor_scale_min=args.scale_min,
            distractor_scale_max=args.scale_max,
            robot_init_high=args.robot_init_high,
            random_rotation=args.random_rotation,
            workspace_x_bounds=args.x_bounds,
            workspace_y_bounds=args.y_bounds,
            collision_free_placement=True,
            reconfiguration_freq=args.reconfiguration_freq
        )
        self.agent = self.env.unwrapped.agent
        if render_mode == "human":
            self.visualizer = WaypointVisualizer(self.env)
            # Register visualizer as plugin
            viewer = self.env.unwrapped.viewer
            if viewer is not None:
                viewer.plugins.append(self.visualizer)
                viewer.init_plugins([self.visualizer])
        else:
            self.visualizer = None

        # Primitives/kinematics/engine set in _reset()
        self.primitives: Optional[AtomicPrimitives] = None
        self.kinematics: Optional[KinematicsHelper] = None
        self._engine: Optional[TrajectoryEngine] = None

        # Manual (tuner) state
        self.tuner_state: str = "IDLE"  # IDLE, READY_TO_EXECUTE
        self.selected_action: str = "pick"
        self.current_params: Dict = {}
        self.trajectory: List[PrimitiveStep] = []
        self.current_object = None
        self.objects: List = []
        self._active_marker = None
        self.param_modifiers = {
            "pick": {"approach_height": 0.0, "grasp_yaw": 0.0},
            "push": {"push_z": 0.0, "push_height_offset": 0.0, "direction_angle": 0.0},
            "place": {"place_z": 0.0, "place_yaw": 0.0},
            "home": {},
        }

        # Auto state
        self._current_plan = None
        self._total_steps = 0
        self._actions_executed = 0
        self._consecutive_failures = 0

        self._reset()

    def _find_all_objects(self) -> None:
        """Find all graspable objects in the scene."""
        self.objects = []
        if hasattr(self.env.unwrapped, "obj") and self.env.unwrapped.obj is not None:
            self.objects.append(self.env.unwrapped.obj)
        if hasattr(self.env.unwrapped, "distractors"):
            self.objects.extend(self.env.unwrapped.distractors)
        if hasattr(self.env.unwrapped, "actors"):
            self.objects.extend(self.env.unwrapped.actors)
        if not self.objects:
            scene = self.env.unwrapped.scene
            for actor in scene.get_all_actors():
                name = actor.name.lower()
                if any(x in name for x in ["panda", "xarm", "robot", "ground", "table", "camera", "goal", "workspace"]):
                    continue
                self.objects.append(actor)
        self.objects = list(set(self.objects))
        self.objects.sort(key=lambda x: x.name)
        if self.objects:
            print(f"Found {len(self.objects)} objects: {[o.name for o in self.objects]}")
            self.current_object = self.objects[0]
        else:
            print("No objects found!")

    def _reset(self) -> None:
        self.env.reset()
        initial_qpos = self.agent.robot.get_qpos().cpu().numpy()
        render_callback = self.env.render if self.visualizer else None
        self.primitives = AtomicPrimitives(
            self.env,
            robot_name=self.args.robot,
            initial_qpos=initial_qpos,
            render_callback=render_callback,
            interpolate_steps=5
        )
        self.kinematics = KinematicsHelper(self.env)
        if self.visualizer:
            self.visualizer.clear()
        
        self._consecutive_failures = 0

        if self.args.mode == "auto":
            self._current_plan = None
            place_bounds = (
                self.args.x_bounds[0], self.args.x_bounds[1],
                self.args.y_bounds[0], self.args.y_bounds[1],
            )
            # Action weights: use explicit weights if all four set, else behavior preset
            if all(
                w is not None
                for w in (
                    self.args.pick_weight,
                    self.args.place_weight,
                    self.args.push_weight,
                    self.args.tool_push_weight,
                )
            ):
                pick_w = self.args.pick_weight
                place_w = self.args.place_weight
                push_w = self.args.push_weight
                tool_push_w = self.args.tool_push_weight
            else:
                pick_w, place_w, push_w, tool_push_w = 0.3, 0.2, 0.4, 0.1
                if self.args.behavior == "pick_place":
                    pick_w, place_w, push_w, tool_push_w = 0.5, 0.5, 0.0, 0.0
                elif self.args.behavior == "push":
                    pick_w, place_w, push_w, tool_push_w = 0.0, 0.0, 0.8, 0.2

            config = TrajectoryConfig(
                waypoint_interval=self.args.waypoint_interval,
                place_bounds=place_bounds,
                pick_weight=pick_w,
                place_weight=place_w,
                push_weight=push_w,
                tool_push_weight=tool_push_w,
                time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            )
            self._engine = TrajectoryEngine(
                self.env,
                primitives=self.primitives,
                config=config,
            )
            self._plan_next_action()
            print("  [Auto] Ready.")
        else:
            self._engine = None
            self._find_all_objects()
            self.trajectory = []
            self.tuner_state = "IDLE"
            print("  [Manual] Ready. TAB/G/T/H/Y/SPACE/ARROWS/ENTER.")

    def _plan_next_action(self) -> None:
        if self._engine is None:
            return

        if self._consecutive_failures >= 3:
            print(f"  [Auto] {self._consecutive_failures} consecutive failures. Forcing HOME recovery.")
            params = self.primitives.sample_home_parameters()
            # Generate home trajectory from current pose (default behavior of generate_home_trajectory)
            trajectory_steps = self.primitives.generate_home_trajectory(params)
            self._current_plan = PlannedAction(
                action_type=ActionType.HOME,
                parameters=params,
                trajectory_steps=trajectory_steps
            )
        else:
            plans = self._engine.plan_episode(steps=1)
            if plans:
                self._current_plan = plans[0]
            else:
                print("  [Warning] No action could be planned. Going to HOME.")
                params = self.primitives.sample_home_parameters()
                trajectory_steps = self.primitives.generate_home_trajectory(params)
                self._current_plan = PlannedAction(
                    action_type=ActionType.HOME,
                    parameters=params,
                    trajectory_steps=trajectory_steps,
                )

        if self._current_plan:
            if self.visualizer:
                trace = self._current_plan.to_trace()
                self.visualizer.clear()
                self.visualizer.visualize_trace(trace)
            print(f"  [Planned] {self._current_plan.action_type.value} action")

    def _execute_current_plan(self) -> None:
        if self._current_plan is None:
            return
        print(f"\n[Executing] {self._current_plan.action_type.value}")
        # Pass visualizer only if available
        result = self._engine.execute_action(self._current_plan)
        if self.visualizer:
            self.visualizer.set_result(result)
        if result.success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            
        self._total_steps += result.steps_taken
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.message}, steps: {result.steps_taken}")
        self._current_plan = None

    def _get_current_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        agent = self.env.unwrapped.agent
        pos, quat = sapien_pose_to_numpy(agent.tcp.pose)
        return pos.astype(np.float32), quat.astype(np.float32)

    # ----- Manual (tuner) helpers -----

    def _cycle_object(self) -> None:
        if not self.objects:
            return
        try:
            idx = self.objects.index(self.current_object)
            self.current_object = self.objects[(idx + 1) % len(self.objects)]
            print(f"Selected Object: {self.current_object.name}")
            self.trajectory = []
            if self.visualizer:
                self.visualizer.clear()
            self.tuner_state = "IDLE"
        except ValueError:
            self.current_object = self.objects[0]

    def _apply_modifiers(self, params: Dict, action_type: str) -> None:
        mods = self.param_modifiers[action_type]
        if action_type == "pick":
            delta_h = mods.get("approach_height", 0.0)
            if delta_h != 0:
                from scipy.spatial.transform import Rotation as R
                rot = R.from_quat(params["approach_quat"])
                z_axis = rot.as_matrix()[:, 2]
                params["approach_pos"] = params["approach_pos"] - z_axis * delta_h
                params["pregrasp_pos"] = params["pregrasp_pos"] - z_axis * delta_h
            delta_yaw = mods.get("grasp_yaw", 0.0)
            if delta_yaw != 0:
                from scipy.spatial.transform import Rotation as R
                rot_delta = R.from_euler("z", delta_yaw, degrees=False)
                base_rot = R.from_quat(params["approach_quat"])
                params["approach_quat"] = (base_rot * rot_delta).as_quat().astype(np.float32)
                base_rot_pg = R.from_quat(params["pregrasp_quat"])
                params["pregrasp_quat"] = (base_rot_pg * rot_delta).as_quat().astype(np.float32)
        elif action_type == "push":
            params["push_z"] = params.get("push_z", 0.05) + mods.get("push_z", 0.0)
            # Apply push_height_offset to primitives so generate_push_trajectory uses config + modifier
            if self.primitives is not None:
                self.primitives.push_height_offset = (
                    self.primitives.config.push_height_offset + mods.get("push_height_offset", 0.0)
                )
            delta_angle = mods.get("direction_angle", 0.0)
            if delta_angle != 0:
                d = params["direction"]
                c, s = np.cos(delta_angle), np.sin(delta_angle)
                params["direction"] = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2]], dtype=np.float32)
        elif action_type == "place":
            delta_z = mods.get("place_z", 0.0)
            if delta_z != 0:
                params["target_pos"][2] = params["target_pos"][2] + delta_z
            delta_yaw = mods.get("place_yaw", 0.0)
            if delta_yaw != 0:
                from scipy.spatial.transform import Rotation as R
                base_rot = R.from_quat(params["place_quat"])
                params["place_quat"] = (base_rot * R.from_euler("z", delta_yaw)).as_quat().astype(np.float32)

    def _sample_action(self) -> None:
        if self.current_object is None:
            print("No object found!")
            return
        bounds = (self.args.x_bounds[0], self.args.x_bounds[1], self.args.y_bounds[0], self.args.y_bounds[1])
        if self.selected_action == "pick":
            base_params = self.primitives.sample_pick_parameters(self.current_object)
            if base_params:
                self._apply_modifiers(base_params, "pick")
                self.current_params = base_params
                self.trajectory = self.primitives.generate_pick_trajectory(self.current_params)
                self.tuner_state = "READY_TO_EXECUTE"
                print(f"Sampled PICK. Steps: {len(self.trajectory)}")
            else:
                print("Failed to find valid grasp.")
        elif self.selected_action == "push":
            base_params = self.primitives.sample_push_parameters(self.current_object)
            self._apply_modifiers(base_params, "push")
            self.current_params = base_params
            self.trajectory = self.primitives.generate_push_trajectory(self.current_params)
            self.tuner_state = "READY_TO_EXECUTE"
            print(f"Sampled PUSH ({base_params.get('push_type')}). Steps: {len(self.trajectory)}")
        elif self.selected_action == "place":
            base_params = self.primitives.sample_place_parameters(bounds=bounds)
            self._apply_modifiers(base_params, "place")
            self.current_params = base_params
            self.trajectory = self.primitives.generate_place_trajectory(self.current_params)
            self.tuner_state = "READY_TO_EXECUTE"
            print(f"Sampled PLACE. Steps: {len(self.trajectory)}")
        elif self.selected_action == "home":
            self.current_params = self.primitives.sample_home_parameters()
            self.trajectory = self.primitives.generate_home_trajectory(self.current_params)
            self.tuner_state = "READY_TO_EXECUTE"
            print(f"Sampled HOME. Steps: {len(self.trajectory)}")
        self._visualize_trajectory()

    def _visualize_trajectory(self) -> None:
        if not self.visualizer:
            return
        self.visualizer.clear()
        if self.trajectory:
            trace = Trace.from_primitive_steps(self.trajectory, metadata={"source": "UnifiedPlay"})
            self.visualizer.visualize_trace(trace)

    def _execute_manual_action(self) -> None:
        if not self.trajectory:
            return
        print("Executing...")
        result = self.primitives.execute_trajectory(
            self.trajectory,
            self.kinematics,
            time_warp_speed_bounds=self.args.time_warp_speed_bounds,
            target_actor=self.current_object if self.selected_action == "pick" else None,
            action_name=self.selected_action,
        )
        if self.visualizer:
            self.visualizer.set_result(result)
        if result.success:
            if self.selected_action == "pick":
                self.primitives._held_object = self.current_object
            elif self.selected_action == "place":
                self.primitives._held_object = None
        print(f"Result: {result.message}")
        self.tuner_state = "IDLE"
        self.trajectory = []
        if self.visualizer:
            self.visualizer.clear()

    def _handle_tuning(self, viewer) -> None:
        changed = False
        step_trans = 0.01
        step_rot = np.radians(5)
        if viewer.window.key_press("up"):
            if self.selected_action == "pick":
                self.param_modifiers["pick"]["approach_height"] += step_trans
            elif self.selected_action == "push":
                self.param_modifiers["push"]["push_z"] += step_trans
            elif self.selected_action == "place":
                self.param_modifiers["place"]["place_z"] += step_trans
            changed = True
        if viewer.window.key_press("down"):
            if self.selected_action == "pick":
                self.param_modifiers["pick"]["approach_height"] -= step_trans
            elif self.selected_action == "push":
                self.param_modifiers["push"]["push_z"] -= step_trans
            elif self.selected_action == "place":
                self.param_modifiers["place"]["place_z"] -= step_trans
            changed = True
        # Push height offset (robot-level modifier): i = up, k = down
        if self.selected_action == "push":
            if viewer.window.key_press("i"):
                self.param_modifiers["push"]["push_height_offset"] += step_trans
                changed = True
            if viewer.window.key_press("k"):
                self.param_modifiers["push"]["push_height_offset"] -= step_trans
                changed = True
        if viewer.window.key_press("left"):
            if self.selected_action == "pick":
                self.param_modifiers["pick"]["grasp_yaw"] += step_rot
            elif self.selected_action == "push":
                self.param_modifiers["push"]["direction_angle"] += step_rot
            elif self.selected_action == "place":
                self.param_modifiers["place"]["place_yaw"] += step_rot
            changed = True
        if viewer.window.key_press("right"):
            if self.selected_action == "pick":
                self.param_modifiers["pick"]["grasp_yaw"] -= step_rot
            elif self.selected_action == "push":
                self.param_modifiers["push"]["direction_angle"] -= step_rot
            elif self.selected_action == "place":
                self.param_modifiers["place"]["place_yaw"] -= step_rot
            changed = True
        if changed and self.tuner_state == "READY_TO_EXECUTE":
            print(f"Modifiers: {self.param_modifiers[self.selected_action]}")
            self._sample_action()

    def _print_modifiers(self) -> None:
        print("\n[bold green]--- Current Modifiers ---[/bold green]")
        print(json.dumps(self.param_modifiers, indent=4))
        print("\n[dim]Copy these values to your primitives tuning section.[/dim]")
        print("-----------------------------")

    def run(self) -> None:
        print("\n--- Unified Play (auto + manual) ---")
        print("  Mode: " + self.args.mode)
        if self.args.mode == "auto":
             print(f"  Actions to execute: {self.args.max_actions}")
        else:
             print("  [A] Switch to Auto   [M] Switch to Manual")

        if self.args.mode == "auto" and not self.args.headless:
             print("  R - Reset   Q - Quit")
        elif self.args.mode == "manual":
            print("  R - Reset   TAB - Cycle object   G/T/H/Y - Pick/Push/Home/Place")
            print("  SPACE - Sample   ARROWS - Tune   I/K - Push height offset   ENTER - Execute   P - Print modifiers   Q - Quit")
        print("--------------------\n")

        # Get viewer if available (might be None in headless)
        viewer = self.env.unwrapped.viewer

        try:
            while True:
                # 1. Update visualization if needed
                if self.args.mode == "auto" and self.visualizer:
                    tcp_pos, tcp_quat = self._get_current_tcp_pose()
                    self.visualizer.update_tcp_pose(tcp_pos, tcp_quat)
                
                # 2. Render
                self.env.render()

                # 3. Handle Auto Mode Logic (Headless or Visual)
                if self.args.mode == "auto":
                    if self._current_plan:
                        if self._actions_executed < self.args.max_actions:
                            self._execute_current_plan()
                            self._actions_executed += 1
                            self._plan_next_action()
                        else: # Max actions reached
                             print(f"\n[Finished] Max actions ({self.args.max_actions}) reached.")
                             break
                    
                    # If headless, we just continue looping (or we could have a sleep to avoid busy loop if waiting for something, but here actions are blocking)
                    # If visual, we check for Q/R below
                
                # 4. Handle Window Interaction (only if viewer exists)
                if viewer is not None and viewer.window is not None:
                     if viewer.window.should_close:
                         break
                     
                     if viewer.window.key_press("q"):
                         break

                     if viewer.window.key_press("r"):
                          self._reset()
                          continue
                     
                     # Check mode switching only if manual/visual-auto allowed it (logic from original code kept for manual)
                     # In original code, auto mode allowed R/Q. Manual allowed switching.
                     # Here we allow switching if in manual.
                     if self.args.mode == "manual":
                         if viewer.window.key_press("a"):
                              self.args.mode = "auto"
                              self._reset()
                              continue
                     if viewer.window.key_press("m"): # Allow switching back to manual from auto if visual?
                          self.args.mode = "manual"
                          self._reset()
                          continue

                     # Manual mode specific keys
                     if self.args.mode == "manual":
                        if viewer.window.key_press("tab"):
                            self._cycle_object()
                        if viewer.window.key_press("g"):
                            self.selected_action = "pick"
                            print("Selected Action: PICK")
                            self._sample_action()
                        if viewer.window.key_press("t"):
                            self.selected_action = "push"
                            print("Selected Action: PUSH")
                            self._sample_action()
                        if viewer.window.key_press("h"):
                            self.selected_action = "home"
                            print("Selected Action: HOME")
                            self._sample_action()
                        if viewer.window.key_press("y"):
                            self.selected_action = "place"
                            print("Selected Action: PLACE")
                            self._sample_action()
                        if viewer.window.key_press(" "):
                            self._sample_action()
                        if viewer.window.key_press("enter"):
                            if self.tuner_state == "READY_TO_EXECUTE":
                                self._execute_manual_action()
                            else:
                                print("Sample an action first (SPACE/G/T).")
                        if viewer.window.key_press("p"):
                            self._print_modifiers()
                        self._handle_tuning(viewer)
                        # Active marker logic
                        if self.current_object:
                            pos, quat = get_actor_world_pose(self.current_object)
                            if self._active_marker is None:
                                self._active_marker = self.env.unwrapped.viewer.add_coordinate_frame(
                                    sapien.Pose(p=pos, q=quat), length=0.2, radius=0.005
                                )
                            else:
                                self._active_marker.set_position(pos)
                                self._active_marker.set_rotation(quat)
                        elif self._active_marker:
                            self._active_marker.set_position(HIDDEN_POS)
                elif self.args.headless and self.args.mode == "manual":
                     # Headless manual mode not really supported/useful, just break
                     print("Manual mode requires a viewer.")
                     break


        finally:
            if self.args.mode == "auto":
                print(f"\nSession steps: {self._total_steps}")
            if self.visualizer:
                self.visualizer.destroy()
            self.env.close()


def main(args: UnifiedArgs) -> None:
    """CLI entry point. Use --mode auto for engine-driven play, --mode manual (default) for tuning."""
    print(f"Creating environment: {args.scene}, robot={args.robot}, num_distractors={args.num_distractors}, mode={args.mode}")
    UnifiedPlay(args).run()


if __name__ == "__main__":
    main(tyro.cli(UnifiedArgs))
