import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs
from rich import print
from . import tasks  # Register all v2 tasks
from . import robots
from datalib.src.camera_dashboard import CameraConfigPlugin


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="PushT-v2",
        help="Task ID (e.g., PushT-v2, RollBall-v2, etc.)",
    )
    parser.add_argument(
        "-r", "--robot", type=str, default="ur10e_stick"
    )  # panda, xarm6_robotiq, ur10e_allegro
    parser.add_argument(
        "-o", "--obs-mode", type=str, default="state+rgb+depth+segmentation"
    )
    parser.add_argument("-s", "--static", action="store_true")
    parser.add_argument("-cw", "--camera-width", type=int, default=512)
    parser.add_argument("-ch", "--camera-height", type=int, default=512)
    parser.add_argument("-sh", "--shader", type=str, default="default")
    parser.add_argument("-d", "--num-distractors", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(
        args.task,
        robot_uids=args.robot,
        obs_mode=args.obs_mode,
        render_mode="human",
        control_mode="pd_joint_delta_pos",
        max_episode_steps=1e10,
        num_distractors=args.num_distractors,
        distractor_types=["cube", "sphere"],
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        shader_dir=args.shader,
        include_all_cameras=True,
    )

    print(f"Initialized Unified Workspace with task: {args.task}, robot: {args.robot}")
    print(f"[bold green]Action Space: {env.action_space}[/bold green]")
    print(f"[bold green]Control Mode: {env.control_mode}[/bold green]")
    print("\nManual Control Keys:")
    print("  [O]: Open Gripper")
    print("  [C]: Close Gripper")
    print("  [W/S/A/D]: XY Plane Movement")
    print("  [R/F]: Z Axis Movement")
    print("  [Space]: Reset Environment")
    print("  [ESC]: Quit")

    obs, _ = env.reset()
    terminated = truncated = False
    
    # Plugin installed flag
    plugin_installed = False

    while True:
        try:
            env.render()
            
            # Install dashboard plugin once viewer is available
            if not plugin_installed:
                viewer = None
                if hasattr(env.unwrapped, "_viewer"):
                    viewer = env.unwrapped._viewer
                elif hasattr(env.unwrapped, "viewer"):
                    viewer = env.unwrapped.viewer

                if viewer:
                    # Append plugin to viewer's plugin list
                    if hasattr(viewer, "plugins") and isinstance(viewer.plugins, list):
                        plugin = CameraConfigPlugin(env)
                        if hasattr(plugin, "init"):
                            plugin.init(viewer)
                        viewer.plugins.append(plugin)
                        plugin_installed = True
                    elif hasattr(viewer, "install"):
                        viewer.install(CameraConfigPlugin(env))
                        plugin_installed = True

                if not plugin_installed:
                    # Plugin failed to install or viewer not found
                    pass
                
        except Exception as e:
            # Use standard print to avoid Rich MarkupError with shader paths
            import builtins
            builtins.print(f"Render failed or window closed: {e}")
            break

        if args.static:
            action = np.zeros(env.action_space.shape)
            if 'stick' not in args.robot and 'close' not in args.robot:
                action[-1] = -1 if 'xarm6' in args.robot else 1
        else:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
