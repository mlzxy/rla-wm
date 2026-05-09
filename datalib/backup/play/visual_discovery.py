import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import cv2
import os
import time
from datalib.play.geometry import sample_interaction_points
from datalib.play.primitives import InteractionPrimitives
import datalib.play.play_env
from mani_skill.utils.structs.types import Array, SimConfig, SceneConfig
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-traces", type=int, default=100, help="Number of interactions to perform")
    parser.add_argument("--headless", action="store_true", help="Save video instead of showing window")
    parser.add_argument("--out-dir", type=str, default="data/discovery_vis")
    args = parser.parse_args()

    print("Initializing Infinite Discovery Visualization...")
    sim_config = SimConfig(scene_config=SceneConfig(enable_ccd=True))
    env = gym.make(
        "PlayEnv-v0",
        num_play_objects=12,
        render_mode="rgb_array", 
        robot_uids="panda",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        sim_config=sim_config
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    video_writer = None
    
    if not args.headless:
        cv2.namedWindow("Autonomous Interaction Discovery", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Autonomous Interaction Discovery", 1024, 1024)

    current_action_text = "Idle"
    
    def render_callback():
        nonlocal current_action_text, video_writer
        frame = env.render()
        if isinstance(frame, list): frame = frame[0]
        if torch.is_tensor(frame):
            frame = frame.cpu().numpy()
        
        # ManiSkill 3 often returns (N, H, W, 3) for rgb_array
        if frame.ndim == 4:
            frame = frame[0]
            
        if frame.ndim != 3:
            print(f"Warning: Unexpected frame shape {frame.shape}")
            return

        # Frame is RGB from ManiSkill
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add Overlay - Professional Header
        h, w = frame_bgr.shape[:2]
        cv2.rectangle(frame_bgr, (0, 0), (w, 40), (45, 45, 45), -1)
        # Glow effect for text
        cv2.putText(frame_bgr, f"AID SYSTEM | ACTIVE DISCOVERY | TASK: {current_action_text}", (20, 27), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"AID SYSTEM | ACTIVE DISCOVERY | TASK: {current_action_text}", (20, 27), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        
        if args.headless:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(args.out_dir, f"discovery_{int(time.time())}.mp4")
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                print(f"Recording to {video_path}")
            video_writer.write(frame_bgr)
        else:
            cv2.imshow("Autonomous Interaction Discovery", frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
                env.close()
                os._exit(0)

    primitives = InteractionPrimitives(env, step_callback=render_callback)
    
    print("Starting interaction loop. Press ESC on the window to exit.")
    
    primitive_types = ["push", "rotate", "poke", "slide", "pick_place", "flip", "stack", "tool_push"]
    
    traces_done = 0
    while traces_done < args.num_traces:
        env.reset()
        current_action_text = "RESETTING SCENE"
        for _ in range(30): # More settling time
            env.step(torch.zeros((1, env.action_space.shape[-1]), device=env.device))
            render_callback()
            
        ptype = np.random.choice(primitive_types)
        current_action_text = ptype.upper()
        
        try:
            actors = env.unwrapped.interactive_objects
            if ptype in ["stack", "tool_push"]:
                if len(actors) < 2: continue
                a, b = np.random.choice(actors, 2, replace=False)
                current_action_text = f"{ptype.upper()}: {a.name} -> {b.name}"
                if ptype == "stack": 
                    primitives.stack(a, b)
                else: 
                    primitives.tool_push(a, b)
            else:
                actor = np.random.choice(actors)
                current_action_text = f"{ptype.upper()}: {actor.name}"
                if ptype in ["push", "poke", "slide"]:
                    points, normals = sample_interaction_points(actor, num_samples=20)
                    if len(points) > 0:
                        idx = np.random.randint(len(points))
                        if ptype == "push": primitives.push(actor, points[idx], normals[idx])
                        elif ptype == "poke": primitives.poke(actor, points[idx], normals[idx])
                        else: primitives.slide(actor, points[idx], normals[idx])
                elif ptype == "rotate":
                    primitives.rotate(actor)
                elif ptype == "pick_place":
                    if primitives.pick(actor):
                        target_xy = np.random.uniform(-0.2, 0.2, 2)
                        primitives.place(target_xy)
                elif ptype == "flip":
                    primitives.flip(actor)
        except Exception as e:
            print(f"Loop error: {e}")
            current_action_text = f"ERROR: {str(e)[:20]}"
        
        traces_done += 1
        time.sleep(0.2)
    
    if video_writer:
        video_writer.release()
    env.close()

if __name__ == "__main__":
    main()
