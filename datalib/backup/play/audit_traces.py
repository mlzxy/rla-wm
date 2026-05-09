import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import os
import json
import argparse
import time
import datalib.play.play_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # 1. Load data
    action_file = os.path.join(args.trace_dir, "action_trace.json")
    meta_file = os.path.join(args.trace_dir, "metadata.json")
    
    if not os.path.exists(action_file):
        print(f"Error: {action_file} not found")
        return

    with open(action_file, "r") as f:
        actions = json.load(f)
    
    with open(meta_file, "r") as f:
        metadata = json.load(f)
        
    print(f"Replaying trace {metadata['trace_id']} on {metadata['target_actor']}")
    print(f"Primitive: {metadata['primitive_type']}")
    print(f"Total actions: {len(actions)}")

    # 2. Setup environment
    env = gym.make(
        "PlayEnv-v0",
        num_play_objects=5,
        render_mode="rgb_array" if args.headless else "human",
        control_mode="pd_ee_delta_pose",
        num_envs=1
    )
    
    obs, _ = env.reset()
    
    # 3. Replay
    # Note: For exact same result, we'd need to restore all object poses from metadata
    # if they were saved. 
    
    try:
        if args.headless:
            import matplotlib.pyplot as plt
            frames = []
        
        for i, action in enumerate(actions):
            action_tensor = torch.from_numpy(np.array(action)).to(env.device)
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            
            if args.headless and i % 10 == 0:
                 img = env.render()
                 if isinstance(img, torch.Tensor): img = img.cpu().numpy()
                 frames.append(img[0] if len(img.shape) == 4 else img)
            elif not args.headless:
                 env.render()
                 time.sleep(0.01)

        if args.headless and len(frames) > 0:
            plt.imsave(f"audit_replay_{metadata['trace_id']}.png", frames[-1])
            print(f"Saved audit_replay_{metadata['trace_id']}.png")

    except Exception as e:
        print(f"Replay error: {e}")
    finally:
        print("Replay finished.")
        env.close()

if __name__ == "__main__":
    main()
