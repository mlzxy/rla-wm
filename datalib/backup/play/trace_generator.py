import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import os
import argparse
import shutil
from datalib.play.geometry import sample_interaction_points
from datalib.play.primitives import InteractionPrimitives
from datalib.play.hybrid_logger import HybridLogger
import datalib.play.play_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-traces", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="data/play_traces")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.num_traces} traces into {args.out_dir}...")

    # We use a single env for the rule-based generator for better control
    env = gym.make(
        "PlayEnv-v0",
        num_play_objects=10,
        render_mode="rgb_array" if not args.render else "human",
        control_mode="pd_ee_delta_pose",
        num_envs=1
    )
    
    primitives = InteractionPrimitives(env)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    traces_completed = 0
    primitive_types = ["push", "rotate", "poke", "slide", "pick_place", "flip", "stack", "tool_push"]
    
    while traces_completed < args.num_traces:
        obs, _ = env.reset()
        env_idx = 0 
        
        # Wait for objects to settle
        for _ in range(30):
            env.step(torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device))
        
        # Select random primitive
        primitive_type = np.random.choice(primitive_types)
        
        # Create logger for this attempt
        trace_id = f"{traces_completed}_{primitive_type}"
        log_dir = os.path.join(args.out_dir, f"trace_{trace_id}")
        logger = HybridLogger(log_dir, num_envs=1)
        
        # Execute interaction
        try:
            actors = env.unwrapped.interactive_objects
            if primitive_type in ["stack", "tool_push"]:
                if len(actors) < 2: continue
                a, b = np.random.choice(actors, 2, replace=False)
                logger.log_metadata({
                    "trace_id": traces_completed,
                    "target_actors": [a.name, b.name],
                    "primitive_type": primitive_type,
                })
                if primitive_type == "stack":
                    success = primitives.stack(a, b, env_idx=env_idx, logger=logger)
                else:
                    success = primitives.tool_push(a, b, env_idx=env_idx, logger=logger)
            else:
                actor = np.random.choice(actors)
                logger.log_metadata({
                    "trace_id": traces_completed,
                    "target_actor": actor.name,
                    "primitive_type": primitive_type,
                })
                
                if primitive_type in ["push", "poke", "slide"]:
                    points, normals = sample_interaction_points(actor, num_samples=20)
                    if len(points) == 0:
                        shutil.rmtree(log_dir)
                        continue
                    pix = np.random.randint(len(points))
                    p, n = points[pix], normals[pix]
                    
                    if primitive_type == "push":
                        success = primitives.push(actor, p, n, env_idx=env_idx, logger=logger)
                    elif primitive_type == "poke":
                        success = primitives.poke(actor, p, n, env_idx=env_idx, logger=logger)
                    else: # slide
                        success = primitives.slide(actor, p, n, env_idx=env_idx, logger=logger)
                elif primitive_type == "rotate":
                    success = primitives.rotate(actor, env_idx=env_idx, logger=logger)
                elif primitive_type == "pick_place":
                    success = primitives.pick(actor, env_idx=env_idx, logger=logger)
                    if success:
                        # Place at a random workspace location
                        target_xy = np.random.uniform(-0.2, 0.2, 2)
                        success = primitives.place(target_xy, env_idx=env_idx, logger=logger)
                elif primitive_type == "flip":
                    success = primitives.flip(actor, env_idx=env_idx, logger=logger)
        except Exception as e:
            print(f"Execution error: {e}")
            success = False
            
        if success:
            logger.save()
            traces_completed += 1
            if traces_completed % 10 == 0:
                print(f"--- Progress: {traces_completed}/{args.num_traces} traces ---")
        else:
            # Clean up failed attempts
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            # print("Interaction failed, retrying...")

    print(f"Dataset generation complete. {traces_completed} traces saved.")
    env.close()

if __name__ == "__main__":
    main()
