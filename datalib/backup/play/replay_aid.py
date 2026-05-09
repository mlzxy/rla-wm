import torch
import json
import os
import argparse
from datalib.play.play_env import PlayEnv

def replay_trajectory(log_dir: str):
    """
    Reconstructs an interaction trajectory from an action trace.
    Used for visual auditing and data validation.
    """
    # Load metadata
    with open(os.path.join(log_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Load action trace
    with open(os.path.join(log_dir, "action_trace.json"), "r") as f:
        action_trace = json.load(f)
        
    print(f"Replaying trajectory from {log_dir}")
    print(f"Agent: {metadata['robot_uids']}")
    
    # Initialize environment with SAME parameters
    # Note: We must ensure the same robot and spawn logic is used.
    env = PlayEnv(
        render_mode="human",
        robot_uids=metadata["robot_uids"],
        control_mode=metadata["control_mode"],
        num_play_objects=metadata["num_play_objects"]
    )
    
    # Set seed to recover identical object shapes/spawns
    env.reset(seed=metadata["seed"])
    
    # Replay actions
    for action_list in action_trace:
        # Convert list back to tensor
        action = torch.tensor(action_list, device=env.device, dtype=torch.float32)
        env.step(action)
        env.render()
        
    print("Replay finished.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the interaction log directory")
    args = parser.parse_args()
    
    replay_trajectory(args.log_dir)
