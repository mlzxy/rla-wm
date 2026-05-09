import os
import tyro
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any, Tuple
from rich import print
from datalib.dataset import ManiSkillTrajectoryDataset, TrajectoryData

@dataclass
class PPOConfig:
    """Arguments for PPO data collection."""
    budget_success: int = 10
    """Number of successful trajectories to collect"""
    budget_failure: int = 0
    """Number of unsuccessful trajectories to collect"""
    checkpoints: Optional[List[str]] = None
    """List of checkpoint paths to sample from. If None, uses pool search."""

@dataclass
class PlayConfig:
    """Arguments for Play (heuristic) data collection."""
    budget: int = 10
    """Number of trajectories to collect"""
    step_interval: int = 1
    """Subsampling interval to align frequencies (1 means no subsampling)"""

@dataclass
class CollectDataArgs:
    """Unified Data Collection Script for PPO and Play agents."""
    mode: Literal["ppo", "play"]
    """Collection mode: ppo or play"""
    env_id: str = "PushT-v1"
    """ManiSkill environment ID"""
    output_dir: str = "data/collected"
    """Directory to save the dataset"""
    seed: int = 42
    """Random seed"""
    device: str = "cuda"
    """Computation device (cuda/cpu)"""
    render_size: Tuple[int, int] = (128, 128)
    """Camera render size (h, w)"""
    
    # Sub-configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    play: PlayConfig = field(default_factory=PlayConfig)

def collect_ppo_data(args: CollectDataArgs, dataset: ManiSkillTrajectoryDataset):
    """Placeholder for PPO data collection."""
    print(f"[bold blue]PPO Mode[/bold blue]: Collecting {args.ppo.budget_success} successes for {args.env_id}")
    
    # Phase 1: Dummy trajectory for verification
    dummy_data = TrajectoryData(
        success=True,
        video_streams={},
        metadata={"mode": "ppo_dummy"}
    )
    dataset.write_trajectory("ppo_dummy_0", dummy_data)
    print(f"  Saved dummy PPO trajectory to {args.output_dir}/traj_ppo_dummy_0")
    return True

def collect_play_data(args: CollectDataArgs, dataset: ManiSkillTrajectoryDataset):
    """Placeholder for Play data collection."""
    print(f"[bold green]Play Mode[/bold green]: Collecting {args.play.budget} trajectories for {args.env_id}")
    
    # Phase 1: Dummy trajectory for verification
    dummy_data = TrajectoryData(
        success=True,
        video_streams={},
        metadata={"mode": "play_dummy"}
    )
    dataset.write_trajectory("play_dummy_0", dummy_data)
    print(f"  Saved dummy Play trajectory to {args.output_dir}/traj_play_dummy_0")
    return True

def main():
    args = tyro.cli(CollectDataArgs)
    
    # Initialize Dataset
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ManiSkillTrajectoryDataset(args.output_dir)
    
    if args.mode == "ppo":
        collect_ppo_data(args, dataset)
    elif args.mode == "play":
        collect_play_data(args, dataset)

if __name__ == "__main__":
    # Needed for tyro to parse Tuple[int, int]
    main()
