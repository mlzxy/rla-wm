"""
Hydra entry point for VLA Diffusion Policy training on v4world.

Usage:
    python policies/train_policy.py --config-name=train_vla_diffusion
    python policies/train_policy.py --config-name=train_vla_diffusion task=PullCube-v2 robot=panda
    python policies/train_policy.py --config-name=train_vla_diffusion training.debug=true
"""

import sys
import os

# Ensure project root is on sys.path so that modules like `policies.*`,
# `src.*`, `utils.*`, `datalib.*` are importable regardless of hydra's
# working-directory changes.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Use line-buffered stdout/stderr (same as atomic_policy)
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib  # noqa: E402

import hydra  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from utils.misc import pretty_print_config  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    if cfg.training.debug:
        cfg.dataloader.num_workers = 0
        cfg.eval.num_episodes = 1
        cfg.logging.enable = False

        if "task_configs" in cfg.dataset:
            for tc in cfg.dataset.task_configs:
                tc.end_traj_id = 10
        elif "robot_dataset_cfg" in cfg.dataset:
            cfg.dataset.robot_dataset_cfg.end_traj_id = min(10, cfg.dataset.robot_dataset_cfg.end_traj_id)
            if "pixel_dataset_cfg" in cfg.dataset and cfg.dataset.pixel_dataset_cfg is not None:
                cfg.dataset.pixel_dataset_cfg.end_traj_id = min(10, cfg.dataset.pixel_dataset_cfg.end_traj_id)
        else:
            cfg.dataset.end_traj_id = min(10, cfg.dataset.end_traj_id)
            
    cls = hydra.utils.get_class(cfg._target_)
    pretty_print_config(cfg)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()



# a more complete version that consider cross tasks
# if cfg.robot == 'ur10e_stick':
#     cfg.action_dim = 5
#     cfg.state_dim = 6
# elif cfg.robot == 'panda':
#     cfg.action_dim = 8
#     if cfg.task in ['PushT-v2', 'RollBall-v1']:
#         cfg.robot = 'panda_closed'
#         cfg.action_dim = 7
#     cfg.state_dim = 9
# elif cfg.robot == 'xarm6_robotiq':
#     cfg.action_dim = 7
#     if cfg.task in ['PushT-v2', 'RollBall-v1']:
#         cfg.robot = 'xarm6_robotiq_closed'
#         cfg.action_dim = 6
#     cfg.state_dim = 12