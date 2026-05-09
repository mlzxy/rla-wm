PY=.venv/bin/python
mode=$1

robot=$2 # 0 1 2
envid=$3 # 0-5
trajs=$4

# robot_list=("panda" "ur10e_stick" "xarm6_robotiq" "panda_closed" "xarm6_robotiq_closed")
# env_list=("PushT-v2" "RollBall-v1" "PokeCube-v2" "PullCubeTool-v1" "PullCube-v2" "PegInsertionSide-v1")


# # in distribution

# CUDA_VISIBLE_DEVICES=0 bash datalib/collect_traj.sh ppo panda PullCube-v2 1000
# CUDA_VISIBLE_DEVICES=1 bash datalib/collect_traj.sh ppo panda PullCubeTool-v1 1000
# CUDA_VISIBLE_DEVICES=2 bash datalib/collect_traj.sh ppo xarm6_robotiq PokeCube-v2 1000
# CUDA_VISIBLE_DEVICES=3 bash datalib/collect_traj.sh ppo xarm6_robotiq PegInsertionSide-v1 1000
# CUDA_VISIBLE_DEVICES=0 bash datalib/collect_traj.sh ppo ur10e_stick PushT-v2 1000
# CUDA_VISIBLE_DEVICES=1 bash datalib/collect_traj.sh ppo ur10e_stick RollBall-v1 1000


# # play data
# CUDA_VISIBLE_DEVICES=0 bash datalib/collect_traj.sh play panda PushT-v2 1000   --play-actions pick_and_place
# CUDA_VISIBLE_DEVICES=1 bash datalib/collect_traj.sh play panda PushT-v2 1000   --play-actions tool_push
# CUDA_VISIBLE_DEVICES=2 bash datalib/collect_traj.sh play panda PushT-v2 1000   --play-actions push_only
# CUDA_VISIBLE_DEVICES=3 bash datalib/collect_traj.sh play xarm6_robotiq PushT-v2 1000   --play-actions pick_and_place
# CUDA_VISIBLE_DEVICES=0 bash datalib/collect_traj.sh play xarm6_robotiq PushT-v2 1000   --play-actions tool_push
# CUDA_VISIBLE_DEVICES=1 bash datalib/collect_traj.sh play xarm6_robotiq PushT-v2 1000   --play-actions push_only
# CUDA_VISIBLE_DEVICES=2 bash datalib/collect_traj.sh play ur10e_stick PushT-v2 1000

# # cross-embodiment

# - PullCube-v2
# CUDA_VISIBLE_DEVICES=0 bash datalib/collect_traj.sh ppo ur10e_stick PullCube-v2 25
# CUDA_VISIBLE_DEVICES=1 bash datalib/collect_traj.sh ppo xarm6_robotiq PullCube-v2 25

# - PullCubeTool-v1
# CUDA_VISIBLE_DEVICES=2 bash datalib/collect_traj.sh ppo xarm6_robotiq PullCubeTool-v1 25

# - PokeCube-v2
# CUDA_VISIBLE_DEVICES=3 bash datalib/collect_traj.sh ppo panda PokeCube-v2 25

# - PegInsertionSide-v1
# CUDA_VISIBLE_DEVICES=0 bash datalib/collect_traj.sh ppo panda PegInsertionSide-v1 25

# - PushT-v2
# CUDA_VISIBLE_DEVICES=1 bash datalib/collect_traj.sh ppo xarm6_robotiq_closed PushT-v2 25
# CUDA_VISIBLE_DEVICES=2 bash datalib/collect_traj.sh ppo panda_closed PushT-v2 25

# - RollBall-v1
# CUDA_VISIBLE_DEVICES=3 bash datalib/collect_traj.sh ppo xarm6_robotiq_closed RollBall-v1 25
# CUDA_VISIBLE_DEVICES=3 bash datalib/collect_traj.sh ppo panda_closed RollBall-v1 25



shift 4

set -xe
$PY datalib/collect_all.py --mode $mode --robot $robot --env_id $envid --force-headless --num-trajectories $trajs $@
