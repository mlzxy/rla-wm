#!/usr/bin/env bash
# WMRL v2 launcher. Same interface as wmrl/train.sh, but:
#   * reads rebuttal/exp1_wmrl_v2/configs/<task>.yaml   (BC initializers re-selected on the
#     offline sweep instead of the noisy train-time score)
#   * --no-eval: no in-training evaluation. The 50-seed eval is not the metric
#     we report; the offline sweep (rebuttal/tools/eval_ckpts.py) is.
#   * writes to runs/wmrl_v2/, so nothing under runs/wmrl_critic is touched
#
# Usage:
#   bash rebuttal/exp1_wmrl_v2/train.sh <task> <reward_mode> <seed1> [seed2 ...]
#
#   bash rebuttal/exp1_wmrl_v2/train.sh pokecube goal          1 2 3     # PokeCube uses goal
#   bash rebuttal/exp1_wmrl_v2/train.sh pusht    corresponding 1 2 3     # everything else
#
# Seeds run sequentially, one process each. Extra flags pass through `flags`:
#   flags="--total-iterations 1 --debug" bash rebuttal/exp1_wmrl_v2/train.sh pusht corresponding 1
#
# Output: runs/wmrl_v2/<task>-<task>-v2-<corr|goal>-lora-seed<N>-run0/<timestamp>/
#           config.yaml  ckpt_{25,50,...,300}.pt  final.pt
set -x
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"   # REBUTTAL_REPO / REBUTTAL_PY / PYTHONPATH
cd "$REBUTTAL_REPO"

task=${1}
reward=${2}
rewardshortname=${reward:0:4}
flags=${flags:-""}
cfg=rebuttal/exp1_wmrl_v2/configs/${task}.yaml

if [ ! -f "${cfg}" ]; then
    echo "no such config: ${cfg}  (expected one of: $(ls rebuttal/exp1_wmrl_v2/configs/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml//' | tr '\n' ' '))" >&2
    exit 1
fi

echo "additional flags: ${flags}"

shift 2

for seed in $@; do
    echo "Running seed $seed";
    "$REBUTTAL_PY" wmrl/train.py --tag "${task}-v2-${rewardshortname}-lora-seed${seed}" --env-kwargs.reward-mode ${reward} --env-kwargs.terminal-success-bonus 0.0 --config-file ${cfg} --total-iterations 300 --no-run-initial-eval --no-eval --bc-loss-weight 0.0 --num-envs 112  --num-steps 4  --mini-batch-size 224 --seed $seed  --use-critic --policy-kwargs.enable-rl-lora  ${flags}
done
