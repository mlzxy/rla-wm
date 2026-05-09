set -x
export PYTHONPATH=.:./third_party/diffusion_policy:$PYTHONPATH

task=${1}
reward=${2}
rewardshortname=${reward:0:4}
flags=${flags:-""}

echo "additional flags: ${flags}"

shift 2

for seed in $@; do
    echo "Running seed $seed";
    .venv/bin/python wmrl/train.py --tag "${task}-apr22-${rewardshortname}-lora-seed${seed}" --env-kwargs.reward-mode ${reward} --env-kwargs.terminal-success-bonus 0.0 --config-file wmrl/configs/${task}.yaml --total-iterations 300 --no-run-initial-eval --bc-loss-weight 0.0 --num-envs 112  --num-steps 4  --mini-batch-size 224 --seed $seed  --use-critic --policy-kwargs.enable-rl-lora  ${flags}
done

