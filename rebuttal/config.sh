# Sourced by every shell script under rebuttal/. Sets the four things that used to
# be hard-coded to the authors' machine. Everything is overridable from the outside:
#
#   REBUTTAL_REPO=/my/checkout bash rebuttal/exp1_wmrl_v2/train.sh pusht corresponding 1
#
# The repo root is derived from this file's own location, so it is correct no matter
# where you cloned to and no matter which directory you launch from.

: "${REBUTTAL_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${REBUTTAL_PY:=$REBUTTAL_REPO/.venv/bin/python}"

# Where exp-2 keeps VLA-Adapter weights and eval logs. Only exp-2 reads these.
: "${REBUTTAL_WEIGHTS:=$REBUTTAL_REPO/runs/weights/vla-adapter}"
: "${REBUTTAL_EVAL_LOGS:=$REBUTTAL_REPO/runs/vla-adapter-eval}"

export REBUTTAL_REPO REBUTTAL_PY REBUTTAL_WEIGHTS REBUTTAL_EVAL_LOGS
export PYTHONPATH="$REBUTTAL_REPO:$REBUTTAL_REPO/third_party/diffusion_policy${PYTHONPATH:+:$PYTHONPATH}"
