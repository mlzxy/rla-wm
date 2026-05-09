#!/bin/bash
set -e
ROBOT="${1:-panda}"
case "$ROBOT" in
    panda|xarm|ur10e) ;;
    *) echo "Usage: $0 [panda|xarm|ur10e]"; exit 1 ;;
esac

export EVAL_ROBOT="$ROBOT"
export PYTHONUNBUFFERED=1

TMPDIR=/tmp .venv/bin/python eval/eval_wrapper.py run-eval \
    --config "configs/rla_wm/${ROBOT}.yaml" \
    --cache-path "data/eval_handles/maniskill/handles.${ROBOT}.json" \
    --module-path "eval/predictors/rla_wm_predictor.py" \
    --output-dir "runs/eval_output/${ROBOT}" \
    --seed 2026 --fail-fast "$@"
