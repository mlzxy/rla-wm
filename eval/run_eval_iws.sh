#!/bin/bash
set -e
SCENE="${1:-pusht}"
case "$SCENE" in
    pusht|box|rope) ;;
    *) echo "Usage: $0 [pusht|box|rope]"; exit 1 ;;
esac

export EVAL_SCENE="$SCENE"
export PYTHONUNBUFFERED=1

shift 1 2>/dev/null || true

TMPDIR=/tmp .venv/bin/python eval/eval_wrapper.py run-eval \
    --config "configs/rla_wm/iws_${SCENE}.yaml" \
    --cache-path "data/eval_handles/iws/handles.${SCENE}.json" \
    --module-path "eval/predictors/rla_wm_predictor_iws.py" \
    --output-dir "runs/eval_output/iws_${SCENE}" \
    --seed 2026 --fail-fast "$@"
