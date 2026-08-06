#!/usr/bin/env bash
# Download VLA-Adapter checkpoints and LIBERO datasets into this repo's runs/ and data/ trees.
#
#   rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh            # ckpts (default) -- all that eval needs
#   rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh ckpts      # the paper's original checkpoints
#   rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh pro        # the enhanced "Pro" checkpoints
#   rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh backbone   # Prismatic VLM, for fine-tuning
#   rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh rlds       # LIBERO RLDS training data
#   rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh all
#
# Resumable and idempotent (`hf download` skips files it already has).
#
# Layout produced:
#   runs/weights/vla-adapter/LIBERO-{Spatial,Object,Goal,Long}/     ~2.7 GB each
#   runs/weights/vla-adapter/prism-qwen25-extra-dinosiglip-224px-0_5b/  ~2.6 GB
#   data/libero/libero_{spatial,object,goal,10}_no_noops/1.0.0/     ~10.2 GB total
#
# NOTE: running an eval rewrites config.json inside a checkpoint directory (update_auto_map) and
# copies the repo's modeling_prismatic.py in (check_model_logic_mismatch). Re-running this script
# restores upstream's versions of those files, which is harmless -- the eval redoes both on startup.
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config.sh"
cd "$REBUTTAL_REPO"

HF="$(dirname "$REBUTTAL_PY")/hf"
WEIGHTS_DIR="$REBUTTAL_WEIGHTS"
DATA_DIR="$REBUTTAL_REPO/data/libero"

# The paper's released (non-Pro) LIBERO checkpoints.
SUITES=(Spatial Object Goal Long)
# Only the final backbone checkpoint -- the repo also holds two earlier ones at 2.6 GB each.
BACKBONE_REPO="Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b"
BACKBONE_CKPT="checkpoints/step-020792-epoch-01-loss=0.5268.pt"
RLDS_REPO="openvla/modified_libero_rlds"
RLDS_SPLITS=(libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops libero_10_no_noops)

WHAT="${1:-ckpts}"
log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[ -x "$HF" ] || { echo "missing $HF -- run 'uv sync' first" >&2; exit 1; }

download_ckpts() {
    log "VLA-Adapter LIBERO checkpoints -> $WEIGHTS_DIR"
    mkdir -p "$WEIGHTS_DIR"
    local suite
    for suite in "${SUITES[@]}"; do
        echo "--- VLA-Adapter/LIBERO-$suite"
        "$HF" download "VLA-Adapter/LIBERO-$suite" --local-dir "$WEIGHTS_DIR/LIBERO-$suite"
    done
}

download_pro() {
    log "VLA-Adapter LIBERO *Pro* checkpoints -> $WEIGHTS_DIR"
    mkdir -p "$WEIGHTS_DIR"
    local suite
    for suite in "${SUITES[@]}"; do
        echo "--- VLA-Adapter/LIBERO-$suite-Pro"
        "$HF" download "VLA-Adapter/LIBERO-$suite-Pro" --local-dir "$WEIGHTS_DIR/LIBERO-$suite-Pro"
    done
}

download_backbone() {
    log "Prismatic VLM backbone -> $WEIGHTS_DIR"
    mkdir -p "$WEIGHTS_DIR"
    "$HF" download "$BACKBONE_REPO" \
        --local-dir "$WEIGHTS_DIR/prism-qwen25-extra-dinosiglip-224px-0_5b" \
        --include "config.json" "config.yaml" "$BACKBONE_CKPT"
}

download_rlds() {
    log "LIBERO RLDS training data -> $DATA_DIR"
    mkdir -p "$DATA_DIR"
    local split
    for split in "${RLDS_SPLITS[@]}"; do
        echo "--- $split"
        "$HF" download "$RLDS_REPO" --repo-type dataset --local-dir "$DATA_DIR" --include "$split/*"
    done
}

case "$WHAT" in
    ckpts)    download_ckpts ;;
    pro)      download_pro ;;
    backbone) download_backbone ;;
    rlds)     download_rlds ;;
    all)      download_ckpts; download_pro; download_backbone; download_rlds ;;
    *) echo "usage: $0 [ckpts|pro|backbone|rlds|all]" >&2; exit 2 ;;
esac

log "done"
du -sh "$WEIGHTS_DIR"/* "$DATA_DIR"/* 2>/dev/null || true
