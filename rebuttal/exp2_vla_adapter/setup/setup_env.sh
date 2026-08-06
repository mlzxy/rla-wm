#!/usr/bin/env bash
# Build VLA-Adapter in a directory and a virtualenv of its own, on upstream's pinned stack.
# This is where both arms in ../results/ were trained -- upstream's code, upstream's versions.
#
#   rebuttal/exp2_vla_adapter/setup/setup_env.sh              # clone + venv + deps + LIBERO + links
#   REF_DIR=/somewhere/else rebuttal/exp2_vla_adapter/setup/setup_env.sh
#
# Then copy ../vla_adapter_overlay/{rla,tools} into the clone and train with tools/train_eval.sh
# (baseline) or tools/train_eval_rla.sh (+ RLA). See ../README.md.
#
# It is a separate environment on purpose. This repo's .venv is on torch 2.8 / transformers 4.57 /
# timm 1.0.22 / numpy 2.x and other work here needs it there; upstream targets torch 2.2 /
# transformers 4.40.1 / timm 0.9.10 / numpy 1.26. The timm pin matters most -- on timm 0.9.x the
# untagged `vit_so400m_patch14_siglip_224` still resolves to SigLIP-1, and on 1.0 it does not.
# Nothing here is installed into this repo's environment.
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config.sh"

# Sibling of the repo by default, so it is outside the checkout and never gets committed.
REF_DIR="${REF_DIR:-$REBUTTAL_REPO/../vla-adapter-ref}"
UPSTREAM_SHA="${UPSTREAM_SHA:-23fa0c9c159e2aa04341cdd3e924f44061311060}"

echo "==> clone upstream @ ${UPSTREAM_SHA:0:8} -> $REF_DIR"
# Refuse to blow away a directory we did not create. REF_DIR defaults next to the repo, and
# on a machine where that happens to be a real checkout an unguarded rm -rf is unrecoverable.
if [ -e "$REF_DIR" ] && [ -n "$(ls -A "$REF_DIR" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "refusing to delete non-empty $REF_DIR; re-run with FORCE=1 if that is really what you want" >&2
    exit 2
fi
rm -rf "$REF_DIR"
git clone -q https://github.com/OpenHelix-Team/VLA-Adapter.git "$REF_DIR"
git -C "$REF_DIR" checkout -q --detach "$UPSTREAM_SHA"

echo "==> venv (python 3.10)"
uv venv --python 3.10 "$REF_DIR/.venv"
PY="$REF_DIR/.venv/bin/python"

# Versions below are from upstream's own `our_envs.txt` (a pip list from the authors' machine),
# except where noted.
echo "==> torch 2.2.0 + cu121"
uv pip install --python "$PY" torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

echo "==> training stack"
uv pip install --python "$PY" \
    numpy==1.26.4 transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    accelerate==1.6.0 peft==0.11.1 draccus==0.8.0 \
    tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics==2021.12.3 \
    einops rich jsonlines matplotlib huggingface_hub sentencepiece \
    "git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972"

# --- four fixes that `our_envs.txt` does not give you -----------------------------------------
# 1+2. tensorflow-metadata: pip resolves it to 1.21, whose generated protobuf stubs import
#      `google.protobuf.runtime_version` (protobuf >= 5.27). TF 2.15 pins protobuf 4.x, so tfds
#      fails at import. 1.14.0 is the TF-2.15-era release and pulls protobuf back to 3.20.3.
# 3.   wandb: >=0.18 also needs protobuf 5.x (`wandb.proto...Imports`), which now conflicts with
#      the protobuf 3.20.3 above. Upstream's list says 0.19.8, which cannot work with their own
#      TF pin; 0.17.5 is the newest that does.
# 4.   json_numpy: imported by experiments/robot/openvla_utils.py, absent from their list.
echo "==> pinned-stack fixes (protobuf / wandb / json_numpy)"
uv pip install --python "$PY" "tensorflow-metadata==1.14.0" "wandb==0.17.5" json_numpy \
    imageio "imageio[ffmpeg]" opencv-python-headless

# 5. flash-attn IS required despite being absent from `our_envs.txt`: prismatic's
#    Qwen25LLMBackbone defaults `use_flash_attention_2=True` (qwen25.py), and transformers 4.40
#    raises ImportError when that is on without the package. Prebuilt wheel matched to
#    torch 2.2 / cu12 / cxx11abiFALSE / cp310 -- never built from source.
echo "==> flash-attn 2.5.6 (prebuilt wheel for torch 2.2)"
uv pip install --python "$PY" --no-build-isolation \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# --- the simulator half ----------------------------------------------------------------------
# robosuite 1.4.1 calls the old `mj_fullM(m, dst, M)`; mujoco >=3.4 changed it to
# `mj_fullM(m, d, dst)` and every env construction dies with a TypeError. 3.3.x is what
# VLA-Adapter's own environment used.
echo "==> LIBERO simulation stack"
# numpy is repeated here on purpose: this install runs after the pinned training stack, and
# without it the resolver is free to pull numpy past 1.26.4, which breaks tensorflow 2.15.
uv pip install --python "$PY" "robosuite==1.4.1" "bddl==3.5.0" "gym==0.26.2" "mujoco==3.3.*" \
    "numpy==1.26.4" cloudpickle termcolor pynput numba

# LIBERO lives *inside* the clone, at $REF_DIR/LIBERO, because that is what
# tools/train_eval.sh and tools/train_eval_rla.sh put on PYTHONPATH.
echo "==> LIBERO @ 8f1084e3 -> $REF_DIR/LIBERO"
git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git "$REF_DIR/LIBERO"
git -C "$REF_DIR/LIBERO" checkout -q --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C "$REF_DIR/LIBERO" apply "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libero-torch-load.patch"

# libero/libero/__init__.py calls input() at import time if this file is missing, which hangs
# any non-interactive run.
echo "==> LIBERO config"
mkdir -p "$REF_DIR/.libero" "$REBUTTAL_REPO/data/libero/libero_hdf5"
grep -v '^#' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libero-config.yaml" \
    | sed -e "s|\${LIBERO_ROOT}|$REF_DIR/LIBERO|g" \
          -e "s|\${LIBERO_DATASETS}|$REBUTTAL_REPO/data/libero/libero_hdf5|g" \
          -e '/^[[:space:]]*$/d' \
    > "$REF_DIR/.libero/config.yaml"

echo "==> link datasets + pretrained VLM from the main tree (no second copy)"
mkdir -p "$REF_DIR/data" "$REF_DIR/outputs"
ln -sfn "$REBUTTAL_REPO/data/libero" "$REF_DIR/data/libero"
rm -rf "$REF_DIR/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
ln -sfn "$REBUTTAL_WEIGHTS/prism-qwen25-extra-dinosiglip-224px-0_5b" \
        "$REF_DIR/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"

echo
echo "==> verify"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 "$PY" - <<'EOF'
import importlib.metadata as md
for p in ("torch", "transformers", "timm", "numpy", "tensorflow", "peft", "protobuf", "wandb", "flash-attn"):
    try:
        print(f"    {p:16s} {md.version(p)}")
    except Exception:
        print(f"    {p:16s} MISSING")
import tensorflow_datasets, wandb, flash_attn  # noqa: F401  -- the imports that used to break

# The LIBERO stack is installed after the training stack, so re-check the pin it could move.
assert md.version("numpy").startswith("1.26"), (
    f"numpy is {md.version('numpy')}; tensorflow 2.15 needs 1.26.x -- something upgraded it"
)

# peft is the one pin nothing else guards. Newer peft expands target_modules="all-linear" to a
# different module set, so LoRA would wrap a different part of the network and the run would
# stop being comparable to upstream's -- silently, with no error anywhere.
assert md.version("peft") == "0.11.1", (
    f"peft is {md.version('peft')}, upstream pins 0.11.1 -- reinstall with peft==0.11.1"
)
print("\n    OK: numpy 1.26.x and peft 0.11.1, upstream's own pins")

from timm.models import get_pretrained_cfg
tag = get_pretrained_cfg("vit_so400m_patch14_siglip_224")
print(f"    untagged SigLIP id -> {tag.hf_hub_id}")
assert "v2" not in (tag.tag or ""), "this timm resolves the untagged id to SigLIP-2; not a valid control"
print("    OK: resolves to SigLIP-1, so this env reproduces upstream's vision tower")
EOF

echo
echo "done. VLA-Adapter env at $REF_DIR"
echo
echo "Next:"
echo "  cp -r rebuttal/exp2_vla_adapter/vla_adapter_overlay/rla   $REF_DIR/"
echo "  cp -r rebuttal/exp2_vla_adapter/vla_adapter_overlay/tools $REF_DIR/"
echo "  cd $REF_DIR && tools/train_eval.sh spatial        # baseline"
echo "  cd $REF_DIR && tools/train_eval_rla.sh spatial    # + RLA"
