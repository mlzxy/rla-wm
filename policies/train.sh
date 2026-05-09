#!/bin/bash
set -xe

config=$1
shift

export FORCE_COLOR=1
unset NO_COLOR

PYTHONPATH=.:./third_party/diffusion_policy .venv/bin/python policies/train_policy.py --config-name=$config "$@"
