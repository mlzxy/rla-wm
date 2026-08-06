"""
rla/eval.py

Evaluation entry point. Runs `experiments/robot/libero/run_libero_eval.py` unmodified, with the
action-head class swapped so a checkpoint trained with RLA tokens loads.

    python rla/eval.py --pretrained_checkpoint outputs/... --task_suite_name libero_10 ...

`RLA_SIDECAR` is **not** needed here: `z` is an output of the policy, never an input, so evaluation
needs the head's *shape* but no targets. `RLA_STEPS` / `RLA_QUERIES` / `RLA_DIM` must match how the
checkpoint was trained -- `get_action_head` loads strict, so a mismatch fails at load rather than
silently randomising half the head.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> None:
    from rla import patch

    # Swap `openvla_utils.L1RegressionActionHead` before anything builds a head. `run_libero_eval`
    # imports `get_action_head` by name, but that function resolves the class from its *own* module
    # globals at call time, so import order does not matter here.
    patch.install_eval()

    import experiments.robot.libero.run_libero_eval as run_libero_eval

    run_libero_eval.eval_libero()


if __name__ == "__main__":
    main()
