"""The Python half of ``rebuttal/config.sh``.

Several scripts ported into ``rebuttal/`` used to open with a line like

    REPO = "/home/me/checkouts/v4world-tmp"

which is obviously useless to anyone else. They now import from here instead.
Every value falls back to a path derived from this file's own location, and every
value can be overridden with the matching environment variable, so the shell
scripts and the Python scripts always agree.
"""

import os
import os.path as osp

#: Repository root (the directory containing ``train.py``).
REPO = os.environ.get("REBUTTAL_REPO") or osp.abspath(
    osp.join(osp.dirname(__file__), "..", "..")
)

#: VLA-Adapter checkpoints (exp-2 only).
WEIGHTS = os.environ.get("REBUTTAL_WEIGHTS") or osp.join(
    REPO, "runs/weights/vla-adapter"
)

#: VLA-Adapter evaluation logs (exp-2 only).
EVAL_LOGS = os.environ.get("REBUTTAL_EVAL_LOGS") or osp.join(
    REPO, "runs/vla-adapter-eval"
)

__all__ = ["REPO", "WEIGHTS", "EVAL_LOGS"]
