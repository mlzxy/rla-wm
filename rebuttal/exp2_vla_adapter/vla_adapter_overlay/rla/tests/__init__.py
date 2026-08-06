"""Gates for the RLA integration -- see `rla/tests/__main__.py` for the table and the entry point.

The two defaults below are set *here*, before any test module can import `rla.config`, because that
module freezes `CFG` from the environment at import time. Without them a bare `python -m rla.tests`
runs every gate against a disabled configuration and fails with shape errors that say nothing about
the real problem, which is a missing env var.

`setdefault`, so an explicit `RLA_STEPS=... RLA_SIDECAR=... python -m rla.tests` still wins -- and so
`tools/train_eval_rla.sh`, which exports both before calling the `join` gate, is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# The gates describe the RLA-*enabled* geometry, and RLA_STEPS has exactly one legal non-zero value.
os.environ.setdefault("RLA_STEPS", str(NUM_ACTIONS_CHUNK))

# The same default `tools/train_eval_rla.sh` uses, applied only if it is actually on disk; the
# sidecar-dependent gates (join, norm, align) skip cleanly rather than fail when it is not.
_SIDECAR = Path(__file__).resolve().parents[2] / "data/libero_rla/16x64-a8-step040000-snapshot"
if _SIDECAR.is_dir():
    os.environ.setdefault("RLA_SIDECAR", str(_SIDECAR))
