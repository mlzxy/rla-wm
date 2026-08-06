"""
rla/train.py

Training entry point. Loads `vla-scripts/finetune.py` unmodified, patches its module globals, and
runs it. Every CLI flag `finetune.py` accepts works here unchanged; the RLA knobs come from the
environment (see `rla/config.py`).

    torchrun --standalone --nproc-per-node 4 rla/train.py --vlm_path ... --dataset_name ...

`vla-scripts` is not an importable package name (the hyphen), so the module is loaded by path. It is
given the name `vla_finetune` rather than `__main__`, which also means its
`if __name__ == "__main__"` guard does not fire and `finetune()` is called exactly once, by us,
after the patches are in place.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_finetune_module():
    path = REPO / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("vla_finetune", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vla_finetune"] = module
    spec.loader.exec_module(module)
    return module


def _rlds_root() -> str:
    """`--data_root_dir` as passed on the command line; the preflight needs it."""
    argv = sys.argv
    for i, token in enumerate(argv):
        if token == "--data_root_dir" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--data_root_dir="):
            return token.split("=", 1)[1]
    return "data/libero"


def _preflight() -> None:
    """Prove the sidecar and the RLDS data are a clean bijection before spending a GPU hour.

    On by default. `tools/train_eval_rla.sh` runs the same check once up front as a separate process
    and then sets `RLA_PREFLIGHT=0` for the launch, so the cost is paid once rather than once per
    rank; anyone invoking this file directly (including a debugger) gets the check for free.
    """
    from rla.config import CFG

    if not CFG.has_targets or os.environ.get("RLA_PREFLIGHT", "1") in ("0", "", "false", "False"):
        return

    from rla.read_rla_sidecar import RlaSidecar, preflight

    sidecar = RlaSidecar(CFG.sidecar, chunk=CFG.steps, num_tokens=CFG.queries, token_dim=CFG.dim)
    preflight(sidecar, _rlds_root(), require_full=True)


def main() -> None:
    _preflight()

    finetune_mod = _load_finetune_module()

    from rla import patch

    patch.install(finetune_mod)

    # draccus reads sys.argv, so this behaves exactly like running finetune.py directly.
    finetune_mod.finetune()


if __name__ == "__main__":
    main()
