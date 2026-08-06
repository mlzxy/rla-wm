"""Gate runner.

    .venv/bin/python -m rla.tests                      # the blocking set
    .venv/bin/python -m rla.tests --gates join
    .venv/bin/python -m rla.tests --gates align --suite libero_goal --rebuild
    .venv/bin/python -m rla.tests --gates stats        # after a run

Exit status is non-zero if any gate fails, so this works in a launcher or in CI.

| gate      | proves | needs |
|-----------|--------|-------|
| join      | sidecar <-> RLDS is a bijection over every suite | data, ~2 min |
| norm      | target scaling is invertible and its loss scale is pinned | sidecar, seconds |
| align     | the block joined to a frame belongs to that frame's episode and chunk | data, ~3 min |
| head      | RLA off is byte-identical to upstream; RLA on has the right shapes and grads | nothing |
| isolation | rla <- action is masked; action <- rla is not | nothing |
| warmstart | stage 2 loaded from stage 1 predicts identical z | nothing |
| eval      | an RLA checkpoint loads through the real rollout path | nothing |
| stats     | normalisation is unchanged vs a vanilla run | two finished runs |
"""

from __future__ import annotations

import argparse
import sys
import traceback

from rla.tests import (test_align, test_eval, test_head, test_isolation, test_join, test_norm,
                       test_stats, test_warmstart)

GATES = {
    "join": test_join.run,
    "norm": test_norm.run,
    "align": test_align.run,
    "head": test_head.run,
    "isolation": test_isolation.run,
    "warmstart": test_warmstart.run,
    "eval": test_eval.run,
    "stats": test_stats.run,
}
DEFAULT_GATES = "join,norm,align,head,isolation,warmstart,eval"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gates", default=DEFAULT_GATES, help=f"comma-separated; {list(GATES)}")
    parser.add_argument("--sidecar", default="", help="override RLA_SIDECAR (join gate)")
    parser.add_argument("--rlds-root", default="data/libero")
    parser.add_argument("--suite", default="libero_spatial", help="suite for the align gate")
    parser.add_argument("--synthetic", default="", help="where the synthetic sidecar lives")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the synthetic sidecar")
    parser.add_argument("--batches", type=int, default=60, help="batches checked by the align gate")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    requested = [g.strip() for g in args.gates.split(",") if g.strip()]
    if unknown := [g for g in requested if g not in GATES]:
        raise SystemExit(f"unknown gate(s) {unknown}; known: {list(GATES)}")

    failures = []
    for name in requested:
        try:
            GATES[name](args)
        except Exception as exc:  # noqa: BLE001 - a gate failure is reported, not raised
            failures.append(name)
            print(f"\nFAILED [{name}]: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    print()
    passed = len(requested) - len(failures)
    if failures:
        print(f"FAILED: {', '.join(failures)}  ({passed}/{len(requested)} passed)")
        return 1
    print(f"OK: {passed}/{len(requested)} gates passed ({', '.join(requested)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
