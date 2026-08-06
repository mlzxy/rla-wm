"""
rla/term.py

A three-function ANSI helper, so the one thing you must be able to trust at a glance -- *did stage 2
actually load stage 1?* -- is impossible to miss in a scrolling four-rank log.

Colour is **on by default even when stderr is not a tty**, which is the opposite of the usual rule.
That is deliberate: `tools/train_eval_rla.sh` pipes the run through `tee`, so `isatty()` is False for
exactly the run whose warm start most needs to be visible. Set `RLA_COLOR=0` (or `NO_COLOR=1`) to get
plain text, e.g. when post-processing a log with a parser that dislikes escape codes.

Nothing that is parsed downstream is coloured -- `tools/plot_loss.py` reads `curr: <float>` out of the
log, and `rla/loss.py`'s per-step metric line stays plain -- so this only ever touches banners.
"""

from __future__ import annotations

import os
import sys

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}

COLOR = (
    not os.environ.get("NO_COLOR")
    and os.environ.get("RLA_COLOR", "1") not in ("0", "false", "False")
    and os.environ.get("TERM", "") != "dumb"
)

WIDTH = 78


def paint(text: str, *styles: str) -> str:
    if not COLOR or not styles:
        return text
    return "\033[" + ";".join(_CODES[s] for s in styles) + "m" + text + "\033[0m"


def is_main() -> bool:
    """Rank 0, or a single-process run. torchrun always sets `RANK`.

    The other ranks run the identical load and raise on the identical checks; they just do not
    reprint the banner, the same convention `finetune.py` uses for its own logging.
    """
    return int(os.environ.get("RANK", 0)) == 0


def mark(good: bool) -> str:
    return paint("OK", "green", "bold") if good else paint("XX", "red", "bold")


def say(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def note(text: str, *styles: str) -> None:
    """A multi-line `[rla] ...` message, painted per line so interleaved output stays legible.

    Rank 0 only, like `block`: every rank derives its config from the same environment, so four
    copies of the same banner is noise rather than confirmation.
    """
    if not is_main():
        return
    for line in text.splitlines():
        say(paint(line, *(styles or ("cyan",))))


def block(title: str, rows, *, style: str = "green") -> None:
    """An aligned, colour-framed key/value block on stderr. Rank 0 only."""
    if not is_main():
        return
    rows = list(rows)
    pad = max((len(key) for key, _ in rows), default=0)
    say(paint(f"+-- {title} " + "-" * max(4, WIDTH - len(title) - 5), style))
    for key, value in rows:
        say(paint("|", style) + " " + paint(key.ljust(pad), "bold") + "  " + value)
    say(paint("+" + "-" * (WIDTH - 1), style))
