"""Checkpoint *step tokens* -- the text between `_step` and `.pt` in a checkpoint filename.

`src/trainers/basic.py:save()` writes `encoder_step0040000.pt` at every `i_save` milestone. Some
runs carry a resumable snapshot beside it, `encoder_step0040000.snapshot.pt`, so what follows
`_step` is only *sometimes* an integer. Everything in this package that pins an RLA version
therefore talks about tokens rather than ints:

    "40000"   "0040000"   ->  (40000, "")            encoder_step0040000.pt
    "40000.snapshot"      ->  (40000, ".snapshot")   encoder_step0040000.snapshot.pt
    40000     (int)       ->  (40000, "")            same as "40000"

Zero padding is not part of a checkpoint's identity, so it is not part of the comparison: a policy
config that says `rla_latent_step: 40000` matches a sidecar built from `0040000`, and hydra's
int-vs-str parsing of a command-line override stops mattering. The suffix *is* part of the identity
-- a snapshot and the milestone beside it are different weights, and silently accepting one for the
other is exactly the failure this package spends so much effort making impossible.

`utils.misc.fetch_state_dict` cannot be used for this. It composes the filename as
`{name}_step{int(step):07d}.pt`, which cannot express a snapshot at all; and given `step=None` it
takes the lexicographically last name, which prefers `...0040000.snapshot.pt` over the milestone
`...0040000.pt`. `find_ckpt` resolves by (number, suffix) and breaks that tie the other way.

The same convention is used by `rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py`.
"""

from __future__ import annotations

import glob
import os

__all__ = [
    "parse_step_token",
    "ckpt_token",
    "canonical_step",
    "same_step",
    "list_ckpts",
    "find_ckpt",
    "find_ckpt_in",
    "load_state_dict_file",
]

_DIGITS = "0123456789"


def parse_step_token(step) -> tuple[int, str]:
    """`40000` / `"0040000"` / `"40000.snapshot"` -> `(40000, "")`, `(40000, "")`, `(40000, ".snapshot")`.

    A token with no leading digits at all (a hand-renamed checkpoint such as `encoder_stepbest.pt`)
    comes back as `(-1, <the literal text>)`, so it can still be matched exactly -- it just never
    compares equal to a numbered one.
    """
    text = str(step).strip()
    cut = len(text) - len(text.lstrip(_DIGITS))
    if cut == 0:
        return -1, text
    return int(text[:cut]), text[cut:]


def ckpt_token(path: str, name: str) -> str:
    """`.../encoder_step0040000.snapshot.pt` -> `"0040000.snapshot"`."""
    base = os.path.basename(path)
    return base[len(f"{name}_step"):-len(".pt")]


def canonical_step(step, pad: int = 7) -> str:
    """Any spelling of a step -> the one the checkpoint file uses: `"0040000.snapshot"`.

    Use this as the dict/set key whenever several sources of truth have to be compared (the four
    policy configs, the sidecar manifest, the shipped checkpoint), so `40000`, `"40000"` and
    `"0040000"` collapse to one entry instead of looking like three different RLA versions.
    """
    number, suffix = parse_step_token(step)
    return str(step).strip() if number < 0 else f"{number:0{pad}d}{suffix}"


def same_step(a, b) -> bool:
    """Do two spellings name the same checkpoint? `None` matches only `None`."""
    if a is None or b is None:
        return a is None and b is None
    return parse_step_token(a) == parse_step_token(b)


def list_ckpts(ckpt_dir: str, name: str) -> list[str]:
    """Every `<name>_step*.pt` in `ckpt_dir`, in "latest last" order.

    So `list_ckpts(...)[-1]` is the checkpoint `find_ckpt(..., step=None)` returns: the
    highest-numbered one, and among those the plain milestone rather than a snapshot.
    """
    files = glob.glob(os.path.join(ckpt_dir, f"{name}_step*.pt"))
    return sorted(files, key=lambda p: _rank(ckpt_token(p, name)))


def _rank(token: str) -> tuple[int, bool, str]:
    number, suffix = parse_step_token(token)
    # `suffix == ""` LAST, because callers take [-1]: at one step the milestone beats the snapshot.
    return number, suffix == "", suffix


def find_ckpt_in(ckpt_dir: str, name: str, step=None) -> str:
    """Resolve one checkpoint inside a `ckpts` directory.

    `step` is a token (`40000`, `"0040000"`, `"40000.snapshot"`) or None. `None` takes the
    highest-numbered checkpoint, preferring the plain milestone when a snapshot sits at the same
    step.
    """
    available = list_ckpts(ckpt_dir, name)
    if not available:
        raise FileNotFoundError(f"no {name}_step*.pt under {ckpt_dir}")
    if step is None:
        return available[-1]

    want = parse_step_token(step)
    for path in available:
        if parse_step_token(ckpt_token(path, name)) == want:
            return path
    raise FileNotFoundError(
        f"{ckpt_dir} has no {name} at step {step!r}. Available: "
        f"{[ckpt_token(p, name) for p in available]}"
    )


def find_ckpt(run_dir: str, name: str, step=None) -> str:
    """`find_ckpt_in` against a training run directory, i.e. `<run_dir>/ckpts`."""
    return find_ckpt_in(os.path.join(run_dir, "ckpts"), name, step)


def load_state_dict_file(path: str, device="cpu"):
    """`utils.misc.fetch_state_dict`'s tail, against a path that has already been resolved.

    Resolving once and loading exactly what was resolved is what lets the manifest make a claim
    about the weights that were actually used.
    """
    import torch

    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob
