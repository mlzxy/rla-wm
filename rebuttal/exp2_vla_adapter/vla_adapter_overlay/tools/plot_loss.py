#!/usr/bin/env python
"""Plot the training loss of one upstream-reference run.

Prefers the run's wandb record (it carries a real `_step` axis); falls back to parsing the
`curr:` prints out of the training log, whose step axis has to be inferred.

  tools/plot_loss.py --run-id REF-Spatial --out curve.png [--log train.log] [--steps 20000]
"""
import argparse
import glob
import json
import os
import re
import sys

# Reading the wandb datastore (and matplotlib) needs this repo's venv. If we were launched by
# some other interpreter -- e.g. through the shebang -- re-exec under it, instead of silently
# degrading to "no loss data found".
_VENV_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(_VENV_PY):
    try:
        import wandb  # noqa: F401
    except Exception:
        os.execv(_VENV_PY, [_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
KEY = "VLA Train/Curr Action L1 Loss"        # what the log prints as `curr:`
FULL = "VLA Train/Loss"                       # full-chunk L1


def from_wandb(run_id):
    """Return {series_name: [(step, value)]} for the newest wandb run named ft+<run_id>."""
    try:
        from wandb.sdk.internal import datastore
        from wandb.proto import wandb_internal_pb2 as pb
    except Exception:
        return {}
    best = None
    for d in sorted(glob.glob(os.path.join(REF, "wandb", "*run-*")), reverse=True):
        f = glob.glob(os.path.join(d, "*.wandb"))
        if not f:
            continue
        try:
            ds = datastore.DataStore()
            ds.open_for_scan(f[0])
        except Exception:
            continue
        name, series = None, {KEY: [], FULL: []}
        while True:
            try:
                data = ds.scan_data()
            except Exception:
                break
            if data is None:
                break
            rec = pb.Record()
            try:
                rec.ParseFromString(data)
            except Exception:
                continue
            kind = rec.WhichOneof("record_type")
            if kind == "run" and rec.run.display_name:
                name = rec.run.display_name
            elif kind == "history":
                step, vals = None, {}
                for it in rec.history.item:
                    if it.key == "_step":
                        try: step = int(json.loads(it.value_json))
                        except Exception: pass
                    elif it.key in series:
                        try: vals[it.key] = float(json.loads(it.value_json))
                        except Exception: pass
                if step is not None:
                    for k, v in vals.items():
                        series[k].append((step, v))
        if name == f"ft+{run_id}" and len(series[KEY]) > 10:
            best = series
            break
    return best or {}


def from_log(path, max_steps):
    """`curr:` prints, mapped onto [0, max_steps] -- prints are uniform in wall-clock."""
    if not path or not os.path.exists(path):
        return []
    txt = open(path, "rb").read().decode("utf8", "replace").replace("\r", "\n")
    v = [float(m) for m in re.findall(r"curr:\s+(\d+\.\d+)", txt)]
    if not v:
        return []
    n = len(v)
    return [(int(max_steps * i / max(1, n - 1)), x) for i, x in enumerate(v)]


def smooth(ys, w=15):
    return [sum(ys[max(0, i - w):min(len(ys), i + w + 1)]) /
            (min(len(ys), i + w + 1) - max(0, i - w)) for i in range(len(ys))]


ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--log", default=None)
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--title", default="")
a = ap.parse_args()

series = from_wandb(a.run_id)
src = "wandb"
if not series.get(KEY):
    series = {KEY: from_log(a.log, a.steps or 1)}
    src = "training log"
if not series.get(KEY):
    print("no loss data found; skipping plot", file=sys.stderr)
    sys.exit(0)

fig, ax = plt.subplots(figsize=(9.5, 5.2))
for key, color, lbl in ((KEY, "#2470b3", "curr-action L1"), (FULL, "#7f8c8d", "full-chunk L1")):
    s = series.get(key) or []
    if len(s) < 10:
        continue
    xs = [p[0] for p in s]; ys = [p[1] for p in s]
    ax.plot(xs, ys, color=color, alpha=0.15, lw=0.8)
    sy = smooth(ys)
    tail = [y for x, y in zip(xs, ys) if x >= max(xs) - max(1, max(xs) // 10)]
    end = sum(tail) / len(tail)
    mn = min(sy)
    warn = "   ** RISING — likely diverged **" if end > mn * 1.25 else ""
    ax.plot(xs, sy, color=color, lw=2.0, label=f"{lbl}   min {mn:.3f} → end {end:.3f}{warn}")
    ax.annotate(f"{end:.3f}", xy=(xs[-1], end), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")

ax.set_xlabel("gradient step")
ax.set_ylabel("L1 loss")
ax.grid(alpha=0.25, ls=":")
ax.legend(fontsize=9, loc="upper right")
ax.set_title(a.title or f"{a.run_id}  (source: {src})", fontsize=12, fontweight="bold", loc="left")
fig.tight_layout()
fig.savefig(a.out, dpi=140)
print(f"loss curve -> {a.out}   (source: {src})")
