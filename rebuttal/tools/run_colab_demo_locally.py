#!/usr/bin/env python
"""Run notebooks/colab_demo.ipynb on a local box, without Colab and without Hugging Face.

The demo notebook is written for a free-tier Colab T4: it clones the repo, pip-installs
everything, and downloads its checkpoints from Hugging Face.  On a machine that already has
the environment and the archives unpacked, all of that is in the way -- but the *rest* of the
notebook is exactly what you want to smoke-test after touching a config, a checkpoint pin or
the world-model code.

So this runs the notebook's own cells, in order, in one namespace, skipping only the cells
whose job is to set up Colab.  It reads the .ipynb directly, so it cannot drift out of sync
with the notebook the way a hand-copied script would.

    export PYTHONPATH=.:./third_party/diffusion_policy
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python rebuttal/tools/run_colab_demo_locally.py --section 1
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python rebuttal/tools/run_colab_demo_locally.py --section 2 --iters 4

What you need on disk first -- all of it from the main release
(../../docs/data-and-checkpoints.md), the same things the notebook downloads for itself:

    runs/weights/rla/maniskill/                      the RLA encoder + decoder
    runs/weights/rla-wm/maniskill/ur10e/             the flow-matching world model
    runs/weights/dino-to-image_unet/maniskill/       the DINO->RGB decoder
    runs/weights/wmrl_checkpoints/0_bc_s2r_nstate/   the BC initializer section 2 starts from
    data/maniskill/ppo/ur10e_stick/PushT-v2/success/ the PushT trajectories

Section 2 is the real thing: 50 PPO iterations inside the world model plus the 50-episode
simulator evaluations.  Budget an hour or so on one A100 and use --iters to cut it down when
you only want to know that it starts.

Pass --list to see which cells would run and which would be skipped, and why.
"""
import argparse
import json
import os
import re
import sys

NB = "notebooks/colab_demo.ipynb"

# A cell is skipped when its source matches one of these.  Each entry is (regex, reason).
# Everything here is Colab plumbing; none of it is part of what the demo demonstrates.
SKIP = [
    (r"google\.colab", "Colab secrets -- set HF_TOKEN in your environment instead"),
    (r"^\s*%pip|^\s*!pip", "package install -- your environment already has these"),
    (r"git\", \"clone\"|REPO_URL", "clones the repo -- you are already in it"),
    (r"snapshot_download|hf_hub_download", "downloads from Hugging Face -- unpack the archives yourself"),
]

# Section 2 starts at the cell that builds the Args object.
SECTION_2_MARKER = r"from wmrl\.train import Args"


def classify(src):
    for pat, why in SKIP:
        if re.search(pat, src, re.M):
            return why
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=NB)
    ap.add_argument("--section", choices=["1", "2", "all"], default="all",
                    help="1 = the t->t+h rollout, 2 = WMRL training, all = both")
    ap.add_argument("--iters", type=int, default=None,
                    help="override total_iterations in section 2 (the notebook uses 50)")
    ap.add_argument("--eval-episodes", type=int, default=None,
                    help="override eval_num_episodes in section 2 (the notebook uses 50)")
    ap.add_argument("--figs", default="runs/colab/figs",
                    help="where to write the plots section 1 would have shown inline")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if not os.path.exists("train.py"):
        sys.exit("run me from the repository root")

    cells = [c for c in json.load(open(args.notebook))["cells"] if c["cell_type"] == "code"]

    sec2_at = next((i for i, c in enumerate(cells)
                    if re.search(SECTION_2_MARKER, "".join(c["source"]))), len(cells))

    plan = []
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        section = 2 if i >= sec2_at else 1
        why = classify(src)
        if why is None and args.section != "all" and str(section) != args.section:
            why = f"section {section}, you asked for {args.section}"
        plan.append((i, section, src, why))

    if args.list:
        for i, section, src, why in plan:
            head = next((l for l in src.splitlines() if l.strip()), "")[:64]
            print(f"cell {i:2d}  §{section}  {'SKIP' if why else 'RUN '}  {head}")
            if why:
                print(f"{'':13}      ^ {why}")
        return

    # There is no inline display outside a notebook, so send plt.show() to a file instead.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.figs, exist_ok=True)
    shown = [0]

    def _show(*_a, **_kw):
        path = os.path.join(args.figs, f"fig{shown[0]:02d}.png")
        plt.savefig(path, dpi=110, bbox_inches="tight")
        print(f"[figure -> {path}]", flush=True)
        shown[0] += 1
        plt.close("all")

    plt.show = _show

    # Section 2 alone still needs `os` and `torch` from the section-1 imports.
    ns = {"__name__": "__main__"}
    exec("import os, sys, torch, numpy as np", ns)

    for i, section, src, why in plan:
        if why:
            print(f"\n--- cell {i} skipped: {why}", flush=True)
            continue
        if args.iters is not None:
            src = re.sub(r"args\.total_iterations\s*=\s*\d+",
                         f"args.total_iterations  = {args.iters}", src)
            src = re.sub(r"args\.eval_freq\s*=\s*\d+",
                         f"args.eval_freq         = {max(1, args.iters // 2)}", src)
        if args.eval_episodes is not None:
            src = re.sub(r"args\.eval_num_episodes\s*=\s*\d+",
                         f"args.eval_num_episodes = {args.eval_episodes}", src)
        print(f"\n=== cell {i} (§{section}) ===", flush=True)
        exec(compile(src, f"<{args.notebook} cell {i}>", "exec"), ns)

    print("\nOK", flush=True)


if __name__ == "__main__":
    main()
