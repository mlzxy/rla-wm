# What the rebuttal work changed outside `rebuttal/`

Everything added for the rebuttal lives under [`rebuttal/`](.) and touches nothing else — with
five exceptions. A self-contained folder cannot, on its own, teach `train.py` about a new model
class, or teach the WMRL agent which of two weight sets in a checkpoint to start from. Those
hooks have to live where the code that reads them lives.

This page lists all of them so you can audit the blast radius in one place.

**The rule every change obeys:** it is additive, it is off by default, and with the new option
absent the code does exactly what it did before. No published number changes.

## The five files

| File | What was added | With the new option absent |
|---|---|---|
| `src/models/__init__.py` | An `__external` dict mapping a model name to a fully-qualified module path, plus one `elif` in `__getattr__`. Registers `MultiViewTokenTransformer`. | Unchanged. Every existing name still resolves through the same lazy relative import. |
| `src/trainers/__init__.py` | The same, registering `RlaAutoencoderMultiViewTrainer`. | Unchanged. |
| `wmrl/agent.py` | `pretrained_weights`, validated to `{"ema", "model"}`. | Defaults to `"ema"` — `ema_model` falling back to `model`, the previous behaviour. |
| `wmrl/train.py` | The same field on `Args`, and one line forwarding it. | Same default, so every existing config builds an identical agent. |
| `policies/workspace/eval_utils.py` | `return_per_episode=False`; when set, adds `metrics["per_episode"]`. Also a missing trailing newline. | Off by default, and deliberately so: `wmrl/train.py` forwards every metric to the logger and `print_eval_table` formats each value with `:.4f`, and both break on a list. |

### Why `pretrained_weights` exists

A BC workspace checkpoint stores two weight sets, `model` and `ema_model`. WMRL always loaded
`ema_model`. But the offline checkpoint selection behind exp-1 and exp-3
([`tools/eval_ckpts.py`](tools/eval_ckpts.py)) scores `model`, so without this option RL would
start from weights that were never the ones ranked. Setting `pretrained_weights: model` makes
the initializer and the thing that picked it agree.

## Two more, neither of them code

- **`configs/**/*.yaml` (13 files)** — dropped a hard-coded personal `wandb.entity`. wandb now
  falls back to whatever account you are logged in as; previously every run tried to log to one
  specific entity that nobody else can write to.
- **`README.md`** — one row in the Docs table, one row in the Repository layout table, so this
  folder is findable.
- **`notebooks/inference_demo.ipynb`** — three absolute paths from the machine it ran on were
  replaced with `<repo root>` in the stored cell output. No code changed.

## What is deliberately *not* changed

**`pyproject.toml` and `uv.lock` are untouched.** VLA-Adapter lives in a clone and a virtualenv
of its own, on upstream's pinned versions — torch 2.2, transformers 4.40.1, timm 0.9.10,
peft 0.11.1. [`exp2_vla_adapter/setup/setup_env.sh`](exp2_vla_adapter/setup/setup_env.sh) builds
it from scratch and asserts the two pins that would otherwise drift silently. None of that stack
goes into this repository's environment, so there is no dependency group to add and no lockfile to
change.

The one exp-2 tool that does run here — `sidecar/extract_rla_sidecar.py`, which needs our RLA
autoencoder — wants TensorFlow to read the RLDS export. That is one `uv pip install`, documented
where the tool is.

## One file was renamed to survive `.gitignore`

The repository's `.gitignore` patterns are unanchored, so they apply inside `rebuttal/` too.
`*.txt`, `*.html`, `*.log`, `data`, `runs`, `logs`, `outputs/`, `lib/`, `env/`, `build/`,
`tmp*`, `.vscode` and `.github` are invisible to git anywhere in the tree. LIBERO's
`.libero/config.yaml` landed on one of those, so it ships as
`exp2_vla_adapter/setup/libero-config.yaml` and `setup_env.sh` writes the real one.

If you add files later: anything you drop into a directory named `logs/` or ending in `.txt`
under `rebuttal/` will silently not be committed.
