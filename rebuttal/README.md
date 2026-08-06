# Rebuttal experiments

Four experiments run during the rebuttal period. Each one is self-contained in its own folder
here; the main repository is unchanged except for a short list of small, default-off hooks listed
in [MAIN_TREE_CHANGES.md](MAIN_TREE_CHANGES.md).

The headline claims of the paper are in the main [README](../README.md) and the
[docs/](../docs/) directory. This folder is the follow-up work.

## What we found

| Experiment | Question | Answer |
|---|---|---|
| [exp-1 · WMRL v2](exp1_wmrl_v2/) | Does WMRL still help once the BC initializer is picked honestly and everything is scored on a large offline evaluation set instead of the noisy in-training one? | Partly. It helps on 2 of 5 tasks (split-half honest Δ over 15 seeds: PokeCube +0.042, PushT +0.013), hurts on 2, and is a wash on the fifth. It also makes successful episodes measurably shorter even where success rate is flat. |
| [exp-2 · VLA-Adapter](exp2_vla_adapter/) | Does the RLA latent objective help a real VLA on LIBERO? | Yes on two suites. Adding RLA as an auxiliary objective improves LIBERO-Object (97.4 → 99.2) and LIBERO-Spatial (97.0 → 97.6); Goal and Long are unchanged. |
| [exp-3 · World-model data diversity](exp3_data_diversity/) | Does training the world model on far more diverse data improve the downstream RL? | Yes, modestly. WMRL on the diverse-data world model scores 0.1851 against 0.1732 for the old one (best-per-seed over 15 seeds; 0.1390 vs 0.1362 on the unselected ladder mean). |
| [exp-4 · Real-world SO-101](exp4_so101/) | Does BC-RLA help on a real arm? | **Untested.** The data, the RLA autoencoder and a matched BC / BC-RLA pair exist and load. No robot evaluation was ever run. |

One convention we hold to everywhere in this folder, because mixing them is what made these
numbers confusing in the first place: **every table says how it aggregates** — best-per-seed,
ladder mean, split-half honest, or single run. A best-per-seed number is biased upward and is not
comparable to a single-run number.

## Layout

| Path | What |
|---|---|
| [exp1_wmrl_v2/](exp1_wmrl_v2/) | v2 configs, launcher, the sweep tables |
| [exp2_vla_adapter/](exp2_vla_adapter/) | The `rla/` overlay, the one-script VLA-Adapter setup, the RLA sidecar, configs, results |
| [exp3_data_diversity/](exp3_data_diversity/) | The autoencoder / world-model / WMRL configs for the diverse Push-T data, and the sweep results |
| [exp4_so101/](exp4_so101/) | The SO-101 policy and dataset code, the RLA latent precompute, and the matched BC / BC-RLA configs |
| [src/](src/) | Two models and one trainer used by exp-2 and exp-4, registered into `src.models` / `src.trainers` |
| [tools/](tools/) | The offline checkpoint scorer, the analysis that turns its output into tables, and a local runner for the Colab demo |
| [config.sh](config.sh), [tools/paths.py](tools/paths.py) | The only two places any path is resolved. Everything else derives from them. |
| [MAIN_TREE_CHANGES.md](MAIN_TREE_CHANGES.md) | Every change outside this folder |
| [DATA.md](DATA.md) | Data and checkpoint downloads |

## Getting set up

The environment is the same one the main repo uses — see [docs/setup.md](../docs/setup.md).
Every command in this folder runs from the repository root with:

```bash
export PYTHONPATH=.:./third_party/diffusion_policy
```

Shell scripts here source [`config.sh`](config.sh), which works out the repository root from its
own location. If you need to point them somewhere else, set `REBUTTAL_REPO` or `REBUTTAL_PY` in
the environment and every script and Python module picks it up.

exp-2 is the exception, and deliberately so: VLA-Adapter lives in **a clone and a virtualenv of
its own, on upstream's pinned versions** — including `peft 0.11.1`. One script builds it and
nothing from that stack touches this repository's environment. See
[exp2_vla_adapter/README.md](exp2_vla_adapter/README.md).

Data and pretrained weights are in [DATA.md](DATA.md).

## Reading the working logs

exp-1 and exp-4 keep a `RUNBOOK.md` next to their README. Those are unedited working logs from
when the experiments were running. They record dead ends and numbers that were later superseded,
on purpose — that history is often the most useful part. Each one carries a header saying so.

Where a log and a README disagree, the README is right.
