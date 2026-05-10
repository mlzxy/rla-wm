# RLA-WM Evaluation

The eval handles ([data/eval_handles/](../data/eval_handles/)) ship pre-computed; no regeneration needed.

```bash
bash eval/run_eval.sh     {panda|xarm|ur10e}   # ManiSkill
bash eval/run_eval_iws.sh {pusht|box|rope}     # IWS
```

## Outputs

```
runs/eval_output/<robot>/                # ManiSkill
runs/eval_output/iws_<scene>/            # IWS
├── eval_summary.json
├── eval_summary.md
├── per-handle metrics
└── rollout videos
```

## Internals

- [eval/eval_wrapper.py](../eval/eval_wrapper.py) loads the cached handles, spawns workers, and dispatches each sample to a predictor module.
- [eval/predictors/rla_wm_predictor.py](../eval/predictors/rla_wm_predictor.py) — ManiSkill predictor.
- [eval/predictors/rla_wm_predictor_iws.py](../eval/predictors/rla_wm_predictor_iws.py) — IWS predictor.

To benchmark a different model, drop in a new predictor module and pass it via `--module-path` (the eval shell scripts show the call site).
