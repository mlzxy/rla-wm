"""
rla/warmstart.py

Stage 2 starts from stage 1's checkpoint. Two wrapped module globals do it.

A saved checkpoint dir (`outputs/<run_id>--<step>_chkpt/`) holds four things that matter here: the
**merged** HF VLM (written by `save_training_checkpoint`'s `merge_lora_during_training` branch), the
LoRA adapter, `action_head--<step>_checkpoint.pt` and `proprio_projector--<step>_checkpoint.pt`.

  * **VLM** -- wrap `get_peft_model`. At that point `vla` is still a plain
    `OpenVLAForActionPrediction` with un-prefixed keys, which is exactly the key set the merged save
    has, so stage 1's weights load into it directly. Stage 2 then starts a *fresh* LoRA on top:
    `init_lora_weights="gaussian"` zero-initialises `lora_B`, so the adapter is the identity at step
    0 and the model begins precisely where stage 1 ended.

    Loading the merged model rather than replaying the adapter is deliberate. PEFT's
    `save_pretrained` writes only adapter weights, and `action_queries` -- 64 learned embeddings that
    `finetune.py:846` force-unfreezes -- is not among them; the merge path has to re-inject it by
    hand (`finetune.py:584`). The merged safetensors already contain it.

  * **Head and proprio projector** -- wrap `init_module`, which is the one place they are built.
    Strict load, because both stages construct the same class with the same `RLA_*`; a mismatch
    should fail here rather than silently randomise.

`--resume` is deliberately not reused for this. It rewrites the run id from `config_file_path`
(`finetune.py:171-176`) and offsets every logged/checkpointed step by `resume_step` (`:1056`), so
stage-2 checkpoints would be named by the stage-1 + stage-2 total.
"""

from __future__ import annotations

from pathlib import Path

import torch

from rla import term
from rla.config import CFG

# Sentinel weights that must survive the load: if the merged checkpoint is missing these, it is not
# a VLA-Adapter checkpoint and every later symptom would be mysterious.
_REQUIRED_KEYS = ("action_queries.weight", "projector.fc1.weight")

# Fraction of the model's own tensors that the checkpoint must cover. On a real merged VLA-Adapter
# save this is 982/983: the sole absentee is `language_model.lm_head.weight`, which HF omits because
# it is tied to `embed_tokens` -- so a genuine warm start sits at 99.9% and anything materially below
# that means the checkpoint and the model are not the same architecture.
_MIN_COVERAGE = 0.99

# How many tensors to read back and byte-compare after the load. Cheap, and it is the difference
# between "load_state_dict did not complain" and "the weights in the model are stage 1's weights".
_VERIFY_SAMPLE = 24


def _load_merged_state_dict(checkpoint_dir: Path) -> dict:
    """Read a `save_pretrained` model dir into a state dict, sharded or not."""
    shards = sorted(checkpoint_dir.glob("*.safetensors"))
    if shards:
        from safetensors.torch import load_file

        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(str(shard)))
        return state_dict

    bins = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if not bins:
        raise FileNotFoundError(
            f"{checkpoint_dir} has no *.safetensors and no pytorch_model*.bin. Stage 1 must run "
            "with --merge_lora_during_training True so the merged VLM lands in the checkpoint dir."
        )
    state_dict = {}
    for shard in bins:
        state_dict.update(torch.load(shard, map_location="cpu", weights_only=True))
    return state_dict


def _verify_loaded(module, state_dict: dict, sample: int = _VERIFY_SAMPLE) -> tuple[int, int]:
    """Read the weights back out of `module` and byte-compare them against `state_dict`.

    `load_state_dict` reporting no complaint is not the same claim as "the tensors in the model are
    the ones in the file": a silent dtype cast, a `_load_from_state_dict` override or a shape-1
    broadcast would all pass quietly. So a deterministic spread of tensors -- the sentinels plus an
    even stride through the rest, which reaches every backbone rather than just the first -- is read
    back and compared exactly, in the dtype the parameter actually holds (which is the dtype
    `load_state_dict` cast to, so the comparison stays exact rather than approximate).
    """
    keys = [k for k in _REQUIRED_KEYS if k in state_dict]
    rest = [k for k in state_dict if k not in keys]
    stride = max(len(rest) // max(sample - len(keys), 1), 1)
    keys += rest[::stride][: max(sample - len(keys), 0)]

    live = module.state_dict()
    same = 0
    for key in keys:
        got = live[key].detach().cpu()
        if torch.equal(got, state_dict[key].to(got.dtype)):
            same += 1
        else:
            raise RuntimeError(
                f"warm start verification FAILED on {key}: the tensor in the model does not match "
                f"the one in {CFG.init_from} after load_state_dict "
                f"(max |d| {(got.float() - state_dict[key].float()).abs().max().item():.4g}). "
                "Training would silently start from something other than stage 1."
            )
    return same, len(keys)


def _tied_to_loaded(model, missing, state_dict: dict) -> list:
    """Which of the missing keys are aliases of a tensor the load *did* populate.

    `save_pretrained` omits weights that are tied to another tensor -- on this model
    `language_model.lm_head.weight`, since `tie_word_embeddings: true` shares it with
    `embed_tokens.weight` -- so they are legitimately absent from the file without being unloaded:
    writing the partner writes them too.

    Storage identity is the test, rather than HF's `_tied_weights_keys`: that attribute lives on the
    inner `Qwen2ForCausalLM`, not on the `OpenVLAForActionPrediction` wrapper, so asking the wrapper
    reports every tied weight as a gap. Comparing `data_ptr()` asks the question that actually
    matters -- does this tensor share memory with one that was just loaded -- and needs no knowledge
    of which keys upstream considers tied.
    """
    live = model.state_dict()
    loaded = {live[k].data_ptr() for k in state_dict if k in live}
    return [k for k in missing if k in live and live[k].data_ptr() in loaded]


def load_stage1_vlm(model) -> None:
    """Load stage 1's merged VLM into a freshly built, pre-LoRA `OpenVLAForActionPrediction`."""
    checkpoint_dir = Path(CFG.init_from)
    state_dict = _load_merged_state_dict(checkpoint_dir)

    missing = [k for k in _REQUIRED_KEYS if k not in state_dict]
    if missing:
        raise RuntimeError(
            f"{checkpoint_dir} is missing {missing} -- it does not look like a merged VLA-Adapter "
            "checkpoint. Check that stage 1 finished a save with merge_lora_during_training=True."
        )

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"{len(result.unexpected_keys)} unexpected keys loading {checkpoint_dir} into the "
            f"stage-2 model, e.g. {result.unexpected_keys[:5]}. The two architectures disagree."
        )

    total = len(state_dict) + len(result.missing_keys)
    coverage = len(state_dict) / max(total, 1)
    tied = _tied_to_loaded(model, result.missing_keys, state_dict)
    untied = [k for k in result.missing_keys if k not in tied]
    if coverage < _MIN_COVERAGE:
        raise RuntimeError(
            f"{checkpoint_dir} covers only {len(state_dict)}/{total} of the stage-2 model's tensors "
            f"({coverage:.1%} < {_MIN_COVERAGE:.0%}); {len(untied)} untied keys would keep their "
            f"random init, e.g. {untied[:5]}. That is not a warm start."
        )

    verified, sampled = _verify_loaded(model, state_dict)
    params = sum(v.numel() for v in state_dict.values())
    shards = len(sorted(checkpoint_dir.glob("*.safetensors"))) or len(
        sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    )
    step = "latest" if CFG.init_step == -1 else CFG.init_step
    term.block(
        "RLA warm start: stage 1 -> stage 2",
        [
            ("from", term.paint(str(checkpoint_dir), "cyan") + f"   step {step}"),
            ("vlm", f"{len(state_dict)} tensors / {params / 1e6:.1f}M params from {shards} shard(s)"),
            ("coverage", f"{term.mark(True)}  {len(state_dict)}/{total} of the model's tensors "
                         f"({coverage:.1%}), 0 unexpected"),
            ("absent", f"{term.mark(not untied)}  "
                       + (f"{len(untied)} LEFT AT RANDOM INIT: {untied[:3]}" if untied
                          else f"{len(tied)} tied weight(s), populated via shared storage"
                               + (f" {tied[:2]}" if tied else " -- none absent"))),
            ("read back", f"{term.mark(verified == sampled)}  {verified}/{sampled} sampled tensors "
                          "bit-identical to the file after load"),
            ("lora", "fresh adapter next, lora_B=0 -> identity at step 0"),
        ],
    )


def _splice_gates(state_dict: dict, module) -> int:
    """Widen stage-1 `gating_factor` tensors when stage 1 ran without the action tokens.

    `RLA_ACTION_TOKENS=0` gives stage 1 a `(num_rla_tokens, 1)` gate; stage 2 wants
    `(NUM_ACTIONS_CHUNK + num_rla_tokens, 1)`, laid out as action positions first. The RLA rows are
    copied into the tail and the action rows are left at their zero init -- the same value the
    upstream scalar starts from, so the vision stream re-opens for the action tokens from scratch.

    Returns how many tensors were adjusted. Zero for the recommended configuration, where both
    stages carry the action tokens and the shapes already match.
    """
    target = module.state_dict()
    adjusted, shapes = 0, ""
    for key, value in list(state_dict.items()):
        if not key.endswith("gating_factor"):
            continue
        want = target[key]
        if want.shape == value.shape:
            continue
        if value.ndim != 2 or want.ndim != 2 or value.shape[1] != want.shape[1]:
            raise RuntimeError(
                f"cannot splice {key}: checkpoint {tuple(value.shape)} into {tuple(want.shape)}. "
                "Stage 1 and stage 2 disagree about RLA_STEPS/RLA_QUERIES, not just about whether "
                "the action tokens were present."
            )
        # The RLA rows are always the tail, so align on the end: widening leaves the new action
        # rows at their zero init, narrowing drops them.
        n_want, n_have = want.shape[0], value.shape[0]
        spliced = torch.zeros_like(want)
        if n_have <= n_want:
            spliced[n_want - n_have:] = value.to(want.dtype)
        else:
            spliced[:] = value[n_have - n_want:].to(want.dtype)
        state_dict[key] = spliced
        # Recorded here rather than read off the loop variables afterwards: the loop walks every key
        # in the state dict, so `key`/`value` end up bound to whatever came last, not to a gate.
        shapes = f"{tuple(value.shape)} -> {tuple(want.shape)}"
        adjusted += 1
    if adjusted:
        term.say(
            term.paint("[rla]", "yellow", "bold")
            + f" warm start: spliced {adjusted} gating_factor tensors {shapes}; the action positions "
            "start at zero. Note the RLA tokens' absolute RoPE positions also shift by "
            "NUM_ACTIONS_CHUNK between the two stages -- see rla/config.py RlaConfig.seq_len."
        )
    return adjusted


def make_get_peft_model(orig):
    """Wrap `peft.get_peft_model` so stage 1's weights land in the base model first."""

    def get_peft_model_rla(model, peft_config, *args, **kwargs):
        if CFG.init_from:
            load_stage1_vlm(model)
        return orig(model, peft_config, *args, **kwargs)

    return get_peft_model_rla


def make_init_module(orig, finetune_mod):
    """Wrap `finetune.init_module` to warm-start the action head and proprio projector.

    Mirrors upstream's body (construct -> optional load -> bf16 -> device -> DDP) because the load
    has to happen between construction and the DDP wrap, and there is no other seam. Upstream's own
    `--resume` branch is left reachable for anyone who wants it.
    """

    def init_module_rla(
        module_class,
        module_name,
        cfg,
        device_id,
        module_args,
        to_bf16=False,
        find_unused_params=False,
    ):
        if not CFG.init_from:
            return orig(
                module_class, module_name, cfg, device_id, module_args, to_bf16, find_unused_params
            )

        module = module_class(**module_args)
        finetune_mod.count_parameters(module, module_name)

        suffix = "latest" if CFG.init_step == -1 else CFG.init_step
        path = Path(CFG.init_from) / f"{module_name}--{suffix}_checkpoint.pt"
        if path.exists():
            state_dict = finetune_mod.remove_ddp_in_checkpoint(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            spliced = _splice_gates(state_dict, module)
            # Strict otherwise: with matching RLA_* the two stages build the identical module, and a
            # shape or name mismatch must fail here rather than silently randomise half the head.
            module.load_state_dict(state_dict)
            verified, sampled = _verify_loaded(module, state_dict)
            term.block(
                f"RLA warm start: {module_name}",
                [
                    ("from", term.paint(str(path), "cyan")),
                    ("load", f"{term.mark(True)}  {len(state_dict)} tensors, strict "
                             f"(every key matched by name and shape)"),
                    ("read back", f"{term.mark(verified == sampled)}  {verified}/{sampled} sampled "
                                  "tensors bit-identical to the file after load"),
                    ("gates", "unchanged -- both stages share the token layout" if not spliced
                              else term.paint(f"{spliced} gating_factor tensors spliced", "yellow")),
                ],
            )
        else:
            term.say(
                term.paint("[rla]", "yellow", "bold")
                + f" warm start: no {path.name} in {CFG.init_from}; {module_name} keeps its "
                "freshly-initialised weights"
            )

        if to_bf16:
            module = module.to(torch.bfloat16)
        module = module.to(device_id)
        return finetune_mod.wrap_ddp(module, device_id, find_unused_params)

    return init_module_rla
