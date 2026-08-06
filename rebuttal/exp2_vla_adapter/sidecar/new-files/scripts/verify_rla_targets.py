#!/usr/bin/env python
"""Gates for the RLA -> VLA-Adapter auxiliary-target integration.

The design note this implements got several things about the pipeline wrong, all of which would
have failed *silently* (see docs/rla-vla-adapter.md §2). These gates turn each of those into a hard
error. Gates 1, 2, 3, 4, 5 and 7 run **without a trained RLA autoencoder**, which is the point: the
plumbing can be locked down before `f_enc` exists, and re-running the extractor is then the only
delta.

    Gate 1  frames and actions are index-aligned; action[k] drives s[k] -> s[k+1]
    Gate 2  sha1(state) is a per-episode bijection (and episode_metadata/file_path is NOT)
    Gate 3  end-to-end index alignment through the real RLDSDataset, via a synthetic sidecar
            whose values encode (episode_index, t, i)          <-- the one that matters
    Gate 4  RLA_STEPS=0 is byte-identical to upstream: same key set, same outputs, strict load
            of a released LIBERO-*-Pro action head
    Gate 5  RLA_STEPS=8 gives the right shapes and finite grads, and *does* change the action
            output -- so the baseline arm must be RLA_STEPS=0, not lambda_rla=0
    Gate 6  the extractor's encode path equals RlaAutoencoderMultiViewTrainer.inference_batch
    Gate 7  the recomputed dataset statistics match the released reference, i.e. editing
            libero_dataset_transform did not perturb action/proprio normalisation

Usage (after `source scripts/vla_adapter_env.sh`):

    .venv/bin/python scripts/verify_rla_targets.py                       # 1,2,3,4,5,7
    .venv/bin/python scripts/verify_rla_targets.py --gates 3
    .venv/bin/python scripts/verify_rla_targets.py --sidecar data/libero_rla/16x64-<stamp>
    .venv/bin/python scripts/verify_rla_targets.py --gates 6 --rla-run runs/<rla-run>/<stamp>

Gates 3/4/5 read RLA_STEPS, which prismatic/vla/constants.py resolves at import time, so each runs
in its own subprocess with the right environment. That is also what a real arm switch does.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "VLA-Adapter"))

# The commit that vendored VLA-Adapter in-tree. Gate 4 diffs the current action head against the
# version at this commit, which is upstream + the eight porting patches and nothing else.
UPSTREAM_REF = "4e05436"
DEFAULT_SUITE = "libero_spatial"
RELEASED_CKPT = "runs/weights/vla-adapter/LIBERO-Spatial-Pro"

GREEN, RED, YELLOW, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"


def banner(msg: str) -> None:
    print(f"\n{BOLD}=== {msg}{RESET}", flush=True)


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}  {msg}", flush=True)


def skip(msg: str) -> None:
    print(f"  {YELLOW}SKIP{RESET}  {msg}", flush=True)


def pin_tf_to_cpu():
    """TF decodes TFRecords, torch owns the GPU. Must happen before TF grabs a device."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    # Grappler's CropAndResize cost-model warnings are harmless and drown out the gate output.
    tf.get_logger().setLevel("ERROR")
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        pass
    return tf


def rlds_dir(suite: str) -> Path:
    return REPO_ROOT / "data" / "libero" / f"{suite}_no_noops" / "1.0.0"


def iter_state_only(suite: str, limit: int = -1):
    """Yield (episode_index, states, actions, file_path) without decoding any JPEG."""
    pin_tf_to_cpu()
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(rlds_dir(suite)))
    total = builder.info.splits["train"].num_examples
    if limit > 0:
        total = min(total, limit)
    no_decode = tfds.decode.SkipDecoding()
    dataset = builder.as_dataset(
        split=f"train[:{total}]",
        shuffle_files=False,
        decoders={"steps": {"observation": {"image": no_decode, "wrist_image": no_decode}}},
    )
    for ep_idx, episode in enumerate(dataset):
        steps = list(episode["steps"])
        states = np.stack([s["observation"]["state"].numpy() for s in steps]).astype(np.float32)
        actions = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
        yield ep_idx, states, actions, episode["episode_metadata"]["file_path"].numpy().decode()


def episode_key(states: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(states, dtype=np.float32).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Gate 1 -- frame/action alignment
# --------------------------------------------------------------------------------------------
def gate_1_alignment(args) -> None:
    banner("Gate 1: N frames <-> N actions, and action[k] drives s[k] -> s[k+1]")
    pin_tf_to_cpu()
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(rlds_dir(args.suite)))
    limit = args.limit if args.limit > 0 else 6
    dataset = builder.as_dataset(split=f"train[:{limit}]", shuffle_files=False)

    lag_corr = {lag: [] for lag in (-2, -1, 0, 1, 2)}
    for episode in dataset:
        steps = list(episode["steps"])
        num = len(steps)
        images = np.stack([s["observation"]["image"].numpy() for s in steps])
        wrists = np.stack([s["observation"]["wrist_image"].numpy() for s in steps])
        states = np.stack([s["observation"]["state"].numpy() for s in steps])
        actions = np.stack([s["action"].numpy() for s in steps])
        assert len(images) == len(wrists) == len(states) == len(actions) == num, (
            f"ragged episode: {len(images)}/{len(wrists)}/{len(states)}/{len(actions)} vs {num}"
        )

        # The discriminating test. The end-effector version (cos between the action's xyz and the
        # achieved delta) does NOT discriminate: consecutive OSC deltas are too correlated for a
        # one-step lag to look different. The gripper does, because the command is near-binary.
        width = states[:, 6] - states[:, 7]
        dwidth = np.diff(width)
        grip = actions[:, 6]
        for lag in lag_corr:
            a = grip[max(0, -lag) : len(grip) - max(0, lag)]
            d = dwidth[max(0, lag) : len(dwidth) - max(0, -lag)]
            n = min(len(a), len(d))
            a, d = a[:n], d[:n]
            if a.std() > 0 and d.std() > 0:
                lag_corr[lag].append(np.corrcoef(a, d)[0, 1])

    means = {lag: float(np.nanmean(v)) for lag, v in lag_corr.items() if v}
    for lag in sorted(means):
        print(f"      corr(gripper_cmd[k], width[k+1+lag] - width[k+lag])  lag {lag:+d}: {means[lag]:+.4f}")
    peak = min(means, key=lambda k: means[k])  # the correlation is negative: close -> width shrinks
    assert peak == 0, (
        f"gripper-lag correlation peaks at lag {peak:+d}, not 0 -- the frame/action convention is "
        "NOT action[k]: s[k] -> s[k+1], and every RLA target is off by one."
    )
    ok(f"{limit} episodes: N frames == N actions == N states; gripper-lag peaks at lag 0")
    ok("=> N frames but only N-1 real transitions; tail targets must clamp to N-1")


# --------------------------------------------------------------------------------------------
# Gate 2 -- the join key
# --------------------------------------------------------------------------------------------
def gate_2_join_key(args) -> None:
    banner("Gate 2: sha1(observation/state) is a per-episode bijection")
    keys, paths, lengths = {}, {}, {}
    for ep_idx, states, _actions, file_path in iter_state_only(args.suite, args.limit):
        key = episode_key(states)
        assert key not in keys, (
            f"state-hash collision: episodes {keys[key]} and {ep_idx} hash to {key}"
        )
        keys[key] = ep_idx
        lengths[ep_idx] = len(states)
        paths.setdefault(file_path, []).append(ep_idx)
    ok(f"{len(keys)} distinct keys over {len(keys)} episodes in {args.suite} -- bijective")

    shared = {p: e for p, e in paths.items() if len(e) > 1}
    assert shared, (
        "episode_metadata/file_path looks unique here; the docs claim it is not. Re-check before "
        "relying on either as a key."
    )
    example = next(iter(shared.items()))
    ok(
        f"episode_metadata/file_path is NOT unique ({len(shared)} paths shared by >1 episode, "
        f"e.g. {len(example[1])} episodes share ...{example[0][-46:]}) -- correctly unused as a key"
    )

    if not args.sidecar:
        skip("no --sidecar given; not cross-checking keys.json")
        return

    keys_path = Path(args.sidecar) / args.suite / "keys.json"
    assert keys_path.exists(), f"{keys_path} missing -- run scripts/extract_rla_latents.py first"
    sidecar_keys = json.loads(keys_path.read_text())
    missing = [k for k in keys if k not in sidecar_keys]
    assert not missing, f"{len(missing)} episodes have no sidecar entry (first: {missing[0]})"
    for key, ep_idx in keys.items():
        entry = sidecar_keys[key]
        assert entry["episode_index"] == ep_idx, (
            f"sidecar says key {key} is episode {entry['episode_index']}, RLDS says {ep_idx}"
        )
        assert entry["traj_len"] == lengths[ep_idx], (
            f"episode {ep_idx}: sidecar traj_len={entry['traj_len']}, RLDS has {lengths[ep_idx]}"
        )
    ok(f"all {len(keys)} keys present in the sidecar with matching episode_index and traj_len")

    # The normalisation statistics must cover every frame on disk. This caught a real bug: the
    # extractor used to accumulate inline, which silently skipped episodes a *resumed* run did not
    # rewrite, so every target got normalised against statistics from a subset of the corpus.
    # Only meaningful when --limit is not restricting the RLDS side.
    if args.limit <= 0:
        manifest = json.loads((Path(args.sidecar) / "manifest.json").read_text())
        stats = json.loads((Path(args.sidecar) / "stats.json").read_text())
        expected = sum(v["traj_len"] for v in sidecar_keys.values()) * manifest["chunk"]
        assert stats["count"] == expected, (
            f"stats.json covers {stats['count']} rows but the sidecar holds {expected} "
            f"({sum(v['traj_len'] for v in sidecar_keys.values())} frames x {manifest['chunk']}). "
            "Re-run the extractor: the normalisation statistics are computed from a subset."
        )
        ok(f"stats.json covers all {expected} rows ({expected // manifest['chunk']} frames x "
           f"{manifest['chunk']} chunk steps)")


# --------------------------------------------------------------------------------------------
# Gate 3 -- end-to-end index alignment through the real dataloader
# --------------------------------------------------------------------------------------------
SYNTHETIC_Q, SYNTHETIC_D, SYNTHETIC_A = 1, 4, 8


def build_synthetic_sidecar(out: Path, suite: str) -> int:
    """A sidecar whose values spell out (episode_index, t, i, traj_len).

    Always covers the **whole** suite -- `--limit` deliberately does not apply here. Partial
    coverage would leave most sampled frames as zero-filled misses, and the gate would then be
    asserting against zeros instead of against the join.

    Q=1/D=4 instead of the real 16/64 is what makes full coverage cheap (<5 MB for a suite). It also
    exercises RlaSidecar's shape-consistency check against the manifest.
    """
    suite_dir = out / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    keys = {}
    for ep_idx, states, _actions, _path in iter_state_only(suite, limit=-1):
        num = len(states)
        z = np.zeros((num, SYNTHETIC_A, SYNTHETIC_Q, SYNTHETIC_D), dtype=np.float16)
        z[:, :, 0, 0] = ep_idx
        z[:, :, 0, 1] = np.arange(num)[:, None]
        z[:, :, 0, 2] = np.arange(SYNTHETIC_A)[None, :]
        z[:, :, 0, 3] = num
        np.save(suite_dir / f"ep_{ep_idx:06d}.npy", z)
        keys[episode_key(states)] = {
            "file": f"ep_{ep_idx:06d}.npy",
            "episode_index": ep_idx,
            "traj_len": num,
        }
    (suite_dir / "keys.json").write_text(json.dumps(keys, indent=1, sort_keys=True))
    # Identity normalisation, so the encoded integers survive RlaSidecar.lookup().
    (out / "stats.json").write_text(
        json.dumps(
            {
                "count": 0,
                "mean": np.zeros((SYNTHETIC_Q, SYNTHETIC_D)).tolist(),
                "std": np.ones((SYNTHETIC_Q, SYNTHETIC_D)).tolist(),
                "std_floor": 1e-3,
                "global_rms": 1.0,
            }
        )
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "chunk": SYNTHETIC_A,
                "num_tokens": SYNTHETIC_Q,
                "token_dim": SYNTHETIC_D,
                "random_encoder": False,
                "synthetic": True,
                "pairs": "anchored",
            }
        )
    )
    return len(keys)


def _child_gate_3(args) -> None:
    """Runs with RLA_STEPS/QUERIES/DIM set for the synthetic sidecar."""
    pin_tf_to_cpu()  # also quiets grappler, before prismatic imports TF
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import RLA_DIM, RLA_QUERIES, RLA_STEPS
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

    config_dir = REPO_ROOT / "third_party" / "VLA-Adapter" / "pretrained_models" / "configs"
    processor = AutoProcessor.from_pretrained(str(config_dir), trust_remote_code=True)
    batch_transform = RLDSBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
        use_minivlm=True,
    )
    # Relative "data/libero" with cwd=third_party/VLA-Adapter, exactly as scripts/train_libero.sh
    # invokes finetune.py. This is not cosmetic: get_dataset_statistics hashes `str(builder.info)`,
    # which embeds the data_dir *string*, so a different spelling of the same physical directory
    # forces a fresh statistics recompute and writes a second cache file.
    dataset = RLDSDataset(
        Path("data/libero"),
        f"{args.suite}_no_noops",
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=1000,
        image_aug=True,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    loader = DataLoader(dataset, batch_size=8, sampler=None, collate_fn=collator, num_workers=0)

    from prismatic.vla import rla_targets

    num_checked, episodes, timesteps = 0, set(), set()
    for batch_idx, batch in enumerate(loader):
        assert "rla" in batch, "the collator did not emit 'rla'"
        rla = batch["rla"]
        assert rla.shape[1:] == (RLA_STEPS, RLA_QUERIES * RLA_DIM), rla.shape
        z = rla.reshape(rla.shape[0], RLA_STEPS, RLA_QUERIES, RLA_DIM).numpy()
        ep, t, i, n = z[:, :, 0, 0], z[:, :, 0, 1], z[:, :, 0, 2], z[:, :, 0, 3]

        # Check misses first: a zero-filled row would otherwise trip the axis assertion below with
        # a much less informative message.
        assert rla_targets.get_sidecar().num_misses == 0, (
            f"{rla_targets.get_sidecar().num_misses} episode(s) missed the sidecar by batch "
            f"{batch_idx}. Either the join is broken, or the sidecar does not cover every episode "
            "of this suite (a partial sidecar makes this gate meaningless)."
        )

        assert np.array_equal(i, np.broadcast_to(np.arange(RLA_STEPS), i.shape)), (
            f"the chunk axis is scrambled -- expected 0..{RLA_STEPS - 1} per row, got\n{i[:2]}"
        )
        assert (ep == ep[:, :1]).all(), "episode index varies along the chunk axis"
        assert (t == t[:, :1]).all(), "timestep varies along the chunk axis"
        # chunk_act_obs truncates to effective_traj_len = N - (NUM_ACTIONS_CHUNK - 1).
        assert (t[:, 0] >= 0).all() and (t[:, 0] <= n[:, 0] - RLA_STEPS).all(), (
            f"timestep outside the effective range: t={t[:, 0]} n={n[:, 0]}"
        )
        episodes.update(ep[:, 0].astype(int).tolist())
        timesteps.update(t[:, 0].astype(int).tolist())
        num_checked += rla.shape[0]
        if batch_idx + 1 >= args.batches:
            break

    assert rla_targets.get_sidecar().num_misses == 0, (
        f"{rla_targets.get_sidecar().num_misses} episodes missed the sidecar -- the join is broken "
        "or the synthetic sidecar does not cover the whole suite"
    )
    assert len(episodes) > 5 and len(timesteps) > 20, (
        f"suspiciously little variety: {len(episodes)} episodes, {len(timesteps)} timesteps"
    )
    print(
        f"  {GREEN}OK{RESET}  {num_checked} samples, {len(episodes)} distinct episodes, "
        f"{len(timesteps)} distinct timesteps (t in [{min(timesteps)}, {max(timesteps)}]), 0 misses"
    )


def gate_3_end_to_end(args) -> None:
    banner("Gate 3: end-to-end (episode, t, i) alignment through the real RLDSDataset")
    workdir = Path(tempfile.mkdtemp(prefix="rla_gate3_"))
    try:
        num = build_synthetic_sidecar(workdir, args.suite)
        print(f"      synthetic sidecar: {num} episodes, Q={SYNTHETIC_Q} D={SYNTHETIC_D} -> {workdir}")
        run_child(
            "3",
            args,
            env={
                "RLA_STEPS": str(SYNTHETIC_A),
                "RLA_QUERIES": str(SYNTHETIC_Q),
                "RLA_DIM": str(SYNTHETIC_D),
                "RLA_SIDECAR": str(workdir),
            },
            cwd=REPO_ROOT / "third_party" / "VLA-Adapter",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# Gate 4 / 5 -- the action head
# --------------------------------------------------------------------------------------------
def _build_head(module, dim=896):
    from prismatic.vla.constants import ACTION_DIM

    return module(input_dim=dim, hidden_dim=dim, action_dim=ACTION_DIM, use_pro_version=True)


def _head_inputs(dim=896, num_patches=512, batch=2, layers=25):
    import torch

    from prismatic.vla.constants import NUM_TOKENS

    torch.manual_seed(7)
    return (
        torch.randn(batch, layers, num_patches + NUM_TOKENS, dim),
        torch.randn(batch, 8),
        torch.nn.Linear(8, dim).to(torch.bfloat16),
    )


def _child_gate_4(args) -> None:
    """Runs with RLA_STEPS=0. Must be indistinguishable from the pre-RLA code."""
    import importlib.util

    import torch

    from prismatic.models.action_heads import L1RegressionActionHead as Modified
    from prismatic.vla.constants import RLA_STEPS

    assert RLA_STEPS == 0, "gate 4 must run with RLA_STEPS=0"

    # (a) bit-identity against the vendored-upstream action head.
    upstream_src = subprocess.run(
        ["git", "show", f"{args.upstream_ref}:third_party/VLA-Adapter/prismatic/models/action_heads.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if upstream_src.returncode != 0:
        skip(f"cannot read action_heads.py at {args.upstream_ref}: {upstream_src.stderr.strip()}")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(upstream_src.stdout)
            upstream_path = handle.name
        try:
            spec = importlib.util.spec_from_file_location("ah_upstream", upstream_path)
            upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream)
        finally:
            os.unlink(upstream_path)

        torch.manual_seed(0)
        head_mod = _build_head(Modified)
        torch.manual_seed(0)
        head_up = _build_head(upstream.L1RegressionActionHead)

        assert sorted(head_mod.state_dict()) == sorted(head_up.state_dict()), (
            "state-dict key sets differ from upstream with RLA_STEPS=0"
        )
        for key, value in head_up.state_dict().items():
            assert torch.equal(value, head_mod.state_dict()[key]), f"init differs at {key}"
        ok(f"key sets and init weights identical to {args.upstream_ref} ({len(head_up.state_dict())} tensors)")

        hidden, proprio, projector = _head_inputs()
        for phase in ("Inference", "Training"):
            torch.manual_seed(99)
            out_up = head_up.predict_action(hidden, proprio, projector, phase=phase)
            torch.manual_seed(99)
            out_mod = head_mod.predict_action(hidden, proprio, projector, phase=phase)
            assert torch.equal(out_up, out_mod), (
                f"phase={phase}: output differs from upstream, max |d| = "
                f"{float((out_up - out_mod).abs().max()):.3e}"
            )
        ok("predict_action bit-identical to upstream for phase=Inference and phase=Training")

    # (b) a released checkpoint still loads strict.
    ckpt_path = REPO_ROOT / args.released_ckpt / "action_head--checkpoint.pt"
    if not ckpt_path.exists():
        skip(f"{ckpt_path} not present; not checking the strict load")
        return
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    _build_head(Modified).load_state_dict(state)  # strict
    ok(f"released {Path(args.released_ckpt).name} action head loads strict ({len(state)} tensors)")


def _child_gate_5(args) -> None:
    """Runs with RLA_STEPS=NUM_ACTIONS_CHUNK."""
    import torch

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, RLA_DIM, RLA_QUERIES, RLA_STEPS

    assert RLA_STEPS == NUM_ACTIONS_CHUNK, "gate 5 must run with RLA_STEPS=NUM_ACTIONS_CHUNK"

    torch.manual_seed(0)
    head = L1RegressionActionHead(input_dim=896, hidden_dim=896, action_dim=ACTION_DIM, use_pro_version=True)
    rla_keys = [k for k in head.state_dict() if "rla" in k]
    assert len(rla_keys) == len(head.model.mlp_resnet_blocks) + 4, (
        f"expected one gating_factor_rla per block plus fc2_rla/layer_norm2_rla, got {rla_keys}"
    )
    ok(f"{len(rla_keys)} RLA-specific tensors ({len(head.model.mlp_resnet_blocks)} gates + 2 heads)")

    hidden, proprio, projector = _head_inputs()
    batch = hidden.shape[0]

    torch.manual_seed(99)
    action, z_hat = head.predict_action(hidden, proprio, projector, phase="Training", return_rla=True)
    assert action.shape == (batch, NUM_ACTIONS_CHUNK, ACTION_DIM), action.shape
    assert z_hat.shape == (batch, RLA_STEPS, RLA_QUERIES * RLA_DIM), z_hat.shape
    ok(f"action {tuple(action.shape)}, z_hat {tuple(z_hat.shape)}")

    torch.manual_seed(99)
    action_only = head.predict_action(hidden, proprio, projector, phase="Training")
    assert torch.is_tensor(action_only), (
        "predict_action must return the bare action tensor by default -- modeling_prismatic.py "
        "calls it that way and is deliberately left unmodified"
    )
    ok("predict_action(...) without return_rla still returns just the action tensor")

    torch.nn.L1Loss()(z_hat, torch.randn_like(z_hat)).backward()
    grad_head = head.model.fc2_rla.weight.grad
    grad_gate = head.model.mlp_resnet_blocks[0].gating_factor_rla.grad
    assert grad_head is not None and torch.isfinite(grad_head).all() and grad_head.abs().sum() > 0
    assert grad_gate is not None and torch.isfinite(grad_gate).all()
    assert head.model.fc2.weight.grad is None, (
        "the RLA loss reached the action output head; the two branches should be disjoint"
    )
    ok(f"aux loss backprops: |grad fc2_rla| = {float(grad_head.abs().sum()):.3f}, "
       f"|grad gating_factor_rla| = {float(grad_gate.abs().sum()):.2e}, action head untouched")

    # The RLA tokens live in the shared self-attention, so the *action* path changes too. Prove it,
    # so nobody uses lambda_rla=0 as the control arm.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_child", "5ref"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "RLA_STEPS": "0", "RLA_SIDECAR": ""},
    )
    assert result.returncode == 0, f"reference child failed:\n{result.stdout}\n{result.stderr}"
    baseline = torch.tensor(json.loads(result.stdout.strip().splitlines()[-1]))
    torch.manual_seed(99)
    with_rla = head.predict_action(hidden, proprio, projector, phase="Inference")
    delta = float((baseline - with_rla.detach()).abs().max())
    assert delta > 1e-6, (
        "the action output is unchanged by the RLA tokens -- they are not in the shared attention"
    )
    ok(f"action output differs from RLA_STEPS=0 by {delta:.4f} even at lambda_rla=0 "
       f"=> the control arm MUST be RLA_STEPS=0")


def _child_gate_5ref() -> None:
    """Emit the RLA_STEPS=0 action output for gate 5 to compare against. Prints JSON on stdout."""
    import torch

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.vla.constants import ACTION_DIM

    torch.manual_seed(0)
    head = L1RegressionActionHead(input_dim=896, hidden_dim=896, action_dim=ACTION_DIM, use_pro_version=True)
    hidden, proprio, projector = _head_inputs()
    torch.manual_seed(99)
    print(json.dumps(head.predict_action(hidden, proprio, projector, phase="Inference").tolist()))


# --------------------------------------------------------------------------------------------
# Gate 6 -- extractor vs trainer
# --------------------------------------------------------------------------------------------
def gate_6_extractor_matches_trainer(args) -> None:
    banner("Gate 6: the extractor's encode path == RlaAutoencoderMultiViewTrainer.inference_batch")
    import torch
    from torch.utils.data import DataLoader

    from src import datasets, models, trainers
    from src.models.plucker_embedding import compute_plucker_rays
    from utils.misc import load_config, move_to_device

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from extract_rla_latents import RlaLatentEncoder

    if not torch.cuda.is_available():
        skip("no GPU; gate 6 needs one for DINOv3")
        return

    from easydict import EasyDict as edict

    cfg = edict(load_config(args.rla_config))
    encoder = RlaLatentEncoder(
        config_path=args.rla_config,
        run_dir=args.rla_run,
        step=args.step,
        device="cuda",
        random_encoder=args.rla_run is None,
    )

    # Same weights on both sides: hand the extractor's own encoder to the trainer.
    model_dict = {
        name: getattr(models, spec.name)(**spec.args).cuda()
        for name, spec in cfg.models.items()
        if name != "encoder"
    }
    model_dict["encoder"] = encoder.encoder

    dataset = getattr(datasets, cfg.dataset.name)(None, **cfg.dataset.args)
    trainer_args = dict(cfg.trainer.args)
    trainer_args.update(
        batch_size=args.batch_size, num_workers=0, max_steps=1, image_decoder_ckpt="", load_dir=None
    )
    with tempfile.TemporaryDirectory(prefix="rla_gate6_") as output_dir:
        trainer = getattr(trainers, cfg.trainer.name)(
            model_dict, dataset, **trainer_args, output_dir=output_dir,
            step=None, wandb_run=None, val_dataset=None, cfg=cfg,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        batch = move_to_device(next(iter(loader)), trainer.device)
        for module in trainer.models.values():
            module.eval()
        with torch.no_grad():
            reference = trainer.inference_batch(trainer.models, batch, training=False)["enc_tokens"]

            # Now the extractor's path over the same pixels.
            rgbs = batch["rgbs"].detach().cpu().numpy()  # [B, K, Cam, 3, H, W] uint8
            tokens_0, patch_hw = encoder.dino_tokens(rgbs[:, 0])
            tokens_1, _ = encoder.dino_tokens(rgbs[:, 1])
            num, cams, lp, ch = tokens_0.shape
            plucker = None
            if encoder.expects_plucker:
                rays = compute_plucker_rays(
                    intrinsics=batch["intrinsics"][:, 0].float(),
                    cam2world=batch["w2c"][:, 0].float(),
                    patch_hw=patch_hw,
                    image_hw=tuple(rgbs.shape[-2:]),
                )
                plucker = rays.reshape(rays.shape[0], -1, 6)
            mine = encoder.encode(
                tokens_0.reshape(num, cams * lp, ch), tokens_1.reshape(num, cams * lp, ch), plucker
            )

    diff = float((reference - mine).abs().max())
    scale = float(reference.abs().max())
    assert torch.allclose(reference, mine, atol=1e-4, rtol=1e-3), (
        f"extractor and trainer disagree: max |d| = {diff:.3e} (|z|max = {scale:.3f})"
    )
    ok(f"enc_tokens match on a real batch: max |d| = {diff:.2e} (|z|max = {scale:.3f}), "
       f"shape {tuple(mine.shape)}")


# --------------------------------------------------------------------------------------------
# Gate 7 -- dataset statistics
# --------------------------------------------------------------------------------------------
def gate_7_dataset_statistics(args) -> None:
    banner("Gate 7: recomputed dataset statistics match the released reference")
    computed = sorted(rlds_dir(args.suite).glob("dataset_statistics_*.json"))
    if not computed:
        skip(
            f"no dataset_statistics_*.json under {rlds_dir(args.suite)} yet -- it is written the "
            "first time the pipeline runs (gate 3 does that)"
        )
        return

    reference_path = REPO_ROOT / args.released_ckpt / "dataset_statistics.json"
    if not reference_path.exists():
        skip(f"{reference_path} not present; nothing to compare against")
        return
    reference = json.loads(reference_path.read_text())
    reference = reference.get(f"{args.suite}_no_noops", reference)

    # Every cached file, not just the newest: the cache key hashes `str(builder.info)`, which
    # embeds the data_dir *string*, so running from a different cwd legitimately produces a second
    # file for the same data. They must all agree with the released reference.
    for path in computed:
        ours = json.loads(path.read_text())
        worst = 0.0
        for field in ("action", "proprio"):
            for stat in ("mean", "std", "q01", "q99", "min", "max"):
                if stat not in reference.get(field, {}) or stat not in ours.get(field, {}):
                    continue
                a = np.asarray(reference[field][stat], dtype=np.float64)
                b = np.asarray(ours[field][stat], dtype=np.float64)
                delta = float(np.abs(a - b).max())
                assert delta < 1e-5, (
                    f"{path.name}: {field}.{stat} drifted by {delta:.3e} from the released reference"
                )
                worst = max(worst, delta)
        for count in ("num_transitions", "num_trajectories"):
            assert reference[count] == ours[count], (
                f"{path.name}: {count} released {reference[count]} vs ours {ours[count]}"
            )
        print(
            f"      {path.name[:34]}...  worst |d| = {worst:.2e}  "
            f"transitions={ours['num_transitions']} trajectories={ours['num_trajectories']}"
        )
    ok(
        f"all {len(computed)} cached statistics file(s) match "
        f"{Path(args.released_ckpt).name} to <1e-5"
    )
    ok("=> editing libero_dataset_transform did not perturb action/proprio normalisation")


# --------------------------------------------------------------------------------------------
def run_child(gate: str, args, env: dict, cwd: Path = REPO_ROOT) -> None:
    """Re-exec this script for a gate that needs a specific RLA_* environment."""
    command = [sys.executable, str(Path(__file__).resolve()), "--_child", gate, "--suite", args.suite]
    if args.limit > 0:
        command += ["--limit", str(args.limit)]
    command += ["--batches", str(args.batches), "--upstream-ref", args.upstream_ref]
    command += ["--released-ckpt", args.released_ckpt]
    sys.stdout.flush()
    result = subprocess.run(command, cwd=cwd, env={**os.environ, **env})
    if result.returncode != 0:
        raise SystemExit(f"{RED}gate {gate} FAILED{RESET} (child exit {result.returncode})")


GATES = {
    "1": ("frame/action alignment", gate_1_alignment),
    "2": ("join key", gate_2_join_key),
    "3": ("end-to-end index alignment", gate_3_end_to_end),
    "4": ("baseline byte-equivalence", None),   # child-only
    "5": ("RLA forward/backward", None),        # child-only
    "6": ("extractor == trainer", gate_6_extractor_matches_trainer),
    "7": ("dataset statistics", gate_7_dataset_statistics),
}
DEFAULT_GATES = "1,2,3,4,5,7"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gates", default=DEFAULT_GATES, help=f"comma-separated; default {DEFAULT_GATES}")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--sidecar", default=None, help="a real sidecar to cross-check in gate 2")
    parser.add_argument("--limit", type=int, default=-1, help="episodes per suite (default: all)")
    parser.add_argument("--batches", type=int, default=60, help="dataloader batches in gate 3")
    parser.add_argument("--rla-config", default="configs/rla/16x64_libero.yaml")
    parser.add_argument("--rla-run", default=None, help="gate 6: RLA run dir (default: random f_enc)")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2, help="gate 6 batch size")
    parser.add_argument("--upstream-ref", default=UPSTREAM_REF)
    parser.add_argument("--released-ckpt", default=RELEASED_CKPT)
    parser.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._child == "3":
        _child_gate_3(args)
        return 0
    if args._child == "4":
        _child_gate_4(args)
        return 0
    if args._child == "5":
        _child_gate_5(args)
        return 0
    if args._child == "5ref":
        _child_gate_5ref()
        return 0

    requested = [g.strip() for g in args.gates.split(",") if g.strip()]
    unknown = [g for g in requested if g not in GATES]
    if unknown:
        raise SystemExit(f"unknown gate(s): {unknown}; known: {sorted(GATES)}")

    for gate in requested:
        name, fn = GATES[gate]
        if gate == "4":
            banner("Gate 4: RLA_STEPS=0 is byte-identical to upstream")
            run_child("4", args, env={"RLA_STEPS": "0", "RLA_SIDECAR": ""})
        elif gate == "5":
            banner("Gate 5: RLA_STEPS=NUM_ACTIONS_CHUNK forward/backward")
            run_child("5", args, env={"RLA_STEPS": "8", "RLA_QUERIES": "16", "RLA_DIM": "64",
                                      "RLA_SIDECAR": "unused-by-gate-5"})
        else:
            fn(args)

    print(f"\n{GREEN}{BOLD}all requested gates passed: {', '.join(requested)}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
