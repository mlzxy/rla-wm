"""Gate `align`: the block joined to a frame really belongs to that frame's episode and chunk.

The strongest gate here, and it needs no `f_enc` and no GPU. A synthetic sidecar encodes
`(episode, t, i, gripper(a_{t+i}), N)` into its latents; the **real** `RLDSDataset` +
`RlaBatchTransform` + `RlaCollator` then run, and the decoded gripper trajectory is compared against
`batch["actions"]` for the same batch element.

That comparison is not circular. Checking the target's own encoded `t` against
`observation["timestep"]` would prove nothing -- the target was *fetched* using that timestep. The
gripper command of `a_{t+i}` travels a completely different path (RLDS -> chunk_act_obs -> collator),
so agreement is real evidence. It catches a wrong episode, an off-by-one in `t`, and a reversed
chunk axis. The gripper dimension is the right channel because `action_normalization_mask` leaves it
unnormalised (`EEF_POS`), so it reaches `batch["actions"]` in comparable units.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from rla.tests.harness import REPO, ok, rule

# Q=1 keeps the fixture at a few MB. Channel layout of the synthetic latents:
SYN_Q, SYN_D = 1, 8
CH_EPISODE, CH_T, CH_I, CH_GRIPPER, CH_N = 0, 1, 2, 3, 4


def build_synthetic_sidecar(out: Path, suite: str, rlds_root: str, chunk: int) -> Path:
    """Write a sidecar whose latents encode where they came from."""
    from rla.read_rla_sidecar import SUITES, episode_key, iter_libero_episodes

    suite_dir = out / suite
    if out.exists():
        shutil.rmtree(out)
    suite_dir.mkdir(parents=True)

    keys, total_frames = {}, 0
    for episode in iter_libero_episodes(rlds_root, suite):
        n = episode.traj_len
        # `libero_dataset_transform` does invert_gripper_actions(clip(a[:, -1:], 0, 1)) = 1 - clip.
        # Reproduced by hand from the raw RLDS actions rather than read back out of the pipeline --
        # that is what makes the later comparison independent evidence.
        gripper = 1.0 - np.clip(episode.actions[:, -1], 0.0, 1.0)
        # `chunk_act_obs` builds the chunk as action[min(max(t + i, 0), N-1)]
        # (traj_transforms.py:32-44), so chunk step `i` at anchor `t` is a_{min(t+i, N-1)}.
        t_grid = np.minimum(np.arange(n)[:, None] + np.arange(chunk)[None, :], n - 1)

        z = np.zeros((n, chunk, SYN_Q, SYN_D), dtype=np.float16)
        z[:, :, 0, CH_EPISODE] = episode.episode_index
        z[:, :, 0, CH_T] = np.arange(n)[:, None]
        z[:, :, 0, CH_I] = np.arange(chunk)[None, :]
        z[:, :, 0, CH_GRIPPER] = gripper[t_grid]
        z[:, :, 0, CH_N] = n

        name = f"ep_{episode.episode_index:06d}.npy"
        np.save(suite_dir / name, z)
        keys[episode_key(episode.states)] = {
            "episode_index": episode.episode_index, "file": name, "traj_len": n
        }
        total_frames += n

    (suite_dir / "keys.json").write_text(json.dumps(keys, indent=1))
    (out / "manifest.json").write_text(json.dumps({
        "chunk": chunk, "num_tokens": SYN_Q, "token_dim": SYN_D, "pairs": "anchored",
        "format": "npy", "synthetic": True, "suites": {suite: {"num_episodes": len(keys)}},
    }, indent=1))
    # Identity normalisation, so the decoded values come back as the integers they went in as.
    (out / "stats.json").write_text(json.dumps({
        "count": total_frames * chunk,
        "mean": np.zeros((SYN_Q, SYN_D)).tolist(),
        "std": np.ones((SYN_Q, SYN_D)).tolist(),
        "std_floor": 1e-3,
    }, indent=1))
    print(f"  built synthetic sidecar: {len(keys)} episodes, {total_frames} frames -> {out}")
    return out


def _real_dataloader(dataset_name: str, rlds_root: str, batch_size: int):
    """The real dataset/transform/collator, exactly as `finetune.py` builds them -- minus the VLM,
    which the join does not touch. `dataset_name` is the RLDS name, not the sidecar suite name."""
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSDataset
    from rla.data import RlaBatchTransform, RlaCollator, install_dataset_hook

    config_dir = REPO / "pretrained_models" / "configs"
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(str(config_dir), trust_remote_code=True)

    install_dataset_hook()
    dataset = RLDSDataset(
        Path(rlds_root),
        dataset_name,
        RlaBatchTransform(
            ActionTokenizer(processor.tokenizer),
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=True,
            use_proprio=True,
            use_minivlm=True,
        ),
        resize_resolution=tuple(json.loads((config_dir / "config.json").read_text())["image_sizes"]),
        shuffle_buffer_size=4096,
        image_aug=True,
    )
    collator = RlaCollator(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)


def run(args) -> None:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from rla.read_rla_sidecar import SUITES
    from rla.sidecar import RlaStore, set_store

    rule("align: (episode, t, i) and the gripper trajectory survive the real dataloader")

    if args.suite not in SUITES:
        raise SystemExit(f"unknown suite {args.suite!r}; known: {list(SUITES)}")

    root = Path(args.synthetic or (REPO / "data" / "libero_rla" / f"synthetic-{args.suite}"))
    if args.rebuild or not (root / "manifest.json").exists():
        build_synthetic_sidecar(root, args.suite, args.rlds_root, NUM_ACTIONS_CHUNK)

    store = RlaStore(str(root), chunk=NUM_ACTIONS_CHUNK, num_tokens=SYN_Q, token_dim=SYN_D)
    set_store(store)
    dataloader = _real_dataloader(SUITES[args.suite], args.rlds_root, args.batch_size)

    episodes, timesteps, samples = set(), set(), 0
    chunk_axis = np.arange(NUM_ACTIONS_CHUNK, dtype=np.float32)

    for index, batch in enumerate(dataloader):
        if index >= args.batches:
            break
        rla, actions = batch["rla"].numpy(), batch["actions"].numpy()
        assert rla.shape[1:] == (NUM_ACTIONS_CHUNK, SYN_Q * SYN_D), (
            f"batch['rla'] is {rla.shape}, expected (B, {NUM_ACTIONS_CHUNK}, {SYN_Q * SYN_D})"
        )
        decoded = rla.reshape(rla.shape[0], NUM_ACTIONS_CHUNK, SYN_Q, SYN_D)

        for b in range(decoded.shape[0]):
            ep, tt = decoded[b, :, 0, CH_EPISODE], decoded[b, :, 0, CH_T]
            ii, grip = decoded[b, :, 0, CH_I], decoded[b, :, 0, CH_GRIPPER]
            traj_len = decoded[b, :, 0, CH_N]

            assert np.array_equal(ii, chunk_axis), (
                f"chunk axis is {ii}, expected {chunk_axis} -- the (A, Q, D) block is not one group "
                "per chunk step, or the reshape order is wrong"
            )
            assert len(set(ep)) == 1, f"one frame's chunk spans {len(set(ep))} episodes: {ep}"
            assert len(set(tt)) == 1, f"one frame's chunk spans {len(set(tt))} anchor frames: {tt}"
            assert len(set(traj_len)) == 1, "traj_len disagrees within one frame's chunk"

            t, n = float(tt[0]), float(traj_len[0])
            assert 0 <= t <= n - NUM_ACTIONS_CHUNK, (
                f"anchor frame t={t} outside [0, N-A] with N={n}; chunk_act_obs drops the last "
                f"{NUM_ACTIONS_CHUNK - 1} frames, so a larger t means the join is misaligned"
            )
            assert np.allclose(grip, actions[b, :, -1], atol=2e-3), (
                f"gripper mismatch at episode {ep[0]:.0f} t={t:.0f}:\n"
                f"    target says {grip}\n    batch says  {actions[b, :, -1]}\n"
                "The block joined to this frame does not belong to this frame's action chunk."
            )
            episodes.add(int(ep[0]))
            timesteps.add(int(t))
            samples += 1

    assert store.num_misses == 0, f"{store.num_misses} join misses"
    assert len(episodes) >= 5, f"only {len(episodes)} distinct episodes seen; raise --batches"
    assert len(timesteps) >= 20, f"only {len(timesteps)} distinct timesteps seen; raise --batches"

    ok(f"{samples} samples, {len(episodes)} distinct episodes, {len(timesteps)} distinct timesteps, "
       "0 misses")
    ok("gripper trajectory of a_t..a_{t+7} matches the joined target on every sample")
    set_store(None)
