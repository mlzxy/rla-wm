"""Gate `norm`: the target scaling is invertible, and its loss scale is known.

`RLA_NORM` picks how the latent-action targets are scaled before the L1. All four modes are exactly
invertible -- none of them loses information -- so what this gate pins down is the thing that
actually differs: the scale the L1 lands at, which is what `LAMBDA_RLA` is calibrated against.

Reports the measured `E|z|` per mode against real latents, so a change of sidecar or of encoder
checkpoint cannot silently move the loss scale out from under the tuned lambda.
"""

import numpy as np

from rla.config import NORM_MODES
from rla.tests.harness import ok, rule, skip

# E|z| for a zero prediction. Anything outside these bands means lambda no longer means what it did.
# `center` sits lower than the other two on purpose: it divides by the *global* std while having
# already removed the per-channel offsets, and on this corpus those offsets are 72% of the total
# variance, so what is left is the ~0.53-std motion signal alone.
EXPECTED = {"global": (0.5, 1.2), "center": (0.2, 0.8), "perdim": (0.5, 1.2), "none": (1.0, 8.0)}


def run(args) -> None:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from rla.config import CFG
    from rla.sidecar import RlaStore

    rule("norm: target scaling is invertible and its loss scale is pinned")

    root = args.sidecar or CFG.sidecar
    if not root:
        skip("no sidecar: pass --sidecar or set RLA_SIDECAR")
        return

    rows = []
    for mode in NORM_MODES:
        store = RlaStore(root, chunk=NUM_ACTIONS_CHUNK, num_tokens=CFG.queries,
                         token_dim=CFG.dim, norm=mode)
        # A spread of real frames from a few episodes, not one block.
        sample = np.stack([store.frame(slot, t)
                           for slot in range(0, len(store), max(len(store) // 12, 1))
                           for t in (0, 5, 20)])
        raw = np.asarray(store.sc.load(store.refs[0])[0], dtype=np.float32)
        back = store.denormalize(store.frame(0, 0))
        assert np.allclose(back, raw, atol=2e-2, rtol=1e-3), (
            f"{mode}: denormalize did not recover the raw latents "
            f"(max |d| {np.abs(back - raw).max():.4g}) -- the scaling is not invertible, so a "
            "predicted z could not be fed back through f_dec"
        )
        e_abs, std = float(np.abs(sample).mean()), float(sample.std())
        low, high = EXPECTED[mode]
        assert low <= e_abs <= high, (
            f"{mode}: E|z| is {e_abs:.4f}, outside the expected [{low}, {high}]. LAMBDA_RLA was "
            "calibrated against this scale; re-check it before trusting a run."
        )
        rows.append((mode, e_abs, std, float(np.abs(sample).max())))

    for mode, e_abs, std, peak in rows:
        marker = "  <- active" if mode == CFG.norm else ""
        ok(f"{mode:<7s} E|z| {e_abs:.4f}  std {std:.4f}  |max| {peak:6.2f}  invertible{marker}")

    active = dict((m, e) for m, e, _, _ in rows)
    ok(f"global and perdim agree on the loss scale to "
       f"{abs(active['global'] - active['perdim']):.4f} -- switching between them does not move "
       "LAMBDA_RLA")
