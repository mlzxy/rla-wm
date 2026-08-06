"""SO-101 real-world pipeline: lerobot dataset -> UNet -> multi-view RLA -> BC / BC-RLA policies.

Exp-4 of the rebuttal release: the `aug2-dual-camera` teleop dataset on an SO-101 follower arm.
Start at README.md -- it says what this experiment does and does not show. RUNBOOK.md is the
end-to-end command sequence.

Nothing under `policies/`, `datalib/` or the main-tree `src/` is modified by this package. It reads
one module from the sibling rebuttal package, `rebuttal.src.models.multiview_token_transformer`.
"""
