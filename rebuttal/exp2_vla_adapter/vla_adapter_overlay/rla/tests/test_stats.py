"""Gate `stats`: the dataset hook did not perturb the action/proprio normalisation.

The hook wraps `normalize_action_and_proprio`, not the per-dataset `standardize_fn`, and
`get_dataset_statistics` hashes `inspect.getsource(standardize_fn)` -- so the cached
`dataset_statistics_<sha>.json` should be reused byte-identically and an RLA run should be
normalisation-comparable to a vanilla run by construction. This checks it rather than trusting it.

Needs a finished run of each kind on the same suite, so it is a follow-up gate, not a blocking one.
"""

import json

import numpy as np

from rla.tests.harness import REPO, ok, rule, skip

FIELDS = ("action", "proprio")
STATS = ("mean", "std", "min", "max", "q01", "q99")


def run(args) -> None:
    rule("stats: the dataset hook did not perturb normalisation")

    reference = sorted(REPO.glob("outputs/REF-*/dataset_statistics.json"))
    candidates = sorted(REPO.glob("outputs/RLA-*/dataset_statistics.json"))
    if not reference or not candidates:
        skip(f"found {len(reference)} REF-* and {len(candidates)} RLA-* statistics files; run this "
             "again after a stage-2 run on a suite you already have a vanilla run for")
        return

    compared = 0
    for candidate in candidates:
        for suite, values in json.loads(candidate.read_text()).items():
            match = next((r for r in reference if suite in json.loads(r.read_text())), None)
            if match is None:
                continue
            expected = json.loads(match.read_text())[suite]
            for field in FIELDS:
                for stat in STATS:
                    delta = float(np.abs(np.asarray(values[field][stat], dtype=np.float64)
                                         - np.asarray(expected[field][stat], dtype=np.float64)).max())
                    assert delta < 1e-5, (
                        f"{candidate}: {suite}/{field}/{stat} differs from {match} by {delta:.3e}"
                    )
            for field in ("num_transitions", "num_trajectories"):
                assert values[field] == expected[field], f"{candidate}: {suite}/{field} differs"
            compared += 1
            ok(f"{suite}: action/proprio statistics match {match.parent.name} to <1e-5")

    if compared == 0:
        skip("no suite appears in both an RLA and a vanilla statistics file")
