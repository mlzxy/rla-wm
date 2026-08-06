"""Gate `join`: the sidecar and the RLDS data are a clean bijection.

Every RLDS episode gets exactly one episode's latents and every sidecar entry is claimed exactly
once, frame counts agree, and `lookup` returns finite correctly-shaped targets. Delegates to
`read_rla_sidecar.preflight`, which is the same code path as the reader's own `join` command, so the
two cannot drift.
"""

from rla.tests.harness import ok, rule


def run(args) -> None:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK
    from rla.config import CFG
    from rla.read_rla_sidecar import RlaSidecar, preflight

    rule("join: every RLDS episode <-> exactly one sidecar entry")

    root = args.sidecar or CFG.sidecar
    if not root:
        raise SystemExit("no sidecar: pass --sidecar or set RLA_SIDECAR")

    # The shape contract is asserted against the constants training will actually use: a sidecar
    # built with a different chunk is a different dataset, not a compatible superset.
    sidecar = RlaSidecar(root, chunk=NUM_ACTIONS_CHUNK, num_tokens=CFG.queries, token_dim=CFG.dim)
    print(f"  {sidecar!r}")

    reports = preflight(sidecar, args.rlds_root, require_full=True, verbose=True)
    for suite, report in reports.items():
        assert report.is_bijection, f"{suite} is not a bijection: {report}"

    total = sum(r.matched for r in reports.values())
    ok(f"{total} episodes over {len(reports)} suites, bijective, frame counts agree, "
       "lookup() finite and correctly shaped")
