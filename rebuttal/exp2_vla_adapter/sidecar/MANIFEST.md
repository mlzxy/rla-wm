# Manifest

Cut from `4e05436bf8950b893cadbe07c47add4447af55db` (`Vendor VLA-Adapter source in-tree instead of cloning + patching it`), 2026-07-25.

Round-trip verified: applying `00-all.patch` to a clean checkout of that commit, and applying
`01`-`07` one at a time, both reproduce all 20 touched files **byte-identically** (`cmp`) and
leave the git index clean. With the bundle in place, `git status` shows nothing but this folder.

`rla-vla-adapter-integration.md` is a moved file, not a patch: it was untracked at
`third_party/VLA-Adapter/`, so git never saw it. Body is byte-identical to the original; the
only change is the 12-line "superseded by" header.

Seven files were touched again when the bundle was packaged for release, so their hashes below are
the released ones, not the ones cut in July. The changes were: an absolute machine path in the
runbook replaced by `$REBUTTAL_REPO` (which is why `07-docs.patch` and `00-all.patch` moved too,
one line each), machine-local paths in `rla-vla-adapter-integration.md` replaced by
`<VLA-Adapter>/`, and a "this is a record, not a step" header on `apply.sh`, on `README.md` and on
`new-files/docs/rla-vla-adapter.md`. No hunk was added or removed — `00-all.patch` is still 3070
lines and still holds exactly the same set of diff lines as `01`–`07` together (it orders the
file-level diffs differently, which is why a plain `cat 01..07 | diff - 00-all.patch` is not empty).

One consequence: `new-files/docs/rla-vla-adapter.md` now carries three header lines that
`07-docs.patch` does not create, so that one file is no longer byte-identical to what the patch
would write. Nothing else in `new-files/` diverges.

Regenerate this table with:

```bash
cd rebuttal/exp2_vla_adapter/sidecar
for f in apply.sh new-files/**/*.* patches/*.patch README.md rla-vla-adapter-integration.md; do
    printf '| `%s` | %s | `%s…` |\n' "$f" "$(wc -l < "$f")" "$(sha256sum "$f" | cut -c1-16)"
done
```

`extract_rla_sidecar.py`, `read_rla_sidecar.py` and `samples/` sit alongside this file and are
**not** part of the July bundle; they are the runnable implementation and are not listed here.

| File | Lines | SHA-256 |
|---|---:|---|
| `apply.sh` | 97 | `e44443d65034780f…` |
| `new-files/datalib/libero_camera.py` | 159 | `fec10ec71bdfaa84…` |
| `new-files/docs/rla-vla-adapter.md` | 337 | `d8ef3a39e4cc1f86…` |
| `new-files/docs/rla-vla-adapter-runbook.md` | 245 | `e02cabc4a6af8e0d…` |
| `new-files/scripts/extract_rla_latents.py` | 598 | `17cbe1fdede26980…` |
| `new-files/scripts/verify_rla_targets.py` | 783 | `b4b3d689d1abe6e0…` |
| `new-files/third_party/VLA-Adapter/prismatic/vla/rla_targets.py` | 151 | `ad5cb61abd3a6933…` |
| `patches/00-all.patch` | 3070 | `8f126ad1c78b9746…` |
| `patches/01-datalib-libero-camera.patch` | 355 | `e98c05c54a024fb2…` |
| `patches/02-extractor.patch` | 604 | `91388e6791bbf42f…` |
| `patches/03-vla-adapter-data-path.patch` | 320 | `710248901486ca59…` |
| `patches/04-action-head-and-loss.patch` | 249 | `10266724f8f81b55…` |
| `patches/05-launchers.patch` | 143 | `7ffb444378722562…` |
| `patches/06-verifier.patch` | 789 | `ee27fa27cd6d941c…` |
| `patches/07-docs.patch` | 610 | `a9c81eeaa9cc7df1…` |
| `README.md` | 168 | `d4592b2e397cee00…` |
| `rla-vla-adapter-integration.md` | 339 | `9cab9bb6d57a373c…` |
