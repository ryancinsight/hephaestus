# Hephaestus provider compatibility checklist

**Target Version**: 0.22.2

## HEPH-LETO-047-COMPAT [patch]

- [x] Audit current defaults, active ownership and the unchanged public provider vocabulary.
- [x] Retain 0.46 and admit 0.47 with matching canonical Leto constraints; update affected package versions and migration records.
- [x] Verify the current-main 0.22.2 source graph, strict diagnostics, native values, all six API comparisons and an unseeded review-source consumer.
- [x] Verify the real Coeus 0.47 review graph and every actual hosted provider check at `e867be0d`; keep publication/required-device evidence separate.


**Previous cohort target**: 0.22.0

## HEPH-EUNOMIA-010-COHORT [minor]

- [x] Audit provider ownership, the public nominal type boundary and ADR 0086.
- [x] Align workspace/internal and provider requirements and document direct-consumer migration.
- [x] Remove stale CI provider substitutions; preserve all backend features, contracts and timeouts.
- [x] Verify one source per provider, unchanged registry records and an unchanged Rust implementation/test tree.
- [x] Pass formatting, strict workspace Clippy, all 959 native tests with required Metal (zero skips), docs and doctests.
- [x] Execute five portable API graph comparisons plus the ROCm comparison in its Linux SDK and applicable hosted backend gates at `4cce2705`; repeat all applicable gates for the refreshed source before merging PR #493.
- [x] Integrate Moirai/Leto and refresh all eight default provider identities without changing registry records.
- [ ] Record final-head fresh-consumer, hosted acceptance and six 0.22.0 registry receipts in [PR #493](https://github.com/ryancinsight/hephaestus/pull/493).

The [macOS verification checklist](docs/verification/macos/checklist.md) retains
its existing hardware and kernel scope; this increment changes dependency
compatibility and the graph used for verification.
