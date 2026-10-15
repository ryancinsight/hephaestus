# Hephaestus provider compatibility checklist

**Target Version**: 0.22.0

## HEPH-EUNOMIA-010-COHORT [minor]

- [x] Audit provider ownership, the public nominal type boundary and ADR 0086.
- [x] Align workspace/internal and provider requirements and document direct-consumer migration.
- [x] Remove stale CI provider substitutions; preserve all backend features, contracts and timeouts.
- [x] Verify one source per provider, unchanged registry records and an unchanged Rust implementation/test tree.
- [ ] Pass formatting, strict Clippy, host/available Metal values, docs and doctests.
- [ ] Pass exact-source API comparison and hosted backend gates on the final head.
- [ ] Integrate upstream providers, refresh the source lock and repeat fresh-consumer verification.

The [macOS verification checklist](docs/verification/macos/checklist.md) retains
its existing hardware and kernel scope; this increment changes dependency
compatibility and the graph used for verification.
