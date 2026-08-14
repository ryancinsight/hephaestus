# hephaestus-conformance

One set of contract clauses that every Atlas accelerator backend is held to
(ADR 0041).

Before this crate, each backend carried a hand-written `tests/contract.rs`, and
the four had diverged: of the 112 entry points declared by all four backends,
only 46 were exercised by all four, and six were exercised by none (Atlas
conformance triage, 2026-07-28). The contract of a substitution seam was in
practice defined by whichever backend's author wrote the most tests.

The clauses here are generic over `hephaestus_core::ComputeDevice` and the
operation seam, so a backend runs them by instantiating rather than by
re-authoring — and a clause added once is executed by every backend from then
on. Every clause is a free function taking the device and the seam value.

`hephaestus-host` provides the always-available CPU instantiation, so a clause
is asserted rather than skipped on machines with no accelerator.

## Documentation

- API reference: [docs.rs/hephaestus-conformance](https://docs.rs/hephaestus-conformance)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
