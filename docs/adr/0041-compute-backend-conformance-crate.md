# ADR 0041 — One generic ComputeBackend conformance crate

- Status: Accepted (retroactive — records the crate as built)
- Date: 2026-07-31
- Refs: atlas `backlog.md#atlas-arch-001`; conformance triage ledger
  `docs/audit/2026-07-28-computebackend-conformance-triage.md`

## Context

Each backend carried a hand-written `tests/contract.rs`, and the four had
diverged: of the 112 public entry points declared by all four backends, only
46 were exercised by all four and six by none. The contract of a
substitution seam was in practice defined by whichever backend's author
wrote the most tests. (The Atlas board referenced this decision as "ADR
0038" before any record existed; hephaestus's 0038 is the blocked-QR ADR —
this document is the backfilled record under the next free number.)

## Decision

`hephaestus-conformance` holds one set of contract clauses, generic over
[`ComputeDevice`] and the operation seam (`AxisReductionOps`,
`ElementwiseOps`, `ConvolutionOps`, `AttentionOps`, …), one module per
seam. A backend runs a clause by instantiating it with its device and seam
value from a thin `tests/*_contracts.rs`; a clause added once is executed
by every backend from then on.

Clause shape: free functions taking `(device, seam)` and panicking with a
located, backend-named message — the shape a test harness expects, without
the crate depending on one. Oracles are exact equalities wherever the
arithmetic admits them; a clause that needs an epsilon states its
derivation at the assertion site.

The clauses drive seam design: prepared-form clauses assert rebind
semantics (re-dispatch is idempotent over unchanged operands and observes
writes to bound inputs), which forced `AxisReductionOps::Prepared` and the
three `ElementwiseOps` prepared types into lending GATs (`Prepared<'op>`)
implemented by borrowing plans on CUDA/ROCm and handle-owning values on
WGPU/Metal.

## Alternatives

- Keeping per-backend hand-written contract files: rejected — measured
  divergence (above) is the failure mode this crate exists to end.
- A macro-generated test matrix: rejected — generic functions monomorphize
  identically with full IDE/type support (macro policy).

## Consequences

Backend `contract.rs` files migrate to instantiation calls as the
112-entry-point ledger burns down; the assertions executed per backend must
remain a superset of the pre-migration set at every step.
