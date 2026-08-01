# ADR 0042 — Device-neutral decomposition seam

- Status: Accepted
- Date: 2026-08-01 (revised same day: scope narrowed to the LU/QR/Cholesky
  trio and the scalar fixed at `f32`, matching the kernels' current
  coverage — the drafted `<T>` parameter and the SVD/eigen families enter
  when their machinery ships; revision driven by implementation evidence,
  all four adapters delivered as-built)
- Refs: atlas `backlog.md#atlas-arch-001` (001e-e1); ADR 0041 (conformance
  crate); the per-backend `MatrixDecompose` traits this seam replaces as the
  cross-backend surface.

## Context

Dense decompositions (LU, full-pivot LU, QR, Cholesky, SVD, rank-revealing
SVD, eigenvalues) exist on WGPU, CUDA, ROCm, and Metal only as per-backend,
device-concrete `MatrixDecompose` traits (metal `linalg_traits.rs`,
cuda/rocm `linalg/matrix.rs`), each returning that backend's
`Gpu*Decomposition` handle types. A consumer — or the conformance suite —
cannot factor a matrix without naming a device type, so the linalg family
is the last conformance ledger group with no shared clauses (ATLAS-ARCH-001e),
and the four hand-written decomposition test sets have already diverged.

## Decision

One `DecompositionOps<D: ComputeDevice, T>` trait in `hephaestus-core`,
mirroring the established seam pattern (zero-sized per-backend implementor,
monomorphized at every call site):

- One method per decomposition family, taking rank-2 `StridedView` operands.
- Associated result types per family (`type Lu<'op>`, `type Qr<'op>`, …),
  lifetime-parameterized like the prepared forms (ADR 0041's rebind
  lesson): backends whose handles borrow operand or workspace state
  implement without erasing the borrow; handle-owning backends ignore
  `'op`.
- Result-handle capability is **oracle-minimal**: each associated type is
  bounded by a small per-family accessor trait (e.g. `LuFactors<D, T>`:
  download L/U, pivot permutation, `solve_into`) covering exactly what
  consumers and the conformance oracles need — reconstruction residual,
  factor structure, permutation validity, and solves. Backend-specific
  extras stay inherent on the concrete handle types.
- `eigenvalues` returns complex values via the `eunomia::Complex` scalar
  the backends already use; SVD's rank-revealing variant keeps its explicit
  tolerance argument, documented as an input contract rather than an
  oracle tolerance.

The conformance clauses (e2) then assert, generically: reconstruction
residuals `‖A − LU‖ ≤ c(n)·ε·‖A‖` (and QR/Cholesky/SVD analogues) with
`c(n)` derived per algorithm from its published error analysis,
orthogonality `‖QᵀQ − I‖`, pivot-permutation validity, SPD fixtures for
Cholesky, and solve round-trips — leto is the differential oracle per the
drop-in substrate rule.

## Alternatives

- Unifying the existing per-backend traits by textual mirroring: rejected —
  that is the drifting shape this seam family exists to end.
- A single opaque `Decomposition` enum result: rejected — collapses the
  per-family contracts and forces runtime matching where associated types
  monomorphize.

## Consequences

e1 lands the trait plus four adapter impls over the existing machinery;
e2 the generic clauses; e3 instantiations and deletion of superseded
hand-written decomposition tests, with per-backend assertion counts staying
a superset throughout.
