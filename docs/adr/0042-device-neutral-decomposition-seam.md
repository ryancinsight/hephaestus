# ADR 0042 — Device-neutral decomposition seam

- Status: Accepted
- Date: 2026-08-01 (revised same day: scope narrowed to the LU/QR/Cholesky
  trio and the scalar fixed at `f32`, matching the kernels' current
  coverage — the drafted `<T>` parameter and the SVD/eigen families enter
  when their machinery ships; revision driven by implementation evidence,
  all four adapters delivered as-built)
- Second revision, 2026-08-01 (late): the remaining 18 decomposition entry
  points are staged into the seam by result-shape family (see Staging
  below), driven by the 001j ledger burn-down. The trio seam and its
  clauses shipped and hardened (pivot convention, R storage convention,
  Householder sign ownership) exactly as this record's e1/e2 predicted.
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

## Staging (second revision): the remaining families

Ordered by (consumer value × machinery readiness); each stage is one
e1-style seam increment plus one e2-style clause increment, and each
handle trait stays oracle-minimal per the original decision.

1. **Pivoted eliminations** — `col_piv_qr{,_blocked}`,
   `full_piv_lu{,_blocked}`: extend `DecompositionOps` with `col_piv_qr`
   and `full_piv_lu` returning handles exposing the permutation(s), rank,
   and solves. Oracles: permutation validity (both sides for full-pivot),
   rank revelation on a constructed rank-deficient fixture, and
   reconstruction through the permutations.
2. **Spectral, symmetric first** — `symmetric_eigen_jacobi`,
   `symmetric_eigenvalues_jacobi`: eigenpairs of a symmetric fixture with
   known closed-form spectrum; orthogonality `‖QᵀQ − I‖` and residual
   `‖AV − VΛ‖` at derived bounds.
3. **SVD family** — `svd_decompose`, `svd_rank_revealing`,
   `singular_values`: singular values of fixtures with known exact
   spectra, `‖A − UΣVᵀ‖`, orthogonality both sides, and the
   rank-revealing tolerance as a documented input contract.
4. **General spectral** — `eigenvalues` (complex results via
   `eunomia::Complex`), `schur`, `hessenberg`, `bidiagonalize`:
   similarity/orthogonality invariants and reconstruction residuals;
   Hessenberg/bidiagonal structure asserted directly (zero patterns).
5. **Symmetric-indefinite** — `bunch_kaufman`, `udu_decompose`: inertia
   against a fixture with known signature, reconstruction, and solves.

**Blocked variants are not seam methods.** `*_blocked` is an execution
strategy of the same mathematical operation, not a distinct contract —
publishing it on the seam would encode an implementation dimension in
API names. The blocked kernels remain per-backend entry points behind
the unblocked seam methods; their equivalence is covered per-backend by
the existing differential tests (blocked vs leto across panel
boundaries), and a backend is free to route the seam method to its
blocked kernel by size heuristic. If a consumer ever needs explicit
strategy selection, it enters as a policy parameter per the standards'
variation mechanisms, never as `*_blocked` seam methods.

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
