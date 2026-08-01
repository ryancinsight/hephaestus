# ADR 0044 — Device-neutral dense product seam

- Status: Accepted
- Date: 2026-08-01
- Refs: atlas `backlog.md#atlas-arch-001` (001i); ADR 0041 (conformance
  crate); ADR 0042 (the decomposition seam this record's staging mirrors).

## Context

The linalg family is the conformance ledger's largest no-seam hole: twelve
entry points (`matmul{,_into}`, `batched_matmul{,_into}`, `kron{,_into}`,
`matexp`, `matpow`, `det`, `pinv`, `matrix_rank{,_with_tolerance}`)
declared by all four backends, none reachable without naming a device
type, so none carry generic conformance clauses.

The family splits on implementation structure:

- **Kernel products** — `matmul`, `batched_matmul`, `kron` and their
  `_into` forms are single device kernels over strided operands, the same
  shape as the elementwise and reduction seams.
- **Host-orchestrated compositions** — `matexp`, `matpow`, `det`, `pinv`,
  `matrix_rank` are algorithms built from the kernel products and the
  decomposition machinery (`det` via LU, `pinv`/`matrix_rank` via SVD,
  `matexp` via scaling-and-squaring over `matmul`).

## Decision

1. One `DenseProductOps<D: ComputeDevice, T: Pod>` trait in
   `hephaestus-core` covering the kernel products: `matmul_into` (rank-2),
   `batched_matmul_into` (rank-3), `kron_into` (rank-2), each over
   `StridedView` operands, zero-sized per-backend implementors,
   monomorphized at call sites. Scalar bounds stay per-impl (each backend
   binds its dialect's requirements), matching every existing seam.
2. Conformance clauses assert exact integer-matrix oracles (products of
   small integer matrices are exact in `f32`), strided traversal, and
   shape rejection without mutation.
3. The host-orchestrated compositions are **staged later**, exactly as
   ADR 0042 staged SVD/eigen: they enter the seam when their increment
   arrives, most naturally as provided methods over `DenseProductOps` +
   `DecompositionOps` bounds rather than per-backend required methods,
   since their logic is backend-invariant composition. Sequencing them
   behind the kernel trio keeps this increment vertical and complete.
4. Prepared forms are omitted: the ledger lists none for this family
   (`prepare_spmm` belongs to sparse), and no consumer requirement exists
   yet. If one appears it follows the established `Prepared<'op>` GAT
   pattern.

## Alternatives

- One monolithic `LinalgOps` covering all twelve now: rejected — the
  compositions need the decomposition seam's SVD extension first, and a
  trait blocked on that would strand the kernel trio's coverage.
- Extending `SparseOperatorOps`'s dense-batch route for matmul: rejected —
  dense-dense products are their own family; conflating them with the
  sparse operator seam couples unrelated contracts.

## Consequences

The three kernel-product pairs become seam-reachable and clause-covered
(six of the twelve entry points); the composition tail is recorded here
and on the board as the follow-up. `hephaestus-conformance` gains a
`dense_product` module instantiated by all four backends.
