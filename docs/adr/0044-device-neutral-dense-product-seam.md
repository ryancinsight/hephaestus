# ADR 0044: Device-neutral dense product seam

- Status: Accepted
- Date: 2026-08-01
- Refs: atlas `backlog.md#atlas-arch-001` (001i); ADR 0041 (conformance
  crate); ADR 0042 (the decomposition seam this record's staging mirrors).

## Revision 2026-09-08: CUDA scalar declarations and arithmetic

Driven by [HEPH-CUDA-DENSE-PRODUCT-SCALARS](../../backlog.md#heph-cuda-dense-product-scalars).
Required-device run `4df024af` reproduces undefined `__half` and
`__nv_bfloat16` in matrix products; `097836c1` then isolates missing SDK
header search paths. The existing four ordinary scalar instantiations pass
the same exact matrix, batched-matrix and Kronecker oracles.

Scalar source declarations belong to `DialectScalar`, beside `TYPE_TOKEN`.
CUDA product generators and fusion consume its `PRELUDE`; the duplicate
`CudaFusionScalar::PRELUDE` declaration is removed. External scalar
implementors move that associated constant to their `DialectScalar<CudaC>`
implementation; qualified callers likewise use that trait. Built-in scalar
types retain the empty default. This is a breaking associated-item migration,
with no compatibility alias and no release/version bump in this increment.

NVRTC compilation targets the acquired device's queried compute capability.
Half source rejects targets below SM53; Bf16 source rejects targets below
SM80. CUDA 13.3 `cuda_fp16.hpp` lines 2613–2643 selects native half operations
at SM53; `cuda_bf16.hpp` lines 2785–2910 selects native Bf16 operations at
SM80 (identity-operand Bf16 FMA) or SM90 (direct add/multiply). Lower targets
use SDK float fallback and therefore cannot satisfy this contract. Rejected
targets remain compilation/dispatch errors rather than changing precision.

The compiler locates headers relative to the canonical path of its loaded
NVRTC module, using the platform loader API. The library directory and its
two parent levels cover toolkit `bin/x64`, `bin`, `lib64`, and `lib/<target>`
layouts. A missing SDK leaves header-free kernels usable and causes NVRTC
to diagnose missing headers for sources that need them. No unrelated toolkit
from `PATH` is silently substituted. Builds still require no CUDA headers;
runtime compilation of header-dependent scalar types requires the matching
toolkit include directory. Loader-query and filesystem failures propagate.

Alternatives rejected: copying half declarations into each operation forks
the scalar contract; imposing fusion bounds on products couples unrelated
operations; selecting an arbitrary installed toolkit risks header/compiler
version mismatch; requiring NVRTC 13.3 bundled-header extraction introduces
a newer runtime floor and persistent header installation without a present
need when the loaded toolkit already supplies those files.

Verification pairs exact small-integer product oracles with native rounding
witnesses: at `A = 2/epsilon`, `[A, 1, -A]` dotted with ones produces zero
under scalar accumulation and one under wider accumulation. Both matrix
generators run this witness. Generated PTX must contain native half/Bf16
arithmetic and no float arithmetic; this distinguishes SDK per-operation
widening that value-only tests cannot detect. Physical-device evidence is
platform-specific; source review of loader APIs does not establish execution
on untested operating systems.

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
