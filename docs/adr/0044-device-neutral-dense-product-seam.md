# ADR 0044: Device-neutral dense product seam

- Status: Accepted
- Date: 2026-08-01
- Refs: atlas `backlog.md#atlas-arch-001` (001i); ADR 0041 (conformance
  crate); ADR 0042 (the decomposition seam this record's staging mirrors).

## Revision 2026-09-24: Host reference and aggregate coverage

`HostDenseProductOps` now implements all three roles over Leto, and
`assert_backend_contract` names the composition and matrix-function seams
alongside the kernel products. The host therefore runs the same dense-linalg
clauses as every accelerator instead of leaving the two new roles opt-in at the
aggregate boundary. Matrix-view construction and array upload are shared host
operand helpers, so decomposition and dense-linalg adapters use one validated
read path and one result-buffer path.

## Revision 2026-09-24: Dense matrix-function role

The final host-delegated pair now has its own `DenseMatrixFunctionOps` role.
`matexp` and `pinv` remain allocation wrappers over the same Leto CPU
operations, but all four providers reach them through static dispatch and one
`assert_dense_matrix_function_contract` clause. The shared contract covers
closed forms, Moore-Penrose identities, general Leto oracles, dense and strided
traversal, rectangular and empty shapes, storage validation, and non-finite or
non-square rejection.

This role is separate from `DenseCompositionOps` because WGPU exposes the pair
only when `decomposition` or `sparse` enables its `leto-ops` dependency.
Making the unconditional power/determinant/rank role feature-dependent would
couple unrelated contracts; making the pair required in WGPU's minimal build
would add a new dependency and unsupported default method. Provider-local
differentials are removed, except WGPU's stable exact diagnostic assertions.

## Revision 2026-09-24: Dense composition role

The follow-up composition stage has landed. `DenseCompositionOps` is a
separate role over the existing provider allocation wrappers for `f32`
matrix power, determinant, and numerical rank. Keeping it separate from
`DenseProductOps` means providers implementing only the kernel tier are not
broken, while the three host-orchestrated entry points become reachable from
one conformance clause without moving their public functions.

`assert_dense_composition_contract` now owns dense and strided matrix power,
zero and empty powers, regular/singular/strided determinants, rectangular
rejection, default and explicit rank tolerances, deficient and rectangular
rank, strided rank, and empty rejection. WGPU, CUDA, ROCm, and Metal all
instantiate the clause. Provider tests retain only edges the role cannot
observe: exact diagnostic text, alternate identity sentinels, and WGPU's
near-singular determinant behavior.

## Revision 2026-09-08: CUDA scalar declarations and arithmetic

Driven by [HEPH-CUDA-DENSE-PRODUCT-SCALARS](../../backlog.md#heph-cuda-dense-product-scalars).
Required-device runs reproduce undefined `__half` and `__nv_bfloat16` in
matrix products and then isolate missing SDK header search paths; the
evidence is recorded at `bdef45b`. The existing four ordinary scalar instantiations pass
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
- **Provider-scheduled compositions** — `matpow`, `det`, and `matrix_rank`
  orchestrate product and decomposition machinery inside each provider.
- **Host-delegated matrix functions** — `matexp` and `pinv` download an
  `f32` operand, run Leto's matrix exponential or SVD-based pseudoinverse on
  the CPU, and upload the allocating result.

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
3. The provider-scheduled compositions enter through a separate
   `DenseCompositionOps<D>` role. Its methods preserve the providers' public
   allocation wrappers while making matrix power, determinant, and rank
   reachable through static dispatch and one conformance clause.
4. The feature-gated host-delegated pair enters through
   `DenseMatrixFunctionOps<D>`. It remains separate from the unconditional
   composition role so WGPU's minimal feature set does not acquire `leto-ops`
   or unsupported default methods.
5. Prepared forms are omitted: the ledger lists none for this family
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

Eleven linalg entry points are now seam-reachable and clause-covered across
all four accelerator backends and the host reference. The kernel tier remains
independently implementable; the composition and matrix-function roles add no
runtime state, virtual dispatch, or public wrapper migration. The aggregate
contract names both roles so a future host omission is a compile error rather
than an opt-in coverage gap. Provider-local matrix-function differentials now
retain only stable provider diagnostics.
