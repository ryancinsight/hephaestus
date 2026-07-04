# ADR 0003 — Blocked-decomposition host-loop consolidation

- Status: **Accepted; in progress** — LU + Cholesky per-panel compute extracted (`factor_lu_panel`, `factor_cholesky_panel`/`panel_cholesky_packed`, 2026-07-03/04); QR compute + loop-structure trait hoist pending
- Date: 2026-07-03
- Scope: `hephaestus-core`, `hephaestus-wgpu`, `hephaestus-cuda`
- Refs: KS-5 (backlog), audit `docs/audit/2026-07-02-hephaestus-gpu-substrate-audit.md` §5.1

## Context

The blocked `lu_decompose_blocked` / `qr_decompose_blocked` / `cholesky_decompose_blocked`
entry points share their host orchestration line-for-line across the wgpu and CUDA backends:
the panel loop, the CPU panel factorization (`panel_lu_packed` / `panel_qr_packed`, already in
`hephaestus_core::domain::decomposition`), the permutation/sign bookkeeping, and the per-panel
region index math. Only three operation *kinds* are backend-specific: the device→device startup
copy, the compact region gather/scatter between device and host, and the trailing-matrix update
kernel (GEMM for LU, Householder application for QR, SYRK for Cholesky).

The prior KS-5 increments (scan, reduction, decomposition validators) hoisted **pure** host logic
(no interleaved backend calls) — a clean function extraction. The blocked loops are different:
they **interleave** shared bookkeeping with backend device calls, so a pure-function hoist is not
possible. The loop must be generic over a trait that abstracts the backend operations.

This is a `[major]` change (public backend surface moves) and touches decomposition, whose
correctness is load-bearing (differential-vs-leto tests). It also carries a live risk: the two LU
loops have **already diverged** — wgpu reuses per-panel host scratch (`download_..._into`,
`write_..._reusable`), CUDA reallocates per panel (`download_..._compact` returning `Vec`). A
naive hoist would have to pick one and silently change the other backend's allocation behavior.
The design must make that choice explicit.

## Decision

Introduce a `BlockedDecompositionBackend` trait in `hephaestus-core` that abstracts the three
backend operation kinds plus the device buffer type. Hoist the three blocked loops into core as
`blocked_lu` / `blocked_qr` / `blocked_cholesky`, generic over the trait. Each backend implements
the trait once, wrapping its existing region and trailing-update functions; the backend entry
points become thin calls into the core loop.

### Trait surface

```rust
// hephaestus-core::domain::decomposition (or a `blocked` submodule)
pub struct PanelRegion { pub row0: usize, pub col0: usize, pub rows: usize, pub cols: usize }

/// Backend operations the blocked decomposition loops need. `Buffer` is the
/// backend's device f32 buffer; the loop owns all host bookkeeping.
pub trait BlockedDecompositionBackend {
    type Buffer;

    /// Allocate an n-element device buffer, zero-initialized.
    fn alloc(&self, len: usize) -> Result<Self::Buffer>;

    /// Device→device copy of the whole `src` into a fresh working buffer.
    fn clone_device(&self, src: &Self::Buffer, len: usize) -> Result<Self::Buffer>;

    /// Gather a compact row-major `region` from `buf` into `out` (host),
    /// resizing `out` to `region.rows * region.cols`. Reuses `out`'s capacity.
    fn download_region(&self, buf: &Self::Buffer, region: PanelRegion, out: &mut Vec<f32>)
        -> Result<()>;

    /// Scatter a compact row-major `data` into `region` of `buf`.
    fn write_region(&self, buf: &Self::Buffer, region: PanelRegion, data: &[f32]) -> Result<()>;

    /// Trailing update on the device. The concrete math differs per
    /// decomposition, so each blocked loop declares the exact method it needs
    /// (see below) rather than one catch-all.
    fn gemm_trailing(&self, buf: &Self::Buffer, spec: TrailingGemm) -> Result<()>;   // LU
    fn householder_trailing(&self, buf: &Self::Buffer, spec: TrailingHh) -> Result<()>; // QR
    fn syrk_trailing(&self, buf: &Self::Buffer, spec: TrailingSyrk) -> Result<()>;   // Cholesky
}
```

`TrailingGemm` / `TrailingHh` / `TrailingSyrk` are POD spec structs (block offsets, dimensions,
the reflector/beta buffers for QR) that the loop fills from its bookkeeping and the backend maps
onto its kernel launch. Splitting the trailing update into three named methods (rather than one
opaque callback) keeps each backend impl a direct wrapper of its existing
`gemm_trailing_update` / Householder / SYRK function with no new dispatch logic.

### Scratch-reuse divergence resolution

The core loop adopts the **wgpu reuse discipline** (host scratch `Vec`s allocated once above the
loop, refilled by `download_region`; the compact transfer buffer allocated once). This is the
better behavior (removes CUDA's O(n/b) per-panel host allocations, audit M-class). CUDA's
`download_region` therefore changes from "allocate and return `Vec`" to "fill caller's `&mut Vec`"
— a behavior improvement, recorded here so it is an intended change, not accidental drift.

### What stays in the backends

Only the trait impl: `alloc`/`clone_device` (a few lines over `ComputeDevice` + the existing
`copy_buffer_to_buffer` / `cuMemcpyDtoD_v2`), the region gather/scatter (already implemented in
each backend's `decomposition/region.rs`), and the trailing kernels (already implemented). No
math or loop logic remains duplicated.

## Alternatives considered

- **One opaque `trailing_update(&self, &dyn Any)` callback** — rejected: erases the per-op spec,
  forces `dyn`/downcast on a hot path, and hides the real contract. Three typed methods are
  zero-cost and self-documenting.
- **Leave the loops per-backend, hoist only the region index math** — rejected: the index math is
  a small fraction; the loop structure and permutation bookkeeping are the bulk of the
  duplication and the source of the observed drift.
- **A full `ExecutionPolicy`-style GAT trait** — rejected as over-engineered for three call
  sites; a plain trait with an associated `Buffer` type suffices and monomorphizes cleanly.

## Failure modes / risks

- **Correctness**: decomposition is differential-tested against leto. The hoist must pass the
  existing `blocked_{lu,qr,cholesky}_matches_leto_reference` and the six dense-operand negative
  tests **unchanged** on live hardware — no threshold or workload changes (integrity: no
  test-gaming). Reduction-order is unchanged (same panel/trailing sequence), so results stay
  bitwise-identical where they were.
- **CUDA hardware verification**: the CUDA path is only verifiable on the WDDM machine; the eight
  formerly-aborting compute tests are green post-KS-8, and the blocked-decomposition tests
  already pass there, so this is testable.
- **Monomorphization**: `blocked_lu<B>` instantiates once per backend — identical to the current
  hand-written code, no fan-out.

## Verification plan

1. Implement the trait + `blocked_lu` in core; wire the CUDA and wgpu LU entry points.
2. Gate: `cargo clippy -D warnings` both backends; `blocked_lu_*` differential + negative tests
   green on live hardware (wgpu + cuda); no reduction/scan/other regressions.
3. Repeat for QR, then Cholesky, each its own commit.
4. Record the net line delta; expected ≈ −300 to −450 lines across the two backends once all
   three land.

## QR / Cholesky prerequisites (found during LU implementation, 2026-07-03)

Unlike LU — whose per-panel compute was verbatim-identical and reused the shared
`panel_lu_packed`, so `factor_lu_panel` was a direct extraction — QR and Cholesky
each need a preparatory step before their per-panel compute can hoist:

- **QR**: the two backends' panel buffers have diverged. wgpu keeps a full `m × b`
  panel (`panel[r*b+j]` over `r in 0..m`); CUDA keeps a compact `panel_rows × b`
  panel (`panel[panel_row*b+j]`). The shared `factor_qr_panel` must first normalize
  both to one compact panel layout (a behavior change, differentially re-verified)
  before the reflector-vector extraction and sub-diagonal zeroing can be shared.
- **Cholesky**: the diagonal-block factor calls `leto_ops::cholesky_decompose`
  inline (not a `core::panel_*` routine). Hoisting `factor_cholesky_panel` needs
  either a new core `panel_cholesky_packed` (a b×b Cholesky matching the leto-ops
  numerics within the reconstruction tolerance — core has no `leto_ops` dep and
  should not gain one for this) or, minimally, extracting just the pure
  off-diagonal triangular solve (verbatim in both backends) while leaving the diag
  factor per-backend.

Both are their own commits with differential re-verification, per the sequencing
below; neither is a mechanical mirror of the LU extraction.

## Sequencing

LU first (cleanest trailing update), then QR (reflector buffers add a spec field), then Cholesky
(SYRK, triangular output). Each decomposition is an independent commit so a regression is
bisectable to one factorization.
