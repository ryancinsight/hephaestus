# ADR 0068: Device-neutral rank-2 adaptive pooling seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-adaptive-pooling-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

The consumer audit named adaptive pooling as a gap: no accelerator path
exists, so coeus falls back to host execution.

`hephaestus_core::PoolingOps` already covers windowed pooling, but its
windows come from a fixed kernel/stride/padding
(`leto::WindowParameters`) — the wrong shape for "map an arbitrary input
extent onto a target output extent" (PyTorch's `AdaptiveAvgPool`/
`AdaptiveMaxPool`), where the window size and stride are *derived* from the
input/output extents rather than supplied. A second, incompatible seam is
warranted rather than overloading `PoolingOps`'s parameters.

## Decision

`AdaptivePoolingOps<D, T>` in `hephaestus-core::domain::adaptive_pooling`
provides `adaptive_pool_axis_into`, matching `InterpolationOps`'s rank-2/
resized-axis shape (not `PoolingOps`'s const-generic spatial rank), since
both this seam and `InterpolationOps` are a per-axis extent remap; only the
per-cell operation differs (weighted sample vs. windowed reduction).

Each output index `i` (of `out_len`) owns the half-open window
`[i * in_len / out_len, ((i + 1) * in_len).div_ceil(out_len))` — the
standard adaptive-pooling formula. This is exact integer arithmetic (no
float, unlike `InterpolationOps`'s coordinate mapping) and, notably,
**windows can overlap** when `out_len` does not evenly divide `in_len`
(verified: `in_len=7, out_len=3` gives `[0,3)`, `[2,5)`, `[4,7)`, sharing one
index between neighbors) — this matches the reference algorithm exactly
(PyTorch's implementation) rather than a naive non-overlapping partition,
which a first hand-derived test fixture in this change incorrectly assumed
before being corrected against the actual host output.

A 2D adaptive-pooled window is the Cartesian product of two independent
per-axis windows, and both average and maximum are associative, commutative
reductions — so a rectangular 2D reduction equals the reduction of the
per-axis 1D reductions, exactly the separability argument ADR 0067 makes
for bilinear interpolation. 2D adaptive pooling is therefore not a separate
mode here either: it composes as two calls to `adaptive_pool_axis_into`,
one per spatial axis.

`AdaptivePoolingMode` (`Average` | `Maximum`) is a runtime parameter on one
method, matching `PoolingOps`'s convention (not `ArgReduceOps`'s two-method
split), since both modes share the same window-derivation logic and differ
only in the per-window reduction. The core trait bounds `T: Pod` only;
`Average`'s division and `Maximum`'s comparison both resolve through
`leto_ops::Scalar` at the host/wgpu impl level (matching `PoolingOps`'s own
`T: Pod + Scalar` bound), so a max-only or avg-only backend still pays no
cost for the unused half — both host and WGPU implement both modes here.

`WgpuAdaptivePoolingOps` dispatches one thread per `(lane, out_idx)` pair
(flat over `lanes * out_len`, matching `InterpolationOps`'s dispatch
shape): each thread computes its own window and reduces it independently,
so the kernel needs no shared state or synchronization.

CUDA, ROCm, and Metal do not implement `AdaptivePoolingOps` yet, and the
seam is not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `adaptive_pool_axis_into` on WGPU (and the host reference
pair) today for 1D adaptive pooling, closing the
`backlog.md#heph-adaptive-pooling-provider-1` item's scope; 2D pooling
composes two calls at the call site (documented above), so no separate
binding is needed for it. CUDA/ROCm/Metal implementations and the aggregate
fold are follow-up items.
