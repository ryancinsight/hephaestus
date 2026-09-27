# ADR 0063: Device-neutral rank-2 argmax/argmin seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-topk-argminmax-provider-1`; coeus accelerator-
  bridge consumer audit (2026-09-26); ADR 0062 (the sibling padding seam this
  record follows the same increment shape as).

## Context

Coeus's consumer audit named `topk`/`argmax`/`argmin` together as one gap:
no accelerator path exists, so coeus downloads full operands to host for
these reductions. `topk` and `argmax`/`argmin` are not the same algorithm
class, though: an extremum-and-its-index reduction is a linear scan (the
same shape as `hephaestus_core::AxisReductionOps`'s existing value
reduction), while a top-`k` selection needs a partial sort or a selection
network — a materially larger, higher-risk increment. Splitting them avoids
forcing the harder algorithm's design risk onto the seam the simpler one
already unblocks.

## Decision

This increment delivers `argmax`/`argmin` only, as `ArgReduceOps<D, T>` in
`hephaestus-core::domain::arg_reduce`, fixed at rank 2 (matching
`AxisReductionOps`'s existing convention: the reduced axis stays at length
one rather than dropping, so the result stays broadcastable against the
input). `topk` remains open under `backlog.md#heph-topk-argminmax-provider-1`,
narrowed to name only the top-`k` selection; this ADR's seam closes the
argmax/argmin half.

Tie-breaking follows `leto::argmax`/`argmin`'s rule exactly: the first
strict improvement wins (`candidate > best` / `candidate < best`, never
`>=`/`<=`), so host and device agree bit-for-bit on integer and exact-float
fixtures without a tolerance.

### Binding an N-D tensor's arbitrary axis onto a rank-2 seam

`ArgReduceOps` takes exactly two axes: the reduced axis and "the other one."
A caller reducing axis `k` of a rank-`M` tensor maps onto it by treating the
tensor as a logical `[outer, axis_len, inner]` three-way split — dimensions
before `k` collapsed into `outer`, dimension `k` itself, dimensions after `k`
collapsed into `inner` — then presenting the seam with whichever *two* of
those three collapse cleanly into one `StridedView` axis:

- **`k` is the last dimension** (`inner` is empty): `outer` merges directly
  into the seam's non-reduced axis (its stride is the tensor's second-to-last
  stride, since `outer`'s dimensions are already contiguous with each other
  in a normal row-major layout) and `axis_len` is the reduced axis, called
  with `axis = 1`.
- **`k` is the first dimension** (`outer` is empty): symmetric, `axis = 0`,
  `inner` merges into the non-reduced axis.
- **`k` is an interior dimension** (`outer > 1` and `inner > 1`): neither
  `outer` nor `inner` alone equals "every other logical position," and the
  two do not merge into one `StridedView` axis unless they happen to be
  stride-mergeable with each other (not true in general — they are separated
  by the reduced axis in the tensor's own memory layout). The caller either
  batches: dispatch once per `outer` index with `inner` as the seam's
  non-reduced axis (`outer` calls, each `[inner, axis_len]`), or transposes
  the axis to an end position first (a `StridedView` re-permutation, no data
  movement) and falls into one of the two cases above. Batching costs
  `outer` dispatches instead of one; transposing costs nothing at the
  `StridedView` level but changes which axis is "last" for whatever the
  caller does next.

This is not a new rule invented for this seam — `AxisReductionOps` (value
reductions) shares the identical rank-2-only contract and the identical
caller-side collapse; `ArgReduceOps` documents it explicitly here because
this ADR is where a caller binding a genuinely N-D tensor for the first time
will look. [`assert_arg_reduce_transposed_view_contract`] is the seam's
evidence that the collapse actually works: it reduces the same fixture
through a transposed (swapped-stride) view — neither axis physically
contiguous in the caller's own indexing — and asserts the result matches the
natural-layout reduction exactly, which is what a caller relies on when it
transposes an interior axis to an end position before calling.

`HostArgReduceOps` does not delegate to `leto::argmax`/`argmin` directly —
those functions drop the reduced axis (rank `N -> N - 1`), a shape this seam
does not return. Reshaping their output back to rank `N` would cost more than
directly walking the two-dimensional index space with the same tie-break
predicate, so the host implementation is a direct loop, documented against
leto's semantics rather than delegating to leto's code.

The WGPU implementation (`WgpuArgReduceOps`,
`hephaestus-wgpu::application::arg_reduce_seam`) runs one thread per output
lane, scanning the full reduced axis in a plain loop — parallelism is across
lanes only, not a tree reduction within one lane's scan. This is the
correctness-first cut; a lane-internal parallel reduction is a follow-up
performance item if a workload's axis length dominates its lane count.
`axis` (0 or 1) and the comparison direction (max vs min) are baked into the
generated WGSL text and folded into the pipeline-cache key, so each of the
four combinations monomorphizes to its own kernel with no per-invocation
branch.

The `ArgReduceMeta` uniform struct uses `vec4`-sized fields throughout even
though only two lanes of `in_shape`/`in_strides`/`out_strides` carry real
data. WGSL's uniform address space requires every struct member aligned to
its own size, and `vec4` needs 16-byte alignment; a tighter `vec2` packing
left the trailing `offsets: vec4<u32>` field at a 24-byte cumulative offset,
which the WGSL compiler silently padded to 32 — desynchronizing the GPU-side
byte layout from the Rust `#[repr(C)]` struct without a build error, and
reproducing as a wrong-answer bug (a fixture's second row read the wrong
`argmax` index) rather than a validation failure. The fix (every cumulative
member offset already a multiple of 16, matching `hephaestus-wgpu`'s
existing `StridedMeta`/`PadMeta` convention of only ever using full
`vec4`-sized chunks) is now the documented reason that convention exists,
not just an inherited habit.

CUDA, ROCm, and Metal do not implement `ArgReduceOps` yet, and the seam is
not folded into `assert_backend_contract`/`BackendUnderTest` this increment,
for the same reason ADR 0062 gives for `PadOps`: that aggregate requires
every named seam to have every backend's implementation.

## Consequences

Coeus can bind `argmax`/`argmin` on WGPU (and the host reference pair)
today. CUDA/ROCm/Metal implementations and the aggregate fold are follow-up
items, as is `topk` itself (a materially different algorithm) and a
lane-internal parallel reduction for large-axis workloads.
