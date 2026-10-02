# ADR 0067: Device-neutral rank-2 axis-resampling seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-interpolation-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

## Context

The consumer audit named interpolation as a gap: no accelerator path exists
for any resampling mode, so coeus falls back to host execution. The
original item asked for linear, nearest, and bilinear resampling.

Bilinear (2D) resize is not a materially different algorithm from linear
(1D) resize — on a regular grid it is separable: resizing width then height
(each an ordinary 1D linear pass) produces the same result as a joint 2D
kernel, at the cost of one intermediate buffer instead of zero. A third,
redundant kernel duplicating that math would violate the canonical
implementation rule (`standards`: one generic zero-cost implementation over
duplicated variants). Bilinear is therefore not a mode of this seam — it is
two calls to it, one per spatial axis.

## Decision

`InterpolationOps<D, T>` in `hephaestus-core::domain::interpolation`
provides `interpolate_axis_into`, matching `ArgReduceOps`'s and
`TopKOps`'s rank-2/axis shape but resizing the axis to a caller-chosen
length (`output`'s shape need not match `input`'s on `axis`) rather than
reducing or selecting from it.

Source coordinates map under the *align-corners* convention (pinned per
`numerical_discipline`: convention pinning): output index `0` always reads
input index `0`, and (when `out_len > 1`) output index `out_len - 1` always
reads input index `in_len - 1`; interior points are evenly spaced between.
This is exact at both endpoints — no boundary-fabricated value — and
well-defined for the two degenerate cases a shape-driven resize must
handle: `in_len == 1` (every output sample reads the sole input element)
and `out_len == 1` (the single output sample reads input index `0`).

Weight arithmetic runs entirely in `T`'s native precision via
`eunomia::TryFromCount::try_from_count` (host) and `T`'s own WGSL type token via a
constructor cast from the loop-index `u32` (WGPU) — no widen-compute-narrow
cast (HARD per `integrity`: fake generics; `numerical_discipline`: concrete
precision contract). `Nearest` mode rounds half down (`frac < 0.5` keeps
the lower index; `frac >= 0.5` advances to the upper, clamped) — an
arbitrary but *documented* tie rule, matching the project's existing
tie-break-toward-the-lower-index convention elsewhere in this seam family
(`ArgReduceOps`, `TopKOps`).

`WgpuInterpolationOps` dispatches one thread per `(lane, out_idx)` pair
(flat over `lanes * out_len`): each thread computes its own source
coordinate and samples or blends independently, so the kernel needs no
shared state or synchronization — unlike `TopKOps`'s per-lane sequential
scan. `mode` and `axis` bake into the generated WGSL at pipeline-cache-key
granularity (matching `ArgReduceOps`'s convention for `axis`/`direction`),
so each combination monomorphizes to its own kernel with no per-invocation
branch on either.

CUDA, ROCm, and Metal do not implement `InterpolationOps` yet, and the seam
is not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `interpolate_axis_into` on WGPU (and the host reference
pair) today for 1D resize, closing the `backlog.md#heph-interpolation-
provider-1` item's linear/nearest scope; bilinear composes two calls at the
call site (documented above), so no separate binding is needed for it.
CUDA/ROCm/Metal implementations and the aggregate fold are follow-up items.
