# ADR 0084: Device-neutral n-D nonzero-coordinate seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26); ADR 0043 (IEEE special
  values per dialect).

- Revised 2026-10-14: the CUDA/ROCm/Metal "not implemented yet" paragraphs below
  are stale 
—
 those backends have since landed (rescue slices 3, 4, 11a). Other
  follow-ups keep their stated status.

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `nonzero` (scanning a strided input in logical
row-major order for the coordinates of its nonzero cells), the last
remaining op after `SortOps`. It is the `torch.nonzero` semantic
`coeus_ops::nonzero` documents — row-major `[N, ndim]` coordinates with
an inequality-against-zero test — except the coordinates are `u32`
rather than `T`-encoded and the count crosses explicitly: `N` is
data-dependent, so the caller sizes the coordinates for the worst case
(`[numel, N]`) and learns `N` from a caller-owned single-cell count.
Returning the count from the seam would force a synchronization point
into every dispatch; the explicit count cell keeps the seam's uniform
`Result<()>` shape and leaves the caller in control of when the count
synchronizes.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. The test interprets
`T` (inequality against zero with a non-finite edge) rather than
copying it, so the seam carries `type Dialect: KernelDialect` — with
the sort seam's wholesale ADR 0043 skip: compaction packs rows densely,
so one unordered `NaN` comparison shifts every later row on a
non-advertising dialect and no assertion about a `NaN`-mixed input can
hold on WGSL. `-0.0` counts as zero on every backend.

## Decision

`NonzeroOps<D, T: Pod>` in `hephaestus-core::domain::nonzero`
provides `nonzero_into`, taking a strided rank-`N` input, caller-owned
rank-2 `u32` coordinates of shape `[numel, N]`, and a caller-owned
rank-1 `u32` count of shape `[1]`, checked by
`validate_nonzero_shape` in a fixed order (coordinates rows, coordinates
columns, count length) over the checked cell-count product. Only the
first `N` coordinate rows are written; tail rows are never touched,
pinned by sentinel pre-fill. All three buffer pairs must not alias —
coordinates/input, count/input, and coordinates/count — each pinned
rejection-before-mutation. The alias clause runs concrete at `T = u32`
so all three buffers share one element type: same-type aliasing is the
only aliasing constructible through the safe buffer API, and it is
exactly what each backend's `aliases` check observes. Empty shapes
scan vacuously and still report the count `0`.

Row order is the input's logical row-major order on every backend,
implemented as an output-parallel scan (each row rescans the input for
its nonzero), the scatter seam's determinism-over-atomics precedent: a
global atomic slot claim would race the row order. `WgpuNonzeroOps`
dispatches one thread per coordinates row in a single launch with at
least one thread, so thread 0 reports the count of even a zero-cell
input. An empty input or coordinates buffer substitutes a one-cell
zeroed dummy for binding — wgpu requires at least one element per
binding — which is sound because the kernel provably touches neither
binding when numel or `N` is zero. To express the cross-type
coordinates/input and count/input checks on the host,
`HostBuffer::aliases` generalized from `&Self` to `&HostBuffer<U>`
over allocation identity, matching the device backends'
pointer-typed `aliases`. The host reference scans flat `0..numel`
through `Layout::offset_of` and tests against
`eunomia::NumericElement::ZERO`, keeping `NaN`-counts-as-nonzero in
Rust's IEEE arithmetic rather than a bitwise test.

CUDA, ROCm, and Metal do not implement `NonzeroOps` yet, and the seam
is not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `nonzero_into` on WGPU (and the host reference pair)
today, converting the `u32` coordinates into its `T`-encoded rows at
the bridge and slicing the leading `N` rows. `HEPH-SHAPE-OPS-PROVIDER-1`
is fully split: no ops remain after this record. CUDA/ROCm/Metal
implementations and the aggregate fold are follow-up items; the CUDA
and ROCm residuals will exercise the `NaN`-counts-as-nonzero cells
their dialects advertise.
