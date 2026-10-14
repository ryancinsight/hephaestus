# ADR 0069: Device-neutral n-D circular-shift seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).
- Revision 2026-10-14: generalized the rank-2 `roll_axis_into` decision to
  the n-D multi-shift `roll_into` seam (rescued from #427, slice 11f);
  CUDA, ROCm, and Metal implementors included. The rank-2 API, its
  `roll_source_index` helper, and the fixtures-based rank-2 oracle are
  superseded and removed.

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `roll` (circularly shifting along one or more
axes with wraparound). It is the `torch.roll` semantic
`coeus_ops::roll` documents, whose per-element host loop with a
flat-to-ND conversion per cell this seam retires for the covered
backends.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. The structural novelty
is the shift pairing: shifts and dims arrive as parallel slices, exactly
as the coeus entry point takes them, and duplicate dims are legal —
rolls along distinct dims commute and rolls along one dim add, so every
backend folds the pairs into one effective per-axis displacement rather
than applying them sequentially, and the result is identical either way.
Shifts of any magnitude are legal and reduce modulo the extent; the
coeus implementation would divide by zero on an empty dim, while this
seam rolls vacuously there.

## Decision

`RollOps<D, T>` in `hephaestus-core::domain::roll` provides `roll_into`,
taking a strided input, `shifts: &[isize]` and `dims: &[usize]`, and a
caller-owned strided output. Input and output share rank `N` and shape
exactly: a roll permutes cells, never resizes.
`validate_roll_shape` checks the geometry in a fixed order (pairing
length, dim range, shape equality). The output buffer must not alias the
input buffer, and the shared clause pins rejection-before-mutation. An
empty shape rolls vacuously with any shift list — no shift normalizes
against a zero modulus on any backend.

`WgpuRollOps` folds the pairs host-side into per-axis effective
back-shifts (summed in `i128`, reduced once per axis with `rem_euclid`,
converted fallibly), packs them into the 144-byte uniform beside the
shapes and strides, and dispatches one thread per output cell in a
single launch: decode the coordinate, move each lane back by its shift
modulo the shared extent, and copy the addressed input cell. The
back-shifts arrive already reduced below their extents, so the kernel
subtracts without underflow. The host reference folds identically and
walks the input's index domain through `leto::ArrayView`, which yields
no indices for an empty shape.

CUDA, ROCm, and Metal implement `RollOps` (`CudaRollOps`, `RocmRollOps`
over C-dialect kernels keyed by `PipelineKey::Roll`; `MetalRollOps`
delegating to the shared WGSL implementation). The seam is not folded
into `assert_backend_contract`/`BackendUnderTest` this increment, for
the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `roll_into` on every backend (and the host reference
pair) today. `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split is
complete for `roll`; the aggregate fold is a follow-up item.
