# ADR 0079: Device-neutral n-D scatter-assignment seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26).

- Revised 2026-10-14: the CUDA/ROCm/Metal "not implemented yet" paragraphs below
  are stale 
—
 those backends have since landed (rescue slices 3, 4, 11a). Other
  follow-ups keep their stated status.

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `index_put` (scatter-assigning values slices at
positions given by a 1-D index along one axis), the next remaining op
after `IndexSelectOps`. It is the inverse of index selection and the
`torch.index_put` semantic `coeus_ops::index_put` documents — except the
coeus entry point fixes `dim` at `0` and round-trips through the host
(copy out, apply, copy back), while this seam selects along any axis and
stays on the device.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. The one structural
novelty is the `accumulate` flag, which the coeus consumer already takes:
replace mode overwrites each target cell, accumulate mode adds onto it.
Duplicate indices in replace mode are left unpinned, exactly as
`torch.index_put` leaves them undefined; only the accumulate path
constrains duplicate behavior (a sum in ascending selection order, the
same accumulation-order rule `ScatterOps` keeps).

## Decision

`IndexPutOps<D, T>` in `hephaestus-core::domain::index_put` provides
`index_put_into`, taking a strided base input, a `dim`, a rank-1 strided
`u32` index, strided values, an `accumulate` flag, and a caller-owned
strided output. Input and output share rank `N` and shape exactly; every
non-`dim` values dimension agrees with the input, and the values' `dim`
extent is the index length. `validate_index_put_shape` checks the
geometry in a fixed order (dim range, output shape, values shape). The
index rank is concrete, so no rank-relation bound is needed. The output
buffer must alias neither the input buffer nor the values buffer, and
the shared clause pins rejection-before-mutation for both. An empty
index updates vacuously and reproduces the base.

Out-of-range indices follow the embedding seam's contract exactly: the
host rejects them while scattering, and device backends report them
through a one-`u32` atomic error flag downloaded after dispatch, failing
with `InvalidConfiguration` and an out-of-range message. No backend
clamps or wraps; the faulting update is skipped.

`WgpuIndexPutOps` runs a copy kernel first (base into output), then one
of two update kernels behind the flag. Replace mode is input-parallel:
one thread per values cell substitutes its indexed position and
overwrites the addressed output cell — duplicates race by design, since
the contract leaves their order undefined. Accumulate mode is
output-parallel: one thread per output cell seeds from the copied base
cell and scans every selection in ascending order, adding the slices
whose index maps to it; output-parallelism is required because repeated
indices would race an input-parallel scatter and WGSL has no
floating-point atomics. The copy and the update are two launches because
the update's four storage buffers already fill the portable
`max_storage_buffers_per_shader_stage` limit of four. The host reference
copies the base through `indexed_iter`, then loops selections ascending
with the `dim` lane pinned, accumulating through `AddAssign`.

CUDA, ROCm, and Metal do not implement `IndexPutOps` yet, and the seam
is not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `index_put_into` on WGPU (and the host reference pair)
today, retiring the host round-trip in `coeus_ops::index_put` for the
covered backends. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further:
masked_fill, roll, tile, where, sort, and nonzero remain. CUDA/ROCm/Metal
implementations and the aggregate fold are follow-up items.
