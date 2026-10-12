# ADR 0078: Device-neutral n-D masked-fill seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-shape-ops-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26); ADR 0043 (IEEE special
  values per dialect).

## Context

Continuing `HEPH-SHAPE-OPS-PROVIDER-1`'s one-op-at-a-time split (ADR 0062),
this record splits out `masked_fill` (writing a scalar wherever a mask is
nonzero, copying the input elsewhere), the next remaining op after
`IndexPutOps`. It is the same-shape contract `coeus_ops::masked_fill`
documents, whose per-element host round-trip this seam retires for the
covered backends.

Like its siblings, there is no leto oracle to delegate to, so the host
reference carries the direct implementation and the shared clauses pin
its value semantics with hand-computed fixtures. Two choices distinguish
this seam. First, the mask carries the same scalar `T` as the input,
binding the coeus entry point directly with no conversion: a cell fills
exactly when its mask value compares unequal to zero. Second, the seam
is the first shape op that *interprets* `T` (a zero comparison with
non-finite edges) rather than copying or finitely accumulating it, which
makes the dialect's IEEE guarantee load-bearing: WGSL permits treating
NaN as absent, and the RTX 5080 Vulkan path observably evaluates
`NaN != f32(0)` to false (cell kept, 2026-10-10).

## Decision

`MaskedFillOps<D, T>` in `hephaestus-core::domain::masked_fill` provides
`masked_fill_into`, taking a strided input, a same-`T` strided mask, a
scalar `value: T`, and a caller-owned strided output. All three shapes
must be equal — no broadcasting — checked by
`validate_masked_fill_shape` in a fixed order (mask against input, then
output against input). The output buffer must alias neither source, and
the shared clause pins rejection-before-mutation for both. Empty shapes
fill vacuously.

The mask test is scalar inequality against zero, pinned for every finite
value on every backend: any nonzero magnitude fills, and negative zero
keeps. A `NaN` mask fills only where the kernel dialect advertises IEEE
special values, following ADR 0043's policy exactly: the trait carries
`type Dialect: KernelDialect` (the elementwise precedent — interpreting
seams name their dialect, copy-only seams do not), and the shared
special-values clause asserts the `NaN` cell behind the same const
branch the typed-elementwise suite uses, skipping by construction on
WGSL. Consumers needing cross-backend determinism must keep masks
finite unless every dialect involved advertises the guarantee.

`WgpuMaskedFillOps` dispatches one thread per output cell in a single
launch: decode the coordinate, read the input and mask cells, and write
the fill value or the input cell. The fill value crosses as a second
uniform buffer, the way the pad seam's fill does. There is no error flag
— the op takes no indices, so no data-dependent fault exists. The host
reference walks the mask through `leto::ArrayView` and tests against
`eunomia::NumericElement::ZERO`, which keeps `-0.0`/`NaN` semantics in
Rust's IEEE arithmetic rather than a bitwise test.

CUDA, ROCm, and Metal do not implement `MaskedFillOps` yet, and the seam
is not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `masked_fill_into` on WGPU (and the host reference pair)
today. `HEPH-SHAPE-OPS-PROVIDER-1` narrows further: roll, tile, where,
sort, and nonzero remain. CUDA/ROCm/Metal implementations and the
aggregate fold are follow-up items; the CUDA and ROCm residuals will
exercise the `NaN`-fills cell their dialects advertise.
