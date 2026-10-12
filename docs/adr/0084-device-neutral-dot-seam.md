# ADR 0084: Device-neutral inner-product seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-device-dot-seam`; coeus
  `backlog.md#coeus-dot-cross-host-copy` (dot resale); ADR 0062
  (one-op-at-a-time split).

## Context

`HEPH-SHAPE-OPS-PROVIDER-1` is fully split (ADR 0083), so the first
sub-seam item is `dot` (the `torch.dot` semantic `coeus_ops::dot`
documents — flatten two equal-numel operands and sum the elementwise
products), filed as `HEPH-DEVICE-DOT-SEAM`. The consumer is
coeus's dot resale, but corrected to fusion rather than residency:
coeus's dot already runs on-device as multiply-then-sum with only the
scalar crossing, so what a seam buys is one pinned device-side fold
with a fixed accumulation order instead of two dispatches and a
bridge-side contract. The seam therefore pins the order as part of
its public semantics, not as an implementation detail.

Like its shape/indexing siblings, there is no leto oracle to
delegate to, so the host reference carries the direct implementation
and the shared clauses pin its value semantics with hand-computed
fixtures. The kernel is pure multiply-add accumulation with no value
comparison, so — unlike sort and nonzero — the seam carries no
`type Dialect: KernelDialect`: non-finite propagation follows the
scalar arithmetic identically on every backend, and the special-float
clause pins `NaN`/`inf`/`-0.0` cells unconditionally rather than
skipping them per dialect. IEEE propagation is order-independent for
the pinned cells (`NaN` input poisons the sum, `inf - inf` reduces
to `NaN`, `-0.0 * x` is a signed zero), while the kernel touches
nothing NaN-bearing that could compare.

## Decision

`DotOps<D, T: Pod>` in `hephaestus-core::domain::dot` provides
`dot_into`, taking strided rank-`N` and rank-`M` inputs of
equal numel (not equal shape — ranks and extents may differ) and a
caller-owned rank-1 `T` output of shape `[1]`, checked by
`validate_dot_shape` in a fixed order (each side's checked numel
product, product agreement, output shape) with numel mismatch a
typed `DispatchFailed` rather than an assert. The accumulation is
native-`T`, single-pass, ascending flat order on every backend —
bitwise-identical by construction — and inputs may alias each other
(a self-dot is legal) while the output may alias neither, each pair
pinned rejection-before-mutation. Empty shapes fold vacuously to
`T`'s additive zero. The `[1]` scalar out follows the nonzero
count-cell precedent: returning the scalar from the seam would
force a synchronization point into every dispatch; the caller-owned
cell keeps the seam's uniform `Result<()>` shape and leaves the
caller in control of when the scalar synchronizes.

`WgpuDotOps` dispatches a single thread — thread 0 of one
workgroup — that folds the whole product in ascending flat order,
decoding each side's logical coordinates with the padded rank-8
odometer through a 160-byte packed `DotMeta` uniform. One thread
is the whole parallel strategy, and that is deliberate: any tree
or tiling would reassociate the sum and break the pinned float
agreement, so the kernel is exactly the host's single-pass fold
running on the device. A blocked parallel fold with a pinned
association order is a follow-up; it must prove bitwise equality
against this fold before it replaces it. An empty input buffer
substitutes a one-cell zeroed dummy for binding — wgpu requires
at least one element per binding — which is sound because the fold
reads neither input when numel is zero, and the consolidated entry
carries a ShaderF64-gated twin clause for the `f64` instantiation,
mirroring the dense-vector precedent.

CUDA, ROCm, and Metal do not implement `DotOps` yet, and the seam
is not folded into `assert_backend_contract`/`BackendUnderTest` this
increment, for the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `dot_into` on WGPU (and the host reference pair)
today wherever it needs one fused device-side fold with ordered
float semantics, while the bridge-side multiply-then-sum resale
stays until the fold lands on the remaining vendors. The CUDA,
ROCm, and Metal implementations and the aggregate fold are
follow-up items; each kernel reuses the pinned single-pass order
rather than inventing a tiled one.
