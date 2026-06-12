# Changelog

SemVer 2.0.0; pre-1.0 minor bumps may include breaking changes (documented).

## Unreleased

## [0.6.0] - 2026-06-12

### Added

- Default `parallel` and `mnemosyne-memory` feature markers in
  `hephaestus-core` and `hephaestus-wgpu`, keeping the GPU substrate aligned
  with the Atlas provider feature contract without changing device dispatch.
- `hephaestus-wgpu`: `binary_elementwise_into`, `unary_elementwise_into`, and
  `scalar_elementwise_into` for caller-owned contiguous output buffers and
  non-default `BlockWidth` selection.
- `hephaestus-wgpu`: `elementwise_into` benchmark target comparing allocating
  contiguous dispatch with caller-owned output dispatch on a real adapter.

### Changed

- Contiguous elementwise allocating APIs now delegate to the caller-owned
  implementations; scalar dispatch reuses the WGPU uniform-buffer pool instead
  of allocating a uniform buffer for every call.
- Pipeline-cache construction is shared by contiguous elementwise, strided
  elementwise, and reduction kernels.

## [0.5.0] - 2026-06-11

Occupancy-planned dispatch: the strided family accepts block widths from the
ADR-0002 planner pipeline instead of a baked-in constant.

### Added

- `hephaestus-core`: `BlockWidth` — a validating `#[repr(transparent)]`
  newtype over `NonZeroU32` (zero-wide blocks are unrepresentable, not
  checked per dispatch), with `DEFAULT` (256), `covering_blocks` saturating
  ceil-division, and `Default`. This is the typed parameter through which
  moirai's occupancy planner (themis `GpuTopology` × mnemosyne
  `KernelResourceBudget`) reaches backend dispatch.
- `hephaestus-wgpu`: `StridedOperand<'_, T, N>` — a `Copy` parameter object
  pairing a device buffer with its leto layout, keeping strided signatures at
  parameter-object altitude.

### Changed (breaking, pre-1.0)

- The strided family (`binary`/`unary`/`scalar_elementwise_strided_into`)
  takes `StridedOperand` bundles plus a `BlockWidth`; WGSL is generated per
  width and the pipeline cache key is now
  `(kernel family, scalar type, width)` (`PipelineKey` alias), so widths
  cache independently and contiguous kernels share the same key space at
  their constant width.

### Tests

- On-hardware proof that a non-default width (128) produces results
  identical to the default at 1027 elements (partial trailing blocks at both
  widths), exercising per-width shader generation and cache keying. 21 tests
  total.

## [0.4.0] - 2026-06-11

ADR 0002 (atlas) provider role: device topology reporting into themis.

### Added

- `hephaestus-wgpu`: `WgpuDevice::topology()` — a themis `GpuTopology`
  snapshot captured at adapter acquisition (`try_default*` paths; the
  Arc-wrapping `new()` has no adapter and reports `None`). wgpu deliberately
  abstracts hardware topology, so only API-reported fields are filled:
  subgroup (warp/wavefront) width from adapter limits, and the memory tier
  inferred from device type (integrated → `Dram`, discrete → the
  technology-unspecified `Device` tier, since wgpu does not expose
  HBM-vs-GDDR). All other capacities are zero per themis's
  "unreported fields are zero, never fabricated" contract; the CUDA backend
  will fill the full set from device attributes.
- `themis` (0.6.0) added as a workspace dependency — hephaestus is the
  topology provider; themis stays stateless law.

### Tests

- On-hardware contract test differentially re-queries the same adapter and
  asserts warp width and tier match the API, unreported capacities are zero,
  and the adapterless constructor reports no topology.

## [0.3.1] - 2026-06-10

### Changed

- Strided dispatch meta uniforms are now pooled (`get_uniform_buffer`/
  `recycle_uniform_buffer`, mirroring the staging pool): contents are written
  with `queue.write_buffer`, which is ordered on the queue timeline relative
  to submissions, so a recycled uniform cannot race in-flight dispatches.
  Eliminates one 80-byte buffer allocation per strided dispatch.

### Docs

- ADR 0001 (Phase 2 gate): the CUDA backend composes **cuda-oxide** (device
  substrate: driver/context/streams/memory/transfers) with **cutile** (tile/
  PTX kernel authoring), preserving the no-toolkit-to-compile property, with
  a strict SoC boundary between the two and differential parity vs CPU and
  wgpu. See `docs/adr/0001-cuda-backend.md`.

## [0.3.0] - 2026-06-10

Strided op-family completion + dispatch consolidation.

### Added

- `hephaestus-wgpu`: `unary_elementwise_strided_into` — unary dispatch over
  leto layout metadata with the same broadcast/validation/caller-owned-output
  contract as the binary form.
- `hephaestus-wgpu`: `scalar_elementwise_strided_into` — **zero new kernels**:
  the scalar uploads as a one-element buffer described by an all-singleton
  leto layout, which the binary kernel broadcasts through zero strides; scalar
  semantics can never drift from binary semantics.

### Changed

- `strided.rs` consolidated to one shared core (SSOT): `StridedMeta` packing,
  WGSL `Meta`/decode fragments, `cached_pipeline`, and `encode_strided` serve
  every strided kernel family; per-family code is reduced to its shader
  expression and validation prologue.

### Tests

- Strided unary (transposed sqrt, broadcast-input neg) and scalar
  (equivalence with binary broadcast semantics over a transposed view)
  coverage; 17 tests total on real hardware.

## [0.2.0] - 2026-06-10

Phase 1 completion: strided-layout-aware dispatch over leto metadata, plus the
op-family/caching/pooling growth landed since 0.1.0.

### Added

- `hephaestus-wgpu`: `binary_elementwise_strided_into` — binary dispatch where
  all three operands are described by leto host-side `Layout<N>` (rank ≤ 4,
  compile-time capped). Inputs broadcast to the output shape with leto's own
  broadcast rules (zero-stride expanded axes; pure metadata, no data
  movement), so transposed, sliced, offset, and broadcast views dispatch with
  no contiguous materialization. The output buffer is caller-owned (allocation
  control stays with the consumer); zero-stride-aliasing output layouts are
  rejected. Shape/strides/offsets travel in one packed 80-byte uniform; the
  shader decomposes the flat index row-major and applies per-operand strides.
- `hephaestus-wgpu`: unary (`AbsOp`/`NegOp`/`ExpOp`/`LnOp`/`SinOp`/`CosOp`/
  `SqrtOp`/`RecipOp`) and scalar-broadcast elementwise dispatch; sum/min/max
  workgroup-tree reductions; pipeline caching keyed by `(kernel, T)` TypeIds;
  staging-buffer pooling on the download path.
- `leto` (core only) added as a dependency of `hephaestus-wgpu` for layout
  metadata — no CPU compute dependency (leto-ops is not pulled).

### Tests

- Strided differential suite vs CPU references over identical layout
  metadata: transposed input, dual broadcast `[2,1]+[1,3]`, offset sub-block
  write isolation, rank-3 inner-transpose, aliasing/short-buffer rejection.
  14 tests total on real hardware.

## [0.1.0] - 2026-06-10

Initial scaffold per atlas ADR 0001 (shared GPU/accelerator substrate).

### Added

- `hephaestus-core`: GPU-dependency-free contracts — `ComputeDevice` seam with
  GAT `Buffer<T: bytemuck::Pod>`, `DeviceBuffer<T>`, and a five-variant error
  vocabulary (adapter, device, length, dispatch, transfer).
- `hephaestus-wgpu`: wgpu 26 backend — `WgpuDevice` acquisition (default and
  custom limits), typed `WgpuBuffer<T>` with copy-alignment padding and a
  `raw()` escape hatch, upload/zeroed-alloc/download transfers, and
  `binary_elementwise::<Op, T>` dispatch driven by ZST `BinaryWgslOp` markers
  (`AddOp`/`SubOp`/`MulOp`) with `WgslScalar` type-token WGSL generation.
- Differential contract tests (GPU vs CPU reference) covering transfer
  round-trips, partial trailing workgroups, integral types, and length
  rejection; environment-gated skip on adapterless hosts.
