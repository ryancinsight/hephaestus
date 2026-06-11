# Changelog

SemVer 2.0.0; pre-1.0 minor bumps may include breaking changes (documented).

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
