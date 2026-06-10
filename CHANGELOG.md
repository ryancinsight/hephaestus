# Changelog

SemVer 2.0.0; pre-1.0 minor bumps may include breaking changes (documented).

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
