# hephaestus-wgpu

The portable wgpu backend of the Atlas accelerator substrate (atlas ADR 0001),
and the reference implementation of the `hephaestus-core` `ComputeDevice` seam.
It runs anywhere wgpu does — Vulkan, Metal, and the rest — with no vendor
toolkit at build time. Most consumers reach it through the `hephaestus` facade
as `hephaestus::wgpu`.

## What it provides

- Adapter and device acquisition over a wgpu device/queue pair, including
  device-preference and adapter-enumeration constructors.
- Typed `WgpuBuffer<T>`, PhantomData-typed over `wgpu::Buffer`, with
  upload/download through a bounded pooled staging path. `WgpuBuffer::raw()` is
  the consumer escape hatch for callers building their own pipelines over
  hephaestus-allocated storage.
- Monomorphized elementwise, reduction, scan, map-reduction, linalg, and sparse
  dispatch through ZST operation markers, with per-`(Op, T, BlockWidth)` WGSL
  generation. No type names appear in API identifiers.
- Prepared plans (`prepare_dot`, `prepare_norm_l2`, `PreparedReduction`,
  `PreparedAxisReduction`) that bind fixed buffers once and re-dispatch without
  reallocating or rebuilding bind groups.
- Prepared dense split-complex 1D, 2D, and 3D FFTs over `f32` or native
  `eunomia::F16`, with radix-four/radix-two power-of-two passes and Bluestein
  non-power-of-two axes. Binary16 requires `ShaderF16` and never falls back to
  host or wider-precision execution. FFT plans own scratch and prebind
  pipelines, parameter buffers, bind groups, and dispatch grids. Bluestein
  phases are range-reduced in `f64` before one scalar narrowing and upload;
  staged and fused roots plus reciprocal scales are likewise prepared once in
  the selected storage scalar. Staged radix-four reflects a compact
  half-circle root table instead of retaining a second half.
  `FftOps::encode_fft` composes a prepared plan into an existing command
  stream without provider-owned transient resources. Fixed operand handles are
  owned by the plan and bound directly, eliminating duplicate volume storage
  and per-transform device copies; WGPU consumers can also encode the plan into
  a provenance-carrying grouped sequence while interleaving raw WGPU commands.
- Deadline-aware stream submission and device readback for bounded hosts and
  measurement harnesses.
- The shared backend-neutral volume ray-integral and 2D Laplacian contracts.

Metal is an adapter preference of this crate, not a separate backend
(ADR 0047): `WgpuDevice::try_metal` acquires a Metal-only adapter and the same
WGSL kernels execute through the native Apple Metal path.

The crate re-exports the exact `wgpu` version it builds against (wgpu 30) so
downstream code can author provider-owned WGPU bindings without adding a second
direct `wgpu` dependency.

## Feature flags

| Feature | Effect |
| --- | --- |
| `decomposition` (default) | dense blocked and pivoted Cholesky/LU/QR contracts |
| `sparse` (default) | CSR upload/download and GPU sparse products |
| `parallel`, `mnemosyne-memory` (default) | forwarded to the contract layer |

## Documentation

- API reference: [docs.rs/hephaestus-wgpu](https://docs.rs/hephaestus-wgpu)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
