# ADR 0001 (hephaestus): CUDA backend composing cuda-oxide + cutile

- Status: Accepted
- Date: 2026-06-10
- Class: [arch] — gates Phase 2 implementation
- Parent decision: atlas ADR 0001 (`atlas/docs/adr/0001-gpu-accelerator-substrate.md`)

## Context

hephaestus is the shared Atlas GPU substrate. Phase 1 delivered the wgpu
backend (portable). Phase 2 adds an NVIDIA-native backend. The stack already
has CUDA experience to draw on: `coeus-cuda` uses the cutile ecosystem
(`cuda-core` + `cuda-async`, dynamic driver loading via `libloading`), and
`mnemosyne` has a dlopen `cuMemAllocManaged` unified-memory backend. The
directive (recorded in atlas ADR 0001) is that **cuda-oxide and cutile
coexist** — composition, not migration.

## Decision

`hephaestus-cuda` implements the `hephaestus-core::ComputeDevice` seam by
composing two libraries along their respective strengths:

1. **cuda-oxide** owns the *device substrate*: driver/runtime initialization,
   context and stream management, device-memory allocation
   (`CUdeviceptr`-backed typed buffers mirroring `WgpuBuffer<T>`'s
   `PhantomData<T>` typing), and host↔device transfer. This is the layer that
   maps one-to-one onto `ComputeDevice` (`alloc_zeroed`/`upload`/`download`).
2. **cutile** owns *kernel authoring*: tile-based kernel definitions and
   PTX/CUBIN generation for the op families (elementwise binary/unary/scalar,
   reductions, strided variants over the same packed layout metadata used by
   the wgpu backend — shape/strides/offsets, rank ≤ 4 padded).

Boundary rule: cuda-oxide types never appear in kernel-authoring code and
cutile types never appear in the device-substrate module — the same
SoC discipline as wgpu's `infrastructure/` vs `application/` split.

## Constraints (inherited, non-negotiable)

- **No CUDA toolkit required to compile.** Driver symbols load dynamically at
  runtime; on hosts without an NVIDIA driver, device acquisition returns
  `HephaestusError::AdapterUnavailable` and contract tests skip exactly as
  the wgpu suite does on adapterless hosts.
- **Differential parity:** every CUDA op family is verified against both the
  CPU reference (same layout metadata) and the wgpu backend on hosts that
  have both.
- **One contracts crate:** `hephaestus-core` stays GPU-API-free; the CUDA
  backend adds no associated-type changes to `ComputeDevice`. If a CUDA
  concept cannot be expressed through the existing seam, the seam discussion
  happens in a follow-up ADR rather than leaking `cuda_oxide::*` types into
  core.

## Alternatives rejected

- **cuda-oxide only:** loses cutile's tile/PTX kernel ergonomics already
  proven in coeus-cuda; kernels would be hand-authored PTX or driver-API
  launches of precompiled cubins.
- **cutile only (status quo of coeus-cuda):** leaves device/memory/stream
  management entangled with the kernel toolkit and duplicates the substrate
  layer hephaestus exists to own.
- **Migrating coeus-cuda wholesale into hephaestus now:** premature; coeus
  re-bases onto hephaestus per coeus MS-60+ Stage D after the substrate
  exists, keeping the co-evolution protocol's one-change-per-unit rule.

## Consequences / staging

1. `hephaestus-cuda` crate: device substrate on cuda-oxide (acquisition,
   typed buffers, transfers) + contract tests (skip-without-driver).
2. Elementwise/reduction kernels via cutile, differential vs CPU and wgpu.
3. Strided variants over the shared packed layout metadata.
4. mnemosyne device pools / melinoe ownership tokens (Phase 3) apply to both
   backends uniformly through the `ComputeDevice` seam.
