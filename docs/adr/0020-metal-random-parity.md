# ADR 0020 (hephaestus): Metal seeded-random parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU, CUDA, and ROCm expose deterministic `uniform_with_seed` and
`normal_with_seed` initializers over real scalar types and const-generic
shapes. Metal exposes the common device and buffer contracts but has no
corresponding random application module, leaving the backend capability matrix
asymmetric.

## Decision

Add Metal-owned `uniform_with_seed` and `normal_with_seed` entry points with
the same scalar, shape, range, distribution, and seed contract. The Metal
module delegates deterministic value generation and validation to the existing
WGPU application path, which uses the shared `leto-ops` initializer contract,
then returns the resulting WGPU buffer wrapped as a `MetalBuffer<T>`.

This keeps the public ownership boundary Metal-specific while retaining one
random-value source of truth. It does not add a device-native PRNG kernel or a
CPU fallback branch; the selected Metal device still owns the returned buffer
and performs the upload through the existing `ComputeDevice` path.

## Alternatives rejected

- Keep random initialization WGPU/CUDA/ROCm-only: this leaves a concrete
  public capability gap for Metal consumers.
- Add a Metal-specific PRNG implementation: it would duplicate seeded value
  generation and risk divergence in ranges, distribution parameters, and
  reproducibility.
- Return host arrays or route to a CPU-only API: the application contract is a
  typed device buffer, so this would change ownership and transfer semantics.

## Verification

The Metal contract compares repeated same-seed uniform outputs, verifies the
uniform half-open bounds, and verifies nonzero normal output for a fixed seed.
The existing macOS Metal CI lane runs the feature, warning-denied Clippy,
Nextest, doctest, and rustdoc gates; required-device enforcement remains the
hardware evidence tier. At code head `96abaac`, CUDA run `30178303257` / job
`89730624710` passed in 7m12s, ROCm run `30178303267` / job `89730603655`
passed in 5m57s, and Metal run `30178303260` / job `89730601126` passed in
6m15s. The AMD and NVIDIA required-device jobs were skipped because hosted GPU
labels were unavailable.
