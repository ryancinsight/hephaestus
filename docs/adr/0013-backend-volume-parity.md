# ADR 0013 (hephaestus): backend-neutral volume ray-integral parity

- Status: Accepted
- Class: [minor]
- Date: 2026-07-25

## Context

`hephaestus-wgpu` owns a device-resident midpoint ray-integral kernel for a
trilinearly sampled three-dimensional field. CUDA and ROCm already expose the
same device, buffer, and matrix operation seams but do not expose this volume
operation. The public geometry and packed-ray contract is independent of a GPU
API, so keeping it in WGPU prevents the other providers from implementing the
same contract without a canonical owner.

## Decision

Move `FieldGeometry` and `RAY_STRIDE` into `hephaestus-core` as GPU-independent
value types, preserve the existing WGPU root paths by re-exporting them, and
implement the same midpoint/trilinear algorithm in CUDA C and HIP C++. The
field and rays remain device-resident for each backend; only one scalar per ray
is written to the output buffer. Metal delegates through its already selected
native WGPU-Metal device and retains the Metal buffer wrapper at its public
boundary.

The validation contract is shared by behavior: packed rays contain six `f32`
values per ray, field storage is exactly `nx * ny * nz`, `step` is finite and
positive, dimensions and ray counts remain exactly representable in `f32`, and
misses or zero-length intersections write zero. The CUDA and HIP kernels use
the same slab intersection, midpoint count, and trilinear interpolation
equations as the WGPU kernel. Reordered floating-point accumulation is not
introduced by this one-thread-per-ray implementation.

## Alternatives rejected

- Keep the volume API WGPU-only: this leaves a concrete capability gap between
  the native providers and forces consumers to select a backend-specific path.
- Upload the field or rays to the host for CUDA/ROCm: this changes the
  device-resident contract and adds a hidden transfer fallback.
- Add a generic runtime kernel dialect abstraction first: the existing CUDA
  and HIP pipeline caches already provide the required compile/launch seam;
  adding a second abstraction would widen this slice without removing the
  operation duplication.

## Verification

The shared core validator covers positive storage, empty output, invalid step,
invalid spacing/dimensions, length mismatches, and the exact-f32 count boundary.
Each device backend contract covers a constant field with a known intersection
length, a miss, an affine field whose trilinear interpolation is exact, and
invalid-step handling; CUDA, ROCm, and Metal also exercise packed-ray mismatch
through their backend entry points. Hosted backend lanes run these focused
contracts with required-device environment variables; adapterless container
runs remain compile/test evidence and are not hardware evidence.
