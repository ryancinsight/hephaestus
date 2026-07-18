# ADR 0010: Eunomia complex provider

- Status: Accepted
- Date: 2026-07-18
- Class: [arch]

## Context

Hephaestus returns general eigenvalues through typed WGPU, CUDA, and Metal
buffers. Those APIs use `num_complex::Complex<f32>`, while the Leto
factorization that produces the values already returns
`eunomia::Complex<f32>`. Backend implementations therefore reconstruct every
value field-by-field before upload.

The Python binding downloads the same provider buffer into
`Vec<num_complex::Complex<f32>>`, then allocates and fills a second
`Vec<numpy::Complex32>`. Eunomia 0.2.0 pins the complex ABI and implements the
NumPy 0.29 `Element` contract directly.

## Decision

- Add Eunomia as the numeric vocabulary dependency for WGPU, CUDA, Metal, and
  Python packages.
- Replace every public and internal `num_complex::Complex` buffer type with
  `eunomia::Complex`.
- Upload Leto eigenvalue vectors directly because Leto re-exports the same
  Eunomia type.
- Return the downloaded `Vec<eunomia::Complex32>` directly as a NumPy array.
- Remove Hephaestus's direct `num-complex` dependency. External NumPy may retain
  its own transitive complex implementation; that type does not enter
  Hephaestus APIs.
- Release the pre-1.0 public type change as 0.17.0.

## Alternatives rejected

- Preserve `num_complex` at the public boundary and map fields: rejected
  because it retains dual vocabulary ownership and allocates in Python.
- Reinterpret slices with `bytemuck`: rejected because direct provider type
  ownership removes the boundary entirely.
- Define a Hephaestus-local complex buffer struct: rejected because Eunomia is
  the stack SSOT for scalar representations.

## Consequences

- Consumers update imports but retain the same real/imaginary layout and value
  semantics.
- Typed GPU upload/download and Python NumPy construction use one complex
  representation end to end.
- Differential or external FFI dependencies may still resolve `num-complex`
  transitively; direct source and manifest ownership is prohibited.
