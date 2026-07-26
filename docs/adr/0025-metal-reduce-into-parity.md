# ADR 0025 (hephaestus): Metal axis-reduction output parity

- Status: accepted
- Class: [minor]
- Date: 2026-07-26

## Context

WGPU, CUDA, and ROCm expose `reduce_axis_into` from their crate roots. Metal
already owns the same rank-2 wrapper in `application::reduction`, but its root
export omits the operation. The omission makes an implemented output-buffer
capability unavailable to Metal consumers and leaves the four public backend
surfaces unequal.

## Decision

Export Metal's existing `reduce_axis_into` wrapper from the crate root and
verify its value contract against the independent CPU axis-sum result. Keep
the existing Metal-owned wrapper and native Metal-selected WGPU dispatch; do
not add a second reduction kernel or re-export a WGPU module.

## Verification

The Metal contract uses a rank-2 C-contiguous input and output buffer, invokes
`reduce_axis_into::<SumOp, f32>`, downloads the output, and compares each
value with the analytical row/column sum. CUDA, ROCm, and macOS Metal CI must
pass the provider gates; hardware jobs remain hardware evidence only when
their hosted labels are available.
