# ADR 0025 (hephaestus): Metal axis-reduction output parity

- Status: Accepted
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

Implementation head `a8ad020` passed the CUDA feature and adapterless
contracts (run `30183081825`, job `89742932396`, 7m31s), ROCm feature and
adapterless contracts (run `30183081834`, job `89742932370`, 5m53s), and
macOS Metal contracts (run `30183081848`, job `89742932459`, 4m57s). NVIDIA
hardware (job `89742932642`) and AMD hardware (job `89742932594`) were
skipped because hosted hardware labels were unavailable; no hardware
execution claim is made. The optional RecurseML analysis check reported only
`Error occurred during analysis (660cb3b4..a8ad0208)` and supplied no source
diagnostic; it is not used as implementation evidence.
