# ADR 0015 (hephaestus): prepared scalar reduction parity

- Status: Accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU and its Metal delegation path expose prepared scalar reduction plans that
retain the reduction tree's scratch buffers and output allocation across
dispatches. CUDA and ROCm currently expose only immediate reductions, which
rebuild the intermediate device buffers for every call. This leaves a public
capability gap for repeated device-resident reductions.

## Decision

Add `PreparedReduction` plans to CUDA and ROCm. Preparation validates the
shared `BlockWidth`, compiles or retrieves the backend-native reduction kernel,
and allocates every partial-output buffer in the reduction tree. Dispatch
reuses those buffers and launches the native CUDA or HIP kernels over the
caller-owned input buffer; it does not upload or download data. Empty inputs
retain the operation identity in a one-element output, and non-empty inputs
use one or more prepared tree passes, including a singleton pass.

Metal delegates the same capability to the existing WGPU implementation on its
native Metal-selected device. The public operation names and value contract
remain aligned across WGPU, CUDA, ROCm, and Metal; backend-specific kernel
launch and buffer ownership stay inside each backend.

## Alternatives rejected

- Re-run the immediate reduction from `dispatch`: this would allocate fresh
  scratch storage and would misrepresent a reusable prepared plan.
- Upload input to the host or route CUDA/ROCm through WGPU: both violate
  device-resident backend ownership.
- Add prepared axis or sparse operations in this increment: those are distinct
  contracts with separate metadata and batching semantics.

## Verification

Each backend contract compares prepared sum, min, and max results with the CPU
operation for empty, singleton, and multi-pass inputs; repeat dispatch and
batch submission assert stable output storage and value semantics; and an
invalid non-power-of-two width is rejected before preparation. At code head
`4279244`, CUDA run `30171447372` / job `89713176239`, ROCm run `30171447427`
/ job `89713176460`, and Metal run `30171447381` / job `89713176351` passed.
Hardware lanes were skipped because hosted GPU labels were unavailable;
required-device enforcement remains enabled on self-hosted hardware lanes.
