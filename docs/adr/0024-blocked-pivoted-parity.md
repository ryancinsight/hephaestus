# ADR 0024 (hephaestus): blocked pivoted-decomposition parity

- Status: Accepted
- Class: [minor]
- Date: 2026-07-25

## Context

ROCm exposes `full_piv_lu_blocked` and `col_piv_qr_blocked` in addition to the
ordinary pivoted decomposition entry points. WGPU, CUDA, and Metal expose the
ordinary operations but omit those blocked-capability names. The providers
already differ internally: ROCm owns HIP factorization kernels, WGPU and CUDA
retain their existing complete-pivot and column-pivot implementations, and
Metal delegates through its native Metal-selected WGPU path.

## Decision

Expose both blocked pivoted-decomposition entry points on all four backends.
Each blocked entry point validates the same dense C-contiguous, zero-offset
operand contract used by device bulk-copy paths, then invokes that backend's
existing pivoted implementation. The operation result remains the backend's
existing typed factor handle, including factors, permutations, rank, solve,
determinant, and inverse behavior. Metal owns its wrappers and delegates only
through the selected Metal WGPU device.

This closes the public capability and validation contract without duplicating
the pivoting algorithms or claiming a blocked kernel where the provider does
not currently own one.

## Alternatives rejected

- Clone the ROCm HIP pivot kernels into WGPU/CUDA: this creates a second
  implementation of a backend-specific algorithm and does not improve the
  shared public contract.
- Make the blocked names silently accept arbitrary strided views: the blocked
  contract requires whole-buffer copies and would make the operation unsafe or
  semantically incorrect for offsets, broadcasts, and transposes.
- Re-export WGPU functions from Metal: this would expose the wrong buffer and
  device ownership boundary instead of a Metal-owned API.

## Verification

Provider contracts compare blocked and ordinary factor values, permutations,
ranks, and applicable solve results on dense matrices. Separate contracts
assert typed dense-layout rejection for transposed, offset, and broadcast
views. CUDA, ROCm, and macOS Metal CI run the feature, warning-denied Clippy,
Nextest, doctest, and rustdoc gates; hosted NVIDIA and AMD jobs remain
hardware evidence only when their labels are available.

The implementation head `5314522` passed the CUDA feature and adapterless
contracts (run `30182486511`, job `89741393411`, 7m35s), ROCm feature and
adapterless contracts (run `30182486506`, job `89741368781`, 5m53s), and
macOS Metal contracts (run `30182486494`, job `89741368870`, 6m14s).
NVIDIA hardware (job `89741391064`) and AMD hardware (job `89741368976`)
were skipped because hosted hardware labels were unavailable; no hardware
execution claim is made.
