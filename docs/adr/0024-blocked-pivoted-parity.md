# ADR 0024: blocked pivoted-decomposition parity

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

### 2026-09-01 revision: packed-factor residency

CUDA now exposes the same device-resident packed-LU split as WGPU. Both
providers turn their packed factor into explicit dense **L** and **U** buffers
without host staging; the copy-and-mask result is bitwise equal to the shared
host oracle.

CUDA ordinary QR now retains the same lazy materialization boundary as WGPU.
Factorization keeps the compact Householder representation and the cleaned
upper-triangular **R** device buffer. `GpuQrDecomposition::accumulate_q`
creates a device identity and applies those reflectors in reverse order, one
256-thread block per Q column. CPU panel factorization does not require host Q
accumulation: the packed tails plus the retained vector heads and scales are
the complete input to the device operation. Bindings request Q through that
method and consume R through `into_r_buffer`, so neither factor is staged
through the host and R is not cloned. Least-squares callers that never request
Q keep the compact representation and allocate no `m × m` factor.

The CUDA operation deliberately uploads the existing packed factor, heads,
and scales when Q is requested instead of retaining duplicate device copies
during factorization. This changes the materialization transfer from `4m²`
bytes of host-built Q to `4mn + 8·min(m,n)` bytes while preserving the lazy
factorization footprint.

## Alternatives rejected

- Clone the ROCm HIP pivot kernels into WGPU/CUDA: this creates a second
  implementation of a backend-specific algorithm and does not improve the
  shared public contract.
- Make the blocked names silently accept arbitrary strided views: the blocked
  contract requires whole-buffer copies and would make the operation unsafe or
  semantically incorrect for offsets, broadcasts, and transposes.
- Re-export WGPU functions from Metal: this would expose the wrong buffer and
  device ownership boundary instead of a Metal-owned API.
- Materialize CUDA Q during every factorization: least-squares and R-only
  consumers do not use Q, so eager O(`m²n`) work and an `m²` buffer would
  regress their execution and memory contracts.
- Retain a second packed factor and reflector set on the device: this would
  save a later upload only for callers that request Q, while charging every
  factorization duplicate retained storage. Lazy direct uploads preserve the
  existing compact handle and remove the transient host Q array.

## Verification

Provider contracts compare blocked and ordinary factor values, permutations,
ranks, and applicable solve results on dense matrices. Separate contracts
assert typed dense-layout rejection for transposed, offset, and broadcast
views. CUDA, ROCm, and macOS Metal CI run the feature, warning-denied Clippy,
Nextest, doctest, and rustdoc gates; hosted NVIDIA and AMD jobs remain
hardware evidence only when their labels are available.

CUDA Q contracts compare device accumulation with Leto's host accumulation on
both public QR entry points using `2·m·min(m,n)·ε` from the two Householder
rounding contributions. A five-panel blocked case independently checks
`QᵀQ ≈ I`, empty `m × 0` decompositions return identity Q, and a second-context
case proves ownership rejection before dispatch while preserving R.

CUDA device-Q source `b6c002a` passes 165 CUDA/Python tests on an RTX 5080 in
35.870 seconds (`cfb70b91-4911-4d54-92ee-e90126a53f5c`). Focused contracts
cover ordinary and blocked factorization, orthogonality, empty shapes, foreign
device provenance, and the Python CUDA binding. The transfer reduction and
absence of host-Q materialization are source-level allocation and residency
evidence; no throughput claim is made without a controlled timing benchmark.

The implementation head `5314522` passed the CUDA feature and adapterless
contracts (run `30182486511`, job `89741393411`, 7m35s), ROCm feature and
adapterless contracts (run `30182486506`, job `89741368781`, 5m53s), and
macOS Metal contracts (run `30182486494`, job `89741368870`, 6m14s).
NVIDIA hardware (job `89741391064`) and AMD hardware (job `89741368976`)
were skipped because hosted hardware labels were unavailable; no hardware
execution claim is made.
