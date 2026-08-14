# ADR 0029: Error-function expression parity

- Status: Accepted
- Date: 2026-07-27
- Scope: Hephaestus unary `erf` and `erfc` expression vocabulary and WGPU,
  CUDA, ROCm, and Metal elementwise providers
- Change class: `[minor]`

## Context

Leto's CPU unary vocabulary includes `erf` and `erfc`. Coeus already has a
WGPU Abramowitz–Stegun expression and CUDA runtime-kernel forms, but the
shared Hephaestus expression seam did not expose either operation. ROCm and
Metal therefore could not consume the same provider-owned vocabulary as the
other accelerator paths.

## Decision

Add `ErfOp` and `ErfcOp` markers to `hephaestus-core` and re-export them from
all four backend elementwise surfaces. WGPU uses the existing
Abramowitz–Stegun expression contract from `coeus-ops`; CUDA C++ and HIP C++
use their native `erf` and `erfc` device intrinsics. Metal delegates the WGPU
expressions through its existing substrate boundary.

The active consumer contract remains f32. `erfc` is represented as
`1 - erf(x)` in WGSL to preserve the existing Coeus WGPU contract; improved
tail-stable approximations are a separate numerical contract and are not
introduced here.

## Alternatives rejected

- Copy the expressions into Coeus ROCm and Metal: rejected because the
  Hephaestus marker seam is the provider vocabulary owner.
- Use a CPU fallback when a provider lacks `erf` or `erfc`: rejected because it
  masks an incomplete accelerator implementation.
- Add a new runtime-parameterized expression seam: rejected because these
  operations have no runtime parameters and fit the existing static marker
  contract.

## Verification

Core tests pin the native CUDA/HIP forms and the shared WGSL compositions.
Provider compile and contract workflows cover WGPU, CUDA, ROCm, and Metal;
Coeus consumer tests compare ROCm and Metal results with the Leto CPU oracle
on the consumer integration head.

## Revisit trigger

Revisit when Coeus requires tail-stable `erfc`, f64, reduced-precision, vector,
or additional special-function contracts, or when a backend needs a different
expression capability than the static `UnaryExpr` seam can represent.
