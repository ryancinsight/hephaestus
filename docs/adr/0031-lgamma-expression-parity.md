# ADR 0031: Log-gamma expression parity

- Status: Accepted
- Date: 2026-07-27
- Scope: Hephaestus `LgammaOp` across WGPU, CUDA, ROCm, and Metal
- Change class: `[minor]`

## Context

Leto and Coeus already expose `lgamma`, defined as `ln|Gamma(x)|`, but the
shared Hephaestus unary vocabulary has no provider marker. Coeus WGPU rejects
the operation while the CUDA, ROCm, and Metal provider integrations cannot
route it through Hephaestus.

## Decision

Add one `LgammaOp` marker to `hephaestus-core` and export it through all four
backend elementwise APIs. CUDA C++ and HIP C++ use their device `lgamma`
overloads. WGPU uses a Lanczos approximation with `g = 7`, reflection for
arguments below one half, and explicit pole/infinity selection. Metal keeps
the same expression through its existing WGPU substrate.

The WGPU expression is intentionally f32-oriented, matching the current
provider contract. It returns positive infinity at non-positive integer poles
and for infinite inputs, and propagates NaN through the expression. The
positive-domain approximation and reflection structure follow the Cephes
log-gamma contract, which defines the result as the natural logarithm of the
absolute gamma value and uses recurrence/reflection plus Stirling-style
asymptotics ([Cephes documentation](https://netlib.org/cephes/doubldoc.html#lgam)).

## Alternatives rejected

- Retain Coeus's typed WGPU rejection: rejected because it leaves a Leto CPU
  operation absent from the shared accelerator seam.
- Copy a separate approximation into every consumer backend: rejected because
  provider expression ownership would drift.
- Route the operation to CPU: rejected because it masks missing accelerator
  capability.

## Verification

Core expression tests pin the native CUDA/HIP forms and the WGSL Lanczos and
pole-selection terms. Coeus provider tests compare f32 outputs against the
Leto CPU oracle for positive, reflected, and pole inputs. Provider PR #118
passed exact-head WGPU, CUDA, ROCm, and Metal adapterless workflows; Coeus PR
#231 passed the matching consumer workflows. Physical-device jobs remained
skipped and are not claimed as execution evidence.

## Revisit trigger

Revisit when f64 or reduced-precision GPU contracts require a separate
precision-aware expression, or when a provider exposes a more accurate native
log-gamma intrinsic with a documented error bound.
