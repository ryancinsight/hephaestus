# ADR 0030: Exact GELU expression parity

- Status: accepted
- Date: 2026-07-27
- Scope: Hephaestus exact GELU forward and gradient markers and WGPU, CUDA,
  ROCm, and Metal elementwise providers
- Change class: `[minor]`

## Context

The Leto CPU vocabulary and Coeus operation enum expose exact `Gelu` and
`GeluGrad` operations. Coeus already has local WGPU and CUDA expressions, while
ROCm and Metal reject the operations. The shared Hephaestus marker seam must
own the accelerator vocabulary so consumers do not copy provider-specific
expressions or fall back to host execution.

## Decision

Add `GeluOp` and `GeluGradOp` to `hephaestus-core` and export them from all four
backend elementwise surfaces. The forward contract is

`GELU(x) = 0.5 x (1 + erf(x / sqrt(2)))`.

The gradient is derived directly from the standard normal density:

`GELU'(x) = 0.5 (1 + erf(x / sqrt(2))) + x exp(-x² / 2) / sqrt(2π)`.

WGPU expands the existing Abramowitz–Stegun `erf` approximation inline because
WGSL has no native error-function intrinsic. CUDA C++ and HIP C++ use `erff`
with the same f32 contract. Metal delegates through the existing WGPU
substrate. Coeus ROCm and Metal route both operations through these markers
and compare results against the Leto CPU implementation.

## Alternatives rejected

- Keep local Coeus ROCm and Metal expressions: rejected because the provider
  marker seam is the canonical operation owner.
- Use a tanh GELU approximation: rejected because it is a different operation
  contract already represented by `GeluTanhOp`.
- Fall back to CPU execution: rejected because it masks missing accelerator
  capability.

## Verification

Core expression tests pin the scaled `erf` argument and the analytic density
term. Coeus provider tests compare f32 ROCm and Metal output with the Leto CPU
oracle. Exact-head WGPU, CUDA, ROCm, and Metal workflows are required before
the item closes; hardware-runner results remain separately reported.

## Revisit trigger

Revisit when consumers require a different precision, vector result, alternate
GELU approximation, or a backend-native expression with a materially different
error contract.
