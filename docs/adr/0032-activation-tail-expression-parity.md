# ADR 0032: Activation-tail expression parity

- Status: accepted
- Date: 2026-07-27
- Scope: Hephaestus `Mish` and `Elu` expression markers and WGPU, CUDA, ROCm,
  and Metal elementwise providers
- Change class: `[minor]`

## Context

Coeus exposes parameter-free `Mish`, `MishGrad`, `Elu`, and `EluGrad` unary
operations in the CPU vocabulary. WGPU currently owns local expressions for
these operations, while the CUDA, ROCm, and Metal provider seams have no
shared marker vocabulary for routing them through Hephaestus.

## Decision

Add `MishOp`, `MishGradOp`, `EluOp`, and `EluGradOp` to `hephaestus-core` and
export them from all four backend elementwise surfaces. The expressions use
the current Coeus f32 contracts:

`Mish(x) = x tanh(log(1 + exp(x)))`.

`Mish'(x) = tanh(softplus(x)) + x (1 - tanh²(softplus(x))) sigmoid(x)`.

`Elu(x) = x` for `x >= 0`, otherwise `exp(x) - 1`.

`Elu'(x) = 1` for `x >= 0`, otherwise `exp(x)`.

WGSL uses `select`; CUDA C++ and HIP C++ use conditional expressions and
single-precision intrinsics. Metal consumes the WGPU dialect through its
existing native Metal-selected WGPU device. Coeus integration remains a
separate consumer change because its backend files are actively owned by
other parity increments.

## Alternatives rejected

- Keep WGPU-local expressions: rejected because the provider marker seam is
  the canonical operation owner and this leaves CUDA/ROCm without parity.
- Add a CPU fallback: rejected because it hides a missing accelerator path.
- Add parameterized activation markers: rejected because `alpha` and other
  parameters require a separate scalar-parameter expression contract.

## Verification

Core expression tests pin the dialect forms. Each provider contract dispatches
all four operations on `[-2, -0.5, 0, 0.5, 2]` and compares output with the
same f32 CPU formulas under the existing error bound. Exact-head WGPU, CUDA,
ROCm, and Metal workflows are required; physical-device jobs remain separate
evidence.

## Revisit trigger

Revisit when Coeus requires parameterized activation expressions, alternate
softplus stabilization, non-f32 scalar contracts, or a backend-native Metal
expression with a materially different numerical bound.
