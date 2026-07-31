# ADR 0032: Activation-tail expression parity

- Status: accepted
- Date: 2026-07-27
- Scope: Hephaestus parameter-free and runtime-parameter activation expression
  markers across WGPU, CUDA, ROCm, and Metal elementwise providers
- Change class: `[minor] [arch]`

## Revision history

- 2026-07-31: add the runtime-parameter unary seam and Hardtanh/Threshold
  forward and gradient markers. Coeus supplied the driving correctness case:
  consumer-authored WGPU source decoded packed `f32` parameters as `f64` bit
  patterns. Runtime values now remain dispatch data owned by one provider
  contract.

## Context

Coeus exposes parameter-free `Mish`, `MishGrad`, `Elu`, and `EluGrad` plus
parameterized Hardtanh and Threshold operations in the CPU vocabulary. WGPU
owned local expressions for these operations, while the CUDA, ROCm, and Metal
provider seams had no shared marker vocabulary. The WGPU-local Hardtanh and
Threshold expressions also decoded two packed `f32` values as `f64` bit
patterns, producing unrelated near-zero parameters.

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
dependency-ordered consumer commit in the same co-evolution delivery.

Add `ParameterizedUnaryExpr<L>` and `ParameterizedUnaryOps<D, T>`. Expressions
read canonical locals `x`, `first`, and `second`; dispatch accepts `[T; 2]`.
Hardtanh interprets the pair as `(minimum, maximum)`, Threshold as `(threshold,
replacement)`, and their gradient markers ignore the unused second value.
The fixed pair keeps one kernel ABI for the current one- and two-parameter
activation family without encoding runtime values in generated source.

WGPU stores the pair in one pooled uniform buffer, CUDA and ROCm pass both
scalars by value, and Metal delegates to the WGPU implementation selected on a
native Metal adapter. Parameter values are absent from pipeline-cache keys, so
changing bounds or replacement values reuses compiled kernels. All paths write
directly to caller-owned output storage without a tensor-sized intermediate.
CUDA and ROCm use a distinct parameterized-unary cache-key variant because the
five-argument kernel ABI must never alias the ordinary three-argument unary
kernel. A core-owned writable-layout validator rejects overlapping outputs
consistently before any backend dispatch. Hardtanh uses the same explicit
comparison chain in every dialect, including when callers reverse its bounds.

## Alternatives rejected

- Keep WGPU-local expressions: rejected because the provider marker seam is
  the canonical operation owner and this leaves CUDA/ROCm without parity.
- Add a CPU fallback: rejected because it hides a missing accelerator path.
- Compile parameter values into generated source: rejected because each value
  would create a distinct source string and pipeline while duplicating
  parameter decoding in consumers.
- Model Hardtanh as two scalar elementwise launches: rejected because it needs
  an intermediate allocation and does not cover Threshold's conditional
  replacement in one pass.
- Pass a device parameter tensor: rejected because two scalar values do not
  justify device-storage ownership or an extra upload/allocation contract.

## Verification

Core expression tests pin the dialect forms. Shared provider conformance uses
non-default dyadic parameters and includes exact Hardtanh and Threshold kink
values, reversed bounds, repeated dispatch with changed parameters, strided
layouts, overlapping writable-layout rejection, and buffer alias rejection.
Values are compared exactly because the operations use only comparison,
selection, and copying over exactly representable inputs. Exact-head WGPU,
CUDA, ROCm, and Metal workflows are required; physical-device jobs remain
separate evidence.

## Revisit trigger

Revisit when an operation requires more than two runtime scalars, parameters
must be device-resident, alternate softplus stabilization is required, or a
backend-native Metal expression has a materially different numerical bound.
