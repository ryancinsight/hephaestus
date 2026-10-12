# ADR 0072: WGSL f64 builtin verdicts keyed by backend, not adapter

- Status: Accepted
- Refs: #414; GLSL.std.450 specification,
  <https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html>.

## Context

The WGPU canary (`hephaestus-wgpu/src/application/canary.rs`) gated f64
`exp`/`log`/`pow`/`tanh` through a table keyed by exact adapter identity,
recording four NVIDIA RTX 5080 Vulkan findings (`log`/`tanh` compile abort,
`pow` dispatch access violation, `exp` at f32 grade) and default-allowing
every other adapter. On hosted CI the Mesa lavapipe adapter, which runs f64
arithmetic shaders correctly (the f64 fusion and cross-entropy backward
contracts pass on it), returned zero for every f64 `exp` sample, and the
canary's truth assertion failed on every master run from its first.

naga 30 lowers those builtins on Vulkan to the GLSL.std.450 instructions
`Exp` (27), `Log` (28), `Pow` (26), `Tanh` (21) at every float width, and the
specification restricts their operands, together with `Radians` (11)
through `Atan2` (25), `Exp2` (29), and `Log2` (30), to 16- or 32-bit floats.
An f64 operand is invalid SPIR-V; each driver's result is undefined, which
is what both adapters showed.

## Decision

The verdict is a rule over backend and width: on `wgpu::Backend::Vulkan`,
every precision-restricted builtin at f64 is blocked. `builtin_verdict`
returns `Blocked` for the four probed builtins with a cause naming the
instruction and the observed behaviors; `require_expr_safe` also refuses the
sixteen restricted builtins that carry no probe. Seams that compile WGSL
containing those builtins (elementwise, scalar, fusion, cross-entropy
forward, attention) return a typed `Unsupported` instead of dispatching.
Other backends and f32 stay supported; the live probe keeps measuring f64
`exp` against CPU truth wherever it is allowed.

## Rejected alternatives

- Add lavapipe to the adapter table: every unlisted Vulkan adapter would
  still compile undefined SPIR-V and return unmeasured values.
- Loosen the canary bound: the error is 1.0 (all-zero output), not a
  tolerance question.
- Software f64 transcendentals built from f64 add, multiply, `fma`, `floor`,
  and `ldexp` (all width-unrestricted in GLSL.std.450): restores capability
  but is a numerical design of its own, with derived error bounds per
  function; it is the follow-up that lifts these refusals.

## Consequences

f64 transcendental ops on Vulkan return `Unsupported` where 0.20.x returned
f32-grade values, zeros, or aborted the process. `BuiltinVerdict::Degraded`
has no producing rule and is kept only for SemVer; it is removed at the next
breaking release.

## Overturning evidence

A naga release that lowers f64 transcendentals to width-valid SPIR-V, or a
Vulkan extension defining 64-bit operands for these instructions, reopens
the rule for the affected builtins.
