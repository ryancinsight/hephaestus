# ADR 0071: Generic stateful-update rules over the scalar type

- Status: Proposed
- Refs: [ADR 0045](0045-provider-owned-stateful-updates.md) (provider-owned
  stateful updates, Accepted); [ADR 0061](0061-operator-value-semantics.md)
  (value semantics, Proposed); the f32/f64 twin split in
  `crates/hephaestus-core/src/domain/stateful_update/`.
- Revision 2026-10-12: the Decision now records the built design. The first
  draft proposed a `UniformLayout` associated type and kept the f64 parameter
  blocks padding-free. Implementation kept explicit padding at both widths, so
  the f64 uniform block changes size (see Consequences). Still Proposed until a
  device run verifies the f64 shader.

## Context

`StatefulUpdateRule` was written as twins. Each rule carried `update` and
`update_f64`, `validate_parameters` and `validate_parameters_f64`, `Parameters`
and `ParametersF64`, `PARAMETER_FIELDS` and `PARAMETER_FIELDS_F64`, `BODY` and
`BODY_F64`. Each result type had an f64 copy (`StatefulUpdateStepF64`), and each
parameter struct had an f64 copy (`SgdParametersF64`, and likewise for Adam,
AdamW, RmsProp and AdaGrad). Every backend repeated the twin: `stateful_update`
and `stateful_update_f64` in core, host, CUDA, ROCm and WGPU. The conformance
crate had `assert_stateful_update_contract_f64`.

These twins are the source of the type-suffixed functions the atlas gate counts
(`type_suffixed_fns`, baseline 7, now 29). A type-suffixed name is a per-type
clone of one algorithm. The standards require one generic entry point
parameterized by the scalar and monomorphized per scalar.

## Decision

1. `StatefulScalar` is a sealed trait implemented for `f32` and `f64`. It
   carries the width facts the rules and backends consume: `ONE_LITERAL`
   (`"1.0f"` or `"1.0"`), `REQUIRED_DEVICE_FEATURE` (`None` or `ShaderF64`), and
   `integer_power`, which delegates to the width's native `powi`. Its
   supertraits are `FloatElement`, `Pod`, and `DialectScalar` for Host, Wgsl,
   CudaC and HipC, so every admitted width is representable on every dialect
   and backend dispatch needs no extra bound.
2. Parameter blocks are generic transparent arrays, `XParameters<T>([T; K])`
   with `K` 4 or 8, derived `Pod` through eunomia. Accessors name the fields.
   Explicit padding fields remain at both widths, so the field list is one
   width-independent `PARAMETER_FIELDS` per rule. `#![forbid(unsafe_code)]` in
   core rules out a manual `Pod` impl, which is why the derive is used.
3. `StatefulUpdateRule` has one method per operation: `update<T>` and
   `validate_parameters<T>`, generic over `T: StatefulScalar`. The `Parameters`
   associated type is a GAT, `Parameters<T>`. `BODY` is one source text with a
   `{one}` placeholder; `body_for<T>()` substitutes `T::ONE_LITERAL`.
4. Backends keep one `stateful_update<T, Rule, const N: usize>` each. The
   f64-only feature check reads `T::REQUIRED_DEVICE_FEATURE`. Metal rejects any
   width that requires a device feature, with its prior `Unsupported` error.
5. Conformance exposes `assert_stateful_update_contract<D, O>`, which runs both
   widths, and `assert_stateful_update_at<T, D, O>` for one width. Backends that
   gate a width call the per-width entry on a device that admits it.

## Rejected alternatives

- **Keep the twins and rename them.** Renaming `_f64` to another suffix leaves
  the same per-type clone and only hides it from the gate. That is gaming.
- **A `UniformLayout` associated type that chooses padding per scalar.** It adds
  a trait for a choice that is one constant per rule and width. Explicit padding
  at both widths keeps one field list and one body without that trait.
- **Padding-free f64 blocks.** This keeps the f64 uniform at its old sizes (16,
  48, 56, 24, 16 bytes), but it reintroduces per-width field lists. Rejected for
  now; reconsider only with device evidence that the f64 block needs the smaller
  size.
- **A runtime precision enum selecting the body.** It is a dispatch in the hot
  path and forfeits monomorphization. Rejected.

## Consequences

- The public `StatefulUpdateRule`, `Parameters*F64`, `StatefulUpdateStepF64`,
  `assert_stateful_update_contract_f64`, and the backend `stateful_update_f64`
  methods are removed. This is a major change for every stack consumer. The
  stack consumers (Coeus `OptimizerOps`, per ADR 0045) migrate in the same
  co-evolution unit.
- f64 uniform blocks change size: 32 bytes (SGD), 64 (Adam, AdamW), 32 (RMSProp,
  AdaGrad), from 16, 48, 56, 24 and 16. Any f64 device consumer built against
  the old layout is an ABI break. The change is in the uniform block only; state
  and parameter buffers keep their element type.
- The WGSL f64 shader text is the same rendering as f32, with `T`'s literal and
  type token. Its uniform block now carries padding fields of type `f64`.

## Verification

- Core: `cargo nextest run -p hephaestus-core -E 'test(/stateful_update/)'`
  passes 16 of 16. Each rule is checked against hand-derived arithmetic at both
  widths, and bias corrections are checked for bitwise equality with the
  closed-form `1 - beta^t`.
- Host: the generic host ops pass the stateful contract at both widths, and the
  dispatch-equals-value-definition and rejection-before-mutation tests run at
  both widths.
- Device shader text: CUDA, ROCm and WGPU render the body and uniform for both
  widths, checked by unit tests that need no device.
- Not verified: the WGSL f64 shader on a device. The shader declares no
  `enable f64;` (`dialect.rs` gives the Wgsl f64 impl no `PRELUDE`). The repo's
  fusion test asserts naga accepts f64 unconditionally, which is evidence but
  not a device run. The conformance contract at f64 runs only on a
  `ShaderF64` device. ROCm feature builds require a Linux ROCm install and were
  not compiled on this host.

## Open question for review

Whether the f64 uniform block needs its old padding-free size, or can keep the
explicit padding fields as the Decision states. This changes the f64 device ABI.
It needs a device run to settle, not a source read.
