# ADR 0071: Generic stateful-update rules over the scalar type

- Status: Proposed
- Refs: [ADR 0045](0045-provider-owned-stateful-updates.md) (provider-owned
  stateful updates, Accepted); [ADR 0061](0061-operator-value-semantics.md)
  (value semantics, Proposed); the f32/f64 twin split in
  `crates/hephaestus-core/src/domain/stateful_update/`.

## Context

`StatefulUpdateRule` is written as twins. Each rule carries `update` and
`update_f64`, `validate_parameters` and `validate_parameters_f64`, `Parameters`
and `ParametersF64`, `PARAMETER_FIELDS` and `PARAMETER_FIELDS_F64`, `BODY` and
`BODY_F64`. Each result type has an f64 copy (`StatefulUpdateStepF64`), and
each parameter struct has an f64 copy (`SgdParametersF64`, and likewise for
Adam, AdamW, RmsProp and AdaGrad). Every backend then repeats the twin:
`stateful_update` and `stateful_update_f64` in core, host, CUDA, ROCm and WGPU.
The conformance crate has an `assert_stateful_update_contract_f64`.

This is the standards violation behind the 22 type-suffixed functions the
atlas gate counts (`type_suffixed_fns`, baseline 7, now 29). A type-suffixed
name is a per-type clone of one algorithm. The standards require one generic
entry point, parameterized by `T: Scalar`, and monomorphized per scalar.

The split is not only naming. The f32 parameter structs carry explicit padding
fields (`padding`, `padding_zero`, `padding_one`) so their sizes stay 16 and 32
bytes; `parameters.rs` documents these as a frozen uniform ABI that device
kernels were built against. The f64 structs are padding-free: 16, 48, 56, 24
and 16 bytes for SGD, Adam, AdamW, RMSProp and AdaGrad (`parameters.rs`
size assertions). The WGSL `Parameters` struct is generated from the per-scalar
field list (`wgpu` `stateful_update.rs`, `parameters_declaration`), so the
layout difference reaches the shader only through that list. No source in the
repository shows a WGSL alignment rule that forces the f64 layout; an earlier
draft of this ADR claimed one, and it was removed.

## Decision

1. `StatefulUpdateRule` has one method per operation: `update<T>` and
   `validate_parameters<T>`, both generic over `T: Scalar` with `Scalar` as in
   `standards`. The `_f64` methods are deleted with no compatibility re-export.
2. The per-scalar layout moves to an associated type on `Scalar`, not on the
   rule: `T::UniformLayout<const N: usize>` or an equivalent that names the
   padded (f32) or padding-free (f64) shape. A rule's parameter block is then
   `Parameters<T>` built from the scalar's layout, one definition for all
   precisions.
3. Each `Parameters*F64` and `StatefulUpdateStepF64` is deleted. Their fields
   become generic over `T`, with the padding chosen by `T`'s layout.
4. `BODY` and `BODY_F64` become one body rendered with the scalar's literal
   spelling (`T::ONE`, `T::ZERO`). The f32 spelling `1.0f` and the f64 spelling
   `1.0` are both `DialectScalar` associated constants, which the codebase
   already uses.
5. Backends keep one `stateful_update<T, Rule, const N: usize>` each. The
   conformance helper becomes `assert_stateful_update_contract<T>`, a breaking
   change to a published crate, so it ships as a major version with the
   migration in the changelog.

## Rejected alternatives

- **Keep the twins and rename them.** Renaming `_f64` to another suffix leaves
  the same per-type clone and only hides it from the gate. That is gaming.
- **One padded layout for both precisions.** The f64 WGSL block would carry
  f32-sized padding it does not need, and the GPU ABI changes for double
  precision. Rejected.
- **A runtime precision enum selecting the body.** It is a dispatch in the hot
  path and forfeits monomorphization. Rejected.

## Consequences

- The trait gets smaller, and its generic bound is the scalar, not a pair of
  method names.
- The f64 WGSL/CUDA/HIP shader text becomes the same rendering as f32, with
  literals chosen by `T`.
- The public `StatefulUpdateRule`, `Parameters*F64`, `StatefulUpdateStepF64`
  and the backend `stateful_update_f64` methods are removed. This is a major
  change for every stack consumer. The stack consumers (Coeus `OptimizerOps`,
  per ADR 0045) migrate in the same co-evolution unit.

## Verification

- Characterization tests pin the current f32 and f64 outputs bitwise before the
  refactor, for each rule, on the host backend.
- After the refactor, the same tests run against the generic path. f32 is
  bitwise identical. f64 is bitwise identical because the arithmetic order is
  unchanged.
- The layout assertions (`size_of`, `align_of`) for the f32 and f64 parameter
  blocks are kept as compile-time checks, so the padding contract cannot drift.

## Open question for review

The layout associated type is the one choice that changes the GPU ABI shape.
Verified by reading source (2026-10-11): the WGSL and CUDA/ROCm `Parameters`
declarations follow the per-scalar field lists, and the field order matches
the Rust `repr(C)` structs for all five rules. Not verified: (1) the WGSL f64
shader declares no `enable f64;` directive (`dialect.rs` gives the Wgsl f64
impl no `PRELUDE`; the fusion test asserts naga accepts f64 unconditionally),
so it has not been compiled on a device; (2) whether the f32 padding fields are
required by any device, or can be dropped as a breaking ABI change. Both need a
device or naga run before implementation starts.
