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

The split is not only naming. The f32 uniform blocks are padded to vec4
alignment for WGSL, and the f64 blocks are padding-free, because the f64 WGSL
type has no matching uniform alignment. Any unified struct therefore has to
express two layouts, one per scalar. That is the reason the twins exist.

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
Review should confirm that the padded-f32 and padding-free-f64 layouts are
what the shaders expect, before implementation starts.
