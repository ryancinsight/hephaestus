# ADR 0045 — Provider-owned stateful elementwise updates

- Status: Accepted
- Date: 2026-08-01
- Refs: `backlog.md#heph-stateful-update-1`; ADR 0041 (generic provider
  conformance); Coeus `OptimizerOps` consumer migration.

## Context

Coeus exposes fused SGD, Adam, RMSProp, AdamW, and AdaGrad updates through a
backend trait. CPU execution owns one generic implementation, WGPU and CUDA
carry separate provider-language formulas, CUDA silently downloads unsupported
inputs for CPU execution, and ROCm and Metal do not implement the contract.
Hephaestus already owns accelerator kernel authoring and dispatch, but its
elementwise seam has one output and prohibits the read/write aliasing required
by optimizer state transitions.

An optimizer update is one elementwise transaction over a parameter, a
gradient, and one or two state buffers. Every writable value must be computed
from the transaction's pre-update values. Validation failure must leave all
writable buffers unchanged.

## Decision

1. `hephaestus-core` owns a `StatefulUpdateOps<D, T>` seam parameterized by a
   zero-sized update-rule marker, scalar type, fixed rank, and const state
   count. The request contains borrowed strided parameter, gradient, and state
   views plus rule-specific POD parameters. Dispatch remains monomorphized;
   there is no trait-object or per-element runtime dispatch.
2. Rule markers own the mathematical update and provider-expression contract
   for SGD, Adam, RMSProp, AdamW, and AdaGrad. A rule declares its state count
   and parameter block, preventing unrelated optional fields and invalid rule
   configurations.
3. One shared planner validates equal logical shapes, checked storage spans,
   backend address limits, finite hyperparameters, positive Adam step, readable
   zero-stride gradients, injective writable layouts, and pairwise non-aliasing
   before kernel preparation or mutation. Empty validated requests are no-ops.
4. WGPU, CUDA, ROCm, and Metal implement the same seam through their existing
   typed authored-kernel and multi-storage machinery. Provider modules own
   dialect source, compilation, metadata packing, launch, and synchronization;
   consumers never supply shader source.
5. `hephaestus-conformance` owns one Leto-differential contract instantiated
   by every admitted backend and scalar. It covers all rules, nonzero state,
   repeated steps, strided and offset views, rank boundaries, empty requests,
   alias and shape rejection, unavailable devices, and failure atomicity.
6. Coeus converts its dynamic layouts into this contract once in
   `coeus-hephaestus`. CPU execution remains directly through Leto-owned
   operations. Accelerator crates select their Hephaestus provider and delete
   local optimizer formulas and host fallback paths.

The initial portable scalar contract is `f32`. Additional scalars enter only
when every selected provider supplies native arithmetic and square-root
semantics with conformance coverage; widening and narrowing are not fallback
implementations.

## Alternatives

- Extend `ElementwiseOps`: rejected because its single-output, non-aliasing
  contract cannot represent atomic parameter plus state updates.
- Let Coeus author kernels through `KernelDevice`: rejected because formulas
  and provider-language source would remain consumer-owned and duplicated.
- Retain CUDA host execution for unsupported cases: rejected because it hides
  backend selection, allocates tensor-sized host buffers, transfers every
  operand twice, and converts provider failure into silent execution elsewhere.
- Add one trait per optimizer: rejected because traversal, validation,
  metadata, and launch structure are one variation family; rule markers encode
  the mathematical variation without cloning the dispatch architecture.

## Consequences

Hephaestus gains one stateful multi-output kernel family and a larger public
surface. Coeus must make `OptimizerOps` and public optimizer steps fallible, a
breaking contract change recorded in its own migration unit. In return,
accelerator dispatch stays device-resident, formulas have one provider-owned
home, ROCm and Metal gain the same extension point, and unsupported capabilities
fail explicitly without compatibility or CPU fallback paths.
