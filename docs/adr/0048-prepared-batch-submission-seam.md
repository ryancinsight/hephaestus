# ADR 0048 — Prepared batch submission seam

- Status: Accepted
- Date: 2026-08-01 (revised same day: implemented as-designed for the
  sparse union; `spmv_dispatch` gained a decoupled plan-borrow lifetime
  (`'plan` vs the operand `'op`) that the single-lifetime draft's
  invariant GAT position made unusable at call sites)
- Refs: atlas `backlog.md#atlas-arch-001` (001k-k2); ADR 0041 (conformance
  crate); the per-backend `submit_prepared_{reduction,axis_reduction,
  sparse}_batch` entry points this seam makes device-neutral.

## Context

Each backend ships batch-submission entry points that amortize
per-dispatch submission overhead: a slice of prepared operations is
encoded into one command buffer (WGPU) or launched back-to-back without
intervening synchronization (CUDA/HIP). The three ledger entry points —
`submit_prepared_reduction_batch`, `submit_prepared_axis_reduction_batch`,
`submit_prepared_sparse_batch` — exist on all four backends but only as
device-concrete functions over backend-local dispatch-handle types
(e.g. wgpu's `PreparedSparseDispatch<'a, T>` enum with `Spmv`/`Spmm`
variants), so no generic conformance clause can drive them and consumers
cannot batch without naming a device type.

## Decision (recommended)

A `BatchSubmitOps<D: ComputeDevice, T: Pod>` trait in `hephaestus-core`
whose dispatch-handle union is an associated type, mirroring the shape
every backend already ships:

```rust
pub trait BatchSubmitOps<D: ComputeDevice, T: Pod> {
    /// One batchable prepared dispatch. Backends define the union of
    /// prepared forms they can encode into a single submission.
    type Dispatch<'op>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Submit every operation in order with one device round-trip.
    fn submit_batch(&self, device: &D, operations: &[Self::Dispatch<'_>]) -> Result<()>;
}
```

Staging: the first increment defines `Dispatch` as the sparse union
(`Spmv`/`Spmm`, wrapping the `PreparedApply` handles ADR-0041-family
seams already expose) because `submit_prepared_sparse_batch` is the entry
point with an existing cross-backend enum; the reduction batches join as
variants in a second increment once their prepared handles are
seam-reachable (the axis/full-reduction `Prepared` GATs already are).
Conformance clause: a batch of heterogeneous prepared dispatches produces
exactly the results of dispatching each individually (exact fixtures),
plus empty-batch as a valid no-op and the rebind contract across a batch.

## Alternatives

- Per-family batch methods on the existing seams
  (`dispatch_apply_batch(&[&PreparedApply])`): rejected — forks the batch
  concept per family and cannot express the mixed batches the backends
  already support (`submit_prepared_mixed_reduction_batch`).
- Routing batching through the `CommandStream` seam: rejected for now —
  CUDA/HIP prepared forms launch eagerly rather than encoding into a
  stream, so a stream-level contract would misdescribe half the
  implementations; revisit if the backends converge on deferred encoding.

## Consequences

The three batch entry points become seam-reachable and clause-covered,
closing the last no-seam group in the 001 ledger. The union associated
type keeps kernel-granularity dispatch and zero-cost monomorphization;
backends whose submission has no batching advantage may implement
`submit_batch` as a loop over individual dispatches without violating the
contract (the contract is result equivalence, not timing).
