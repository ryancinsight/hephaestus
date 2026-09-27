# ADR 0066: Device-neutral rank-2 top-k selection seam

- Status: Accepted
- Refs: atlas `backlog.md#heph-topk-argminmax-provider-1`; coeus
  accelerator-bridge consumer audit (2026-09-26); the sibling argmax/argmin
  seam this record narrows the same backlog item alongside.

## Context

The original consumer-audit item bundled `topk`/`argmax`/`argmin` together.
The argmax/argmin half (a linear-scan reduction, the same shape as
`AxisReductionOps`) landed separately; `topk` is a selection algorithm —
materially different, since the result set's size (`k`) is neither 1 (an
extremum) nor the full axis (a sort), and a device kernel cannot allocate a
per-thread scratch array whose size is chosen at call time.

## Decision

`TopKOps<D, T>` in `hephaestus-core::domain::topk` provides `topk_axis_into`,
fixed at rank 2 (matching `AxisReductionOps`'s and the argmax/argmin seam's
convention). `k` is validated against `input`'s *static* shape
(`axis_len`, known before dispatch) rather than device-resident data, so —
unlike `EmbeddingOps`'s indices — no device-side error-flag mechanism is
needed for it.

Both `HostTopKOps` and `WgpuTopKOps` use the same incremental-insertion
algorithm rather than a full sort: fill the first `k` slots in
sorted-descending order via insertion, then for each remaining element,
displace the current minimum (position `k-1`) whenever a later element is
*strictly* greater, re-establishing sorted order by insertion. This is
`O(axis_len * k)`, appropriate for the modest `k` an embedding/attention-style
top-k call uses; a heap-based `O(axis_len * log k)` variant is a follow-up if
a workload needs larger `k`. Matching the algorithm (not just the output
contract) between host and device is what makes their tie-break agree by
construction: a tie (`val == current_min`) never displaces, so the earlier
index always wins, on both backends, without a documented-but-unenforced
convention.

The device kernel needs no dynamically-sized per-thread scratch: the `k`-slot
result lives directly in the `values`/`indices` output buffers the caller
already sized to `k`, and the kernel's insertion writes through them in
place. `k` itself is an ordinary runtime loop bound read from the uniform,
not baked into the shader text — unlike `axis`, it does not change which
shader instructions are legal, only how many times a loop runs.

CUDA, ROCm, and Metal do not implement `TopKOps` yet, and the seam is not
folded into `assert_backend_contract`/`BackendUnderTest` this increment, for
the reason ADR 0062 gives for `PadOps`.

## Consequences

Coeus can bind `topk_axis_into` on WGPU (and the host reference pair) today,
closing the `backlog.md#heph-topk-argminmax-provider-1` item's remaining
scope. CUDA/ROCm/Metal implementations, the aggregate fold, and a
larger-`k` heap-based algorithm are follow-up items.
