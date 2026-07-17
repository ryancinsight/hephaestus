# ADR 0009: Order-preserving tiled axis scans

Status: Accepted
Date: 2026-07-17
Driver: HEPH-SCAN-TILED-1 / KS-5b

## Context

The axis-scan kernels previously assigned one thread to each complete line.
That removed the old quadratic work, but a long line still serialized all
element folds. The core planner also expressed the dispatch as a covering
block count, which is no longer the right contract once a block owns one line.
Apollo, Kwavers, and other consumers must continue to receive one provider
owned scan implementation through Hephaestus; no consumer-side WGPU kernel is
introduced.

## Decision

Both WGPU and CUDA dispatch exactly one workgroup/block per scan line. With
`W` lanes, a line of length `L` is partitioned into contiguous logical chunks
of length `ceil(L/W)`. Each lane computes its local prefix in sequence and
stores its local total in workgroup/shared memory. Lane zero folds those
totals from the first chunk to the last and writes each lane's preceding
prefix back to the same tile. After a barrier, each lane combines that
preceding prefix with its local outputs.

The operation vocabulary remains generic (`CombineExpr` and `IdentityToken`);
the scalar type and operation are still selected at the pipeline cache
boundary. The CUDA launch requests exactly `W * size_of::<T>()` bytes of
dynamic shared memory. WGPU declares the equivalent statically in WGSL.

## Theorem and evidence

For an associative operation with identity `e`, let `C_j` be the ordered fold
of chunk `j`, and let `P_j = C_0 ⊗ ... ⊗ C_{j-1}`. Every element `x` in chunk
`j` has local prefix `Q_j(x)`, so the kernel writes `P_j ⊗ Q_j(x)`, equal to
the sequential fold by associativity. The chunk partition is contiguous and
the lane-zero fold is ordered, so no element changes logical direction,
including reverse scans. Integer contracts therefore remain exact. Floating
point addition and multiplication are not associative; their reassociation is
intentional and must be checked with the derived `O(log2(L) * eps * sum|x|)`
bound when a floating differential contract is added.

Evidence for this increment is compile-time source-contract tests in both
backends, core dispatch value tests, and the existing real-device integer scan
contracts. A backend contract that exercises `L > W` is required before the
KS-5b multi-pass extension is accepted.

The existing WGPU and CUDA `L = 513`, `W = 256` integer contracts satisfy that
condition. For `L >= 1`, the kernel uses `W` shared partials and each lane
performs at most `ceil(L/W)` sequential folds, so `shared_bytes = W *
size_of(T)` is independent of `L`. The current implementation therefore has
no line-length-driven shared-memory overflow; KS-5b is a performance follow-up
that reopens only on a measured device limit or latency-budget failure.

## Consequences

- Long lines expose lane-level work without duplicating scan algorithms in
  consumers or reintroducing raw WGPU ownership.
- Dispatch planning now reports the exact number of lines, not a covering
  count for one-thread-per-line blocks.
- Shared-memory limits remain provider concerns; a future multi-pass path is
  tracked separately under KS-5b and must preserve the same generic operation
  vocabulary and derived floating-point tolerance policy.
