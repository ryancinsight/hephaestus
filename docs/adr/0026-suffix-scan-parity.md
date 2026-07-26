# ADR 0026: Reverse cumulative-sum convenience parity

- Status: accepted
- Date: 2026-07-26
- Scope: WGPU, CUDA, ROCm, and Metal scan application modules
- Change class: `[minor]`

## Context

Leto exposes reverse cumulative sum as `scan_axis::<CumSumOp, _, 2>` with
`ScanDirection::Reverse`. Hephaestus already had the shared rank-2 scan
planner and native reverse scan kernels, but its crate roots exposed only the
generic route and reverse cumulative product convenience helpers. Consumers
therefore had no backend-neutral suffix-sum name to bind against.

## Decision

Expose allocated and caller-owned `suffix_sum` operations in every provider.
Each wrapper calls the existing `CumSumOp` scan with reverse direction. Metal
delegates through the WGPU substrate, preserving one kernel implementation and
the native Metal-selected device path. The API remains rank-2 and strided,
matching the existing scan contract; dynamic-rank expansion is separate work.

## Rejected alternatives

- Require consumers to construct `ScanDirection::Reverse` themselves: this
  leaves the common operation vocabulary asymmetric with `cumsum` and
  `cumprod`.
- Add a second suffix kernel per provider: the existing reverse scan is the
  canonical implementation and already carries the direction contract.
- Implement suffix sum in Coeus: Hephaestus owns the accelerator operation and
  Coeus should consume the provider API.

## Verification

Each provider contract compares allocated and caller-owned results with the
Leto reverse `CumSumOp` oracle, including the existing strided and empty-layout
coverage where applicable. Provider CI is the authoritative compilation and
device-independent gate while the checkout-local Leto/Hermes graph is stale.

## Revisit trigger

Add a higher-rank convenience form only after all providers expose the same
rank-generic scan contract and its CI matrix carries equivalent value tests.
