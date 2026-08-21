# Decomposition Seam

`DecompositionOps<D>` is the device-neutral factorization seam. The current
core contract uses rank-2 `f32` operands and returns typed associated handles;
it does not expose a boxed, scalar-suffixed, or backend-specific result type.

The primary entry points are:

- `lu` for partial-pivoted `P·A = L·U`;
- `qr` for `A = Q·R` with `m >= n`;
- `col_piv_qr` for rank-revealing column-pivoted QR; and
- `full_piv_lu` for rank-revealing fully pivoted LU.

The returned `LuHandle`, `QrHandle`, and related handle traits expose only
the factor, shape, pivot, determinant, and solve capabilities required by
consumers and conformance tests. A solve returns a typed result and rejects
length or singularity failures.

The core also owns the `BlockedDecompositionBackend` seam and the
`blocked_lu` orchestration. Backend implementations supply the panel and
trailing-update operations; the host-side blocked algorithm is not copied once
per device.

Consumer crates bind to these traits and keep their own domain semantics. A
consumer must not reconstruct factorization state from raw buffers or teach the
book an API that is absent from `hephaestus-core`.
