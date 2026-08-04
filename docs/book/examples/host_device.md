# Example: Host Device

**Crate**: `hephaestus-host`
**Source**: `crates/hephaestus-host/examples/book_host_device.rs`

The `HostDevice` implements `ComputeDevice` over plain host memory — no GPU
required.  This example demonstrates the full buffer lifecycle: allocate,
write, sub-write, download, and copy.

## Source

```rust
{{#include ../../../crates/hephaestus-host/examples/book_host_device.rs}}
```

## Output

```text
backend: host
allocated 8 f32 slots
round-trip: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
after sub-write: [0.0, 1.0, 99.0, 88.0, 77.0, 5.0, 6.0, 7.0]
length mismatch returned LengthMismatch { host_len: 4, device_len: 8 }
copy verified: [0.0, 1.0, 99.0, 88.0, 77.0, 5.0, 6.0, 7.0]
all host-device assertions passed
```

## What to notice

- `HostDevice::new()` returns a zero-sized unit struct — no allocation, no
  I/O, no runtime cost.  Every GPU backend constructor similarly returns a
  lightweight handle rather than copying device state.

- `alloc_zeroed_with_hint` takes a `PlacementHint` from `themis`.  On the
  host device the hint is ignored (no NUMA on a single-socket system); on a
  real GPU backend it selects the memory tier (HBM vs GDDR vs unified).

- `write_sub_buffer` is range-checked: the `offset + data.len()` must fit
  inside the buffer, otherwise `HephaestusError::LengthMismatch` is returned.
  No unsafe indexing is possible through the contract.

- `copy_buffer` into itself is a no-op, detected by comparing `Arc` pointers.
