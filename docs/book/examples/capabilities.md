# Example: Capabilities

**Crate**: `hephaestus-host`
**Source**: `crates/hephaestus-host/examples/book_capabilities.rs`

Query `backend_name`, `topology`, and `device_limits` from a device that is
generic over the backend.  The same `print_backend` function works for
`HostDevice`, `WgpuDevice`, or any future backend.

## Source

```rust
{{#include ../../../crates/hephaestus-host/examples/book_capabilities.rs}}
```

## Output

```text
backend: host
topology available: false
max_buffer_size: 18446744073709551615
HostDevice is Copy: no heap cost to pass around
backend: host
topology available: false
all capability assertions passed
```

## What to notice

- `print_backend` is generic over `impl ComputeDevice`, so it accepts any
  backend without knowing which one.  This is the primary use case for the
  contract layer: physics kernels write `fn compute<D: ComputeDevice>(device:
  &D, ...)` and let the caller choose the backend.

- `topology()` returns `None` for `HostDevice` because it is not a GPU;
  real backends return `Some(&GpuTopology)` populated from driver queries.

- `synchronize()` on the host is always `Ok(())` — host operations complete
  before the function returns, so there is no asynchronous queue to drain.

- `HostDevice` is `Copy` and zero-sized; real backends hold an `Arc`-shared
  device handle.  Passing by value never deep-copies device state.
