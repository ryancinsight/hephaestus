# Example: Capabilities

**Crate**: `hephaestus-host`
**Source**: `crates/hephaestus-host/examples/book_capabilities.rs`

Query `backend_name`, `topology`, and `device_limits` from a device generic
over the `ComputeDevice` contract. The example uses `HostDevice`, which
gives deterministic values without requiring an accelerator.

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
  backend that implements the core transfer contract.

- `topology()` returns `None` for `HostDevice` because it is not a GPU;
  real backends return `Some(&GpuTopology)` populated from driver queries.

- `synchronize()` on the host is always `Ok(())` — host operations complete
  before the function returns, so there is no asynchronous queue to drain.

- `HostDevice` is `Copy` and zero-sized. Accelerator handles retain their
  own provider-specific ownership and synchronization state.
