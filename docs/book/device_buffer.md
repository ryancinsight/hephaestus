# Device Buffer

`DeviceBuffer<T>` is the core storage abstraction for accelerator-resident
data in Hephaestus.

## Purpose

Every Hephaestus op trait operates on `DeviceBuffer<T>` references.
The buffer may reside in GPU global memory, CUDA unified memory, host-pinned
memory, or host RAM (for the host backend). The caller never needs to know
which backend is active.

## Lifecycle

```rust,ignore
// Allocate a device buffer of 1024 f32 elements
let mut buf: Box<dyn DeviceBuffer<f32>> = device.allocate(1024)?;

// Upload data from host to device
device.upload(&host_slice, &mut *buf)?;

// Run a kernel that reads/writes buf
device.elementwise_unary(&ReluOp, &*buf, &mut *output_buf)?;

// Download results back to host
device.download(&*buf, &mut host_slice)?;
```

## `StridedView`

`StridedView` describes a non-contiguous view into a `DeviceBuffer`:
offset, shape, and per-axis strides. Op traits accept `StridedView` to
support transposed, sliced, and broadcast tensor operands without copying.

## Memory Tiers

The Mnemosyne allocator backend determines the physical memory tier:

| Backend | Memory Tier |
|---------|-------------|
| `hephaestus-cuda` | `MemoryTier::Gddr` (CUDA device memory) |
| `hephaestus-wgpu` | `MemoryTier::Device` (wgpu storage buffer) |
| `hephaestus-host` | `MemoryTier::Dram` (host RAM) |
| CUDA unified memory | `MemoryTier::HostPinned` |

## No Backend Lock-In

Code that takes `&dyn DeviceBuffer<T>` and dispatches through a Hephaestus
op trait is backend-agnostic. The backend is selected at device acquisition
time; thereafter all dispatch is monomorphic or dynamically dispatched
through the trait object.
