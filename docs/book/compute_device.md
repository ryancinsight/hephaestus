# Compute Device

`ComputeDevice` is the handle to an acquired accelerator device. It carries
device identity and the capability surface that all Hephaestus op traits
dispatch through.

## Acquisition

```rust,ignore
use hephaestus_core::{ComputeDeviceAcquisition, DevicePreference};

let acq = ComputeDeviceAcquisition::acquire(DevicePreference::Discrete)?;
let device = acq.device;
```

`DevicePreference` guides selection:

| Variant | Description |
|---------|-------------|
| `Any` | First available device |
| `Integrated` | Integrated GPU (shared DRAM) |
| `Discrete` | Discrete GPU (own VRAM) |
| `Cuda` | NVIDIA CUDA device |
| `Hip` | AMD HIP/ROCm device |
| `Metal` | Apple Metal device |
| `Wgpu` | wgpu cross-platform device |

## `ComputeDeviceCapabilities`

Reports the feature set of the acquired device:

```rust,ignore
let caps = acq.capabilities;
if caps.supports(DeviceFeature::F16) {
    // half-precision compute available
}
```

`DeviceLimits` exposes hardware constants: max buffer size, max workgroup
size, max compute units, etc.

## `DeviceBuffer<T>`

`DeviceBuffer<T>` is the core buffer abstraction for device-resident
storage. It is the input/output type for all Hephaestus ops:

```rust,ignore
let buf: Box<dyn DeviceBuffer<f32>> = device.allocate(1024)?;
```

All ops take `&dyn DeviceBuffer<T>` or `&mut dyn DeviceBuffer<T>` as
operand references.
