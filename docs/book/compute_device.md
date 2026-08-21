# Compute Device

`ComputeDevice` is the backend-neutral transfer and synchronization
contract. It owns a typed associated buffer family, so a buffer allocated by
one device implementation cannot be passed to a kernel for another device
type.

## Acquisition

`ComputeDeviceAcquisition` exposes two fallible entry points:

- `try_acquire_device` acquires one device from a label, a
  `DevicePreference`, optional `DeviceFeature` values, and required
  `DeviceLimits`.
- `try_acquire_devices` requests up to a bounded number of matching devices.

`DevicePreference` has two variants: `HighPerformance` and `LowPower`.
They are selection hints; they do not claim that a particular vendor or
device type is present.

## Transfers and lifetime

`ComputeDevice` exposes `alloc_zeroed`, `alloc_uninitialized`, `upload`,
`download`, `download_owned`, `write_buffer`, `write_sub_buffer`,
`copy_buffer`, `topology`, and `synchronize`. Length and byte-size
failures are returned as typed `HephaestusError` values.
`alloc_uninitialized` is reserved for producers that overwrite every
element before a read; callers needing defined contents use `alloc_zeroed`.

The host reference backend makes this contract executable without accelerator
hardware:

```rust
extern crate hephaestus_core;
extern crate hephaestus_host;

use hephaestus_core::ComputeDevice;
use hephaestus_host::HostDevice;

fn main() -> hephaestus_core::Result<()> {
    let device = HostDevice::new();
    let source = [1.0_f32, 2.0, 3.0];
    let buffer = device.upload(&source)?;
    let mut restored = [0.0_f32; 3];
    device.download(&buffer, &mut restored)?;
    assert_eq!(restored, source);
    Ok(())
}
```

The complete allocation, subrange-write, copy, and length-mismatch example is
the [Host Device example](examples/host_device.md).
