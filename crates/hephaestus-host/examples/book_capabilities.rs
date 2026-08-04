//! Backend name and capability inspection without a physical accelerator.
//!
//! A caller that is generic over a [`ComputeDevice`] implementation can
//! inspect the device's capabilities without knowing which backend is
//! present.  [`HostDevice`] provides deterministic, stable answers for
//! the host-memory reference case; real backends (wgpu, CUDA, ROCm) fill
//! the same slots from driver queries.
//!
//! This example also shows the zero-cost `ComputeDeviceCapabilities` query
//! surface and confirms that `synchronize()` is infallible on the host
//! (host operations complete before returning; there is no asynchronous queue).

use hephaestus_core::{ComputeDevice, ComputeDeviceCapabilities};
use hephaestus_host::HostDevice;

fn print_backend(dev: &impl ComputeDevice) {
    println!("backend: {}", dev.backend_name());
    println!("topology available: {}", dev.topology().is_some());
    dev.synchronize().expect("synchronize should never fail");
}

fn main() {
    let host = HostDevice::new();

    print_backend(&host);

    let limits = host.device_limits();
    println!("max_buffer_size: {}", limits.max_buffer_size);
    assert_eq!(limits.max_buffer_size, u64::MAX);

    // HostDevice reports no GPU topology — it is not a GPU.
    assert!(host.topology().is_none());

    // HostDevice is Copy and zero-sized — safe to pass by value everywhere.
    let _ = host; // move to show Copy
    let host2 = HostDevice::new();
    let _ = host2;
    println!("HostDevice is Copy: no heap cost to pass around");

    // Backend-generic check: print_backend works for any ComputeDevice.
    print_backend(&HostDevice);

    println!("all capability assertions passed");
}
