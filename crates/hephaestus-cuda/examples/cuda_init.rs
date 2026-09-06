//! Acquire CUDA through the public Atlas device contract and verify a transfer.

use hephaestus_core::{ComputeDevice, ComputeDeviceCapabilities, Result};
use hephaestus_cuda::CudaDevice;

fn main() -> Result<()> {
    let device = CudaDevice::try_default()?;
    println!("CUDA device 0: {:?}", device.device_limits());
    println!("Free device memory: {} bytes", device.free_memory_bytes()?);
    let expected = [1_u32, 2, 3, 4];
    let buffer = device.upload(&expected)?;
    let mut actual = [0; 4];
    device.download(&buffer, &mut actual)?;
    assert_eq!(actual, expected, "CUDA transfer preserves each value");
    println!("Device transfer: {actual:?}");
    Ok(())
}
