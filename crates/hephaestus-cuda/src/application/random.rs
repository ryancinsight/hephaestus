//! Seeded host-delegated PRNG initializers.

use crate::application::cuda_type::CudaScalar;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, Result};
use leto_ops::RealScalar;

/// Fill a GPU-resident buffer of `shape` with i.i.d. uniform samples in
/// `[low, high)`, derived deterministically from `seed`.
pub fn uniform_with_seed<T: CudaScalar + RealScalar + Pod, const N: usize>(
    device: &CudaDevice,
    shape: [usize; N],
    low: T,
    high: T,
    seed: u64,
) -> Result<CudaBuffer<T>> {
    let arr = leto_ops::uniform_with_seed(shape, low, high, seed).map_err(|e| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("RNG uniform failed: {e}"),
        }
    })?;
    device.upload(leto::Storage::as_slice(arr.storage()))
}

/// Fill a GPU-resident buffer of `shape` with i.i.d. normal samples of the
/// given `mean` and `std_dev`, derived deterministically from `seed`.
pub fn normal_with_seed<T: CudaScalar + RealScalar + Pod, const N: usize>(
    device: &CudaDevice,
    shape: [usize; N],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<CudaBuffer<T>> {
    let arr = leto_ops::normal_with_seed(shape, mean, std_dev, seed).map_err(|e| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("RNG normal failed: {e}"),
        }
    })?;
    device.upload(leto::Storage::as_slice(arr.storage()))
}
