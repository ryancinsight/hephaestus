//! Seeded host-delegated random initializers for ROCm buffers.

use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, DialectScalar, HipC, Result};
use leto_ops::RealScalar;

use crate::RocmDevice;
use crate::infrastructure::RocmBuffer;

/// Fill a ROCm buffer with deterministic uniform samples in `[low, high)`.
///
/// Random value generation is delegated explicitly to `leto-ops`, matching
/// the CUDA and WGPU backend contracts; the resulting values are uploaded to
/// device storage in one transfer.
///
/// # Errors
///
/// Returns a typed dispatch error when `leto-ops` rejects the range or when
/// the device upload fails.
pub fn uniform_with_seed<T: DialectScalar<HipC> + RealScalar + Pod, const N: usize>(
    device: &RocmDevice,
    shape: [usize; N],
    low: T,
    high: T,
    seed: u64,
) -> Result<RocmBuffer<T>> {
    let array = leto_ops::uniform_with_seed(shape, low, high, seed).map_err(|error| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("RNG uniform failed: {error}"),
        }
    })?;
    device.upload(leto::Storage::as_slice(array.storage()))
}

/// Fill a ROCm buffer with deterministic normal samples.
///
/// Random value generation is delegated explicitly to `leto-ops`, matching
/// the CUDA and WGPU backend contracts; the resulting values are uploaded to
/// device storage in one transfer.
///
/// # Errors
///
/// Returns a typed dispatch error when `leto-ops` rejects the distribution or
/// when the device upload fails.
pub fn normal_with_seed<T: DialectScalar<HipC> + RealScalar + Pod, const N: usize>(
    device: &RocmDevice,
    shape: [usize; N],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<RocmBuffer<T>> {
    let array = leto_ops::normal_with_seed(shape, mean, std_dev, seed).map_err(|error| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("RNG normal failed: {error}"),
        }
    })?;
    device.upload(leto::Storage::as_slice(array.storage()))
}
