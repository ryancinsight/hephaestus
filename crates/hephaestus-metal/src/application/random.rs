//! Seeded host-delegated random initializers.

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, Wgsl};
use hephaestus_wgpu as wgpu_backend;
use leto_ops::RealScalar;

/// Fill a Metal buffer of `shape` with deterministic uniform samples in
/// `[low, high)`.
pub fn uniform_with_seed<T: DialectScalar<Wgsl> + RealScalar + Pod, const N: usize>(
    device: &MetalDevice,
    shape: [usize; N],
    low: T,
    high: T,
    seed: u64,
) -> Result<MetalBuffer<T>> {
    let inner = wgpu_backend::uniform_with_seed(&device.inner, shape, low, high, seed)?;
    Ok(MetalBuffer { inner })
}

/// Fill a Metal buffer of `shape` with deterministic normal samples.
pub fn normal_with_seed<T: DialectScalar<Wgsl> + RealScalar + Pod, const N: usize>(
    device: &MetalDevice,
    shape: [usize; N],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<MetalBuffer<T>> {
    let inner = wgpu_backend::normal_with_seed(&device.inner, shape, mean, std_dev, seed)?;
    Ok(MetalBuffer { inner })
}
