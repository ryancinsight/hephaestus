//! Provider-owned seeded random initialization seam for Wgpu.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, RandomInitOps, Result, Wgsl};
use leto_ops::RealScalar;

use crate::WgpuBuffer;
use crate::WgpuDevice;
use crate::application::random::{normal_with_seed, uniform_with_seed};

/// Provider-owned implementation of [`RandomInitOps`] for Wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuRandomOps;

impl<T> RandomInitOps<WgpuDevice, T> for WgpuRandomOps
where
    T: DialectScalar<Wgsl> + RealScalar + Pod,
{
    fn uniform_with_seed<const N: usize>(
        &self,
        device: &WgpuDevice,
        shape: [usize; N],
        low: T,
        high: T,
        seed: u64,
    ) -> Result<WgpuBuffer<T>> {
        uniform_with_seed(device, shape, low, high, seed)
    }

    fn normal_with_seed<const N: usize>(
        &self,
        device: &WgpuDevice,
        shape: [usize; N],
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<WgpuBuffer<T>> {
        normal_with_seed(device, shape, mean, std_dev, seed)
    }
}
