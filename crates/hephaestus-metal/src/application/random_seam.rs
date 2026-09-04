//! Provider-owned seeded random initialization seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, RandomInitOps, Result, Wgsl};
use leto_ops::RealScalar;

use crate::MetalBuffer;
use crate::MetalDevice;
use crate::application::random::{normal_with_seed, uniform_with_seed};

/// Provider-owned implementation of [`RandomInitOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalRandomOps;

impl<T> RandomInitOps<MetalDevice, T> for MetalRandomOps
where
    T: DialectScalar<Wgsl> + RealScalar + Pod,
{
    fn uniform_with_seed<const N: usize>(
        &self,
        device: &MetalDevice,
        shape: [usize; N],
        low: T,
        high: T,
        seed: u64,
    ) -> Result<MetalBuffer<T>> {
        uniform_with_seed(device, shape, low, high, seed)
    }

    fn normal_with_seed<const N: usize>(
        &self,
        device: &MetalDevice,
        shape: [usize; N],
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<MetalBuffer<T>> {
        normal_with_seed(device, shape, mean, std_dev, seed)
    }
}
