//! Provider-owned seeded random initialization seam for Cuda.

use eunomia::Pod;
use hephaestus_core::{CudaC, DialectScalar, RandomInitOps, Result};
use leto_ops::RealScalar;

use crate::CudaBuffer;
use crate::CudaDevice;
use crate::application::random::{normal_with_seed, uniform_with_seed};

/// Provider-owned implementation of [`RandomInitOps`] for Cuda.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRandomOps;

impl<T> RandomInitOps<CudaDevice, T> for CudaRandomOps
where
    T: DialectScalar<CudaC> + RealScalar + Pod,
{
    fn uniform_with_seed<const N: usize>(
        &self,
        device: &CudaDevice,
        shape: [usize; N],
        low: T,
        high: T,
        seed: u64,
    ) -> Result<CudaBuffer<T>> {
        uniform_with_seed(device, shape, low, high, seed)
    }

    fn normal_with_seed<const N: usize>(
        &self,
        device: &CudaDevice,
        shape: [usize; N],
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<CudaBuffer<T>> {
        normal_with_seed(device, shape, mean, std_dev, seed)
    }
}
