//! Provider-owned seeded random initialization seam for Rocm.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, HipC, RandomInitOps, Result};
use leto_ops::RealScalar;

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::random::{normal_with_seed, uniform_with_seed};

/// Provider-owned implementation of [`RandomInitOps`] for Rocm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmRandomOps;

impl<T> RandomInitOps<RocmDevice, T> for RocmRandomOps
where
    T: DialectScalar<HipC> + RealScalar + Pod,
{
    fn uniform_with_seed<const N: usize>(
        &self,
        device: &RocmDevice,
        shape: [usize; N],
        low: T,
        high: T,
        seed: u64,
    ) -> Result<RocmBuffer<T>> {
        uniform_with_seed(device, shape, low, high, seed)
    }

    fn normal_with_seed<const N: usize>(
        &self,
        device: &RocmDevice,
        shape: [usize; N],
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<RocmBuffer<T>> {
        normal_with_seed(device, shape, mean, std_dev, seed)
    }
}
