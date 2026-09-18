//! Leto as a seeded-random-initialization-seam implementor (ADR 0046).
//!
//! Every backend delegates seeded generation to leto-ops and uploads the
//! result, so the host is the generator itself: [`HostRandomOps`] returns
//! leto-ops' sequence in a host buffer, and the conformance clause's bitwise
//! oracle is the same computation.

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, RandomInitOps, Result};
use leto_ops::RealScalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Seeded uniform and normal initialization for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostRandomOps;

impl<T> RandomInitOps<HostDevice, T> for HostRandomOps
where
    T: Pod + RealScalar,
{
    fn uniform_with_seed<const N: usize>(
        &self,
        device: &HostDevice,
        shape: [usize; N],
        low: T,
        high: T,
        seed: u64,
    ) -> Result<HostBuffer<T>> {
        let samples =
            leto_ops::uniform_with_seed(shape, low, high, seed).map_err(map_leto_error)?;
        device.upload(leto::Storage::as_slice(samples.storage()))
    }

    fn normal_with_seed<const N: usize>(
        &self,
        device: &HostDevice,
        shape: [usize; N],
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<HostBuffer<T>> {
        let samples =
            leto_ops::normal_with_seed(shape, mean, std_dev, seed).map_err(map_leto_error)?;
        device.upload(leto::Storage::as_slice(samples.storage()))
    }
}
