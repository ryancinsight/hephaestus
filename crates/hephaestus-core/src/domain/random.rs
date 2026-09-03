//! Device-neutral seeded random initialization.
//!
//! Every backend delegates seeded initialization to the host generator
//! (leto-ops) and uploads the result, so identical seeds produce
//! identical device contents across backends by construction. The seam
//! exists so consumers and the conformance suite can request seeded
//! buffers without naming a device type.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::Result;

/// Device-neutral seeded random buffer initialization.
///
/// Implementors are zero-sized per-backend markers. Determinism is part of
/// the contract: the same `(shape, parameters, seed)` yields the same
/// buffer contents on every call and every backend.
pub trait RandomInitOps<D: ComputeDevice, T: Pod> {
    /// Allocate a buffer of `shape` filled with i.i.d. uniform samples in
    /// `[low, high)`, derived deterministically from `seed`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-parameter rejection (`low >= high`, zero-sized
    /// shape where unsupported) or an allocation or transfer failure.
    fn uniform_with_seed<const N: usize>(
        &self,
        device: &D,
        shape: [usize; N],
        low: T,
        high: T,
        seed: u64,
    ) -> Result<D::Buffer<T>>;

    /// Allocate a buffer of `shape` filled with i.i.d. normal samples of
    /// the given mean and standard deviation, derived deterministically
    /// from `seed`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-parameter rejection (non-positive standard
    /// deviation) or an allocation or transfer failure.
    fn normal_with_seed<const N: usize>(
        &self,
        device: &D,
        shape: [usize; N],
        mean: T,
        std_dev: T,
        seed: u64,
    ) -> Result<D::Buffer<T>>;
}
