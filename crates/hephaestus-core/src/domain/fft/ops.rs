use bytemuck::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;

use super::{FftDirection, FftOperands};

/// Monomorphized accelerator dense complex FFT operations.
///
/// Implementors are backend-owned zero-sized markers. Preparation validates
/// fixed operands, selects every axis strategy, compiles pipelines, and owns
/// all scratch required by repeated dispatch.
pub trait FftOps<D: ComputeDevice, T: Pod> {
    /// Prepared resources bound to fixed split-complex operands and one rank.
    type Prepared<'a, const R: usize>
    where
        D: 'a,
        T: 'a;

    /// Transform caller-owned split-complex device storage in place.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn fft_in_place<const R: usize>(
        &self,
        device: &D,
        operands: FftOperands<'_, D::Buffer<T>, R>,
        direction: FftDirection,
    ) -> Result<()> {
        let prepared = self.prepare_fft(device, operands, direction)?;
        self.dispatch_fft(device, &prepared)
    }

    /// Validate and prepare one transform over fixed device operands.
    ///
    /// # Errors
    ///
    /// Returns before operand mutation on rank, shape, storage, layout, alias,
    /// capability, allocation, or compilation failure.
    fn prepare_fft<'a, const R: usize>(
        &self,
        device: &'a D,
        operands: FftOperands<'a, D::Buffer<T>, R>,
        direction: FftDirection,
    ) -> Result<Self::Prepared<'a, R>>;

    /// Dispatch a prepared transform without changing its provider or shape.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed command-encoding, submission, or device
    /// execution failure.
    fn dispatch_fft<const R: usize>(
        &self,
        device: &D,
        prepared: &Self::Prepared<'_, R>,
    ) -> Result<()>;
}
