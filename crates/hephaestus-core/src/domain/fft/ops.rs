use bytemuck::Pod;

use crate::domain::error::{HephaestusError, Result};
use crate::domain::stream::{CommandStream, KernelDevice};

use super::{FftDirection, FftOperands};

/// Monomorphized accelerator dense complex FFT operations.
///
/// Implementors are backend-owned zero-sized markers. Preparation validates
/// fixed operands, selects every axis strategy, compiles pipelines, and owns
/// all scratch required by repeated dispatch.
pub trait FftOps<D: KernelDevice, T: Pod> {
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

    /// Transform selected axes of caller-owned split-complex storage in place.
    ///
    /// # Errors
    ///
    /// Returns a typed axis, validation, preparation, or device-dispatch
    /// failure.
    fn fft_axes_in_place<const R: usize>(
        &self,
        device: &D,
        operands: FftOperands<'_, D::Buffer<T>, R>,
        direction: FftDirection,
        axes: &[usize],
    ) -> Result<()> {
        let prepared = self.prepare_fft_axes(device, operands, direction, axes)?;
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

    /// Validate and prepare a transform over selected axes of fixed operands.
    ///
    /// Existing providers inherit all-axis behavior. Providers supporting
    /// selected axes override this method and reject invalid selections before
    /// allocating provider state or mutating operands.
    ///
    /// # Errors
    ///
    /// Returns a typed axis, validation, preparation, allocation, compilation,
    /// or unsupported-capability failure.
    fn prepare_fft_axes<'a, const R: usize>(
        &self,
        device: &'a D,
        operands: FftOperands<'a, D::Buffer<T>, R>,
        direction: FftDirection,
        axes: &[usize],
    ) -> Result<Self::Prepared<'a, R>> {
        let selects_every_axis = axes.len() == R && (0..R).all(|axis| axes.contains(&axis));
        if selects_every_axis {
            self.prepare_fft(device, operands, direction)
        } else {
            Err(HephaestusError::InvalidConfiguration {
                message: "selected-axis FFT is unsupported by this provider".to_owned(),
            })
        }
    }

    /// Encode a prepared transform into an existing provider command stream.
    ///
    /// This is the provider-neutral composition boundary for consumers that
    /// combine an FFT with adjacent accelerator kernels in one submission.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed ownership or command-encoding failure.
    fn encode_fft<const R: usize>(
        &self,
        device: &D,
        prepared: &Self::Prepared<'_, R>,
        stream: &mut D::Stream<'_>,
    ) -> Result<()>;

    /// Dispatch a prepared transform without changing its provider or shape.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed command-encoding or submission failure.
    /// Completion and asynchronous execution failures are observed through
    /// [`crate::ComputeDevice::synchronize`] or a synchronizing transfer.
    fn dispatch_fft<const R: usize>(
        &self,
        device: &D,
        prepared: &Self::Prepared<'_, R>,
    ) -> Result<()> {
        let mut stream = device.stream()?;
        self.encode_fft(device, prepared, &mut stream)?;
        stream.submit()
    }
}
