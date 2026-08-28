use hephaestus_core::{FftDirection, FftOperands, FftOps, HephaestusError, Result, plan_fft};

use crate::application::prepared::{device_owner, validate_device_owner};
use crate::infrastructure::device::PipelineCache;
use crate::infrastructure::{buffer::WgpuBuffer, device::WgpuDevice};
use crate::{WgpuCommandStream, WgpuGroupedSequence};

use super::{plan::WgpuFftPlan, scalar::WgpuFftScalar};

/// WGPU implementation of the device-neutral dense complex FFT seam.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFftOps;

/// Prepared WGPU FFT resources bound to fixed split-complex device buffers.
///
/// Inputs are not filtered for IEEE special values. NaNs propagate through
/// butterfly arithmetic, infinities remain non-finite, and a transform of
/// signed zeros is numerically zero but does not preserve zero signs.
pub struct WgpuPreparedFft<const R: usize, T = f32> {
    plan: WgpuFftPlan<T>,
    owner: PipelineCache,
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

impl<const R: usize, T: WgpuFftScalar> WgpuPreparedFft<R, T> {
    fn validate_device(&self, device: &WgpuDevice) -> Result<()> {
        validate_device_owner(&self.owner, device, "FFT")
    }

    /// Encode this prepared FFT into an existing WGPU command stream.
    ///
    /// This composes FFT execution with a larger accelerator operation without
    /// forcing an intermediate queue submission or rebuilding provider state.
    ///
    /// # Errors
    ///
    /// Returns a typed ownership or command-encoding failure.
    pub(crate) fn encode_into(&self, stream: &mut WgpuCommandStream<'_>) -> Result<()> {
        self.validate_device(stream.device())?;
        self.plan.encode(stream)
    }

    /// Encode the prepared FFT into an active WGPU grouped sequence.
    ///
    /// This WGPU-specific composition boundary lets a provider consumer keep
    /// adjacent kernels and every FFT stage in one provenance-carrying compute
    /// pass. The prepared plan owns cloned handles to its fixed operands, so no
    /// Hephaestus-managed resource allocation, pipeline compilation, bind-group
    /// construction, or device copy occurs.
    ///
    /// # Errors
    ///
    /// Returns a typed ownership error when the sequence was not opened by the
    /// preparation device.
    pub fn encode_in_sequence(&self, sequence: &mut WgpuGroupedSequence<'_>) -> Result<()> {
        self.validate_device(sequence.device())?;
        self.plan.encode_in_pass(sequence.raw_pass_mut());
        Ok(())
    }
}

impl<T: WgpuFftScalar> FftOps<WgpuDevice, T> for WgpuFftOps {
    type Prepared<'a, const R: usize>
        = WgpuPreparedFft<R, T>
    where
        T: 'a;

    fn prepare_fft<'a, const R: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: FftOperands<'a, WgpuBuffer<T>, R>,
        direction: FftDirection,
    ) -> Result<Self::Prepared<'a, R>> {
        T::validate_fft_capability(device)?;
        if !operands.real.buffer.belongs_to(&device.pipeline_cache)
            || !operands.imaginary.buffer.belongs_to(&device.pipeline_cache)
        {
            return Err(invalid(
                "FFT component buffers must belong to the preparation device",
            ));
        }
        let logical = plan_fft::<T, _, R>(
            &operands,
            direction,
            operands.real.buffer.aliases(operands.imaginary.buffer),
        )?;
        logical.validate_address_limit(u32::MAX as usize)?;
        Ok(WgpuPreparedFft {
            plan: WgpuFftPlan::new(
                device,
                logical,
                operands.real.buffer.clone(),
                operands.imaginary.buffer.clone(),
            )?,
            owner: device_owner(device),
        })
    }

    fn encode_fft<const R: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::Prepared<'_, R>,
        stream: &mut WgpuCommandStream<'_>,
    ) -> Result<()> {
        prepared.validate_device(device)?;
        prepared.encode_into(stream)
    }
}

#[cfg(test)]
#[path = "seam_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod scalar_tests;
