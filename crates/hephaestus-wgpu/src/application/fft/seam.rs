use hephaestus_core::{FftDirection, FftOperands, FftOps, HephaestusError, Result, plan_fft};

use crate::WgpuCommandStream;
use crate::application::prepared::{device_owner, validate_device_owner};
use crate::infrastructure::device::PipelineCache;
use crate::infrastructure::{buffer::WgpuBuffer, device::WgpuDevice};

use super::plan::WgpuFftPlan;

/// WGPU implementation of the device-neutral dense complex FFT seam.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFftOps;

/// Prepared WGPU FFT resources bound to fixed split-complex device buffers.
pub struct WgpuPreparedFft<'a, const R: usize> {
    plan: WgpuFftPlan,
    real: &'a WgpuBuffer<f32>,
    imaginary: &'a WgpuBuffer<f32>,
    owner: PipelineCache,
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

impl<const R: usize> WgpuPreparedFft<'_, R> {
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
        self.plan.encode(stream, self.real, self.imaginary)
    }
}

impl FftOps<WgpuDevice, f32> for WgpuFftOps {
    type Prepared<'a, const R: usize> = WgpuPreparedFft<'a, R>;

    fn prepare_fft<'a, const R: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: FftOperands<'a, WgpuBuffer<f32>, R>,
        direction: FftDirection,
    ) -> Result<Self::Prepared<'a, R>> {
        if !operands.real.buffer.belongs_to(&device.pipeline_cache)
            || !operands.imaginary.buffer.belongs_to(&device.pipeline_cache)
        {
            return Err(invalid(
                "FFT component buffers must belong to the preparation device",
            ));
        }
        let logical = plan_fft::<f32, _, R>(
            &operands,
            direction,
            operands.real.buffer.aliases(operands.imaginary.buffer),
        )?;
        logical.validate_address_limit(u32::MAX as usize)?;
        Ok(WgpuPreparedFft {
            plan: WgpuFftPlan::new(device, logical)?,
            real: operands.real.buffer,
            imaginary: operands.imaginary.buffer,
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
