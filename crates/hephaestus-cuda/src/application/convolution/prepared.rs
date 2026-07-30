use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{BlockWidth, ConvolutionForwardOperands, Result};

use super::metadata::ConvolutionMeta;
use super::resources::same_device;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

const BLOCK_WIDTH: BlockWidth = BlockWidth::DEFAULT;

pub(super) fn compile<T: 'static>(
    device: &CudaDevice,
    entry: &'static str,
    spatial_rank: usize,
    bias: bool,
    source: String,
) -> Result<Arc<SafeCachedKernel>> {
    let key = PipelineKey::Convolution {
        entry,
        scalar: TypeId::of::<T>(),
        spatial_rank,
        bias,
    };
    cached_kernel(device, key, entry, || source)
}

/// Prepared CUDA forward convolution with borrowed device operands.
pub struct PreparedConvolutionForward<'a, T> {
    device: &'a CudaDevice,
    kernel: Option<Arc<SafeCachedKernel>>,
    input: &'a CudaBuffer<T>,
    weight: &'a CudaBuffer<T>,
    bias: Option<&'a CudaBuffer<T>>,
    output: &'a CudaBuffer<T>,
    metadata: ConvolutionMeta,
    elements: usize,
}

impl<'a, T> PreparedConvolutionForward<'a, T> {
    pub(super) fn new<const R: usize>(
        device: &'a CudaDevice,
        kernel: Option<Arc<SafeCachedKernel>>,
        operands: ConvolutionForwardOperands<'a, CudaBuffer<T>, R>,
        metadata: ConvolutionMeta,
        elements: usize,
    ) -> Self {
        Self {
            device,
            kernel,
            input: operands.input.buffer,
            weight: operands.weight.buffer,
            bias: operands.bias.map(|bias| bias.buffer),
            output: operands.output.buffer,
            metadata,
            elements,
        }
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut input = self.input.raw();
        let mut weight = self.weight.raw();
        let mut output = self.output.raw();
        let mut metadata = self.metadata;
        let mut bias = self.bias.map(CudaBuffer::raw);
        if let Some(bias) = bias.as_mut() {
            let mut args = [
                argument(&mut input),
                argument(&mut weight),
                argument(bias),
                argument(&mut output),
                argument(&mut metadata),
            ];
            return launch(device, kernel, self.elements, &mut args);
        }
        let mut args = [
            argument(&mut input),
            argument(&mut weight),
            argument(&mut output),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args)
    }
}

pub(super) struct PreparedConvolutionGradient<'a, T> {
    kernel: Option<Arc<SafeCachedKernel>>,
    first: &'a CudaBuffer<T>,
    second: Option<&'a CudaBuffer<T>>,
    target: &'a CudaBuffer<T>,
    metadata: ConvolutionMeta,
    elements: usize,
}

impl<'a, T> PreparedConvolutionGradient<'a, T> {
    const fn new(
        kernel: Option<Arc<SafeCachedKernel>>,
        first: &'a CudaBuffer<T>,
        second: Option<&'a CudaBuffer<T>>,
        target: &'a CudaBuffer<T>,
        metadata: ConvolutionMeta,
        elements: usize,
    ) -> Self {
        Self {
            kernel,
            first,
            second,
            target,
            metadata,
            elements,
        }
    }

    fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut first = self.first.raw();
        let mut second = self.second.map(CudaBuffer::raw);
        let mut target = self.target.raw();
        let mut metadata = self.metadata;
        if let Some(second) = second.as_mut() {
            let mut args = [
                argument(&mut first),
                argument(second),
                argument(&mut target),
                argument(&mut metadata),
            ];
            return launch(device, kernel, self.elements, &mut args);
        }
        let mut args = [
            argument(&mut first),
            argument(&mut target),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args)
    }
}

/// Every selected CUDA additive-gradient kernel prepared as one unit.
pub struct PreparedConvolutionBackward<'a, T> {
    device: &'a CudaDevice,
    input: Option<PreparedConvolutionGradient<'a, T>>,
    weight: Option<PreparedConvolutionGradient<'a, T>>,
    bias: Option<PreparedConvolutionGradient<'a, T>>,
}

impl<'a, T> PreparedConvolutionBackward<'a, T> {
    pub(super) const fn new(
        device: &'a CudaDevice,
        input: Option<PreparedConvolutionGradient<'a, T>>,
        weight: Option<PreparedConvolutionGradient<'a, T>>,
        bias: Option<PreparedConvolutionGradient<'a, T>>,
    ) -> Self {
        Self {
            device,
            input,
            weight,
            bias,
        }
    }

    pub(super) fn gradient(
        kernel: Option<Arc<SafeCachedKernel>>,
        first: &'a CudaBuffer<T>,
        second: Option<&'a CudaBuffer<T>>,
        target: &'a CudaBuffer<T>,
        metadata: ConvolutionMeta,
        elements: usize,
    ) -> PreparedConvolutionGradient<'a, T> {
        PreparedConvolutionGradient::new(kernel, first, second, target, metadata, elements)
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        for gradient in [&self.input, &self.weight, &self.bias]
            .into_iter()
            .flatten()
        {
            gradient.dispatch(device)?;
        }
        Ok(())
    }
}

fn validate_device(owner: &CudaDevice, requested: &CudaDevice) -> Result<()> {
    if same_device(owner, requested) {
        Ok(())
    } else {
        Err(hephaestus_core::HephaestusError::DispatchFailed {
            message: "prepared CUDA convolution belongs to a different device".to_string(),
        })
    }
}

fn launch(
    device: &CudaDevice,
    kernel: &SafeCachedKernel,
    elements: usize,
    args: &mut [*mut core::ffi::c_void],
) -> Result<()> {
    let blocks = grid_size(elements, BLOCK_WIDTH)?;
    launch_kernel(
        device,
        kernel,
        LaunchConfig::linear(blocks, BLOCK_WIDTH),
        args,
    )
}

fn argument<T>(value: &mut T) -> *mut core::ffi::c_void {
    (value as *mut T).cast()
}
