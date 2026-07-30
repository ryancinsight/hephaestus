use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{BlockWidth, Result};

use super::metadata::ConvolutionMeta;
use super::resources::same_device;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::RocmBuffer;
use crate::infrastructure::device::RocmDevice;

const BLOCK_WIDTH: BlockWidth = BlockWidth::DEFAULT;

pub(super) fn compile<T: 'static>(
    device: &RocmDevice,
    entry: &'static str,
    spatial_rank: usize,
    bias: bool,
    source: String,
) -> Result<Arc<RocmKernel>> {
    let key = PipelineKey::Convolution {
        entry,
        scalar: TypeId::of::<T>(),
        spatial_rank,
        bias,
    };
    cached_kernel(device, key, entry, || source)
}

/// Prepared ROCm forward convolution with borrowed device operands.
pub struct PreparedConvolutionForward<'a, T> {
    device: &'a RocmDevice,
    kernel: Option<Arc<RocmKernel>>,
    input: &'a RocmBuffer<T>,
    weight: &'a RocmBuffer<T>,
    bias: Option<&'a RocmBuffer<T>>,
    output: &'a RocmBuffer<T>,
    metadata: ConvolutionMeta,
    elements: usize,
}

impl<'a, T> PreparedConvolutionForward<'a, T> {
    pub(super) const fn new(
        device: &'a RocmDevice,
        kernel: Option<Arc<RocmKernel>>,
        input: &'a RocmBuffer<T>,
        weight: &'a RocmBuffer<T>,
        bias: Option<&'a RocmBuffer<T>>,
        output: &'a RocmBuffer<T>,
        metadata: ConvolutionMeta,
        elements: usize,
    ) -> Self {
        Self {
            device,
            kernel,
            input,
            weight,
            bias,
            output,
            metadata,
            elements,
        }
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        validate_device(self.device, device)?;
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut input = self.input.raw();
        let mut weight = self.weight.raw();
        let mut output = self.output.raw();
        let mut metadata = self.metadata;
        let mut bias = self.bias.map(RocmBuffer::raw);
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
    kernel: Option<Arc<RocmKernel>>,
    first: &'a RocmBuffer<T>,
    second: Option<&'a RocmBuffer<T>>,
    target: &'a RocmBuffer<T>,
    metadata: ConvolutionMeta,
    elements: usize,
}

impl<'a, T> PreparedConvolutionGradient<'a, T> {
    const fn new(
        kernel: Option<Arc<RocmKernel>>,
        first: &'a RocmBuffer<T>,
        second: Option<&'a RocmBuffer<T>>,
        target: &'a RocmBuffer<T>,
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

    fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut first = self.first.raw();
        let mut second = self.second.map(RocmBuffer::raw);
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

/// Every selected ROCm additive-gradient kernel prepared as one unit.
pub struct PreparedConvolutionBackward<'a, T> {
    device: &'a RocmDevice,
    input: Option<PreparedConvolutionGradient<'a, T>>,
    weight: Option<PreparedConvolutionGradient<'a, T>>,
    bias: Option<PreparedConvolutionGradient<'a, T>>,
}

impl<'a, T> PreparedConvolutionBackward<'a, T> {
    pub(super) const fn new(
        device: &'a RocmDevice,
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
        kernel: Option<Arc<RocmKernel>>,
        first: &'a RocmBuffer<T>,
        second: Option<&'a RocmBuffer<T>>,
        target: &'a RocmBuffer<T>,
        metadata: ConvolutionMeta,
        elements: usize,
    ) -> PreparedConvolutionGradient<'a, T> {
        PreparedConvolutionGradient::new(kernel, first, second, target, metadata, elements)
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
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

fn validate_device(owner: &RocmDevice, requested: &RocmDevice) -> Result<()> {
    if same_device(owner, requested) {
        Ok(())
    } else {
        Err(hephaestus_core::HephaestusError::DispatchFailed {
            message: "prepared ROCm convolution belongs to a different device".to_string(),
        })
    }
}

fn launch(
    device: &RocmDevice,
    kernel: &RocmKernel,
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
