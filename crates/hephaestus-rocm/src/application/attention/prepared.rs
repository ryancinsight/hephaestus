use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{BlockWidth, Result};

use super::metadata::AttentionMeta;
use super::resources::same_device;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::{RocmBuffer, RocmDevice};

const BLOCK_WIDTH: BlockWidth = BlockWidth::DEFAULT;

pub(super) fn compile<T: 'static>(
    device: &RocmDevice,
    entry: &'static str,
    source: String,
) -> Result<Arc<RocmKernel>> {
    cached_kernel(
        device,
        PipelineKey::Attention {
            entry,
            scalar: TypeId::of::<T>(),
        },
        entry,
        || source,
    )
}

/// Prepared ROCm attention forward pass with borrowed device operands.
pub struct PreparedAttentionForward<'a, T> {
    device: &'a RocmDevice,
    kernel: Option<Arc<RocmKernel>>,
    query: &'a RocmBuffer<T>,
    key: &'a RocmBuffer<T>,
    value: &'a RocmBuffer<T>,
    keep: &'a RocmBuffer<T>,
    output: &'a RocmBuffer<T>,
    weights: &'a RocmBuffer<T>,
    scale: T,
    metadata: AttentionMeta,
    rows: usize,
}

impl<'a, T: Copy> PreparedAttentionForward<'a, T> {
    #[expect(
        clippy::too_many_arguments,
        reason = "prepared attention retains each validated borrowed launch operand"
    )]
    pub(super) const fn new(
        device: &'a RocmDevice,
        kernel: Option<Arc<RocmKernel>>,
        query: &'a RocmBuffer<T>,
        key: &'a RocmBuffer<T>,
        value: &'a RocmBuffer<T>,
        keep: &'a RocmBuffer<T>,
        output: &'a RocmBuffer<T>,
        weights: &'a RocmBuffer<T>,
        scale: T,
        metadata: AttentionMeta,
        rows: usize,
    ) -> Self {
        Self {
            device,
            kernel,
            query,
            key,
            value,
            keep,
            output,
            weights,
            scale,
            metadata,
            rows,
        }
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        validate_device(self.device, device)?;
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut query = self.query.raw();
        let mut key = self.key.raw();
        let mut value = self.value.raw();
        let mut keep = self.keep.raw();
        let mut output = self.output.raw();
        let mut weights = self.weights.raw();
        let mut scale = self.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut query),
            argument(&mut key),
            argument(&mut value),
            argument(&mut keep),
            argument(&mut output),
            argument(&mut weights),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.rows, &mut args)
    }
}

pub(super) struct PreparedAttentionGradient<'a, T> {
    kernel: Option<Arc<RocmKernel>>,
    grad_output: &'a RocmBuffer<T>,
    query: &'a RocmBuffer<T>,
    key: &'a RocmBuffer<T>,
    value: &'a RocmBuffer<T>,
    weights: &'a RocmBuffer<T>,
    destination: &'a RocmBuffer<T>,
    scale: T,
    metadata: AttentionMeta,
    elements: usize,
}

impl<'a, T: Copy> PreparedAttentionGradient<'a, T> {
    #[expect(
        clippy::too_many_arguments,
        reason = "one additive gradient launch retains all validated borrowed operands"
    )]
    const fn new(
        kernel: Option<Arc<RocmKernel>>,
        grad_output: &'a RocmBuffer<T>,
        query: &'a RocmBuffer<T>,
        key: &'a RocmBuffer<T>,
        value: &'a RocmBuffer<T>,
        weights: &'a RocmBuffer<T>,
        destination: &'a RocmBuffer<T>,
        scale: T,
        metadata: AttentionMeta,
        elements: usize,
    ) -> Self {
        Self {
            kernel,
            grad_output,
            query,
            key,
            value,
            weights,
            destination,
            scale,
            metadata,
            elements,
        }
    }

    fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut grad_output = self.grad_output.raw();
        let mut query = self.query.raw();
        let mut key = self.key.raw();
        let mut value = self.value.raw();
        let mut weights = self.weights.raw();
        let mut destination = self.destination.raw();
        let mut scale = self.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut query),
            argument(&mut key),
            argument(&mut value),
            argument(&mut weights),
            argument(&mut destination),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args)
    }
}

/// Every selected ROCm additive attention gradient prepared as one unit.
pub struct PreparedAttentionBackward<'a, T> {
    device: &'a RocmDevice,
    query: Option<PreparedAttentionGradient<'a, T>>,
    key: Option<PreparedAttentionGradient<'a, T>>,
    value: Option<PreparedAttentionGradient<'a, T>>,
}

impl<'a, T: Copy> PreparedAttentionBackward<'a, T> {
    pub(super) const fn new(
        device: &'a RocmDevice,
        query: Option<PreparedAttentionGradient<'a, T>>,
        key: Option<PreparedAttentionGradient<'a, T>>,
        value: Option<PreparedAttentionGradient<'a, T>>,
    ) -> Self {
        Self {
            device,
            query,
            key,
            value,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "prepared gradient construction preserves the complete validated launch state"
    )]
    pub(super) const fn gradient(
        kernel: Option<Arc<RocmKernel>>,
        grad_output: &'a RocmBuffer<T>,
        query: &'a RocmBuffer<T>,
        key: &'a RocmBuffer<T>,
        value: &'a RocmBuffer<T>,
        weights: &'a RocmBuffer<T>,
        destination: &'a RocmBuffer<T>,
        scale: T,
        metadata: AttentionMeta,
        elements: usize,
    ) -> PreparedAttentionGradient<'a, T> {
        PreparedAttentionGradient::new(
            kernel,
            grad_output,
            query,
            key,
            value,
            weights,
            destination,
            scale,
            metadata,
            elements,
        )
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        validate_device(self.device, device)?;
        for gradient in [&self.query, &self.key, &self.value].into_iter().flatten() {
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
            message: "prepared ROCm attention belongs to a different device".to_string(),
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
