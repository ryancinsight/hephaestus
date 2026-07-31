use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{BlockWidth, HephaestusError, Result};

use super::kernel::GradientKernel;
use super::metadata::{BackwardMeta, ForwardMeta};
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
    source: String,
) -> Result<Arc<SafeCachedKernel>> {
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

/// Prepared CUDA attention forward with borrowed device operands.
pub struct PreparedAttentionForward<'a, T> {
    device: &'a CudaDevice,
    kernel: Option<Arc<SafeCachedKernel>>,
    query: &'a CudaBuffer<T>,
    key: &'a CudaBuffer<T>,
    value: &'a CudaBuffer<T>,
    keep: Option<&'a CudaBuffer<T>>,
    output: &'a CudaBuffer<T>,
    weights: &'a CudaBuffer<T>,
    scale: T,
    metadata: ForwardMeta,
    rows: usize,
}

impl<'a, T: Copy> PreparedAttentionForward<'a, T> {
    pub(super) fn new(
        device: &'a CudaDevice,
        kernel: Option<Arc<SafeCachedKernel>>,
        operands: hephaestus_core::AttentionForwardOperands<'a, CudaBuffer<T>, T>,
        metadata: ForwardMeta,
        rows: usize,
    ) -> Self {
        Self {
            device,
            kernel,
            query: operands.query.buffer,
            key: operands.key.buffer,
            value: operands.value.buffer,
            keep: match operands.mask.grouped_keep() {
                Some(keep) => Some(keep.view().buffer),
                None => None,
            },
            output: operands.output.buffer,
            weights: operands.weights.buffer,
            scale: operands.scale,
            metadata,
            rows,
        }
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut query = self.query.raw();
        let mut key = self.key.raw();
        let mut value = self.value.raw();
        let mut keep = self.keep.map_or(0, CudaBuffer::raw);
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

struct PreparedGradient<'a, T> {
    kind: GradientKernel,
    kernel: Option<Arc<SafeCachedKernel>>,
    target: &'a CudaBuffer<T>,
    metadata: BackwardMeta,
    elements: usize,
}

/// Every selected CUDA additive-attention kernel prepared as one unit.
pub struct PreparedAttentionBackward<'a, T> {
    device: &'a CudaDevice,
    score_kernel: Option<Arc<SafeCachedKernel>>,
    score_gradient: CudaBuffer<T>,
    grad_output: &'a CudaBuffer<T>,
    query: &'a CudaBuffer<T>,
    key: &'a CudaBuffer<T>,
    value: &'a CudaBuffer<T>,
    weights: &'a CudaBuffer<T>,
    scale: T,
    score_metadata: BackwardMeta,
    score_rows: usize,
    gradients: [Option<PreparedGradient<'a, T>>; 3],
}

pub(super) struct PreparedGradientSpec<'a, T> {
    pub(super) kind: GradientKernel,
    pub(super) kernel: Option<Arc<SafeCachedKernel>>,
    pub(super) target: &'a CudaBuffer<T>,
    pub(super) metadata: BackwardMeta,
    pub(super) elements: usize,
}

impl<'a, T: Copy> PreparedAttentionBackward<'a, T> {
    pub(super) fn new(
        device: &'a CudaDevice,
        score_kernel: Option<Arc<SafeCachedKernel>>,
        score_gradient: CudaBuffer<T>,
        operands: hephaestus_core::AttentionBackwardOperands<'a, CudaBuffer<T>, T>,
        score_metadata: BackwardMeta,
        score_rows: usize,
        specs: [Option<PreparedGradientSpec<'a, T>>; 3],
    ) -> Self {
        Self {
            device,
            score_kernel,
            score_gradient,
            grad_output: operands.grad_output.buffer,
            query: operands.query.buffer,
            key: operands.key.buffer,
            value: operands.value.buffer,
            weights: operands.weights.buffer,
            scale: operands.scale,
            score_metadata,
            score_rows,
            gradients: specs.map(|spec| {
                spec.map(|spec| PreparedGradient {
                    kind: spec.kind,
                    kernel: spec.kernel,
                    target: spec.target,
                    metadata: spec.metadata,
                    elements: spec.elements,
                })
            }),
        }
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        if let Some(kernel) = self.score_kernel.as_ref() {
            self.dispatch_score(device, kernel)?;
        }
        for gradient in self.gradients.iter().flatten() {
            gradient.dispatch(device, self)?;
        }
        Ok(())
    }

    fn dispatch_score(&self, device: &CudaDevice, kernel: &SafeCachedKernel) -> Result<()> {
        let mut grad_output = self.grad_output.raw();
        let mut value = self.value.raw();
        let mut weights = self.weights.raw();
        let mut score_gradient = self.score_gradient.raw();
        let mut metadata = self.score_metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut value),
            argument(&mut weights),
            argument(&mut score_gradient),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.score_rows, &mut args)
    }
}

impl<T: Copy> PreparedGradient<'_, T> {
    fn dispatch(
        &self,
        device: &CudaDevice,
        backward: &PreparedAttentionBackward<'_, T>,
    ) -> Result<()> {
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut grad_output = backward.grad_output.raw();
        let mut query = backward.query.raw();
        let mut key = backward.key.raw();
        let mut weights = backward.weights.raw();
        let mut score_gradient = backward.score_gradient.raw();
        let mut target = self.target.raw();
        let mut scale = backward.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut query),
            argument(&mut key),
            argument(&mut weights),
            argument(&mut score_gradient),
            argument(&mut target),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("CUDA {:?} attention gradient failed: {error}", self.kind),
            }
        })
    }
}

fn validate_device(owner: &CudaDevice, requested: &CudaDevice) -> Result<()> {
    if same_device(owner, requested) {
        Ok(())
    } else {
        Err(HephaestusError::DispatchFailed {
            message: "prepared CUDA attention belongs to a different device".to_string(),
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
