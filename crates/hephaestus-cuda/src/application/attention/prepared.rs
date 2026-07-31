use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{
    AttentionSemanticStatus, BlockWidth, ComputeDevice, HephaestusError, Result,
};

use super::kernel::GradientKernel;
use super::metadata::{BackwardMeta, BackwardPreflightMeta, ForwardMeta};
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
    preflight_kernel: Arc<SafeCachedKernel>,
    kernel: Option<Arc<SafeCachedKernel>>,
    status: CudaBuffer<u32>,
    query: &'a CudaBuffer<T>,
    key: &'a CudaBuffer<T>,
    value: &'a CudaBuffer<T>,
    keep: Option<&'a CudaBuffer<T>>,
    output: &'a CudaBuffer<T>,
    weights: &'a CudaBuffer<T>,
    scale: T,
    metadata: ForwardMeta,
    preflight_elements: usize,
    rows: usize,
}

pub(super) struct PreparedForwardSpec {
    pub(super) preflight_kernel: Arc<SafeCachedKernel>,
    pub(super) kernel: Option<Arc<SafeCachedKernel>>,
    pub(super) status: CudaBuffer<u32>,
    pub(super) metadata: ForwardMeta,
    pub(super) preflight_elements: usize,
    pub(super) rows: usize,
}

impl<'a, T: Copy> PreparedAttentionForward<'a, T> {
    pub(super) fn new(
        device: &'a CudaDevice,
        operands: hephaestus_core::AttentionForwardOperands<'a, CudaBuffer<T>, T>,
        spec: PreparedForwardSpec,
    ) -> Self {
        Self {
            device,
            preflight_kernel: spec.preflight_kernel,
            kernel: spec.kernel,
            status: spec.status,
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
            metadata: spec.metadata,
            preflight_elements: spec.preflight_elements,
            rows: spec.rows,
        }
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        reset_status(device, &self.status)?;
        self.dispatch_preflight(device)?;
        check_status(device, &self.status)?;
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

    fn dispatch_preflight(&self, device: &CudaDevice) -> Result<()> {
        let mut query = self.query.raw();
        let mut key = self.key.raw();
        let mut value = self.value.raw();
        let mut keep = self.keep.map_or(0, CudaBuffer::raw);
        let mut status = self.status.raw();
        let mut scale = self.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut query),
            argument(&mut key),
            argument(&mut value),
            argument(&mut keep),
            argument(&mut status),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(
            device,
            &self.preflight_kernel,
            self.preflight_elements,
            &mut args,
        )
    }
}

struct PreparedGradient<'a, T> {
    kind: GradientKernel,
    preflight_kernel: Option<Arc<SafeCachedKernel>>,
    kernel: Option<Arc<SafeCachedKernel>>,
    target: &'a CudaBuffer<T>,
    metadata: BackwardMeta,
    elements: usize,
}

/// Every selected CUDA additive-attention kernel prepared as one unit.
pub struct PreparedAttentionBackward<'a, T> {
    device: &'a CudaDevice,
    validation_kernel: Arc<SafeCachedKernel>,
    score_kernel: Option<Arc<SafeCachedKernel>>,
    score_gradient: CudaBuffer<T>,
    status: CudaBuffer<u32>,
    grad_output: &'a CudaBuffer<T>,
    query: &'a CudaBuffer<T>,
    key: &'a CudaBuffer<T>,
    value: &'a CudaBuffer<T>,
    weights: &'a CudaBuffer<T>,
    query_gradient: Option<&'a CudaBuffer<T>>,
    key_gradient: Option<&'a CudaBuffer<T>>,
    value_gradient: Option<&'a CudaBuffer<T>>,
    scale: T,
    score_metadata: BackwardMeta,
    preflight_metadata: BackwardPreflightMeta,
    validation_elements: usize,
    score_rows: usize,
    gradients: [Option<PreparedGradient<'a, T>>; 3],
}

pub(super) struct PreparedGradientSpec<'a, T> {
    pub(super) kind: GradientKernel,
    pub(super) preflight_kernel: Option<Arc<SafeCachedKernel>>,
    pub(super) kernel: Option<Arc<SafeCachedKernel>>,
    pub(super) target: &'a CudaBuffer<T>,
    pub(super) metadata: BackwardMeta,
    pub(super) elements: usize,
}

pub(super) struct PreparedBackwardSpec<'a, T> {
    pub(super) validation_kernel: Arc<SafeCachedKernel>,
    pub(super) score_kernel: Option<Arc<SafeCachedKernel>>,
    pub(super) score_gradient: CudaBuffer<T>,
    pub(super) status: CudaBuffer<u32>,
    pub(super) score_metadata: BackwardMeta,
    pub(super) preflight_metadata: BackwardPreflightMeta,
    pub(super) validation_elements: usize,
    pub(super) score_rows: usize,
    pub(super) gradients: [Option<PreparedGradientSpec<'a, T>>; 3],
}

impl<'a, T: Copy> PreparedAttentionBackward<'a, T> {
    pub(super) fn new(
        device: &'a CudaDevice,
        operands: hephaestus_core::AttentionBackwardOperands<'a, CudaBuffer<T>, T>,
        spec: PreparedBackwardSpec<'a, T>,
    ) -> Self {
        Self {
            device,
            validation_kernel: spec.validation_kernel,
            score_kernel: spec.score_kernel,
            score_gradient: spec.score_gradient,
            status: spec.status,
            grad_output: operands.grad_output.buffer,
            query: operands.query.buffer,
            key: operands.key.buffer,
            value: operands.value.buffer,
            weights: operands.weights.buffer,
            query_gradient: operands.gradients.query.map(|gradient| gradient.buffer),
            key_gradient: operands.gradients.key.map(|gradient| gradient.buffer),
            value_gradient: operands.gradients.value.map(|gradient| gradient.buffer),
            scale: operands.scale,
            score_metadata: spec.score_metadata,
            preflight_metadata: spec.preflight_metadata,
            validation_elements: spec.validation_elements,
            score_rows: spec.score_rows,
            gradients: spec.gradients.map(|spec| {
                spec.map(|spec| PreparedGradient {
                    kind: spec.kind,
                    preflight_kernel: spec.preflight_kernel,
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
        reset_status(device, &self.status)?;
        self.dispatch_validation(device)?;
        if let Some(kernel) = self.score_kernel.as_ref() {
            self.dispatch_score(device, kernel)?;
        }
        for gradient in self.gradients.iter().flatten() {
            gradient.dispatch_preflight(device, self)?;
        }
        check_status(device, &self.status)?;
        for gradient in self.gradients.iter().flatten() {
            gradient.dispatch(device, self)?;
        }
        Ok(())
    }

    fn dispatch_validation(&self, device: &CudaDevice) -> Result<()> {
        let mut grad_output = self.grad_output.raw();
        let mut query = self.query.raw();
        let mut key = self.key.raw();
        let mut value = self.value.raw();
        let mut weights = self.weights.raw();
        let mut query_gradient = self.query_gradient.map_or(0, CudaBuffer::raw);
        let mut key_gradient = self.key_gradient.map_or(0, CudaBuffer::raw);
        let mut value_gradient = self.value_gradient.map_or(0, CudaBuffer::raw);
        let mut status = self.status.raw();
        let mut metadata = self.preflight_metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut query),
            argument(&mut key),
            argument(&mut value),
            argument(&mut weights),
            argument(&mut query_gradient),
            argument(&mut key_gradient),
            argument(&mut value_gradient),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(
            device,
            &self.validation_kernel,
            self.validation_elements,
            &mut args,
        )
    }

    fn dispatch_score(&self, device: &CudaDevice, kernel: &SafeCachedKernel) -> Result<()> {
        let mut grad_output = self.grad_output.raw();
        let mut value = self.value.raw();
        let mut weights = self.weights.raw();
        let mut score_gradient = self.score_gradient.raw();
        let mut status = self.status.raw();
        let mut metadata = self.score_metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut value),
            argument(&mut weights),
            argument(&mut score_gradient),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.score_rows, &mut args)
    }
}

impl<T: Copy> PreparedGradient<'_, T> {
    fn dispatch_preflight(
        &self,
        device: &CudaDevice,
        backward: &PreparedAttentionBackward<'_, T>,
    ) -> Result<()> {
        let Some(kernel) = self.preflight_kernel.as_ref() else {
            return Ok(());
        };
        let mut grad_output = backward.grad_output.raw();
        let mut query = backward.query.raw();
        let mut key = backward.key.raw();
        let mut weights = backward.weights.raw();
        let mut score_gradient = backward.score_gradient.raw();
        let mut target = self.target.raw();
        let mut status = backward.status.raw();
        let mut scale = backward.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut query),
            argument(&mut key),
            argument(&mut weights),
            argument(&mut score_gradient),
            argument(&mut target),
            argument(&mut status),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args)
    }

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

fn reset_status(device: &CudaDevice, status: &CudaBuffer<u32>) -> Result<()> {
    device.write_sub_buffer(status, 0, &[u32::MAX])
}

fn check_status(device: &CudaDevice, status: &CudaBuffer<u32>) -> Result<()> {
    let mut code = [u32::MAX];
    device.download(status, &mut code)?;
    let code = if code[0] == u32::MAX {
        AttentionSemanticStatus::Valid.code()
    } else {
        code[0]
    };
    AttentionSemanticStatus::check(code)
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
