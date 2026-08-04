use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

use super::metadata::{BackwardMeta, ForwardMeta};
use super::resources::same_device;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use hephaestus_core::CrossEntropyStatus;

const BLOCK_WIDTH: BlockWidth = BlockWidth::DEFAULT;

pub(super) fn compile(
    device: &CudaDevice,
    entry: &'static str,
    source: impl FnOnce() -> String,
) -> Result<Arc<SafeCachedKernel>> {
    cached_kernel(
        device,
        PipelineKey::CrossEntropy {
            entry,
            scalar: TypeId::of::<f32>(),
        },
        entry,
        source,
    )
}

/// Prepared CUDA mean cross-entropy forward resources.
pub struct PreparedCrossEntropyForward<'a> {
    device: &'a CudaDevice,
    preflight_kernel: Arc<SafeCachedKernel>,
    forward_kernel: Arc<SafeCachedKernel>,
    mean_kernel: Arc<SafeCachedKernel>,
    status: CudaBuffer<u32>,
    row_losses: CudaBuffer<f32>,
    logits: &'a CudaBuffer<f32>,
    targets: &'a CudaBuffer<u32>,
    loss: &'a CudaBuffer<f32>,
    probabilities: &'a CudaBuffer<f32>,
    metadata: ForwardMeta,
    batch: usize,
}

pub(super) struct PreparedForwardSpec {
    pub(super) preflight_kernel: Arc<SafeCachedKernel>,
    pub(super) forward_kernel: Arc<SafeCachedKernel>,
    pub(super) mean_kernel: Arc<SafeCachedKernel>,
    pub(super) status: CudaBuffer<u32>,
    pub(super) row_losses: CudaBuffer<f32>,
    pub(super) metadata: ForwardMeta,
    pub(super) batch: usize,
}

impl<'a> PreparedCrossEntropyForward<'a> {
    pub(super) fn new(
        device: &'a CudaDevice,
        operands: hephaestus_core::CrossEntropyForwardOperands<
            'a,
            CudaBuffer<f32>,
            CudaBuffer<u32>,
        >,
        spec: PreparedForwardSpec,
    ) -> Self {
        Self {
            device,
            preflight_kernel: spec.preflight_kernel,
            forward_kernel: spec.forward_kernel,
            mean_kernel: spec.mean_kernel,
            status: spec.status,
            row_losses: spec.row_losses,
            logits: operands.logits.buffer,
            targets: operands.targets.buffer,
            loss: operands.loss.buffer,
            probabilities: operands.probabilities.buffer,
            metadata: spec.metadata,
            batch: spec.batch,
        }
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        reset_status(device, &self.status)?;
        self.dispatch_preflight(device)?;
        check_status(device, &self.status)?;
        self.dispatch_forward(device)?;
        self.dispatch_mean(device)
    }

    fn dispatch_preflight(&self, device: &CudaDevice) -> Result<()> {
        let mut logits = self.logits.raw();
        let mut targets = self.targets.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut logits),
            argument(&mut targets),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(device, &self.preflight_kernel, self.batch, &mut args)
    }

    fn dispatch_forward(&self, device: &CudaDevice) -> Result<()> {
        let mut logits = self.logits.raw();
        let mut targets = self.targets.raw();
        let mut probabilities = self.probabilities.raw();
        let mut row_losses = self.row_losses.raw();
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut logits),
            argument(&mut targets),
            argument(&mut probabilities),
            argument(&mut row_losses),
            argument(&mut metadata),
        ];
        launch(device, &self.forward_kernel, self.batch, &mut args)
    }

    fn dispatch_mean(&self, device: &CudaDevice) -> Result<()> {
        let mut row_losses = self.row_losses.raw();
        let mut loss = self.loss.raw();
        let mut batch = i64::try_from(self.batch).map_err(|_| HephaestusError::DispatchFailed {
            message: "CUDA cross-entropy batch exceeds i64 during mean launch".to_string(),
        })?;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut row_losses),
            argument(&mut loss),
            argument(&mut batch),
            argument(&mut metadata),
        ];
        launch(device, &self.mean_kernel, 1, &mut args)
    }
}

/// Prepared CUDA additive mean cross-entropy backward resources.
pub struct PreparedCrossEntropyBackward<'a> {
    device: &'a CudaDevice,
    preflight_kernel: Arc<SafeCachedKernel>,
    kernel: Arc<SafeCachedKernel>,
    status: CudaBuffer<u32>,
    output_gradient: &'a CudaBuffer<f32>,
    probabilities: &'a CudaBuffer<f32>,
    targets: &'a CudaBuffer<u32>,
    logit_gradient: &'a CudaBuffer<f32>,
    metadata: BackwardMeta,
    batch: usize,
    elements: usize,
}

pub(super) struct PreparedBackwardSpec {
    pub(super) preflight_kernel: Arc<SafeCachedKernel>,
    pub(super) kernel: Arc<SafeCachedKernel>,
    pub(super) status: CudaBuffer<u32>,
    pub(super) metadata: BackwardMeta,
    pub(super) batch: usize,
    pub(super) elements: usize,
}

impl<'a> PreparedCrossEntropyBackward<'a> {
    pub(super) fn new(
        device: &'a CudaDevice,
        operands: hephaestus_core::CrossEntropyBackwardOperands<
            'a,
            CudaBuffer<f32>,
            CudaBuffer<u32>,
        >,
        spec: PreparedBackwardSpec,
    ) -> Self {
        Self {
            device,
            preflight_kernel: spec.preflight_kernel,
            kernel: spec.kernel,
            status: spec.status,
            output_gradient: operands.output_gradient.buffer,
            probabilities: operands.probabilities.buffer,
            targets: operands.targets.buffer,
            logit_gradient: operands.logit_gradient.buffer,
            metadata: spec.metadata,
            batch: spec.batch,
            elements: spec.elements,
        }
    }

    pub(super) fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        validate_device(self.device, device)?;
        reset_status(device, &self.status)?;
        self.dispatch_preflight(device)?;
        check_status(device, &self.status)?;
        self.dispatch_backward(device)
    }

    fn dispatch_preflight(&self, device: &CudaDevice) -> Result<()> {
        let mut output_gradient = self.output_gradient.raw();
        let mut probabilities = self.probabilities.raw();
        let mut targets = self.targets.raw();
        let mut logit_gradient = self.logit_gradient.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut output_gradient),
            argument(&mut probabilities),
            argument(&mut targets),
            argument(&mut logit_gradient),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(device, &self.preflight_kernel, self.batch, &mut args)
    }

    fn dispatch_backward(&self, device: &CudaDevice) -> Result<()> {
        let mut output_gradient = self.output_gradient.raw();
        let mut probabilities = self.probabilities.raw();
        let mut targets = self.targets.raw();
        let mut logit_gradient = self.logit_gradient.raw();
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut output_gradient),
            argument(&mut probabilities),
            argument(&mut targets),
            argument(&mut logit_gradient),
            argument(&mut metadata),
        ];
        launch(device, &self.kernel, self.elements, &mut args)
    }
}

fn reset_status(device: &CudaDevice, status: &CudaBuffer<u32>) -> Result<()> {
    device.write_sub_buffer(status, 0, &[u32::MAX])
}

fn check_status(device: &CudaDevice, status: &CudaBuffer<u32>) -> Result<()> {
    let mut code = [u32::MAX];
    device.download(status, &mut code)?;
    CrossEntropyStatus::check(if code[0] == u32::MAX { 0 } else { code[0] })
}

fn validate_device(owner: &CudaDevice, requested: &CudaDevice) -> Result<()> {
    if same_device(owner, requested) {
        Ok(())
    } else {
        Err(HephaestusError::DispatchFailed {
            message: "prepared CUDA cross-entropy belongs to a different device".to_string(),
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
