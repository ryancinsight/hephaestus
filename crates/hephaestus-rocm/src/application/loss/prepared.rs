use std::sync::Arc;

use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

use super::metadata::CrossEntropyMeta;
use super::resources::same_device;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::{RocmBuffer, RocmDevice};
use hephaestus_core::CrossEntropyStatus;

const BLOCK_WIDTH: BlockWidth = BlockWidth::DEFAULT;

pub(super) fn compile(
    device: &RocmDevice,
    entry: &'static str,
    source: impl FnOnce() -> String,
) -> Result<Arc<RocmKernel>> {
    cached_kernel(device, PipelineKey::CrossEntropy { entry }, entry, source)
}

/// Prepared ROCm mean cross-entropy forward dispatch.
pub struct PreparedRocmCrossEntropyForward<'a> {
    device: &'a RocmDevice,
    preflight: Arc<RocmKernel>,
    forward: Arc<RocmKernel>,
    mean: Arc<RocmKernel>,
    status: RocmBuffer<u32>,
    row_losses: RocmBuffer<f32>,
    logits: &'a RocmBuffer<f32>,
    targets: &'a RocmBuffer<u32>,
    loss: &'a RocmBuffer<f32>,
    probabilities: &'a RocmBuffer<f32>,
    metadata: CrossEntropyMeta,
    rows: usize,
}

impl<'a> PreparedRocmCrossEntropyForward<'a> {
    #[expect(
        clippy::too_many_arguments,
        reason = "prepared dispatch retains compiled stages and every borrowed device operand"
    )]
    pub(super) const fn new(
        device: &'a RocmDevice,
        preflight: Arc<RocmKernel>,
        forward: Arc<RocmKernel>,
        mean: Arc<RocmKernel>,
        status: RocmBuffer<u32>,
        row_losses: RocmBuffer<f32>,
        logits: &'a RocmBuffer<f32>,
        targets: &'a RocmBuffer<u32>,
        loss: &'a RocmBuffer<f32>,
        probabilities: &'a RocmBuffer<f32>,
        metadata: CrossEntropyMeta,
        rows: usize,
    ) -> Self {
        Self {
            device,
            preflight,
            forward,
            mean,
            status,
            row_losses,
            logits,
            targets,
            loss,
            probabilities,
            metadata,
            rows,
        }
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        validate_device(self.device, device)?;
        reset_status(device, &self.status)?;

        let mut logits = self.logits.raw();
        let mut targets = self.targets.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        launch_rows(
            device,
            &self.preflight,
            self.rows,
            &mut [
                argument(&mut logits),
                argument(&mut targets),
                argument(&mut status),
                argument(&mut metadata),
            ],
        )?;
        check_status(device, &self.status)?;

        let mut logits = self.logits.raw();
        let mut targets = self.targets.raw();
        let mut probabilities = self.probabilities.raw();
        let mut row_losses = self.row_losses.raw();
        let mut metadata = self.metadata;
        launch_rows(
            device,
            &self.forward,
            self.rows,
            &mut [
                argument(&mut logits),
                argument(&mut targets),
                argument(&mut probabilities),
                argument(&mut row_losses),
                argument(&mut metadata),
            ],
        )?;

        let mut row_losses = self.row_losses.raw();
        let mut loss = self.loss.raw();
        let mut metadata = self.metadata;
        launch_rows(
            device,
            &self.mean,
            1,
            &mut [
                argument(&mut row_losses),
                argument(&mut loss),
                argument(&mut metadata),
            ],
        )
    }
}

/// Prepared ROCm additive mean cross-entropy backward dispatch.
pub struct PreparedRocmCrossEntropyBackward<'a> {
    device: &'a RocmDevice,
    preflight: Arc<RocmKernel>,
    backward: Arc<RocmKernel>,
    status: RocmBuffer<u32>,
    output_gradient: &'a RocmBuffer<f32>,
    probabilities: &'a RocmBuffer<f32>,
    targets: &'a RocmBuffer<u32>,
    logit_gradient: &'a RocmBuffer<f32>,
    metadata: CrossEntropyMeta,
    rows: usize,
    elements: usize,
}

impl<'a> PreparedRocmCrossEntropyBackward<'a> {
    #[expect(
        clippy::too_many_arguments,
        reason = "prepared dispatch retains compiled stages and every borrowed device operand"
    )]
    pub(super) const fn new(
        device: &'a RocmDevice,
        preflight: Arc<RocmKernel>,
        backward: Arc<RocmKernel>,
        status: RocmBuffer<u32>,
        output_gradient: &'a RocmBuffer<f32>,
        probabilities: &'a RocmBuffer<f32>,
        targets: &'a RocmBuffer<u32>,
        logit_gradient: &'a RocmBuffer<f32>,
        metadata: CrossEntropyMeta,
        rows: usize,
        elements: usize,
    ) -> Self {
        Self {
            device,
            preflight,
            backward,
            status,
            output_gradient,
            probabilities,
            targets,
            logit_gradient,
            metadata,
            rows,
            elements,
        }
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        validate_device(self.device, device)?;
        reset_status(device, &self.status)?;

        let mut output_gradient = self.output_gradient.raw();
        let mut probabilities = self.probabilities.raw();
        let mut targets = self.targets.raw();
        let mut logit_gradient = self.logit_gradient.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        launch_rows(
            device,
            &self.preflight,
            self.rows,
            &mut [
                argument(&mut output_gradient),
                argument(&mut probabilities),
                argument(&mut targets),
                argument(&mut logit_gradient),
                argument(&mut status),
                argument(&mut metadata),
            ],
        )?;
        check_status(device, &self.status)?;

        let mut output_gradient = self.output_gradient.raw();
        let mut probabilities = self.probabilities.raw();
        let mut targets = self.targets.raw();
        let mut logit_gradient = self.logit_gradient.raw();
        let mut metadata = self.metadata;
        launch_rows(
            device,
            &self.backward,
            self.elements,
            &mut [
                argument(&mut output_gradient),
                argument(&mut probabilities),
                argument(&mut targets),
                argument(&mut logit_gradient),
                argument(&mut metadata),
            ],
        )
    }
}

fn reset_status(device: &RocmDevice, status: &RocmBuffer<u32>) -> Result<()> {
    device.write_buffer(status, &[u32::MAX])
}

fn check_status(device: &RocmDevice, status: &RocmBuffer<u32>) -> Result<()> {
    let mut host = [u32::MAX];
    device.download(status, &mut host)?;
    let code = if host[0] == u32::MAX {
        CrossEntropyStatus::Valid.code()
    } else {
        host[0]
    };
    CrossEntropyStatus::check(code)
}

fn validate_device(owner: &RocmDevice, requested: &RocmDevice) -> Result<()> {
    if same_device(owner, requested) {
        Ok(())
    } else {
        Err(HephaestusError::DispatchFailed {
            message: "prepared ROCm cross-entropy belongs to a different device".to_string(),
        })
    }
}

fn launch_rows(
    device: &RocmDevice,
    kernel: &RocmKernel,
    work_items: usize,
    args: &mut [*mut core::ffi::c_void],
) -> Result<()> {
    let blocks = grid_size(work_items, BLOCK_WIDTH)?;
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

#[cfg(test)]
mod tests {
    use crate::application::loss::kernel::{
        BACKWARD_ENTRY, BACKWARD_PREFLIGHT_ENTRY, FORWARD_ENTRY, FORWARD_MEAN_ENTRY,
        FORWARD_PREFLIGHT_ENTRY,
    };

    #[test]
    fn every_kernel_entry_has_a_distinct_pipeline_identity() {
        let entries = [
            FORWARD_PREFLIGHT_ENTRY,
            FORWARD_ENTRY,
            FORWARD_MEAN_ENTRY,
            BACKWARD_PREFLIGHT_ENTRY,
            BACKWARD_ENTRY,
        ];
        for (index, entry) in entries.iter().enumerate() {
            assert!(!entries[..index].contains(entry));
        }
    }
}
