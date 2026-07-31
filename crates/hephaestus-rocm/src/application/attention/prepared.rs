use std::any::TypeId;
use std::sync::Arc;

use hephaestus_core::{
    AttentionBackwardOperands, AttentionSemanticStatus, BlockWidth, ComputeDevice, Result,
};

use super::kernel::GradientTarget;
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

pub(super) struct PreparedFinite<'a, T> {
    kernel: Option<Arc<RocmKernel>>,
    source: &'a RocmBuffer<T>,
    metadata: AttentionMeta,
    elements: usize,
}

impl<'a, T> PreparedFinite<'a, T> {
    pub(super) const fn new(
        kernel: Option<Arc<RocmKernel>>,
        source: &'a RocmBuffer<T>,
        metadata: AttentionMeta,
        elements: usize,
    ) -> Self {
        Self {
            kernel,
            source,
            metadata,
            elements,
        }
    }

    fn dispatch(&self, device: &RocmDevice, status: &RocmBuffer<u32>) -> Result<()> {
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut source = self.source.raw();
        let mut status = status.raw();
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut source),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args)
    }
}

/// Prepared ROCm attention forward pass with parallel semantic preflight.
pub struct PreparedAttentionForward<'a, T> {
    device: &'a RocmDevice,
    finite: [PreparedFinite<'a, T>; 4],
    arithmetic_kernel: Option<Arc<RocmKernel>>,
    kernel: Option<Arc<RocmKernel>>,
    status: RocmBuffer<u32>,
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
        reason = "prepared attention retains every compiled stage and validated launch operand"
    )]
    pub(super) const fn new(
        device: &'a RocmDevice,
        finite: [PreparedFinite<'a, T>; 4],
        arithmetic_kernel: Option<Arc<RocmKernel>>,
        kernel: Option<Arc<RocmKernel>>,
        status: RocmBuffer<u32>,
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
            finite,
            arithmetic_kernel,
            kernel,
            status,
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
        reset_status(device, &self.status)?;
        if self.kernel.is_none() {
            return Ok(());
        }
        for finite in &self.finite {
            finite.dispatch(device, &self.status)?;
        }
        if let Some(kernel) = self.arithmetic_kernel.as_ref() {
            let mut query = self.query.raw();
            let mut key = self.key.raw();
            let mut value = self.value.raw();
            let mut keep = self.keep.raw();
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
            launch(device, kernel, self.rows, &mut args)?;
        }
        check_status(device, &self.status)?;

        let kernel = self
            .kernel
            .as_ref()
            .expect("invariant: nonempty forward retains a mutation kernel");
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
    target: GradientTarget,
    preflight_kernel: Option<Arc<RocmKernel>>,
    kernel: Option<Arc<RocmKernel>>,
    source: &'a RocmBuffer<T>,
    destination: &'a RocmBuffer<T>,
    metadata: AttentionMeta,
    elements: usize,
}

impl<'a, T: Copy> PreparedAttentionGradient<'a, T> {
    pub(super) const fn new(
        target: GradientTarget,
        preflight_kernel: Option<Arc<RocmKernel>>,
        kernel: Option<Arc<RocmKernel>>,
        source: &'a RocmBuffer<T>,
        destination: &'a RocmBuffer<T>,
        metadata: AttentionMeta,
        elements: usize,
    ) -> Self {
        Self {
            target,
            preflight_kernel,
            kernel,
            source,
            destination,
            metadata,
            elements,
        }
    }

    fn preflight(
        &self,
        device: &RocmDevice,
        backward: &PreparedAttentionBackward<'_, T>,
    ) -> Result<()> {
        let Some(kernel) = self.preflight_kernel.as_ref() else {
            return Ok(());
        };
        let mut grad_output = backward.grad_output.raw();
        let mut weights = backward.weights.raw();
        let mut score_gradient = backward
            .score_gradient
            .as_ref()
            .map_or_else(|| backward.weights.raw(), RocmBuffer::raw);
        let mut source = self.source.raw();
        let mut destination = self.destination.raw();
        let mut status = backward.status.raw();
        let mut scale = backward.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut weights),
            argument(&mut score_gradient),
            argument(&mut source),
            argument(&mut destination),
            argument(&mut status),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args)
    }

    fn dispatch(
        &self,
        device: &RocmDevice,
        backward: &PreparedAttentionBackward<'_, T>,
    ) -> Result<()> {
        let Some(kernel) = self.kernel.as_ref() else {
            return Ok(());
        };
        let mut grad_output = backward.grad_output.raw();
        let mut query = backward.query.raw();
        let mut key = backward.key.raw();
        let mut weights = backward.weights.raw();
        let mut score_gradient = backward
            .score_gradient
            .as_ref()
            .map_or_else(|| backward.weights.raw(), RocmBuffer::raw);
        let mut destination = self.destination.raw();
        let mut scale = backward.scale;
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut grad_output),
            argument(&mut query),
            argument(&mut key),
            argument(&mut weights),
            argument(&mut score_gradient),
            argument(&mut destination),
            argument(&mut scale),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.elements, &mut args).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("ROCm {:?} attention gradient failed: {error}", self.target),
            }
        })
    }
}

/// Every selected ROCm additive attention gradient prepared as one unit.
pub struct PreparedAttentionBackward<'a, T> {
    device: &'a RocmDevice,
    finite: [PreparedFinite<'a, T>; 5],
    probability_kernel: Option<Arc<RocmKernel>>,
    candidate_kernel: Option<Arc<RocmKernel>>,
    score_kernel: Option<Arc<RocmKernel>>,
    status: RocmBuffer<u32>,
    candidate: Option<RocmBuffer<T>>,
    score_gradient: Option<RocmBuffer<T>>,
    grad_output: &'a RocmBuffer<T>,
    query: &'a RocmBuffer<T>,
    key: &'a RocmBuffer<T>,
    value: &'a RocmBuffer<T>,
    weights: &'a RocmBuffer<T>,
    scale: T,
    metadata: AttentionMeta,
    rows: usize,
    score_elements: usize,
    gradients: [Option<PreparedAttentionGradient<'a, T>>; 3],
}

impl<'a, T: Copy> PreparedAttentionBackward<'a, T> {
    #[expect(
        clippy::too_many_arguments,
        reason = "prepared backward owns the complete preflight DAG and borrowed operands"
    )]
    pub(super) const fn new(
        device: &'a RocmDevice,
        finite: [PreparedFinite<'a, T>; 5],
        probability_kernel: Option<Arc<RocmKernel>>,
        candidate_kernel: Option<Arc<RocmKernel>>,
        score_kernel: Option<Arc<RocmKernel>>,
        status: RocmBuffer<u32>,
        candidate: Option<RocmBuffer<T>>,
        score_gradient: Option<RocmBuffer<T>>,
        operands: &AttentionBackwardOperands<'a, RocmBuffer<T>, T>,
        metadata: AttentionMeta,
        rows: usize,
        score_elements: usize,
        gradients: [Option<PreparedAttentionGradient<'a, T>>; 3],
    ) -> Self {
        Self {
            device,
            finite,
            probability_kernel,
            candidate_kernel,
            score_kernel,
            status,
            candidate,
            score_gradient,
            grad_output: operands.grad_output.buffer,
            query: operands.query.buffer,
            key: operands.key.buffer,
            value: operands.value.buffer,
            weights: operands.weights.buffer,
            scale: operands.scale,
            metadata,
            rows,
            score_elements,
            gradients,
        }
    }

    pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        validate_device(self.device, device)?;
        reset_status(device, &self.status)?;
        for finite in &self.finite {
            finite.dispatch(device, &self.status)?;
        }
        self.dispatch_probability(device)?;
        self.dispatch_score_preflight(device)?;
        for gradient in self.gradients.iter().flatten() {
            gradient.preflight(device, self)?;
        }
        check_status(device, &self.status)?;
        for gradient in self.gradients.iter().flatten() {
            gradient.dispatch(device, self)?;
        }
        Ok(())
    }

    fn dispatch_probability(&self, device: &RocmDevice) -> Result<()> {
        let Some(kernel) = self.probability_kernel.as_ref() else {
            return Ok(());
        };
        let mut weights = self.weights.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        let mut args = [
            argument(&mut weights),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(device, kernel, self.rows, &mut args)
    }

    fn dispatch_score_preflight(&self, device: &RocmDevice) -> Result<()> {
        let (Some(candidate_kernel), Some(score_kernel), Some(candidate), Some(score_gradient)) = (
            self.candidate_kernel.as_ref(),
            self.score_kernel.as_ref(),
            self.candidate.as_ref(),
            self.score_gradient.as_ref(),
        ) else {
            return Ok(());
        };
        let mut grad_output = self.grad_output.raw();
        let mut value = self.value.raw();
        let mut candidate_ptr = candidate.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        let mut candidate_args = [
            argument(&mut grad_output),
            argument(&mut value),
            argument(&mut candidate_ptr),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(
            device,
            candidate_kernel,
            self.score_elements,
            &mut candidate_args,
        )?;

        let mut candidate_ptr = candidate.raw();
        let mut weights = self.weights.raw();
        let mut score_gradient_ptr = score_gradient.raw();
        let mut status = self.status.raw();
        let mut metadata = self.metadata;
        let mut score_args = [
            argument(&mut candidate_ptr),
            argument(&mut weights),
            argument(&mut score_gradient_ptr),
            argument(&mut status),
            argument(&mut metadata),
        ];
        launch(device, score_kernel, self.rows, &mut score_args)
    }
}

fn reset_status(device: &RocmDevice, status: &RocmBuffer<u32>) -> Result<()> {
    device.write_buffer(status, &[u32::MAX])
}

fn check_status(device: &RocmDevice, status: &RocmBuffer<u32>) -> Result<()> {
    let mut host = [u32::MAX];
    device.download(status, &mut host)?;
    let code = if host[0] == u32::MAX {
        AttentionSemanticStatus::Valid.code()
    } else {
        host[0]
    };
    AttentionSemanticStatus::check(code)
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

#[cfg(test)]
mod tests {
    #[test]
    fn repeated_dispatch_resets_sentinel_and_reads_only_status() {
        let source = include_str!("prepared.rs");

        assert!(source.contains("device.write_buffer(status, &[u32::MAX])"));
        assert!(source.contains("if host[0] == u32::MAX"));
        assert!(source.contains("device.download(status"));
        for operand in ["query", "value", "weights"] {
            assert!(!source.contains(&format!("download(self.{operand}")));
        }

        let forward = source
            .split("pub(super) fn dispatch(&self, device: &RocmDevice) -> Result<()> {")
            .nth(1)
            .expect("forward dispatch source");
        let reset = forward.find("reset_status").expect("status reset");
        let empty = forward.find("self.kernel.is_none()").expect("empty guard");
        assert!(reset < empty, "empty dispatch must reset semantic status");
    }
}
