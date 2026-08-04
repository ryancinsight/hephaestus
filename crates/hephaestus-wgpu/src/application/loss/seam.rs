use core::any::TypeId;

use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    Result, plan_cross_entropy_backward, plan_cross_entropy_forward,
};

use super::metadata::CrossEntropyMeta;
use super::prepared::{
    PreparedCrossEntropyBackward, PreparedCrossEntropyForward, PreparedCrossEntropyKernel,
};
use super::resources::{
    address_limit, backward_aliases, binding, forward_aliases, metadata_buffer, raw_binding,
    validate_backward_owners, validate_forward_owners,
};
use super::shader::{
    BACKWARD_ACCUMULATE, BACKWARD_ARITHMETIC, BACKWARD_ROWS, FORWARD_MEAN, FORWARD_PREFLIGHT,
    FORWARD_ROWS, shader,
};
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP_WIDTH: hephaestus_core::BlockWidth = hephaestus_core::BlockWidth::DEFAULT;

struct ForwardPreflight;
struct ForwardRows;
struct ForwardMean;
struct BackwardRows;
struct BackwardArithmetic;
struct BackwardAccumulate;

/// WGPU implementation of provider-owned mean cross-entropy.
///
/// Runtime adapter selection makes this implementation serve native Vulkan,
/// DirectX, browser WebGPU, and Metal devices without a parallel Metal path.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCrossEntropyOps;

impl CrossEntropyOps<WgpuDevice, f32> for WgpuCrossEntropyOps {
    type PreparedForward<'a>
        = PreparedCrossEntropyForward
    where
        WgpuDevice: 'a,
        f32: 'a;
    type PreparedBackward<'a>
        = PreparedCrossEntropyBackward
    where
        WgpuDevice: 'a,
        f32: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: CrossEntropyForwardOperands<'a, WgpuBuffer<f32>, WgpuBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        validate_forward_owners(device, &operands)?;
        let plan = plan_cross_entropy_forward(&operands, forward_aliases(&operands))?;
        plan.validate_address_limit(address_limit())?;
        let metadata = CrossEntropyMeta::forward(
            plan,
            operands.logits.layout,
            operands.targets.layout,
            operands.loss.layout,
            operands.probabilities.layout,
        )?;
        let status = device.alloc_zeroed::<u32>(1)?;
        let row_losses = device.alloc_zeroed::<f32>(plan.batch)?;
        let preflight = prepare::<ForwardPreflight>(
            device,
            &metadata,
            plan.batch,
            FORWARD_PREFLIGHT,
            "hephaestus-cross-entropy-forward-preflight",
            &[
                binding(0, operands.logits.buffer),
                binding(1, operands.targets.buffer),
                binding(2, &status),
            ],
        )?;
        let probabilities = prepare::<ForwardRows>(
            device,
            &metadata,
            plan.batch,
            FORWARD_ROWS,
            "hephaestus-cross-entropy-forward-rows",
            &[
                binding(0, operands.logits.buffer),
                binding(1, operands.targets.buffer),
                binding(2, operands.probabilities.buffer),
                binding(3, &row_losses),
            ],
        )?;
        let mean = prepare::<ForwardMean>(
            device,
            &metadata,
            1,
            FORWARD_MEAN,
            "hephaestus-cross-entropy-forward-mean",
            &[binding(0, &row_losses), binding(1, operands.loss.buffer)],
        )?;
        Ok(PreparedCrossEntropyForward::new(
            preflight,
            status,
            probabilities,
            mean,
            row_losses,
        ))
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: CrossEntropyBackwardOperands<'a, WgpuBuffer<f32>, WgpuBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        validate_backward_owners(device, &operands)?;
        let plan = plan_cross_entropy_backward(&operands, backward_aliases(&operands))?;
        plan.validate_address_limit(address_limit())?;
        let metadata = CrossEntropyMeta::backward(
            plan,
            operands.output_gradient.layout,
            operands.probabilities.layout,
            operands.targets.layout,
            operands.logit_gradient.layout,
        )?;
        let status = device.alloc_zeroed::<u32>(2)?;
        let row_preflight = prepare::<BackwardRows>(
            device,
            &metadata,
            plan.batch,
            BACKWARD_ROWS,
            "hephaestus-cross-entropy-backward-row-preflight",
            &[
                binding(0, operands.output_gradient.buffer),
                binding(1, operands.probabilities.buffer),
                binding(2, operands.targets.buffer),
                binding(3, &status),
            ],
        )?;
        let arithmetic_preflight = prepare::<BackwardArithmetic>(
            device,
            &metadata,
            plan.elements,
            BACKWARD_ARITHMETIC,
            "hephaestus-cross-entropy-backward-arithmetic-preflight",
            &[
                binding(0, operands.probabilities.buffer),
                binding(1, operands.targets.buffer),
                binding(2, operands.logit_gradient.buffer),
                binding(3, &status),
            ],
        )?;
        let backward = prepare::<BackwardAccumulate>(
            device,
            &metadata,
            plan.elements,
            BACKWARD_ACCUMULATE,
            "hephaestus-cross-entropy-backward-accumulate",
            &[
                binding(0, operands.output_gradient.buffer),
                binding(1, operands.probabilities.buffer),
                binding(2, operands.targets.buffer),
                binding(3, operands.logit_gradient.buffer),
            ],
        )?;
        Ok(PreparedCrossEntropyBackward::new(
            row_preflight,
            arithmetic_preflight,
            status,
            backward,
        ))
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn prepare<K: 'static>(
    device: &WgpuDevice,
    metadata: &CrossEntropyMeta,
    elements: usize,
    stage: u8,
    label: &'static str,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
) -> Result<PreparedCrossEntropyKernel> {
    let pipeline = try_cached_pipeline(
        device,
        (
            TypeId::of::<K>(),
            TypeId::of::<f32>(),
            WORKGROUP_WIDTH.get(),
        ),
        label,
        || shader(stage, WORKGROUP_WIDTH.get()),
    )?;
    let metadata_buffer = metadata_buffer(device, metadata)?;
    let mut entries: smallvec::SmallVec<[wgpu::BindGroupEntry<'_>; 6]> =
        storage_entries.iter().cloned().collect();
    entries.push(raw_binding(
        u32::try_from(storage_entries.len())
            .expect("invariant: cross-entropy binding count fits u32"),
        &metadata_buffer,
    ));
    let bind_group = checked_bind_group(device, &pipeline, label, &entries)?;
    drop(entries);
    Ok(PreparedCrossEntropyKernel::new(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(elements, WORKGROUP_WIDTH)?,
        label,
    ))
}
