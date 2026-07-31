use core::any::TypeId;

use hephaestus_core::{
    AttentionBackwardOperands, AttentionCausality, AttentionForwardOperands, AttentionOps,
    AttentionPlan, ComputeDevice, Result, StridedView, plan_attention_backward,
    plan_attention_forward,
};

use super::metadata::AttentionMeta;
use super::preflight::{
    prepare_backward as prepare_backward_preflight, prepare_forward as prepare_forward_preflight,
};
use super::prepared::{
    PreparedAttentionBackward, PreparedAttentionForward, PreparedAttentionKernel,
};
use super::resources::{
    address_limit, backward_aliases, binding, forward_aliases, metadata_buffer, raw_binding,
    validate_backward_owners, validate_forward_owners,
};
use super::shader::{BackwardStage, ForwardStage, backward_shader, forward_shader};
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

pub(super) const WORKGROUP_WIDTH: hephaestus_core::BlockWidth =
    hephaestus_core::BlockWidth::DEFAULT;

struct ForwardWeights;
struct ForwardOutput;
struct BackwardQuery;
struct BackwardKey;
struct BackwardValue;

/// WGPU implementation of provider-owned scaled dot-product attention.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuAttentionOps;

impl AttentionOps<WgpuDevice, f32> for WgpuAttentionOps {
    type PreparedForward<'a>
        = PreparedAttentionForward
    where
        WgpuDevice: 'a,
        f32: 'a;
    type PreparedBackward<'a>
        = PreparedAttentionBackward
    where
        WgpuDevice: 'a,
        f32: 'a;

    fn prepare_attention_forward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: AttentionForwardOperands<'a, WgpuBuffer<f32>, f32>,
    ) -> Result<Self::PreparedForward<'a>> {
        validate_forward_owners(device, &operands)?;
        let plan = plan_attention_forward(&operands, forward_aliases(&operands))?;
        plan.validate_address_limit(address_limit())?;
        let (keep_layout, heads_per_batch) =
            operands.mask.grouped_keep().map_or((None, 1), |keep| {
                (Some(keep.view().layout), keep.heads_per_batch().get())
            });
        let metadata = AttentionMeta::new(
            plan,
            operands.query.layout,
            operands.key.layout,
            operands.value.layout,
            operands.weights.layout,
            None,
            operands.output.layout,
            keep_layout,
            heads_per_batch,
            operands.mask.causality() == AttentionCausality::Causal,
            operands.scale,
        )?;
        let mask_buffer = operands
            .mask
            .grouped_keep()
            .map_or(operands.query.buffer, |keep| keep.view().buffer);
        let rows = checked_product(plan.batch, plan.query_sequence, "forward row count")?;
        let preflight =
            prepare_forward_preflight(device, &operands, plan, &metadata, mask_buffer, rows)?;
        let weights = prepare_kernel::<ForwardWeights>(
            device,
            ForwardStage::Weights,
            &metadata,
            rows,
            "hephaestus-attention-forward-weights",
            &[
                binding(0, operands.query.buffer),
                binding(1, operands.key.buffer),
                binding(2, mask_buffer),
                binding(3, operands.weights.buffer),
            ],
        )?;
        let output_elements = checked_product(rows, plan.value_feature, "forward output count")?;
        let output = prepare_kernel::<ForwardOutput>(
            device,
            ForwardStage::Output,
            &metadata,
            output_elements,
            "hephaestus-attention-forward-output",
            &[
                binding(0, operands.weights.buffer),
                binding(1, operands.value.buffer),
                binding(2, operands.output.buffer),
            ],
        )?;
        Ok(PreparedAttentionForward::new(
            preflight.kernels,
            preflight.status,
            weights,
            output,
        ))
    }

    fn dispatch_attention_forward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_attention_backward<'a>(
        &self,
        device: &'a WgpuDevice,
        operands: AttentionBackwardOperands<'a, WgpuBuffer<f32>, f32>,
    ) -> Result<Self::PreparedBackward<'a>> {
        validate_backward_owners(device, &operands)?;
        let plan = plan_attention_backward(&operands, backward_aliases(&operands))?;
        plan.validate_address_limit(address_limit())?;
        let needs_score = operands.gradients.query.is_some() || operands.gradients.key.is_some();
        let score_workspace = needs_score
            .then(|| device.alloc_zeroed::<f32>(plan.score_elements))
            .transpose()?;
        let preflight =
            prepare_backward_preflight(device, &operands, plan, score_workspace.as_ref())?;
        let query = operands
            .gradients
            .query
            .map(|target| {
                prepare_gradient::<BackwardQuery>(
                    device,
                    &operands,
                    plan,
                    target,
                    score_workspace.as_ref(),
                    BackwardStage::Query,
                    "hephaestus-attention-backward-query",
                )
            })
            .transpose()?;
        let key = operands
            .gradients
            .key
            .map(|target| {
                prepare_gradient::<BackwardKey>(
                    device,
                    &operands,
                    plan,
                    target,
                    score_workspace.as_ref(),
                    BackwardStage::Key,
                    "hephaestus-attention-backward-key",
                )
            })
            .transpose()?;
        let value = operands
            .gradients
            .value
            .map(|target| prepare_value(device, &operands, plan, target))
            .transpose()?;
        Ok(PreparedAttentionBackward::new(
            preflight.kernels,
            preflight.status,
            None,
            query,
            key,
            value,
            score_workspace,
        ))
    }

    fn dispatch_attention_backward(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn prepare_gradient<K: 'static>(
    device: &WgpuDevice,
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<f32>, f32>,
    plan: AttentionPlan,
    target: StridedView<'_, WgpuBuffer<f32>, 3>,
    workspace: Option<&WgpuBuffer<f32>>,
    stage: BackwardStage,
    label: &'static str,
) -> Result<PreparedAttentionKernel> {
    let workspace = workspace.expect("invariant: query and key gradients prepare score workspace");
    let metadata = backward_metadata(operands, plan, target)?;
    let elements = target.layout.checked_size().map_err(|error| {
        super::resources::invalid(format!("attention gradient layout rejected: {error}"))
    })?;
    let source = match stage {
        BackwardStage::Query => operands.key.buffer,
        BackwardStage::Key => operands.query.buffer,
        BackwardStage::Score | BackwardStage::Value => {
            return Err(super::resources::invalid(
                "attention gradient stage does not use score workspace",
            ));
        }
    };
    prepare_backward_kernel::<K>(
        device,
        stage,
        &metadata,
        elements,
        label,
        &[
            binding(0, workspace),
            binding(1, source),
            binding(2, target.buffer),
        ],
    )
}

fn prepare_value(
    device: &WgpuDevice,
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<f32>, f32>,
    plan: AttentionPlan,
    target: StridedView<'_, WgpuBuffer<f32>, 3>,
) -> Result<PreparedAttentionKernel> {
    let metadata = backward_metadata(operands, plan, target)?;
    let elements = target.layout.checked_size().map_err(|error| {
        super::resources::invalid(format!("attention value-gradient layout rejected: {error}"))
    })?;
    prepare_backward_kernel::<BackwardValue>(
        device,
        BackwardStage::Value,
        &metadata,
        elements,
        "hephaestus-attention-backward-value",
        &[
            binding(0, operands.weights.buffer),
            binding(1, operands.grad_output.buffer),
            binding(2, target.buffer),
        ],
    )
}

pub(super) fn backward_metadata(
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<f32>, f32>,
    plan: AttentionPlan,
    destination: StridedView<'_, WgpuBuffer<f32>, 3>,
) -> Result<AttentionMeta> {
    AttentionMeta::new(
        plan,
        operands.query.layout,
        operands.key.layout,
        operands.value.layout,
        operands.weights.layout,
        Some(operands.grad_output.layout),
        destination.layout,
        None,
        1,
        false,
        operands.scale,
    )
}

fn prepare_kernel<K: 'static>(
    device: &WgpuDevice,
    stage: ForwardStage,
    metadata: &AttentionMeta,
    elements: usize,
    label: &'static str,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
) -> Result<PreparedAttentionKernel> {
    prepare(
        device,
        metadata,
        elements,
        label,
        storage_entries,
        || forward_shader(stage, WORKGROUP_WIDTH.get()),
        TypeId::of::<K>(),
    )
}

fn prepare_backward_kernel<K: 'static>(
    device: &WgpuDevice,
    stage: BackwardStage,
    metadata: &AttentionMeta,
    elements: usize,
    label: &'static str,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
) -> Result<PreparedAttentionKernel> {
    prepare(
        device,
        metadata,
        elements,
        label,
        storage_entries,
        || backward_shader(stage, WORKGROUP_WIDTH.get()),
        TypeId::of::<K>(),
    )
}

pub(super) fn prepare(
    device: &WgpuDevice,
    metadata: &AttentionMeta,
    elements: usize,
    label: &'static str,
    storage_entries: &[wgpu::BindGroupEntry<'_>],
    shader: impl FnOnce() -> String,
    kernel: TypeId,
) -> Result<PreparedAttentionKernel> {
    if elements == 0 {
        return Ok(PreparedAttentionKernel::empty(device));
    }
    let pipeline = try_cached_pipeline(
        device,
        (kernel, TypeId::of::<f32>(), WORKGROUP_WIDTH.get()),
        label,
        shader,
    )?;
    let metadata_buffer = metadata_buffer(device, metadata)?;
    let mut entries: smallvec::SmallVec<[wgpu::BindGroupEntry<'_>; 5]> =
        storage_entries.iter().cloned().collect();
    entries.push(raw_binding(
        u32::try_from(storage_entries.len()).expect("invariant: attention binding count fits u32"),
        &metadata_buffer,
    ));
    let bind_group = checked_bind_group(device, &pipeline, label, &entries)?;
    drop(entries);
    Ok(PreparedAttentionKernel::ready(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(elements, WORKGROUP_WIDTH)?,
        label,
    ))
}

pub(super) fn checked_product(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        super::resources::invalid(format!("attention {name} overflows: {left} * {right}"))
    })
}
