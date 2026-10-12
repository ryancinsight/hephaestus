use core::any::TypeId;

use hephaestus_core::{
    AttentionBackwardOperands, AttentionPlan, AttentionSemanticStatus, ComputeDevice, Result,
    StridedView,
};

use super::super::WgslAttentionScalar;
use super::super::prepared::PreparedAttentionKernel;
use super::super::resources::binding;
use super::super::seam::{WORKGROUP_WIDTH, backward_metadata, checked_product, prepare};
use super::super::shader::{
    BackwardStage, GradientPreflightStage, backward_gradient_preflight_shader,
    backward_probability_preflight_shader, backward_shader, linear_finite_preflight_shader,
};
use super::kernel::{prepare_finite, prepare_status};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

struct GradOutputFinite;
struct QueryFinite;
struct KeyFinite;
struct ValueFinite;
struct WeightsFinite;
struct Probability;
struct ScoreWorkspace;
struct ScoreFinite;
struct QueryGradient;
struct KeyGradient;
struct ValueGradient;

pub(in crate::application::attention) struct BackwardPreflight {
    pub(in crate::application::attention) kernels:
        smallvec::SmallVec<[PreparedAttentionKernel; 12]>,
    pub(in crate::application::attention) status: WgpuBuffer<u32>,
}

pub(in crate::application::attention) fn prepare_backward<T: WgslAttentionScalar>(
    device: &WgpuDevice,
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<T>, T>,
    plan: AttentionPlan,
    score_workspace: Option<&WgpuBuffer<T>>,
) -> Result<BackwardPreflight> {
    let status = device.alloc_zeroed::<u32>(1)?;
    let destination = operands
        .gradients
        .query
        .or(operands.gradients.key)
        .or(operands.gradients.value)
        .expect("invariant: attention plan requires one gradient destination");
    let metadata = backward_metadata(operands, plan, destination)?;
    let rows = checked_product(plan.batch, plan.query_sequence, "backward row count")?;
    let key_rows = checked_product(plan.batch, plan.key_sequence, "backward key row count")?;
    let mut kernels = smallvec::SmallVec::new();
    kernels.push(prepare_finite::<GradOutputFinite, T>(
        device,
        &metadata,
        checked_product(rows, plan.value_feature, "output-gradient element count")?,
        "grad_output",
        3,
        AttentionSemanticStatus::NonFiniteOutputGradient,
        operands.grad_output.buffer,
        &status,
    )?);
    kernels.push(prepare_finite::<QueryFinite, T>(
        device,
        &metadata,
        checked_product(rows, plan.key_feature, "query element count")?,
        "query",
        3,
        AttentionSemanticStatus::NonFiniteQuery,
        operands.query.buffer,
        &status,
    )?);
    kernels.push(prepare_finite::<KeyFinite, T>(
        device,
        &metadata,
        checked_product(key_rows, plan.key_feature, "key element count")?,
        "key",
        3,
        AttentionSemanticStatus::NonFiniteKey,
        operands.key.buffer,
        &status,
    )?);
    kernels.push(prepare_finite::<ValueFinite, T>(
        device,
        &metadata,
        checked_product(key_rows, plan.value_feature, "value element count")?,
        "value",
        3,
        AttentionSemanticStatus::NonFiniteValue,
        operands.value.buffer,
        &status,
    )?);
    kernels.push(prepare_finite::<WeightsFinite, T>(
        device,
        &metadata,
        plan.score_elements,
        "weights",
        3,
        AttentionSemanticStatus::NonFiniteWeights,
        operands.weights.buffer,
        &status,
    )?);
    kernels.push(prepare_status::<Probability, T>(
        device,
        &metadata,
        rows,
        "hephaestus-attention-backward-probability-preflight",
        &[binding(0, operands.weights.buffer), binding(1, &status)],
        || backward_probability_preflight_shader::<T>(WORKGROUP_WIDTH.get()),
    )?);
    if let Some(workspace) = score_workspace {
        kernels.push(prepare_score_workspace(
            device, operands, plan, workspace, &metadata,
        )?);
        kernels.push(prepare_status::<ScoreFinite, T>(
            device,
            &metadata,
            plan.score_elements,
            "hephaestus-attention-backward-score-finite-preflight",
            &[binding(0, workspace), binding(1, &status)],
            || {
                linear_finite_preflight_shader::<T>(
                    AttentionSemanticStatus::NonFiniteWeightsArithmetic,
                    WORKGROUP_WIDTH.get(),
                )
            },
        )?);
    }
    if let Some(target) = operands.gradients.query {
        let workspace = score_workspace
            .expect("invariant: query gradient preparation allocated score workspace");
        kernels.push(prepare_gradient::<QueryGradient, T>(
            device,
            operands,
            plan,
            target,
            workspace,
            operands.key.buffer,
            &status,
            GradientPreflightStage::Query,
            "hephaestus-attention-backward-query-preflight",
        )?);
    }
    if let Some(target) = operands.gradients.key {
        let workspace =
            score_workspace.expect("invariant: key gradient preparation allocated score workspace");
        kernels.push(prepare_gradient::<KeyGradient, T>(
            device,
            operands,
            plan,
            target,
            workspace,
            operands.query.buffer,
            &status,
            GradientPreflightStage::Key,
            "hephaestus-attention-backward-key-preflight",
        )?);
    }
    if let Some(target) = operands.gradients.value {
        kernels.push(prepare_gradient::<ValueGradient, T>(
            device,
            operands,
            plan,
            target,
            operands.weights.buffer,
            operands.grad_output.buffer,
            &status,
            GradientPreflightStage::Value,
            "hephaestus-attention-backward-value-preflight",
        )?);
    }
    Ok(BackwardPreflight { kernels, status })
}

fn prepare_score_workspace<T: WgslAttentionScalar>(
    device: &WgpuDevice,
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<T>, T>,
    plan: AttentionPlan,
    workspace: &WgpuBuffer<T>,
    metadata: &super::super::metadata::AttentionMeta<T>,
) -> Result<PreparedAttentionKernel> {
    prepare(
        device,
        metadata,
        plan.score_elements,
        "hephaestus-attention-backward-score-preflight",
        &[
            binding(0, operands.grad_output.buffer),
            binding(1, operands.value.buffer),
            binding(2, operands.weights.buffer),
            binding(3, workspace),
        ],
        || backward_shader::<T>(BackwardStage::Score, WORKGROUP_WIDTH.get()),
        TypeId::of::<ScoreWorkspace>(),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "gradient preflight preparation enumerates the complete read-only kernel ABI"
)]
fn prepare_gradient<K: 'static, T: WgslAttentionScalar>(
    device: &WgpuDevice,
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<T>, T>,
    plan: AttentionPlan,
    target: StridedView<'_, WgpuBuffer<T>, 3>,
    first: &WgpuBuffer<T>,
    second: &WgpuBuffer<T>,
    status: &WgpuBuffer<u32>,
    stage: GradientPreflightStage,
    label: &'static str,
) -> Result<PreparedAttentionKernel> {
    let metadata = backward_metadata(operands, plan, target)?;
    let elements = target.layout.checked_size().map_err(|error| {
        super::super::resources::invalid(format!("attention gradient layout rejected: {error}"))
    })?;
    prepare_status::<K, T>(
        device,
        &metadata,
        elements,
        label,
        &[
            binding(0, first),
            binding(1, second),
            binding(2, target.buffer),
            binding(3, status),
        ],
        || backward_gradient_preflight_shader::<T>(stage, WORKGROUP_WIDTH.get()),
    )
}
