use hephaestus_core::{
    AttentionForwardOperands, AttentionPlan, AttentionSemanticStatus, ComputeDevice, Result,
};

use super::super::metadata::AttentionMeta;
use super::super::prepared::PreparedAttentionKernel;
use super::super::resources::binding;
use super::super::seam::{WORKGROUP_WIDTH, checked_product};
use super::super::shader::forward_arithmetic_preflight_shader;
use super::kernel::{prepare_finite, prepare_status};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

struct QueryFinite;
struct KeyFinite;
struct ValueFinite;
struct KeepFinite;
struct WeightArithmetic;

pub(in crate::application::attention) struct ForwardPreflight {
    pub(in crate::application::attention) kernels: [PreparedAttentionKernel; 5],
    pub(in crate::application::attention) status: WgpuBuffer<u32>,
}

pub(in crate::application::attention) fn prepare_forward(
    device: &WgpuDevice,
    operands: &AttentionForwardOperands<'_, WgpuBuffer<f32>, f32>,
    plan: AttentionPlan,
    metadata: &AttentionMeta,
    mask_buffer: &WgpuBuffer<f32>,
    rows: usize,
) -> Result<ForwardPreflight> {
    let status = device.alloc_zeroed::<u32>(1)?;
    let query_finite = prepare_finite::<QueryFinite>(
        device,
        metadata,
        checked_product(rows, plan.key_feature, "query element count")?,
        "query",
        3,
        AttentionSemanticStatus::NonFiniteQuery,
        operands.query.buffer,
        &status,
    )?;
    let key_rows = checked_product(plan.batch, plan.key_sequence, "key row count")?;
    let key_finite = prepare_finite::<KeyFinite>(
        device,
        metadata,
        checked_product(key_rows, plan.key_feature, "key element count")?,
        "key",
        3,
        AttentionSemanticStatus::NonFiniteKey,
        operands.key.buffer,
        &status,
    )?;
    let value_finite = prepare_finite::<ValueFinite>(
        device,
        metadata,
        checked_product(key_rows, plan.value_feature, "value element count")?,
        "value",
        3,
        AttentionSemanticStatus::NonFiniteValue,
        operands.value.buffer,
        &status,
    )?;
    let keep_elements = operands
        .mask
        .grouped_keep()
        .map(|keep| keep.view().layout.checked_size())
        .transpose()
        .map_err(|error| {
            super::super::resources::invalid(format!("attention keep layout rejected: {error}"))
        })?
        .unwrap_or(0);
    let keep_finite = prepare_finite::<KeepFinite>(
        device,
        metadata,
        keep_elements,
        "keep_mask",
        2,
        AttentionSemanticStatus::NonFiniteKeep,
        mask_buffer,
        &status,
    )?;
    let arithmetic = prepare_status::<WeightArithmetic>(
        device,
        metadata,
        rows,
        "hephaestus-attention-forward-preflight",
        &[
            binding(0, operands.query.buffer),
            binding(1, operands.key.buffer),
            binding(2, mask_buffer),
            binding(3, &status),
        ],
        || forward_arithmetic_preflight_shader(WORKGROUP_WIDTH.get()),
    )?;
    Ok(ForwardPreflight {
        kernels: [
            query_finite,
            key_finite,
            value_finite,
            keep_finite,
            arithmetic,
        ],
        status,
    })
}
