//! The forward preflight: operand finiteness, then weight arithmetic.

use core::num::NonZeroUsize;

use hephaestus_core::{AttentionForwardOperands, AttentionPlan, AttentionSemanticStatus};
use leto::ArrayView;

use super::read::{all_finite, attention_score, is_kept};
use crate::HostBuffer;

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/attention/resources.rs::forward_aliases`.
pub(super) fn forward_aliases(
    operands: &AttentionForwardOperands<'_, HostBuffer<f32>, f32>,
) -> bool {
    let reads = [
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
    ];
    let mask = operands.mask.grouped_keep().map(|keep| keep.view().buffer);
    operands.output.buffer.aliases(operands.weights.buffer)
        || [operands.output.buffer, operands.weights.buffer]
            .into_iter()
            .any(|target| {
                reads.into_iter().any(|read| target.aliases(read))
                    || mask.is_some_and(|mask| target.aliases(mask))
            })
}

/// Recompute `hephaestus-wgpu`'s five forward preflight kernels and reduce
/// them to their minimum, exactly as their shared `atomicMin` status word
/// does.
pub(super) fn forward_preflight_status(
    query: &ArrayView<'_, f32, 3>,
    key: &ArrayView<'_, f32, 3>,
    value: &ArrayView<'_, f32, 3>,
    grouped: Option<(ArrayView<'_, f32, 2>, NonZeroUsize)>,
    causal: bool,
    plan: AttentionPlan,
    scale: f32,
) -> u32 {
    let mut status = u32::MAX;
    if !all_finite(query) {
        status = status.min(AttentionSemanticStatus::NonFiniteQuery.code());
    }
    if !all_finite(key) {
        status = status.min(AttentionSemanticStatus::NonFiniteKey.code());
    }
    if !all_finite(value) {
        status = status.min(AttentionSemanticStatus::NonFiniteValue.code());
    }
    if let Some((view, _)) = grouped
        && !all_finite(&view)
    {
        status = status.min(AttentionSemanticStatus::NonFiniteKeep.code());
    }
    status.min(forward_arithmetic_status(
        query, key, grouped, causal, plan, scale,
    ))
}

/// Recompute `hephaestus-wgpu`'s `forward_arithmetic_preflight_shader`.
fn forward_arithmetic_status(
    query: &ArrayView<'_, f32, 3>,
    key: &ArrayView<'_, f32, 3>,
    grouped: Option<(ArrayView<'_, f32, 2>, NonZeroUsize)>,
    causal: bool,
    plan: AttentionPlan,
    scale: f32,
) -> u32 {
    let mut status = u32::MAX;
    for batch in 0..plan.batch {
        for query_index in 0..plan.query_sequence {
            let mut maximum = f32::MIN;
            let mut kept_count = 0usize;
            for key_index in 0..plan.key_sequence {
                if !is_kept(grouped, causal, batch, query_index, key_index) {
                    continue;
                }
                kept_count += 1;
                let (score, nonfinite) = attention_score(
                    query,
                    key,
                    batch,
                    query_index,
                    key_index,
                    plan.key_feature,
                    scale,
                );
                if nonfinite {
                    status = status.min(AttentionSemanticStatus::NonFiniteWeightsArithmetic.code());
                }
                if score > maximum {
                    maximum = score;
                }
            }
            if kept_count == 0 {
                continue;
            }
            let mut denominator = 0.0_f32;
            for key_index in 0..plan.key_sequence {
                if !is_kept(grouped, causal, batch, query_index, key_index) {
                    continue;
                }
                let (score, _) = attention_score(
                    query,
                    key,
                    batch,
                    query_index,
                    key_index,
                    plan.key_feature,
                    scale,
                );
                denominator += (score - maximum).exp();
                if !denominator.is_finite() {
                    status = status.min(AttentionSemanticStatus::NonFiniteWeightsArithmetic.code());
                }
            }
        }
    }
    status
}
