//! The backward preflight: probability rows, the score-gradient workspace,
//! and one destination/arithmetic pair per requested gradient target.

use hephaestus_core::{
    AttentionBackwardOperands, AttentionGradientViews, AttentionPlan, AttentionSemanticStatus,
    Result,
};
use leto::ArrayView;

use super::read::{all_finite, read3, score_index};
use crate::{HostBuffer, map_leto_error};

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/attention/resources.rs::backward_aliases`.
pub(super) fn backward_aliases(
    operands: &AttentionBackwardOperands<'_, HostBuffer<f32>, f32>,
) -> bool {
    let reads = [
        operands.grad_output.buffer,
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
        operands.weights.buffer,
    ];
    let targets = [
        operands.gradients.query.map(|view| view.buffer),
        operands.gradients.key.map(|view| view.buffer),
        operands.gradients.value.map(|view| view.buffer),
    ];
    targets
        .into_iter()
        .flatten()
        .any(|target| reads.into_iter().any(|read| target.aliases(read)))
        || targets.iter().enumerate().any(|(index, target)| {
            target.is_some_and(|target| {
                targets[index + 1..]
                    .iter()
                    .flatten()
                    .any(|other| target.aliases(other))
            })
        })
}

/// Recompute `hephaestus-wgpu`'s `backward_probability_preflight_shader`.
fn backward_probability_status(weights: &ArrayView<'_, f32, 3>, plan: AttentionPlan) -> u32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "mirrors the WGSL shader's f32(dimensions.z) key-sequence scale"
    )]
    let tolerance = 4.0_f32 * f32::EPSILON * plan.key_sequence as f32;
    let mut status = u32::MAX;
    for batch in 0..plan.batch {
        for query_index in 0..plan.query_sequence {
            let mut sum = 0.0_f32;
            let mut out_of_range = false;
            for key_index in 0..plan.key_sequence {
                let weight = read3(weights, batch, query_index, key_index);
                if !(0.0..=1.0).contains(&weight) {
                    out_of_range = true;
                }
                sum += weight;
            }
            if out_of_range
                || !tolerance.is_finite()
                || tolerance >= 0.5
                || (sum != 0.0 && (sum - 1.0).abs() > tolerance)
            {
                status = status.min(AttentionSemanticStatus::InvalidWeights.code());
            }
        }
    }
    status
}

/// Recompute `hephaestus-wgpu`'s `BackwardStage::Score` kernel plus its
/// follow-up whole-buffer finiteness scan
/// (`shader/preflight.rs::linear_finite_preflight_shader`), reusing leto's
/// store-once algorithm rather than the shader's per-invocation
/// recomputation: both perform the identical per-feature summation, so their
/// results agree bit for bit.
fn score_gradient_workspace(
    grad_output: &ArrayView<'_, f32, 3>,
    value: &ArrayView<'_, f32, 3>,
    weights: &ArrayView<'_, f32, 3>,
    plan: AttentionPlan,
) -> (Vec<f32>, u32) {
    let mut status = u32::MAX;
    let mut workspace = vec![0.0_f32; plan.score_elements];
    for batch in 0..plan.batch {
        for query_index in 0..plan.query_sequence {
            for key_index in 0..plan.key_sequence {
                let mut candidate_gradient = 0.0_f32;
                for feature in 0..plan.value_feature {
                    candidate_gradient += read3(grad_output, batch, query_index, feature)
                        * read3(value, batch, key_index, feature);
                }
                workspace[score_index(plan, batch, query_index, key_index)] = candidate_gradient;
            }
            let mut projection = 0.0_f32;
            for candidate in 0..plan.key_sequence {
                projection += read3(weights, batch, query_index, candidate)
                    * workspace[score_index(plan, batch, query_index, candidate)];
            }
            for key_index in 0..plan.key_sequence {
                let index = score_index(plan, batch, query_index, key_index);
                let weight = read3(weights, batch, query_index, key_index);
                let value = weight * (workspace[index] - projection);
                workspace[index] = value;
                if !value.is_finite() {
                    status = status.min(AttentionSemanticStatus::NonFiniteWeightsArithmetic.code());
                }
            }
        }
    }
    (workspace, status)
}

/// Recompute `hephaestus-wgpu`'s `backward_gradient_preflight_shader` for the
/// query-gradient stage.
fn query_gradient_status(
    key: &ArrayView<'_, f32, 3>,
    score_gradient: &[f32],
    destination: &[f32],
    destination_layout: &leto::Layout<3>,
    plan: AttentionPlan,
    scale: f32,
) -> Result<u32> {
    let destination_view =
        ArrayView::try_new(*destination_layout, destination).map_err(map_leto_error)?;
    let mut status = u32::MAX;
    for batch in 0..plan.batch {
        for query_index in 0..plan.query_sequence {
            for feature in 0..plan.key_feature {
                let mut accumulated = 0.0_f32;
                for key_index in 0..plan.key_sequence {
                    accumulated += score_gradient[score_index(plan, batch, query_index, key_index)]
                        * read3(key, batch, key_index, feature);
                }
                let increment = scale * accumulated;
                let current = read3(&destination_view, batch, query_index, feature);
                if !current.is_finite() {
                    status = status.min(AttentionSemanticStatus::NonFiniteQueryGradient.code());
                }
                if !increment.is_finite() || !(current + increment).is_finite() {
                    status = status
                        .min(AttentionSemanticStatus::NonFiniteQueryGradientArithmetic.code());
                }
            }
        }
    }
    Ok(status)
}

/// Recompute `hephaestus-wgpu`'s `backward_gradient_preflight_shader` for the
/// key-gradient stage.
fn key_gradient_status(
    query: &ArrayView<'_, f32, 3>,
    score_gradient: &[f32],
    destination: &[f32],
    destination_layout: &leto::Layout<3>,
    plan: AttentionPlan,
    scale: f32,
) -> Result<u32> {
    let destination_view =
        ArrayView::try_new(*destination_layout, destination).map_err(map_leto_error)?;
    let mut status = u32::MAX;
    for batch in 0..plan.batch {
        for key_index in 0..plan.key_sequence {
            for feature in 0..plan.key_feature {
                let mut accumulated = 0.0_f32;
                for query_index in 0..plan.query_sequence {
                    accumulated += score_gradient[score_index(plan, batch, query_index, key_index)]
                        * read3(query, batch, query_index, feature);
                }
                let increment = scale * accumulated;
                let current = read3(&destination_view, batch, key_index, feature);
                if !current.is_finite() {
                    status = status.min(AttentionSemanticStatus::NonFiniteKeyGradient.code());
                }
                if !increment.is_finite() || !(current + increment).is_finite() {
                    status =
                        status.min(AttentionSemanticStatus::NonFiniteKeyGradientArithmetic.code());
                }
            }
        }
    }
    Ok(status)
}

/// Recompute `hephaestus-wgpu`'s `backward_gradient_preflight_shader` for the
/// value-gradient stage (unscaled, matching `value_body`'s formula).
fn value_gradient_status(
    grad_output: &ArrayView<'_, f32, 3>,
    weights: &ArrayView<'_, f32, 3>,
    destination: &[f32],
    destination_layout: &leto::Layout<3>,
    plan: AttentionPlan,
) -> Result<u32> {
    let destination_view =
        ArrayView::try_new(*destination_layout, destination).map_err(map_leto_error)?;
    let mut status = u32::MAX;
    for batch in 0..plan.batch {
        for key_index in 0..plan.key_sequence {
            for feature in 0..plan.value_feature {
                let mut accumulated = 0.0_f32;
                for query_index in 0..plan.query_sequence {
                    accumulated += read3(weights, batch, query_index, key_index)
                        * read3(grad_output, batch, query_index, feature);
                }
                let current = read3(&destination_view, batch, key_index, feature);
                if !current.is_finite() {
                    status = status.min(AttentionSemanticStatus::NonFiniteValueGradient.code());
                }
                if !accumulated.is_finite() || !(current + accumulated).is_finite() {
                    status = status
                        .min(AttentionSemanticStatus::NonFiniteValueGradientArithmetic.code());
                }
            }
        }
    }
    Ok(status)
}

/// Recompute every applicable `hephaestus-wgpu` backward preflight kernel and
/// reduce to the minimum status, matching their shared `atomicMin` word.
/// Unlike leto-ops' own sequential `validate_backward` (which checks
/// `grad_output` before `query`/`key`), every check here runs independently
/// so the numerically lowest triggered code always wins.
#[expect(
    clippy::too_many_arguments,
    reason = "the backward preflight recomputation enumerates every readable operand and selected gradient destination, mirroring hephaestus-wgpu's independently dispatched kernels"
)]
pub(super) fn backward_preflight_status(
    grad_output: &ArrayView<'_, f32, 3>,
    query: &ArrayView<'_, f32, 3>,
    key: &ArrayView<'_, f32, 3>,
    value: &ArrayView<'_, f32, 3>,
    weights: &ArrayView<'_, f32, 3>,
    gradients: &AttentionGradientViews<'_, HostBuffer<f32>>,
    query_grad_cells: Option<&[f32]>,
    key_grad_cells: Option<&[f32]>,
    value_grad_cells: Option<&[f32]>,
    plan: AttentionPlan,
    scale: f32,
) -> Result<u32> {
    let mut status = u32::MAX;
    if !all_finite(grad_output) {
        status = status.min(AttentionSemanticStatus::NonFiniteOutputGradient.code());
    }
    if !all_finite(query) {
        status = status.min(AttentionSemanticStatus::NonFiniteQuery.code());
    }
    if !all_finite(key) {
        status = status.min(AttentionSemanticStatus::NonFiniteKey.code());
    }
    if !all_finite(value) {
        status = status.min(AttentionSemanticStatus::NonFiniteValue.code());
    }
    if !all_finite(weights) {
        status = status.min(AttentionSemanticStatus::NonFiniteWeights.code());
    }
    status = status.min(backward_probability_status(weights, plan));

    let needs_score = gradients.query.is_some() || gradients.key.is_some();
    let workspace =
        needs_score.then(|| score_gradient_workspace(grad_output, value, weights, plan));
    if let Some((_, workspace_status)) = &workspace {
        status = status.min(*workspace_status);
    }

    if let (Some(target), Some(cells)) = (gradients.query, query_grad_cells) {
        let workspace = &workspace
            .as_ref()
            .expect("invariant: query gradient preparation computed the score workspace")
            .0;
        status = status.min(query_gradient_status(
            key,
            workspace,
            cells,
            target.layout,
            plan,
            scale,
        )?);
    }
    if let (Some(target), Some(cells)) = (gradients.key, key_grad_cells) {
        let workspace = &workspace
            .as_ref()
            .expect("invariant: key gradient preparation computed the score workspace")
            .0;
        status = status.min(key_gradient_status(
            query,
            workspace,
            cells,
            target.layout,
            plan,
            scale,
        )?);
    }
    if let (Some(target), Some(cells)) = (gradients.value, value_grad_cells) {
        status = status.min(value_gradient_status(
            grad_output,
            weights,
            cells,
            target.layout,
            plan,
        )?);
    }
    Ok(status)
}
