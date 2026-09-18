//! Leto as a scaled dot-product attention seam implementor (ADR 0046).
//!
//! [`HostAttentionOps`] mirrors the read-only preflight contract
//! `hephaestus-wgpu`'s `application/attention` module shares across its
//! forward and backward kernels: every triggered
//! [`AttentionSemanticStatus`](hephaestus_core::AttentionSemanticStatus)
//! reduces to its numerically lowest code, exactly as the WGSL kernels
//! combine per-invocation failures through `atomicMin` on one shared status
//! word, and only a `Valid` preflight reaches leto-ops'
//! `scaled_dot_product_attention_into` /
//! `scaled_dot_product_attention_backward_accumulate`, the only code paths
//! that touch a destination buffer.
//!
//! Forward's preflight is order-independent by construction: `atomicMin`
//! commutes, so the host recomputes every WGSL preflight kernel's status
//! (`query`/`key`/`value`/`keep`-mask finiteness, then weight-arithmetic
//! finiteness) and folds them with [`u32::min`] rather than short-circuiting.
//! Backward mirrors the same five independent finiteness kernels plus the
//! probability-row check, the score-gradient workspace finiteness check, and
//! one destination/arithmetic pair per requested gradient target — again
//! folded rather than short-circuited, since leto-ops' own sequential
//! `validate_backward` checks `grad_output` before `query`/`key` (codes 5
//! before 1/2), which would pick the wrong winner whenever both fail.

use core::num::NonZeroUsize;

use hephaestus_core::{
    AttentionBackwardOperands, AttentionCausality, AttentionForwardOperands,
    AttentionGradientViews, AttentionOps, AttentionPlan, AttentionSemanticStatus, Result,
    plan_attention_backward, plan_attention_forward,
};
use leto::{ArrayView, ArrayViewMut};
use leto_ops::{
    AttentionGradients as LetoGradients, AttentionMask as LetoMask,
    GroupedKeepMask as LetoGroupedKeepMask,
};

use crate::operands::with_operand_reads;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Scaled dot-product attention for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostAttentionOps;

/// Prepared host attention forward resources.
///
/// The host dispatches no kernel, so preparation only runs the host-visible
/// structural plan; the value-dependent preflight status runs at dispatch
/// time, once the caller's buffers hold their final contents.
pub struct HostAttentionForward<'a> {
    operands: AttentionForwardOperands<'a, HostBuffer<f32>, f32>,
    plan: AttentionPlan,
}

/// Prepared host attention additive backward resources.
pub struct HostAttentionBackward<'a> {
    operands: AttentionBackwardOperands<'a, HostBuffer<f32>, f32>,
    plan: AttentionPlan,
}

impl AttentionOps<HostDevice, f32> for HostAttentionOps {
    type PreparedForward<'a>
        = HostAttentionForward<'a>
    where
        HostDevice: 'a,
        f32: 'a;
    type PreparedBackward<'a>
        = HostAttentionBackward<'a>
    where
        HostDevice: 'a,
        f32: 'a;

    fn prepare_attention_forward<'a>(
        &self,
        _device: &'a HostDevice,
        operands: AttentionForwardOperands<'a, HostBuffer<f32>, f32>,
    ) -> Result<Self::PreparedForward<'a>> {
        let plan = plan_attention_forward(&operands, forward_aliases(&operands))?;
        Ok(HostAttentionForward { operands, plan })
    }

    fn dispatch_attention_forward(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        let operands = &prepared.operands;
        let plan = prepared.plan;
        let causal = operands.mask.causality() == AttentionCausality::Causal;
        let keep = operands.mask.grouped_keep();

        let mut reads: Vec<&HostBuffer<f32>> = vec![
            operands.query.buffer,
            operands.key.buffer,
            operands.value.buffer,
        ];
        if let Some(keep) = keep {
            reads.push(keep.view().buffer);
        }

        with_operand_reads(&reads, |slices| -> Result<()> {
            let query_view =
                ArrayView::try_new(*operands.query.layout, slices[0]).map_err(map_leto_error)?;
            let key_view =
                ArrayView::try_new(*operands.key.layout, slices[1]).map_err(map_leto_error)?;
            let value_view =
                ArrayView::try_new(*operands.value.layout, slices[2]).map_err(map_leto_error)?;
            let keep_view = keep
                .map(|keep| {
                    ArrayView::try_new(*keep.view().layout, slices[3]).map_err(map_leto_error)
                })
                .transpose()?;
            let grouped = keep_view.map(|view| {
                (
                    view,
                    keep.expect("invariant: keep view presence matches grouped-keep presence")
                        .heads_per_batch(),
                )
            });

            // Preflight completes, read-only, before either destination is
            // touched: a failure here leaves `output` and `weights`
            // untouched, matching the accelerator seams' atomicity contract.
            let status = forward_preflight_status(
                &query_view,
                &key_view,
                &value_view,
                grouped,
                causal,
                plan,
                operands.scale,
            );
            AttentionSemanticStatus::check(finalize_status(status))?;

            let leto_mask = grouped.map_or(
                if causal {
                    LetoMask::Causal
                } else {
                    LetoMask::Unmasked
                },
                |(view, heads_per_batch)| {
                    let grouped = LetoGroupedKeepMask::new(view, heads_per_batch);
                    if causal {
                        LetoMask::CausalGroupedKeep(grouped)
                    } else {
                        LetoMask::GroupedKeep(grouped)
                    }
                },
            );

            let mut output_cells = operands.output.buffer.write();
            let mut weights_cells = operands.weights.buffer.write();
            let mut output_view = ArrayViewMut::try_new(*operands.output.layout, &mut output_cells)
                .map_err(map_leto_error)?;
            let mut weights_view =
                ArrayViewMut::try_new(*operands.weights.layout, &mut weights_cells)
                    .map_err(map_leto_error)?;
            leto_ops::scaled_dot_product_attention_into(
                &query_view,
                &key_view,
                &value_view,
                leto_mask,
                operands.scale,
                &mut output_view,
                &mut weights_view,
            )
            .map_err(map_leto_error)
        })
    }

    fn prepare_attention_backward<'a>(
        &self,
        _device: &'a HostDevice,
        operands: AttentionBackwardOperands<'a, HostBuffer<f32>, f32>,
    ) -> Result<Self::PreparedBackward<'a>> {
        let plan = plan_attention_backward(&operands, backward_aliases(&operands))?;
        Ok(HostAttentionBackward { operands, plan })
    }

    fn dispatch_attention_backward(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        let operands = &prepared.operands;
        let plan = prepared.plan;
        let reads: Vec<&HostBuffer<f32>> = vec![
            operands.grad_output.buffer,
            operands.query.buffer,
            operands.key.buffer,
            operands.value.buffer,
            operands.weights.buffer,
        ];

        with_operand_reads(&reads, |slices| -> Result<()> {
            let grad_output_view = ArrayView::try_new(*operands.grad_output.layout, slices[0])
                .map_err(map_leto_error)?;
            let query_view =
                ArrayView::try_new(*operands.query.layout, slices[1]).map_err(map_leto_error)?;
            let key_view =
                ArrayView::try_new(*operands.key.layout, slices[2]).map_err(map_leto_error)?;
            let value_view =
                ArrayView::try_new(*operands.value.layout, slices[3]).map_err(map_leto_error)?;
            let weights_view =
                ArrayView::try_new(*operands.weights.layout, slices[4]).map_err(map_leto_error)?;

            let mut query_grad_cells = operands.gradients.query.map(|target| target.buffer.write());
            let mut key_grad_cells = operands.gradients.key.map(|target| target.buffer.write());
            let mut value_grad_cells = operands.gradients.value.map(|target| target.buffer.write());

            let status = backward_preflight_status(
                &grad_output_view,
                &query_view,
                &key_view,
                &value_view,
                &weights_view,
                &operands.gradients,
                query_grad_cells.as_ref().map(|cells| cells.as_slice()),
                key_grad_cells.as_ref().map(|cells| cells.as_slice()),
                value_grad_cells.as_ref().map(|cells| cells.as_slice()),
                plan,
                operands.scale,
            )?;
            AttentionSemanticStatus::check(finalize_status(status))?;

            let query_gradient = match (&mut query_grad_cells, operands.gradients.query) {
                (Some(cells), Some(target)) => {
                    Some(ArrayViewMut::try_new(*target.layout, cells).map_err(map_leto_error)?)
                }
                _ => None,
            };
            let key_gradient = match (&mut key_grad_cells, operands.gradients.key) {
                (Some(cells), Some(target)) => {
                    Some(ArrayViewMut::try_new(*target.layout, cells).map_err(map_leto_error)?)
                }
                _ => None,
            };
            let value_gradient = match (&mut value_grad_cells, operands.gradients.value) {
                (Some(cells), Some(target)) => {
                    Some(ArrayViewMut::try_new(*target.layout, cells).map_err(map_leto_error)?)
                }
                _ => None,
            };
            leto_ops::scaled_dot_product_attention_backward_accumulate(
                &grad_output_view,
                &query_view,
                &key_view,
                &value_view,
                &weights_view,
                operands.scale,
                LetoGradients::new(query_gradient, key_gradient, value_gradient),
            )
            .map_err(map_leto_error)
        })
    }
}

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/attention/resources.rs::forward_aliases`.
fn forward_aliases(operands: &AttentionForwardOperands<'_, HostBuffer<f32>, f32>) -> bool {
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

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/attention/resources.rs::backward_aliases`.
fn backward_aliases(operands: &AttentionBackwardOperands<'_, HostBuffer<f32>, f32>) -> bool {
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

/// Map the `u32::MAX` no-failure sentinel used by the status-reduction
/// helpers below to [`AttentionSemanticStatus::Valid`]'s protocol code,
/// mirroring `hephaestus-wgpu`'s preflight status buffers, which reset to
/// `u32::MAX` before dispatch and are read back the same way.
fn finalize_status(status: u32) -> u32 {
    if status == u32::MAX {
        AttentionSemanticStatus::Valid.code()
    } else {
        status
    }
}

fn all_finite<const N: usize>(view: &ArrayView<'_, f32, N>) -> bool {
    view.iter().all(|value| value.is_finite())
}

fn read3(view: &ArrayView<'_, f32, 3>, batch: usize, row: usize, col: usize) -> f32 {
    *view
        .get([batch, row, col])
        .expect("invariant: plan-validated attention index is in bounds")
}

fn read2(view: &ArrayView<'_, f32, 2>, row: usize, col: usize) -> f32 {
    *view
        .get([row, col])
        .expect("invariant: plan-validated attention mask index is in bounds")
}

/// Whether `(batch, query_index, key_index)` survives causal and keep-mask
/// gating, mirroring `hephaestus-wgpu`'s `shader/preflight.rs::kept`.
fn is_kept(
    grouped: Option<(ArrayView<'_, f32, 2>, NonZeroUsize)>,
    causal: bool,
    batch: usize,
    query_index: usize,
    key_index: usize,
) -> bool {
    if causal && key_index > query_index {
        return false;
    }
    match grouped {
        None => true,
        Some((view, heads_per_batch)) => {
            let mask_batch = batch / heads_per_batch.get();
            read2(&view, mask_batch, key_index) != 0.0
        }
    }
}

/// Recompute `hephaestus-wgpu`'s five forward preflight kernels and reduce
/// them to their minimum, exactly as their shared `atomicMin` status word
/// does.
fn forward_preflight_status(
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

/// Score and dot-product accumulation, checked for finiteness at every step
/// exactly as `hephaestus-wgpu`'s `attention_score` does.
fn attention_score(
    query: &ArrayView<'_, f32, 3>,
    key: &ArrayView<'_, f32, 3>,
    batch: usize,
    query_index: usize,
    key_index: usize,
    key_feature: usize,
    scale: f32,
) -> (f32, bool) {
    let mut dot = 0.0_f32;
    let mut nonfinite = false;
    for feature in 0..key_feature {
        dot += read3(query, batch, query_index, feature) * read3(key, batch, key_index, feature);
        nonfinite |= !dot.is_finite();
    }
    let scaled = dot * scale;
    nonfinite |= !scaled.is_finite();
    (scaled, nonfinite)
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

fn score_index(plan: AttentionPlan, batch: usize, query_index: usize, key_index: usize) -> usize {
    (batch * plan.query_sequence + query_index) * plan.key_sequence + key_index
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
fn backward_preflight_status(
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
