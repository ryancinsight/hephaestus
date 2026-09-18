//! Leto as a mean cross-entropy seam implementor (ADR 0046).
//!
//! [`HostCrossEntropyOps`] mirrors the two-phase preflight contract every
//! accelerator backend shares (see `hephaestus-wgpu`'s
//! `application/loss/prepared.rs` and `application/loss/shader.rs`): a
//! read-only preflight pass reduces every triggered `CrossEntropyStatus`
//! down to its numerically lowest code, exactly as the WGSL kernels combine
//! per-invocation failures through `atomicMin` on a shared status word, and
//! only a `Valid` preflight reaches leto-ops' `cross_entropy_forward_into` /
//! `cross_entropy_backward_accumulate`, which perform the real softmax/NLL
//! arithmetic and are the only code paths that touch a destination buffer.
//!
//! The preflight is reimplemented here rather than delegated to leto-ops'
//! own `validate_forward`/`validate_backward`: leto-ops checks the upstream
//! gradient's finiteness before the target range
//! (`leto-ops::application::loss::validation`), while the shared conformance
//! clause (`assert_cross_entropy_contract`) requires a target-range failure
//! to win over a non-finite upstream, matching the WGSL shader's priority.
//! Once a dispatch's preflight is `Valid`, leto-ops' own validation inside
//! `cross_entropy_forward_into`/`cross_entropy_backward_accumulate` is
//! necessarily satisfied too, so the real computation always proceeds.

use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps, CrossEntropyPlan,
    CrossEntropyStatus, Result, plan_cross_entropy_backward, plan_cross_entropy_forward,
};
use leto::{ArrayView, ArrayViewMut, Layout};

use crate::operands::with_operands;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Mean cross-entropy for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostCrossEntropyOps;

/// Prepared host cross-entropy forward resources.
///
/// The host dispatches no kernel, so preparation only runs the host-visible
/// structural plan; the value-dependent preflight status runs at dispatch
/// time, once the caller's buffers hold their final contents.
pub struct HostCrossEntropyForward<'a> {
    operands: CrossEntropyForwardOperands<'a, HostBuffer<f32>, HostBuffer<u32>>,
    plan: CrossEntropyPlan,
}

/// Prepared host cross-entropy additive backward resources.
pub struct HostCrossEntropyBackward<'a> {
    operands: CrossEntropyBackwardOperands<'a, HostBuffer<f32>, HostBuffer<u32>>,
    plan: CrossEntropyPlan,
}

impl CrossEntropyOps<HostDevice, f32> for HostCrossEntropyOps {
    type PreparedForward<'a>
        = HostCrossEntropyForward<'a>
    where
        HostDevice: 'a,
        f32: 'a;
    type PreparedBackward<'a>
        = HostCrossEntropyBackward<'a>
    where
        HostDevice: 'a,
        f32: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        _device: &'a HostDevice,
        operands: CrossEntropyForwardOperands<'a, HostBuffer<f32>, HostBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        let plan = plan_cross_entropy_forward(&operands, forward_aliases(&operands))?;
        Ok(HostCrossEntropyForward { operands, plan })
    }

    fn dispatch_cross_entropy_forward(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        let operands = &prepared.operands;
        let plan = prepared.plan;

        let logits_cells = operands.logits.buffer.read();
        let targets_cells = operands.targets.buffer.read();
        let logits_view =
            ArrayView::try_new(*operands.logits.layout, &logits_cells).map_err(map_leto_error)?;
        let targets = gather_targets(&targets_cells, operands.targets.layout, plan.batch)?;

        // Preflight completes, read-only, before either destination is
        // touched: a failure here leaves `loss` and `probabilities`
        // untouched, matching the accelerator seams' atomicity contract.
        CrossEntropyStatus::check(finalize_status(forward_preflight_status(
            &logits_view,
            &targets,
            plan.classes,
        )))?;
        let targets = resolved_targets(&targets);

        let mut loss_cells = operands.loss.buffer.write();
        let mut probabilities_cells = operands.probabilities.buffer.write();
        let mut loss_view = ArrayViewMut::try_new(*operands.loss.layout, &mut loss_cells)
            .map_err(map_leto_error)?;
        let mut probabilities_view =
            ArrayViewMut::try_new(*operands.probabilities.layout, &mut probabilities_cells)
                .map_err(map_leto_error)?;
        leto_ops::cross_entropy_forward_into(
            &logits_view,
            &targets,
            &mut loss_view,
            &mut probabilities_view,
        )
        .map_err(map_leto_error)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        _device: &'a HostDevice,
        operands: CrossEntropyBackwardOperands<'a, HostBuffer<f32>, HostBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        let plan = plan_cross_entropy_backward(&operands, backward_aliases(&operands))?;
        Ok(HostCrossEntropyBackward { operands, plan })
    }

    fn dispatch_cross_entropy_backward(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        let operands = &prepared.operands;
        let plan = prepared.plan;

        let targets_cells = operands.targets.buffer.read();
        let targets = gather_targets(&targets_cells, operands.targets.layout, plan.batch)?;
        let mut logit_gradient_cells = operands.logit_gradient.buffer.write();

        // `output_gradient` and `probabilities` are both `f32` reads that
        // are never proven disjoint by the plan, so they share one guard
        // when they alias, per this crate's `HostBuffer` lock discipline.
        with_operands(
            operands.output_gradient.buffer,
            operands.probabilities.buffer,
            |output_gradient_cells, probabilities_cells| -> Result<()> {
                let output_gradient_view =
                    ArrayView::try_new(*operands.output_gradient.layout, output_gradient_cells)
                        .map_err(map_leto_error)?;
                let probabilities_view =
                    ArrayView::try_new(*operands.probabilities.layout, probabilities_cells)
                        .map_err(map_leto_error)?;
                let destination_view =
                    ArrayView::try_new(*operands.logit_gradient.layout, &logit_gradient_cells)
                        .map_err(map_leto_error)?;
                let upstream = *output_gradient_view
                    .get([0])
                    .expect("invariant: plan-validated output-gradient index is in bounds");

                let row_status = backward_row_status(
                    upstream,
                    &probabilities_view,
                    &targets,
                    plan.classes,
                    plan.probability_tolerance,
                );
                let arithmetic_status = backward_arithmetic_status(
                    upstream,
                    &probabilities_view,
                    &targets,
                    &destination_view,
                    plan.classes,
                    plan.batch,
                );
                // The two WGSL preflight kernels write the same shared
                // atomic status word; taking the minimum of their
                // independently computed codes reproduces that reduction.
                CrossEntropyStatus::check(finalize_status(row_status.min(arithmetic_status)))?;
                let targets = resolved_targets(&targets);

                let mut logit_gradient_view = ArrayViewMut::try_new(
                    *operands.logit_gradient.layout,
                    &mut logit_gradient_cells,
                )
                .map_err(map_leto_error)?;
                leto_ops::cross_entropy_backward_accumulate(
                    &output_gradient_view,
                    &probabilities_view,
                    &targets,
                    &mut logit_gradient_view,
                )
                .map_err(map_leto_error)
            },
        )
    }
}

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/loss/resources.rs::forward_aliases`: a writable destination
/// must not alias a readable operand or the other destination. `targets` is
/// `HostBuffer<u32>` and can never alias an `f32` buffer by construction, so
/// only the `f32`-typed pairs need checking.
fn forward_aliases(
    operands: &CrossEntropyForwardOperands<'_, HostBuffer<f32>, HostBuffer<u32>>,
) -> bool {
    operands.loss.buffer.aliases(operands.logits.buffer)
        || operands.loss.buffer.aliases(operands.probabilities.buffer)
        || operands
            .probabilities
            .buffer
            .aliases(operands.logits.buffer)
}

/// Illegal-aliasing predicate mirroring `hephaestus-wgpu`'s
/// `application/loss/resources.rs::backward_aliases`.
fn backward_aliases(
    operands: &CrossEntropyBackwardOperands<'_, HostBuffer<f32>, HostBuffer<u32>>,
) -> bool {
    operands
        .logit_gradient
        .buffer
        .aliases(operands.output_gradient.buffer)
        || operands
            .logit_gradient
            .buffer
            .aliases(operands.probabilities.buffer)
}

/// Materialize one `u32` target per batch row into a plain, logically
/// ordered slice. [`Layout::offset_of`] is documented total on in-shape
/// indices, so an in-range row index cannot fail here.
fn gather_targets(cells: &[u32], layout: &Layout<1>, batch: usize) -> Result<Vec<u32>> {
    let view = ArrayView::try_new(*layout, cells).map_err(map_leto_error)?;
    Ok((0..batch)
        .map(|row| {
            *view
                .get([row])
                .expect("invariant: plan-validated target index is in bounds")
        })
        .collect())
}

/// Convert preflight-validated targets (each proven `< classes`) into the
/// `usize` slice leto-ops' forward/backward kernels require.
fn resolved_targets(targets: &[u32]) -> Vec<usize> {
    targets
        .iter()
        .map(|&target| {
            usize::try_from(target).expect("invariant: preflight-validated target index fits usize")
        })
        .collect()
}

/// Map the `u32::MAX` no-failure sentinel used by the status-reduction
/// helpers below to [`CrossEntropyStatus::Valid`]'s protocol code, mirroring
/// `hephaestus-wgpu`'s `check_status` (`application/loss/prepared.rs`),
/// which resets the shared atomic to `u32::MAX` before dispatch and reads it
/// back the same way.
fn finalize_status(status: u32) -> u32 {
    if status == u32::MAX {
        CrossEntropyStatus::Valid.code()
    } else {
        status
    }
}

/// Read a validated `[row, class]` element; the plan already proved every
/// such index is in bounds for the layout it was built from.
fn read_row_major(view: &ArrayView<'_, f32, 2>, row: usize, class: usize) -> f32 {
    *view
        .get([row, class])
        .expect("invariant: plan-validated row-major index is in bounds")
}

/// Recompute `hephaestus-wgpu`'s `forward_preflight` shader
/// (`application/loss/shader.rs`) over the whole batch, reducing every
/// triggered [`CrossEntropyStatus`] code to its minimum exactly as the
/// shader's shared `atomicMin` status word does: whichever failure carries
/// the numerically lowest protocol code is canonical, regardless of which
/// row raised it. A row whose target is out of range skips its own
/// logit/arithmetic checks, exactly as the shader returns early for that
/// row; every other row is still evaluated.
fn forward_preflight_status(
    logits: &ArrayView<'_, f32, 2>,
    targets: &[u32],
    classes: usize,
) -> u32 {
    let class_bound = u32::try_from(classes)
        .expect("invariant: cross-entropy plan bounds classes within u32 range");
    let mut status = u32::MAX;
    for (row, &target) in targets.iter().enumerate() {
        if target >= class_bound {
            status = status.min(CrossEntropyStatus::TargetOutOfRange.code());
            continue;
        }
        let mut maximum = f32::MIN;
        for class in 0..classes {
            let value = read_row_major(logits, row, class);
            if !value.is_finite() {
                status = status.min(CrossEntropyStatus::NonFiniteLogits.code());
            }
            if value > maximum {
                maximum = value;
            }
        }
        let mut denominator = 0.0_f32;
        for class in 0..classes {
            denominator += (read_row_major(logits, row, class) - maximum).exp();
        }
        let target_class = usize::try_from(target).expect("invariant: in-range target fits usize");
        let target_logit = read_row_major(logits, row, target_class);
        let row_loss = denominator.ln() + (maximum - target_logit);
        if !denominator.is_finite() || denominator <= 0.0 || !row_loss.is_finite() {
            status = status.min(CrossEntropyStatus::NonFiniteForwardArithmetic.code());
        }
    }
    status
}

/// Recompute `hephaestus-wgpu`'s `backward_rows` preflight shader over every
/// row: the upstream scalar's finiteness (checked once, since it is the same
/// value for every row), each row's target range, and each row's saved
/// probabilities. Unlike the forward preflight, no row is skipped early,
/// matching the shader.
fn backward_row_status(
    upstream: f32,
    probabilities: &ArrayView<'_, f32, 2>,
    targets: &[u32],
    classes: usize,
    tolerance: f32,
) -> u32 {
    let class_bound = u32::try_from(classes)
        .expect("invariant: cross-entropy plan bounds classes within u32 range");
    let mut status = u32::MAX;
    if !upstream.is_finite() {
        status = status.min(CrossEntropyStatus::NonFiniteOutputGradient.code());
    }
    for (row, &target) in targets.iter().enumerate() {
        if target >= class_bound {
            status = status.min(CrossEntropyStatus::TargetOutOfRange.code());
        }
        let mut sum = 0.0_f32;
        let mut row_invalid = false;
        for class in 0..classes {
            let probability = read_row_major(probabilities, row, class);
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                row_invalid = true;
            }
            sum += probability;
        }
        if row_invalid || !sum.is_finite() || (sum - 1.0).abs() > tolerance {
            status = status.min(CrossEntropyStatus::InvalidProbabilities.code());
        }
    }
    status
}

/// Recompute `hephaestus-wgpu`'s `backward_arithmetic` preflight shader over
/// every element: the pre-existing destination value's finiteness and the
/// finiteness of the increment the accumulate pass would add.
fn backward_arithmetic_status(
    upstream: f32,
    probabilities: &ArrayView<'_, f32, 2>,
    targets: &[u32],
    destination: &ArrayView<'_, f32, 2>,
    classes: usize,
    batch: usize,
) -> u32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "mirrors the WGSL shader's f32(dimensions.x) batch scale"
    )]
    let scale = upstream / batch as f32;
    let mut status = u32::MAX;
    for (row, &target) in targets.iter().enumerate() {
        for class in 0..classes {
            let class_index = u32::try_from(class)
                .expect("invariant: cross-entropy plan bounds classes within u32 range");
            let indicator = if class_index == target { 1.0 } else { 0.0 };
            let probability = read_row_major(probabilities, row, class);
            let increment = scale * (probability - indicator);
            let current = read_row_major(destination, row, class);
            if !current.is_finite() {
                status = status.min(CrossEntropyStatus::NonFiniteGradientDestination.code());
            }
            if !increment.is_finite() || !(current + increment).is_finite() {
                status = status.min(CrossEntropyStatus::NonFiniteBackwardArithmetic.code());
            }
        }
    }
    status
}
