//! The prepared resources and the [`AttentionOps`] implementation.

use hephaestus_core::{
    AttentionBackwardOperands, AttentionCausality, AttentionForwardOperands, AttentionOps,
    AttentionPlan, AttentionSemanticStatus, Result, plan_attention_backward,
    plan_attention_forward,
};
use leto::{ArrayView, ArrayViewMut};
use leto_ops::{
    AttentionGradients as LetoGradients, AttentionMask as LetoMask,
    GroupedKeepMask as LetoGroupedKeepMask,
};

use super::backward::{backward_aliases, backward_preflight_status};
use super::forward::{forward_aliases, forward_preflight_status};
use super::read::finalize_status;
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
