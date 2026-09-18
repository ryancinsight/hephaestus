//! Stateful parameter updates for the host reference device (ADR 0061).
//!
//! [`HostStatefulUpdateOps`] validates through the same device-neutral
//! [`plan_stateful_update`](hephaestus_core::plan_stateful_update) every
//! accelerator backend calls (shape, rank-eight bound, storage span, and
//! pairwise operand aliasing), then applies
//! [`StatefulUpdateRule::update`](hephaestus_core::StatefulUpdateRule::update)
//! elementwise. Unlike the elementwise and combine seams, no
//! `unsupported_operator` fallback exists here: `StatefulUpdateRule` is
//! sealed (ADR 0061 Decision 1), so `update` is a required method every
//! implementor already carries for every dialect, including
//! [`Host`](hephaestus_core::Host).
//!
//! # Aliasing
//!
//! [`plan_stateful_update`](hephaestus_core::plan_stateful_update) rejects
//! the operation unless the parameter,
//! gradient, and both persistent-state buffers are pairwise distinct
//! (`StatefulUpdateAliasing::any`), so by the time validation succeeds every
//! [`HostBuffer`] touched below is backed by a different lock — one `write`
//! or `read` guard is taken per buffer, never a second guard on a lock this
//! thread already holds.

use hephaestus_core::{
    Host, Result, StatefulUpdateAliasing, StatefulUpdateOperands, StatefulUpdateOps,
    StatefulUpdateRule, plan_stateful_update,
};
use leto::{ArrayView, ArrayViewMut};

use crate::elementwise::map_layout_err;
use crate::{HostBuffer, HostDevice};

/// Pairwise storage-alias facts among a stateful update's operands, mirroring
/// `hephaestus-wgpu`'s `aliasing`.
fn aliasing<const N: usize>(
    operands: &StatefulUpdateOperands<'_, HostBuffer<f32>, N>,
) -> StatefulUpdateAliasing {
    let state_zero = operands.states.first();
    let state_one = operands.states.get(1);
    StatefulUpdateAliasing {
        parameter_gradient: operands.parameter.buffer.aliases(operands.gradient.buffer),
        parameter_state_zero: state_zero
            .is_some_and(|state| operands.parameter.buffer.aliases(state.buffer)),
        parameter_state_one: state_one
            .is_some_and(|state| operands.parameter.buffer.aliases(state.buffer)),
        gradient_state_zero: state_zero
            .is_some_and(|state| operands.gradient.buffer.aliases(state.buffer)),
        gradient_state_one: state_one
            .is_some_and(|state| operands.gradient.buffer.aliases(state.buffer)),
        states: state_zero
            .zip(state_one)
            .is_some_and(|(left, right)| left.buffer.aliases(right.buffer)),
    }
}

/// Stateful parameter updates for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostStatefulUpdateOps;

impl StatefulUpdateOps<HostDevice> for HostStatefulUpdateOps {
    type Dialect = Host;

    fn stateful_update<Rule, const N: usize>(
        &self,
        _device: &HostDevice,
        operands: StatefulUpdateOperands<'_, HostBuffer<f32>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        Rule::validate_parameters(&parameters)?;
        let plan = plan_stateful_update(operands, Rule::STATE_COUNT, aliasing(&operands))?;
        if plan.is_empty() {
            return Ok(());
        }

        let gradient_cells = operands.gradient.buffer.read();
        let gradient_view = ArrayView::try_new(*operands.gradient.layout, &gradient_cells)
            .map_err(map_layout_err)?;

        let mut parameter_cells = operands.parameter.buffer.write();
        let parameter_view =
            ArrayViewMut::try_new(*operands.parameter.layout, &mut parameter_cells)
                .map_err(map_layout_err)?;
        let mut parameter_iter = parameter_view.try_iter_mut().map_err(map_layout_err)?;

        let state_zero = operands
            .states
            .first()
            .expect("invariant: planner validated state zero");
        let mut state_zero_cells = state_zero.buffer.write();
        let state_zero_view = ArrayViewMut::try_new(*state_zero.layout, &mut state_zero_cells)
            .map_err(map_layout_err)?;
        let mut state_zero_iter = state_zero_view.try_iter_mut().map_err(map_layout_err)?;

        if let Some(state_one) = operands.states.get(1) {
            let mut state_one_cells = state_one.buffer.write();
            let state_one_view = ArrayViewMut::try_new(*state_one.layout, &mut state_one_cells)
                .map_err(map_layout_err)?;
            let mut state_one_iter = state_one_view.try_iter_mut().map_err(map_layout_err)?;

            for &gradient in &gradient_view {
                let parameter_slot = parameter_iter
                    .next()
                    .expect("invariant: plan validated matching operand shapes");
                let state_zero_slot = state_zero_iter
                    .next()
                    .expect("invariant: plan validated matching operand shapes");
                let state_one_slot = state_one_iter
                    .next()
                    .expect("invariant: plan validated matching operand shapes");
                let step = Rule::update(
                    *parameter_slot,
                    gradient,
                    [*state_zero_slot, *state_one_slot],
                    &parameters,
                );
                *parameter_slot = step.parameter;
                *state_zero_slot = step.states[0];
                *state_one_slot = step.states[1];
            }
        } else {
            for &gradient in &gradient_view {
                let parameter_slot = parameter_iter
                    .next()
                    .expect("invariant: plan validated matching operand shapes");
                let state_zero_slot = state_zero_iter
                    .next()
                    .expect("invariant: plan validated matching operand shapes");
                let step = Rule::update(
                    *parameter_slot,
                    gradient,
                    [*state_zero_slot, 0.0],
                    &parameters,
                );
                *parameter_slot = step.parameter;
                *state_zero_slot = step.states[0];
            }
        }
        Ok(())
    }
}
