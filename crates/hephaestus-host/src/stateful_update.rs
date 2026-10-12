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
fn aliasing<T, const N: usize>(
    operands: &StatefulUpdateOperands<'_, HostBuffer<T>, N>,
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

/// Shared elementwise driver: both widths run the same loop over the
/// planner-validated views. The `update` closure adapts one width's rule
/// value function to in-place slot mutation; one-state rules receive
/// `None` for the second slot and the closure substitutes that width's
/// zero for the carried-through unused value.
fn run_update<T, P, const N: usize>(
    operands: StatefulUpdateOperands<'_, HostBuffer<T>, N>,
    parameters: &P,
    update: impl Fn(&mut T, T, &mut T, Option<&mut T>, &P),
) -> Result<()>
where
    T: Copy,
{
    let gradient_cells = operands.gradient.buffer.read();
    let gradient_view =
        ArrayView::try_new(*operands.gradient.layout, &gradient_cells).map_err(map_layout_err)?;

    let mut parameter_cells = operands.parameter.buffer.write();
    let parameter_view = ArrayViewMut::try_new(*operands.parameter.layout, &mut parameter_cells)
        .map_err(map_layout_err)?;
    let mut parameter_iter = parameter_view.try_iter_mut().map_err(map_layout_err)?;

    let state_zero = operands
        .states
        .first()
        .expect("invariant: planner validated state zero");
    let mut state_zero_cells = state_zero.buffer.write();
    let state_zero_view =
        ArrayViewMut::try_new(*state_zero.layout, &mut state_zero_cells).map_err(map_layout_err)?;
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
            update(
                parameter_slot,
                gradient,
                state_zero_slot,
                Some(state_one_slot),
                parameters,
            );
        }
    } else {
        for &gradient in &gradient_view {
            let parameter_slot = parameter_iter
                .next()
                .expect("invariant: plan validated matching operand shapes");
            let state_zero_slot = state_zero_iter
                .next()
                .expect("invariant: plan validated matching operand shapes");
            update(parameter_slot, gradient, state_zero_slot, None, parameters);
        }
    }
    Ok(())
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
        run_update(
            operands,
            &parameters,
            |parameter, gradient, state_zero, state_one, parameters| {
                let states = [*state_zero, state_one.as_deref().copied().unwrap_or(0.0)];
                let step = Rule::update(*parameter, gradient, states, parameters);
                *parameter = step.parameter;
                *state_zero = step.states[0];
                if let Some(slot) = state_one {
                    *slot = step.states[1];
                }
            },
        )
    }

    fn stateful_update_f64<Rule, const N: usize>(
        &self,
        _device: &HostDevice,
        operands: StatefulUpdateOperands<'_, HostBuffer<f64>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::ParametersF64,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        Rule::validate_parameters_f64(&parameters)?;
        let plan = plan_stateful_update(operands, Rule::STATE_COUNT, aliasing(&operands))?;
        if plan.is_empty() {
            return Ok(());
        }
        run_update(
            operands,
            &parameters,
            |parameter, gradient, state_zero, state_one, parameters| {
                let states = [*state_zero, state_one.as_deref().copied().unwrap_or(0.0)];
                let step = Rule::update_f64(*parameter, gradient, states, parameters);
                *parameter = step.parameter;
                *state_zero = step.states[0];
                if let Some(slot) = state_one {
                    *slot = step.states[1];
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::HostStatefulUpdateOps;
    use eunomia::Zeroable;
    use hephaestus_core::{
        Adam, AdamParametersF64, ComputeDevice, Host, Sgd, SgdParameters, SgdParametersF64,
        StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, StridedView,
    };
    use leto::Layout;

    #[test]
    fn host_sgd_f64_matches_hand_derived_arithmetic() {
        // lr = 0.25, momentum = 0.5: every intermediate is binary-exact.
        // elem0: s = 0*0.5+2 = 2.0, p = 10-0.25*2 = 9.5.
        // elem1: s = 1*0.5+1 = 1.5, p = 4-0.25*1.5 = 3.625.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter = device.upload(&[10.0_f64, 4.0]).expect("upload succeeds");
        let gradient = device.upload(&[2.0_f64, 1.0]).expect("upload succeeds");
        let state = device.upload(&[0.0_f64, 1.0]).expect("upload succeeds");
        let states = [StridedView::new(&state, &layout)];
        let parameters = SgdParametersF64::new(0.25, 0.5).expect("valid f64 SGD parameters");

        HostStatefulUpdateOps
            .stateful_update_f64::<Sgd, 1>(
                &device,
                StatefulUpdateOperands {
                    parameter: StridedView::new(&parameter, &layout),
                    gradient: StridedView::new(&gradient, &layout),
                    states: &states,
                },
                parameters,
            )
            .expect("host f64 SGD succeeds");

        let mut actual_parameter = [0.0_f64; 2];
        let mut actual_state = [0.0_f64; 2];
        device
            .download(&parameter, &mut actual_parameter)
            .expect("download succeeds");
        device
            .download(&state, &mut actual_state)
            .expect("download succeeds");
        assert_eq!(actual_parameter, [9.5, 3.625]);
        assert_eq!(actual_state, [2.0, 1.5]);
    }

    #[test]
    fn host_adam_f64_dispatch_equals_the_value_definition() {
        // Two-state path: the seam must equal `update_f64` elementwise.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter = device.upload(&[1.0_f64, -2.0]).expect("upload succeeds");
        let gradient = device.upload(&[2.0_f64, 0.5]).expect("upload succeeds");
        let state_zero = device.upload(&[0.5_f64, 0.25]).expect("upload succeeds");
        let state_one = device.upload(&[0.25_f64, 0.125]).expect("upload succeeds");
        let states = [
            StridedView::new(&state_zero, &layout),
            StridedView::new(&state_one, &layout),
        ];
        let parameters =
            AdamParametersF64::new(0.1, 0.9, 0.999, 1.0e-6, 3).expect("valid f64 Adam parameters");

        HostStatefulUpdateOps
            .stateful_update_f64::<Adam, 1>(
                &device,
                StatefulUpdateOperands {
                    parameter: StridedView::new(&parameter, &layout),
                    gradient: StridedView::new(&gradient, &layout),
                    states: &states,
                },
                parameters,
            )
            .expect("host f64 Adam succeeds");

        for (index, ((&p, &g), (&s0, &s1))) in [1.0_f64, -2.0]
            .iter()
            .zip([2.0_f64, 0.5].iter())
            .zip([0.5_f64, 0.25].iter().zip([0.25_f64, 0.125].iter()))
            .enumerate()
        {
            let expected =
                <Adam as StatefulUpdateRule<Host>>::update_f64(p, g, [s0, s1], &parameters);
            let mut actual_parameter = [0.0_f64; 2];
            let mut actual_zero = [0.0_f64; 2];
            let mut actual_one = [0.0_f64; 2];
            device
                .download(&parameter, &mut actual_parameter)
                .expect("download succeeds");
            device
                .download(&state_zero, &mut actual_zero)
                .expect("download succeeds");
            device
                .download(&state_one, &mut actual_one)
                .expect("download succeeds");
            assert_eq!(actual_parameter[index], expected.parameter);
            assert_eq!(actual_zero[index], expected.states[0]);
            assert_eq!(actual_one[index], expected.states[1]);
        }
    }

    #[test]
    fn host_sgd_f32_still_matches_after_the_shared_runner_refactor() {
        // The f32 path now runs through the shared `run_update` driver;
        // pin its arithmetic so the refactor cannot drift it.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter = device.upload(&[10.0_f32, 4.0]).expect("upload succeeds");
        let gradient = device.upload(&[2.0_f32, 1.0]).expect("upload succeeds");
        let state = device.upload(&[0.0_f32, 1.0]).expect("upload succeeds");
        let states = [StridedView::new(&state, &layout)];
        let parameters = SgdParameters::new(0.25, 0.5).expect("valid SGD parameters");

        HostStatefulUpdateOps
            .stateful_update::<Sgd, 1>(
                &device,
                StatefulUpdateOperands {
                    parameter: StridedView::new(&parameter, &layout),
                    gradient: StridedView::new(&gradient, &layout),
                    states: &states,
                },
                parameters,
            )
            .expect("host f32 SGD succeeds");

        let mut actual_parameter = [0.0_f32; 2];
        let mut actual_state = [0.0_f32; 2];
        device
            .download(&parameter, &mut actual_parameter)
            .expect("download succeeds");
        device
            .download(&state, &mut actual_state)
            .expect("download succeeds");
        assert_eq!(actual_parameter, [9.5, 3.625]);
        assert_eq!(actual_state, [2.0, 1.5]);
    }

    #[test]
    fn host_f64_rejects_invalid_parameters_before_mutation() {
        // A zeroed Adam block carries epsilon = 0, which validation must
        // reject before any buffer is touched.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter = device.upload(&[10.0_f64, 4.0]).expect("upload succeeds");
        let gradient = device.upload(&[2.0_f64, 1.0]).expect("upload succeeds");
        let state_zero = device.upload(&[0.0_f64, 1.0]).expect("upload succeeds");
        let state_one = device.upload(&[0.0_f64, 1.0]).expect("upload succeeds");
        let states = [
            StridedView::new(&state_zero, &layout),
            StridedView::new(&state_one, &layout),
        ];
        let parameters = AdamParametersF64::zeroed();

        let result = HostStatefulUpdateOps.stateful_update_f64::<Adam, 1>(
            &device,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states: &states,
            },
            parameters,
        );
        assert!(result.is_err(), "zeroed Adam block must be rejected");

        let mut actual_parameter = [0.0_f64; 2];
        device
            .download(&parameter, &mut actual_parameter)
            .expect("download succeeds");
        assert_eq!(
            actual_parameter,
            [10.0, 4.0],
            "rejected dispatch must not mutate"
        );
    }
}
