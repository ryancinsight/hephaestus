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
    Host, Result, StatefulScalar, StatefulUpdateAliasing, StatefulUpdateOperands,
    StatefulUpdateOps, StatefulUpdateRule, plan_stateful_update,
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

    fn stateful_update<T, Rule, const N: usize>(
        &self,
        _device: &HostDevice,
        operands: StatefulUpdateOperands<'_, HostBuffer<T>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters<T>,
    ) -> Result<()>
    where
        T: StatefulScalar,
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
                let states = [
                    *state_zero,
                    state_one.as_deref().copied().unwrap_or(T::ZERO),
                ];
                let step = Rule::update(*parameter, gradient, states, parameters);
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
        Adam, AdamParameters, ComputeDevice, HephaestusError, Host, Sgd, SgdParameters,
        StatefulScalar, StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, StridedView,
    };
    use leto::Layout;

    fn sgd_matches_hand_derived_arithmetic<T: StatefulScalar>() {
        // lr = 0.25, momentum = 0.5: every intermediate is binary-exact.
        // elem0: s = 0*0.5+2 = 2.0, p = 10-0.25*2 = 9.5.
        // elem1: s = 1*0.5+1 = 1.5, p = 4-0.25*1.5 = 3.625.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter = device
            .upload(&[T::from_f64(10.0), T::from_f64(4.0)])
            .expect("upload succeeds");
        let gradient = device
            .upload(&[T::from_f64(2.0), T::from_f64(1.0)])
            .expect("upload succeeds");
        let state = device.upload(&[T::ZERO, T::ONE]).expect("upload succeeds");
        let states = [StridedView::new(&state, &layout)];
        let parameters =
            SgdParameters::new(T::from_f64(0.25), T::from_f64(0.5)).expect("valid SGD parameters");

        HostStatefulUpdateOps
            .stateful_update::<T, Sgd, 1>(
                &device,
                StatefulUpdateOperands {
                    parameter: StridedView::new(&parameter, &layout),
                    gradient: StridedView::new(&gradient, &layout),
                    states: &states,
                },
                parameters,
            )
            .expect("host SGD succeeds");

        let mut actual_parameter = [T::ZERO; 2];
        let mut actual_state = [T::ZERO; 2];
        device
            .download(&parameter, &mut actual_parameter)
            .expect("download succeeds");
        device
            .download(&state, &mut actual_state)
            .expect("download succeeds");
        assert_eq!(actual_parameter, [T::from_f64(9.5), T::from_f64(3.625)]);
        assert_eq!(actual_state, [T::from_f64(2.0), T::from_f64(1.5)]);
    }

    fn adam_dispatch_equals_value_definition<T: StatefulScalar>() {
        // Two-state path: the seam must equal `update` elementwise.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter_initial = [T::ONE, T::from_f64(-2.0)];
        let gradient_initial = [T::from_f64(2.0), T::from_f64(0.5)];
        let state_zero_initial = [T::from_f64(0.5), T::from_f64(0.25)];
        let state_one_initial = [T::from_f64(0.25), T::from_f64(0.125)];
        let parameter = device.upload(&parameter_initial).expect("upload succeeds");
        let gradient = device.upload(&gradient_initial).expect("upload succeeds");
        let state_zero = device.upload(&state_zero_initial).expect("upload succeeds");
        let state_one = device.upload(&state_one_initial).expect("upload succeeds");
        let states = [
            StridedView::new(&state_zero, &layout),
            StridedView::new(&state_one, &layout),
        ];
        let parameters = AdamParameters::new(
            T::from_f64(0.1),
            T::from_f64(0.9),
            T::from_f64(0.999),
            T::from_f64(1.0e-6),
            3,
        )
        .expect("valid Adam parameters");

        HostStatefulUpdateOps
            .stateful_update::<T, Adam, 1>(
                &device,
                StatefulUpdateOperands {
                    parameter: StridedView::new(&parameter, &layout),
                    gradient: StridedView::new(&gradient, &layout),
                    states: &states,
                },
                parameters,
            )
            .expect("host Adam succeeds");

        let mut actual_parameter = [T::ZERO; 2];
        let mut actual_zero = [T::ZERO; 2];
        let mut actual_one = [T::ZERO; 2];
        device
            .download(&parameter, &mut actual_parameter)
            .expect("download succeeds");
        device
            .download(&state_zero, &mut actual_zero)
            .expect("download succeeds");
        device
            .download(&state_one, &mut actual_one)
            .expect("download succeeds");
        for index in 0..2 {
            let expected = <Adam as StatefulUpdateRule<Host>>::update(
                parameter_initial[index],
                gradient_initial[index],
                [state_zero_initial[index], state_one_initial[index]],
                &parameters,
            );
            assert_eq!(actual_parameter[index], expected.parameter);
            assert_eq!(actual_zero[index], expected.states[0]);
            assert_eq!(actual_one[index], expected.states[1]);
        }
    }

    fn rejects_invalid_parameters_before_mutation<T: StatefulScalar>() {
        // A zeroed Adam block carries epsilon = 0, which validation must
        // reject before any buffer is touched.
        let device = super::HostDevice::new();
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let initial = [T::from_f64(10.0), T::from_f64(4.0)];
        let parameter = device.upload(&initial).expect("upload succeeds");
        let gradient = device
            .upload(&[T::from_f64(2.0), T::ONE])
            .expect("upload succeeds");
        let state_zero = device.upload(&[T::ZERO, T::ONE]).expect("upload succeeds");
        let state_one = device.upload(&[T::ZERO, T::ONE]).expect("upload succeeds");
        let states = [
            StridedView::new(&state_zero, &layout),
            StridedView::new(&state_one, &layout),
        ];
        let parameters = AdamParameters::<T>::zeroed();

        let result = HostStatefulUpdateOps.stateful_update::<T, Adam, 1>(
            &device,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states: &states,
            },
            parameters,
        );
        let Err(HephaestusError::InvalidConfiguration { message }) = result else {
            panic!("zeroed Adam block was accepted");
        };
        assert_eq!(
            message,
            "Adam beta values must lie in [0, 1), epsilon and bias corrections must be positive"
        );

        let mut actual_parameter = [T::ZERO; 2];
        device
            .download(&parameter, &mut actual_parameter)
            .expect("download succeeds");
        assert_eq!(
            actual_parameter, initial,
            "rejected dispatch must not mutate"
        );
    }

    #[test]
    fn host_sgd_matches_hand_derived_arithmetic_at_both_widths() {
        sgd_matches_hand_derived_arithmetic::<f32>();
        sgd_matches_hand_derived_arithmetic::<f64>();
    }

    #[test]
    fn host_adam_dispatch_equals_the_value_definition_at_both_widths() {
        adam_dispatch_equals_value_definition::<f32>();
        adam_dispatch_equals_value_definition::<f64>();
    }

    #[test]
    fn host_rejects_invalid_parameters_before_mutation_at_both_widths() {
        rejects_invalid_parameters_before_mutation::<f32>();
        rejects_invalid_parameters_before_mutation::<f64>();
    }
}
