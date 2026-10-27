//! The stateful-update clauses: the rank-boundary layout sweep, the
//! rejection-atomicity clause, and the aggregate entry point.

use hephaestus_core::{
    AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW, AdamWParameters, ComputeDevice,
    HephaestusError, RmsProp, RmsPropParameters, Sgd, SgdParameters, StatefulUpdateOperands,
    StatefulUpdateOps, StatefulUpdateRule, StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{
    AdaGrad as CpuAdaGrad, AdaGradParameters as CpuAdaGradParameters, Adam as CpuAdam,
    AdamParameters as CpuAdamParameters, AdamW as CpuAdamW, AdamWParameters as CpuAdamWParameters,
    RmsProp as CpuRmsProp, RmsPropParameters as CpuRmsPropParameters, Sgd as CpuSgd,
    SgdParameters as CpuSgdParameters, stateful_update as cpu_update,
};

use super::harness::{assert_close, assert_rule};
use super::width::StatefulUpdateWidth;

/// Width-matched SGD parameters for the boundary clause: the backend twin
/// plus the Leto oracle parameters. One argument instead of two so the
/// boundary clause stays under the argument-count lint.
struct SgdParameterPair<T, P> {
    backend: P,
    host: CpuSgdParameters<T>,
}

impl<T, P> SgdParameterPair<T, P> {
    #[must_use]
    fn new(backend: P, host: CpuSgdParameters<T>) -> Self {
        Self { backend, host }
    }
}

/// Assert one SGD step over an arbitrary layout agrees with the oracle.
///
/// The clause takes the layout as a parameter so the rank boundary
/// sweep can run it at rank zero and at rank eight — the widest rank
/// the seam admits — without duplicating the body.
fn assert_sgd_layout<D, O, T, const N: usize>(
    device: &D,
    operations: &O,
    layout: Layout<N>,
    parameter_initial: &[T],
    gradient_initial: &[T],
    state_initial: &[T],
    parameters: SgdParameterPair<T, <Sgd as StatefulUpdateRule<O::Dialect>>::Parameters<T>>,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect>,
    T: StatefulUpdateWidth,
{
    let mut expected_parameter = parameter_initial.to_vec();
    let mut expected_state = state_initial.to_vec();
    cpu_update::<T, CpuSgd, N>(
        ArrayViewMut::new(layout, &mut expected_parameter),
        ArrayView::new(layout, gradient_initial),
        ArrayViewMut::new(layout, &mut expected_state),
        parameters.host,
    )
    .expect("Leto boundary oracle");

    let parameter = device.upload(parameter_initial).expect("parameter upload");
    let gradient = device.upload(gradient_initial).expect("gradient upload");
    let state = device.upload(state_initial).expect("state upload");
    let states = [StridedView::new(&state, &layout)];
    T::dispatch::<D, O, Sgd, N>(
        device,
        operations,
        StatefulUpdateOperands {
            parameter: StridedView::new(&parameter, &layout),
            gradient: StridedView::new(&gradient, &layout),
            states: &states,
        },
        parameters.backend,
    )
    .expect("boundary dispatch");
    let mut actual_parameter = vec![T::ZERO; parameter_initial.len()];
    let mut actual_gradient = vec![T::ZERO; gradient_initial.len()];
    let mut actual_state = vec![T::ZERO; state_initial.len()];
    device
        .download(&parameter, &mut actual_parameter)
        .expect("parameter download");
    device
        .download(&gradient, &mut actual_gradient)
        .expect("gradient download");
    device
        .download(&state, &mut actual_state)
        .expect("state download");
    assert_eq!(actual_gradient, gradient_initial, "gradient changed");
    assert_close(
        &format!("{} boundary parameter", device.backend_name()),
        &actual_parameter,
        &expected_parameter,
    );
    assert_close(
        &format!("{} boundary state", device.backend_name()),
        &actual_state,
        &expected_state,
    );
}

/// A rejected dispatch must leave every operand untouched.
///
/// Two rejections are exercised: a state whose shape disagrees with the
/// parameter, and a state that aliases the parameter's storage. Both
/// must fail typed, and neither may have mutated the parameter or the
/// state before failing.
fn assert_rejections_are_atomic<D, O, T>(
    device: &D,
    operations: &O,
    backend_parameters: <Sgd as StatefulUpdateRule<O::Dialect>>::Parameters<T>,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect>,
    T: StatefulUpdateWidth,
{
    let layout = Layout::c_contiguous([2]).expect("layout");
    let state_layout = Layout::c_contiguous([1]).expect("state layout");
    let parameter_initial = [T::from_f32(1.0), T::from_f32(2.0)];
    let state_initial = [T::from_f32(0.5), T::from_f32(0.6)];
    let parameter = device.upload(&parameter_initial).expect("parameter upload");
    let gradient = device
        .upload(&[T::from_f32(0.1), T::from_f32(0.2)])
        .expect("gradient upload");
    let state = device.upload(&state_initial).expect("state upload");
    let states = [StridedView::new(&state, &state_layout)];
    let error = T::dispatch::<D, O, Sgd, 1>(
        device,
        operations,
        StatefulUpdateOperands {
            parameter: StridedView::new(&parameter, &layout),
            gradient: StridedView::new(&gradient, &layout),
            states: &states,
        },
        backend_parameters,
    )
    .expect_err("shape mismatch must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{}: expected shape dispatch failure, got {error}",
        device.backend_name()
    );
    let mut parameter_after = [T::ZERO; 2];
    let mut state_after = [T::ZERO; 2];
    device
        .download(&parameter, &mut parameter_after)
        .expect("parameter download");
    device
        .download(&state, &mut state_after)
        .expect("state download");
    assert_eq!(parameter_after, parameter_initial);
    assert_eq!(state_after, state_initial);

    let aliased_states = [StridedView::new(&parameter, &layout)];
    let error = T::dispatch::<D, O, Sgd, 1>(
        device,
        operations,
        StatefulUpdateOperands {
            parameter: StridedView::new(&parameter, &layout),
            gradient: StridedView::new(&gradient, &layout),
            states: &aliased_states,
        },
        backend_parameters,
    )
    .expect_err("aliased state must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{}: expected alias dispatch failure, got {error}",
        device.backend_name()
    );
}

/// Run all stateful-update value and rejection clauses against one backend.
///
/// Fixture scalars lift through [`FloatElement::from_f32`](eunomia::FloatElement::from_f32),
/// so f32 clauses run bit-identical values while f64 clauses widen the same
/// f32-rounded fixtures; backend and Leto oracle always share one parameter
/// value.
///
/// # Panics
///
/// Panics with the backend and violated differential, repeated-dispatch,
/// striding, guard-storage, rank-boundary, alias, or failure-atomicity clause
/// when the provider diverges from the Leto CPU contract.
pub fn assert_stateful_update_contract<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    T: StatefulUpdateWidth,
    Sgd: StatefulUpdateRule<O::Dialect, Parameters<T> = SgdParameters<T>>,
    Adam: StatefulUpdateRule<O::Dialect, Parameters<T> = AdamParameters<T>>,
    AdamW: StatefulUpdateRule<O::Dialect, Parameters<T> = AdamWParameters<T>>,
    RmsProp: StatefulUpdateRule<O::Dialect, Parameters<T> = RmsPropParameters<T>>,
    AdaGrad: StatefulUpdateRule<O::Dialect, Parameters<T> = AdaGradParameters<T>>,
{
    assert_rule::<D, O, Sgd, CpuSgd, T>(
        device,
        operations,
        SgdParameters::<T>::new(T::from_f32(0.05), T::from_f32(0.9)).expect("SGD parameters"),
        CpuSgdParameters::<T>::new(T::from_f32(0.05), T::from_f32(0.9))
            .expect("Leto SGD parameters"),
    );
    assert_rule::<D, O, Adam, CpuAdam, T>(
        device,
        operations,
        AdamParameters::<T>::new(
            T::from_f32(0.01),
            T::from_f32(0.9),
            T::from_f32(0.99),
            T::from_f32(1.0e-6),
            3,
        )
        .expect("Adam parameters"),
        CpuAdamParameters::<T>::new(
            T::from_f32(0.01),
            T::from_f32(0.9),
            T::from_f32(0.99),
            T::from_f32(1.0e-6),
            3,
        )
        .expect("Leto Adam parameters"),
    );
    assert_rule::<D, O, AdamW, CpuAdamW, T>(
        device,
        operations,
        AdamWParameters::<T>::new(
            T::from_f32(0.01),
            T::from_f32(0.9),
            T::from_f32(0.99),
            T::from_f32(1.0e-6),
            T::from_f32(0.1),
            3,
        )
        .expect("AdamW parameters"),
        CpuAdamWParameters::<T>::new(
            T::from_f32(0.01),
            T::from_f32(0.9),
            T::from_f32(0.99),
            T::from_f32(1.0e-6),
            T::from_f32(0.1),
            3,
        )
        .expect("Leto AdamW parameters"),
    );
    assert_rule::<D, O, RmsProp, CpuRmsProp, T>(
        device,
        operations,
        RmsPropParameters::<T>::new(T::from_f32(0.05), T::from_f32(0.9), T::from_f32(1.0e-6))
            .expect("RMSProp parameters"),
        CpuRmsPropParameters::<T>::new(T::from_f32(0.05), T::from_f32(0.9), T::from_f32(1.0e-6))
            .expect("Leto RMSProp parameters"),
    );
    assert_rule::<D, O, AdaGrad, CpuAdaGrad, T>(
        device,
        operations,
        AdaGradParameters::<T>::new(T::from_f32(0.05), T::from_f32(1.0e-6))
            .expect("AdaGrad parameters"),
        CpuAdaGradParameters::<T>::new(T::from_f32(0.05), T::from_f32(1.0e-6))
            .expect("Leto AdaGrad parameters"),
    );

    assert_sgd_layout::<D, O, T, 0>(
        device,
        operations,
        Layout::c_contiguous([]).expect("scalar layout"),
        &[T::from_f32(2.0)],
        &[T::from_f32(0.5)],
        &[T::from_f32(0.25)],
        SgdParameterPair::new(
            SgdParameters::<T>::new(T::from_f32(0.1), T::from_f32(0.5)).expect("SGD parameters"),
            CpuSgdParameters::<T>::new(T::from_f32(0.1), T::from_f32(0.5))
                .expect("Leto SGD parameters"),
        ),
    );
    assert_sgd_layout::<D, O, T, 8>(
        device,
        operations,
        Layout::c_contiguous([1, 1, 1, 0, 1, 1, 1, 1]).expect("empty rank-eight layout"),
        &[T::from_f32(9.0)],
        &[T::from_f32(8.0)],
        &[T::from_f32(7.0)],
        SgdParameterPair::new(
            SgdParameters::<T>::new(T::from_f32(0.1), T::from_f32(0.5)).expect("SGD parameters"),
            CpuSgdParameters::<T>::new(T::from_f32(0.1), T::from_f32(0.5))
                .expect("Leto SGD parameters"),
        ),
    );
    assert_rejections_are_atomic::<D, O, T>(
        device,
        operations,
        SgdParameters::<T>::new(T::from_f32(0.1), T::from_f32(0.0)).expect("SGD parameters"),
    );
}
