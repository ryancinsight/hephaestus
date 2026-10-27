//! The differential harness: run a rule on the host oracle and on a
//! backend, then compare the two within the rule's rounding budget.

use eunomia::{FloatElement, RealField};
use hephaestus_core::{
    ComputeDevice, StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, StridedView,
};
use leto::Layout;

use super::oracle::{CpuRule, HostOperands};
use super::width::StatefulUpdateWidth;

/// The number of repeated steps every differential run performs.
///
/// One step is not enough to catch a provider that mis-orders its own
/// accumulators: the second step reads back the state the first step
/// wrote, so a rule that drops or double-counts the update shows up.
pub(super) const REPEATED_STEPS: usize = 2;

/// Run the host oracle for [`REPEATED_STEPS`] steps.
pub(super) fn run_cpu<Rule: CpuRule<T>, T: FloatElement>(
    parameters: Rule::Parameters,
) -> HostOperands<T> {
    let mut operands = HostOperands::seeded();
    for _ in 0..REPEATED_STEPS {
        Rule::step(&mut operands, parameters);
    }
    operands
}

/// Run the backend for [`REPEATED_STEPS`] steps over the same seeded
/// fixture, returning the resulting operands.
pub(super) fn run_backend<D, O, Rule, T>(
    device: &D,
    operations: &O,
    parameters: <Rule as StatefulUpdateRule<O::Dialect>>::Parameters<T>,
    state_count: usize,
) -> HostOperands<T>
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Rule: StatefulUpdateRule<O::Dialect>,
    T: StatefulUpdateWidth,
{
    let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
    let seeded = HostOperands::seeded();
    let parameter = device.upload(&seeded.parameter).expect("parameter upload");
    let gradient = device.upload(&seeded.gradient).expect("gradient upload");
    let state_zero = device
        .upload(&seeded.state_zero)
        .expect("state-zero upload");
    let state_one = device.upload(&seeded.state_one).expect("state-one upload");
    let one_state = [StridedView::new(&state_zero, &layout)];
    let two_states = [
        StridedView::new(&state_zero, &layout),
        StridedView::new(&state_one, &layout),
    ];
    let states = if state_count == 1 {
        &one_state[..]
    } else {
        &two_states[..]
    };
    for _ in 0..REPEATED_STEPS {
        T::dispatch::<D, O, Rule, 2>(
            device,
            operations,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states,
            },
            parameters,
        )
        .expect("stateful update dispatch");
    }
    let mut actual = seeded;
    device
        .download(&parameter, &mut actual.parameter)
        .expect("parameter download");
    device
        .download(&gradient, &mut actual.gradient)
        .expect("gradient download");
    device
        .download(&state_zero, &mut actual.state_zero)
        .expect("state-zero download");
    device
        .download(&state_one, &mut actual.state_one)
        .expect("state-one download");
    actual
}

/// Assert two operand vectors agree within the rule's rounding budget.
///
/// Each provider rule has at most 24 rounded operations per step per
/// width. Two repeated steps give a forward bound of 48 epsilon; 64
/// epsilon includes final comparison rounding without masking formula
/// defects. The bound scales with the width's own epsilon.
pub(super) fn assert_close<T: RealField>(name: &str, actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let magnitude = expected.abs();
        let scale = if magnitude > T::ONE {
            magnitude
        } else {
            T::ONE
        };
        let tolerance = T::from_f32(64.0) * T::EPSILON * scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}[{index}]: got {actual:?}, expected {expected:?}, tolerance {tolerance:?}"
        );
    }
}

/// Compare one provider rule against its host oracle.
///
/// The gradient is compared exactly — no rule mutates it — while the
/// parameter and the state vectors are compared within the rounding
/// budget.
pub(super) fn assert_rule<D, O, BackendRule, HostRule, T>(
    device: &D,
    operations: &O,
    backend_parameters: <BackendRule as StatefulUpdateRule<O::Dialect>>::Parameters<T>,
    host_parameters: HostRule::Parameters,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    BackendRule: StatefulUpdateRule<O::Dialect>,
    HostRule: CpuRule<T>,
    T: StatefulUpdateWidth,
{
    let name = device.backend_name();
    let actual = run_backend::<D, O, BackendRule, T>(
        device,
        operations,
        backend_parameters,
        HostRule::STATE_COUNT,
    );
    let expected = run_cpu::<HostRule, T>(host_parameters);
    assert_close(
        &format!("{name} parameter"),
        &actual.parameter,
        &expected.parameter,
    );
    assert_eq!(
        actual.gradient, expected.gradient,
        "{name} gradient changed"
    );
    assert_close(
        &format!("{name} state zero"),
        &actual.state_zero,
        &expected.state_zero,
    );
    if HostRule::STATE_COUNT == 2 {
        assert_close(
            &format!("{name} state one"),
            &actual.state_one,
            &expected.state_one,
        );
    }
}
