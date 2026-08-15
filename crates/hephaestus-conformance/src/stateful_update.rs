//! Contract clauses for provider-owned stateful parameter updates.

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

const REPEATED_STEPS: usize = 2;

#[derive(Clone)]
struct HostOperands {
    parameter: [f32; 6],
    gradient: [f32; 6],
    state_zero: [f32; 6],
    state_one: [f32; 6],
}

impl HostOperands {
    fn seeded() -> Self {
        Self {
            parameter: [91.0, 1.0, 2.0, 92.0, 3.0, 4.0],
            gradient: [81.0, 0.1, 0.2, 82.0, 0.3, 0.4],
            state_zero: [71.0, 0.5, 0.6, 72.0, 0.7, 0.8],
            state_one: [61.0, 0.25, 0.36, 62.0, 0.49, 0.64],
        }
    }
}

trait CpuRule {
    type Parameters: Copy;
    const STATE_COUNT: usize;

    fn step(operands: &mut HostOperands, parameters: Self::Parameters);
}

impl CpuRule for CpuSgd {
    type Parameters = CpuSgdParameters<f32>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<f32, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto SGD oracle");
    }
}

impl CpuRule for CpuAdam {
    type Parameters = CpuAdamParameters<f32>;
    const STATE_COUNT: usize = 2;

    fn step(operands: &mut HostOperands, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<f32, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            (
                ArrayViewMut::new(layout, &mut operands.state_zero),
                ArrayViewMut::new(layout, &mut operands.state_one),
            ),
            parameters,
        )
        .expect("Leto Adam oracle");
    }
}

impl CpuRule for CpuAdamW {
    type Parameters = CpuAdamWParameters<f32>;
    const STATE_COUNT: usize = 2;

    fn step(operands: &mut HostOperands, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<f32, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            (
                ArrayViewMut::new(layout, &mut operands.state_zero),
                ArrayViewMut::new(layout, &mut operands.state_one),
            ),
            parameters,
        )
        .expect("Leto AdamW oracle");
    }
}

impl CpuRule for CpuRmsProp {
    type Parameters = CpuRmsPropParameters<f32>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<f32, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto RMSProp oracle");
    }
}

impl CpuRule for CpuAdaGrad {
    type Parameters = CpuAdaGradParameters<f32>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<f32, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto AdaGrad oracle");
    }
}

fn run_cpu<Rule: CpuRule>(parameters: Rule::Parameters) -> HostOperands {
    let mut operands = HostOperands::seeded();
    for _ in 0..REPEATED_STEPS {
        Rule::step(&mut operands, parameters);
    }
    operands
}

fn run_backend<D, O, Rule>(
    device: &D,
    operations: &O,
    parameters: Rule::Parameters,
    state_count: usize,
) -> HostOperands
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Rule: StatefulUpdateRule<O::Dialect>,
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
        operations
            .stateful_update::<Rule, 2>(
                device,
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

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        // Each provider rule has at most 24 rounded f32 operations per step.
        // Two repeated steps give a forward bound of 48 epsilon; 64 epsilon
        // includes final comparison rounding without masking formula defects.
        let tolerance = 64.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}[{index}]: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn assert_rule<D, O, BackendRule, HostRule>(
    device: &D,
    operations: &O,
    backend_parameters: BackendRule::Parameters,
    host_parameters: HostRule::Parameters,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    BackendRule: StatefulUpdateRule<O::Dialect>,
    HostRule: CpuRule,
{
    let name = device.backend_name();
    let actual = run_backend::<D, O, BackendRule>(
        device,
        operations,
        backend_parameters,
        HostRule::STATE_COUNT,
    );
    let expected = run_cpu::<HostRule>(host_parameters);
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

fn assert_sgd_layout<D, O, const N: usize>(
    device: &D,
    operations: &O,
    layout: Layout<N>,
    parameter_initial: &[f32],
    gradient_initial: &[f32],
    state_initial: &[f32],
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect, Parameters = SgdParameters>,
{
    let mut expected_parameter = parameter_initial.to_vec();
    let mut expected_state = state_initial.to_vec();
    cpu_update::<f32, CpuSgd, N>(
        ArrayViewMut::new(layout, &mut expected_parameter),
        ArrayView::new(layout, gradient_initial),
        ArrayViewMut::new(layout, &mut expected_state),
        CpuSgdParameters::new(0.1, 0.5).expect("Leto SGD parameters"),
    )
    .expect("Leto boundary oracle");

    let parameter = device.upload(parameter_initial).expect("parameter upload");
    let gradient = device.upload(gradient_initial).expect("gradient upload");
    let state = device.upload(state_initial).expect("state upload");
    let states = [StridedView::new(&state, &layout)];
    operations
        .stateful_update::<Sgd, N>(
            device,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states: &states,
            },
            SgdParameters::new(0.1, 0.5).expect("SGD parameters"),
        )
        .expect("boundary dispatch");
    let mut actual_parameter = vec![0.0; parameter_initial.len()];
    let mut actual_gradient = vec![0.0; gradient_initial.len()];
    let mut actual_state = vec![0.0; state_initial.len()];
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

fn assert_rejections_are_atomic<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect, Parameters = SgdParameters>,
{
    let layout = Layout::c_contiguous([2]).expect("layout");
    let state_layout = Layout::c_contiguous([1]).expect("state layout");
    let parameter_initial = [1.0_f32, 2.0];
    let state_initial = [0.5_f32, 0.6];
    let parameter = device.upload(&parameter_initial).expect("parameter upload");
    let gradient = device.upload(&[0.1_f32, 0.2]).expect("gradient upload");
    let state = device.upload(&state_initial).expect("state upload");
    let states = [StridedView::new(&state, &state_layout)];
    let error = operations
        .stateful_update::<Sgd, 1>(
            device,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states: &states,
            },
            SgdParameters::new(0.1, 0.0).expect("SGD parameters"),
        )
        .expect_err("shape mismatch must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{}: expected shape dispatch failure, got {error}",
        device.backend_name()
    );
    let mut parameter_after = [0.0; 2];
    let mut state_after = [0.0; 2];
    device
        .download(&parameter, &mut parameter_after)
        .expect("parameter download");
    device
        .download(&state, &mut state_after)
        .expect("state download");
    assert_eq!(parameter_after, parameter_initial);
    assert_eq!(state_after, state_initial);

    let aliased_states = [StridedView::new(&parameter, &layout)];
    let error = operations
        .stateful_update::<Sgd, 1>(
            device,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states: &aliased_states,
            },
            SgdParameters::new(0.1, 0.0).expect("SGD parameters"),
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
/// # Panics
///
/// Panics with the backend and violated differential, repeated-dispatch,
/// striding, guard-storage, rank-boundary, alias, or failure-atomicity clause
/// when the provider diverges from the Leto CPU contract.
pub fn assert_stateful_update_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect, Parameters = SgdParameters>,
    Adam: StatefulUpdateRule<O::Dialect, Parameters = AdamParameters>,
    AdamW: StatefulUpdateRule<O::Dialect, Parameters = AdamWParameters>,
    RmsProp: StatefulUpdateRule<O::Dialect, Parameters = RmsPropParameters>,
    AdaGrad: StatefulUpdateRule<O::Dialect, Parameters = AdaGradParameters>,
{
    assert_rule::<D, O, Sgd, CpuSgd>(
        device,
        operations,
        SgdParameters::new(0.05, 0.9).expect("SGD parameters"),
        CpuSgdParameters::new(0.05, 0.9).expect("Leto SGD parameters"),
    );
    assert_rule::<D, O, Adam, CpuAdam>(
        device,
        operations,
        AdamParameters::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Adam parameters"),
        CpuAdamParameters::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Leto Adam parameters"),
    );
    assert_rule::<D, O, AdamW, CpuAdamW>(
        device,
        operations,
        AdamWParameters::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("AdamW parameters"),
        CpuAdamWParameters::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("Leto AdamW parameters"),
    );
    assert_rule::<D, O, RmsProp, CpuRmsProp>(
        device,
        operations,
        RmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("RMSProp parameters"),
        CpuRmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("Leto RMSProp parameters"),
    );
    assert_rule::<D, O, AdaGrad, CpuAdaGrad>(
        device,
        operations,
        AdaGradParameters::new(0.05, 1.0e-6).expect("AdaGrad parameters"),
        CpuAdaGradParameters::new(0.05, 1.0e-6).expect("Leto AdaGrad parameters"),
    );

    assert_sgd_layout(
        device,
        operations,
        Layout::c_contiguous([]).expect("scalar layout"),
        &[2.0],
        &[0.5],
        &[0.25],
    );
    assert_sgd_layout(
        device,
        operations,
        Layout::c_contiguous([1, 1, 1, 0, 1, 1, 1, 1]).expect("empty rank-eight layout"),
        &[9.0],
        &[8.0],
        &[7.0],
    );
    assert_rejections_are_atomic(device, operations);
}
