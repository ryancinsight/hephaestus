//! Contract clauses for provider-owned stateful parameter updates.

use eunomia::{FloatElement, Pod, RealField};
use hephaestus_core::{
    AdaGrad, AdaGradParameters, AdaGradParametersF64, Adam, AdamParameters, AdamParametersF64,
    AdamW, AdamWParameters, AdamWParametersF64, ComputeDevice, HephaestusError, KernelDialect,
    RmsProp, RmsPropParameters, RmsPropParametersF64, Sgd, SgdParameters, SgdParametersF64,
    StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{
    AdaGrad as CpuAdaGrad, AdaGradParameters as CpuAdaGradParameters, Adam as CpuAdam,
    AdamParameters as CpuAdamParameters, AdamW as CpuAdamW, AdamWParameters as CpuAdamWParameters,
    RealScalar, RmsProp as CpuRmsProp, RmsPropParameters as CpuRmsPropParameters, Sgd as CpuSgd,
    SgdParameters as CpuSgdParameters, stateful_update as cpu_update,
};

const REPEATED_STEPS: usize = 2;

/// Width-indexed backend entry point.
///
/// Selects `stateful_update` (f32) or `stateful_update_f64` (f64) plus the
/// matching parameter struct, so every clause below runs for both widths from
/// one body instead of a duplicated f64 file.
trait Width: Pod + RealField + RealScalar {
    /// Backend parameter struct for `Rule` at this width.
    type BackendParameters<Rule, Dialect>: Pod
    where
        Rule: StatefulUpdateRule<Dialect>,
        Dialect: KernelDialect;

    /// Dispatch one update at this width.
    fn dispatch<D, O, Rule, const N: usize>(
        device: &D,
        operations: &O,
        operands: StatefulUpdateOperands<'_, D::Buffer<Self>, N>,
        parameters: Self::BackendParameters<Rule, O::Dialect>,
    ) -> Result<(), HephaestusError>
    where
        D: ComputeDevice,
        O: StatefulUpdateOps<D>,
        Rule: StatefulUpdateRule<O::Dialect>;
}

impl Width for f32 {
    type BackendParameters<Rule, Dialect>
        = Rule::Parameters
    where
        Rule: StatefulUpdateRule<Dialect>,
        Dialect: KernelDialect;

    fn dispatch<D, O, Rule, const N: usize>(
        device: &D,
        operations: &O,
        operands: StatefulUpdateOperands<'_, D::Buffer<Self>, N>,
        parameters: Self::BackendParameters<Rule, O::Dialect>,
    ) -> Result<(), HephaestusError>
    where
        D: ComputeDevice,
        O: StatefulUpdateOps<D>,
        Rule: StatefulUpdateRule<O::Dialect>,
    {
        operations.stateful_update::<Rule, N>(device, operands, parameters)
    }
}

impl Width for f64 {
    type BackendParameters<Rule, Dialect>
        = Rule::ParametersF64
    where
        Rule: StatefulUpdateRule<Dialect>,
        Dialect: KernelDialect;

    fn dispatch<D, O, Rule, const N: usize>(
        device: &D,
        operations: &O,
        operands: StatefulUpdateOperands<'_, D::Buffer<Self>, N>,
        parameters: Self::BackendParameters<Rule, O::Dialect>,
    ) -> Result<(), HephaestusError>
    where
        D: ComputeDevice,
        O: StatefulUpdateOps<D>,
        Rule: StatefulUpdateRule<O::Dialect>,
    {
        operations.stateful_update_f64::<Rule, N>(device, operands, parameters)
    }
}

#[derive(Clone)]
struct HostOperands<T> {
    parameter: [T; 6],
    gradient: [T; 6],
    state_zero: [T; 6],
    state_one: [T; 6],
}

impl<T: FloatElement> HostOperands<T> {
    fn seeded() -> Self {
        Self {
            parameter: [91.0, 1.0, 2.0, 92.0, 3.0, 4.0].map(T::from_f32),
            gradient: [81.0, 0.1, 0.2, 82.0, 0.3, 0.4].map(T::from_f32),
            state_zero: [71.0, 0.5, 0.6, 72.0, 0.7, 0.8].map(T::from_f32),
            state_one: [61.0, 0.25, 0.36, 62.0, 0.49, 0.64].map(T::from_f32),
        }
    }
}

trait CpuRule<T> {
    type Parameters: Copy;
    const STATE_COUNT: usize;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters);
}

impl<T: RealScalar> CpuRule<T> for CpuSgd {
    type Parameters = CpuSgdParameters<T>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto SGD oracle");
    }
}

impl<T: RealScalar> CpuRule<T> for CpuAdam {
    type Parameters = CpuAdamParameters<T>;
    const STATE_COUNT: usize = 2;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
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

impl<T: RealScalar> CpuRule<T> for CpuAdamW {
    type Parameters = CpuAdamWParameters<T>;
    const STATE_COUNT: usize = 2;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
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

impl<T: RealScalar> CpuRule<T> for CpuRmsProp {
    type Parameters = CpuRmsPropParameters<T>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto RMSProp oracle");
    }
}

impl<T: RealScalar> CpuRule<T> for CpuAdaGrad {
    type Parameters = CpuAdaGradParameters<T>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto AdaGrad oracle");
    }
}

fn run_cpu<Rule: CpuRule<T>, T: FloatElement>(parameters: Rule::Parameters) -> HostOperands<T> {
    let mut operands = HostOperands::seeded();
    for _ in 0..REPEATED_STEPS {
        Rule::step(&mut operands, parameters);
    }
    operands
}

fn run_backend<D, O, Rule, T>(
    device: &D,
    operations: &O,
    parameters: T::BackendParameters<Rule, O::Dialect>,
    state_count: usize,
) -> HostOperands<T>
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Rule: StatefulUpdateRule<O::Dialect>,
    T: Width,
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

fn assert_close<T: RealField>(name: &str, actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        // Each provider rule has at most 24 rounded operations per step per
        // width. Two repeated steps give a forward bound of 48 epsilon; 64
        // epsilon includes final comparison rounding without masking formula
        // defects. The bound scales with the width's own epsilon.
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

fn assert_rule<D, O, BackendRule, HostRule, T>(
    device: &D,
    operations: &O,
    backend_parameters: T::BackendParameters<BackendRule, O::Dialect>,
    host_parameters: HostRule::Parameters,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    BackendRule: StatefulUpdateRule<O::Dialect>,
    HostRule: CpuRule<T>,
    T: Width,
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

fn assert_sgd_layout<D, O, T, const N: usize>(
    device: &D,
    operations: &O,
    layout: Layout<N>,
    parameter_initial: &[T],
    gradient_initial: &[T],
    state_initial: &[T],
    parameters: SgdParameterPair<T, T::BackendParameters<Sgd, O::Dialect>>,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect>,
    T: Width,
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

fn assert_rejections_are_atomic<D, O, T>(
    device: &D,
    operations: &O,
    backend_parameters: T::BackendParameters<Sgd, O::Dialect>,
) where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect>,
    T: Width,
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
    assert_rule::<D, O, Sgd, CpuSgd, f32>(
        device,
        operations,
        SgdParameters::new(0.05, 0.9).expect("SGD parameters"),
        CpuSgdParameters::new(0.05, 0.9).expect("Leto SGD parameters"),
    );
    assert_rule::<D, O, Adam, CpuAdam, f32>(
        device,
        operations,
        AdamParameters::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Adam parameters"),
        CpuAdamParameters::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Leto Adam parameters"),
    );
    assert_rule::<D, O, AdamW, CpuAdamW, f32>(
        device,
        operations,
        AdamWParameters::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("AdamW parameters"),
        CpuAdamWParameters::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("Leto AdamW parameters"),
    );
    assert_rule::<D, O, RmsProp, CpuRmsProp, f32>(
        device,
        operations,
        RmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("RMSProp parameters"),
        CpuRmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("Leto RMSProp parameters"),
    );
    assert_rule::<D, O, AdaGrad, CpuAdaGrad, f32>(
        device,
        operations,
        AdaGradParameters::new(0.05, 1.0e-6).expect("AdaGrad parameters"),
        CpuAdaGradParameters::new(0.05, 1.0e-6).expect("Leto AdaGrad parameters"),
    );

    assert_sgd_layout(
        device,
        operations,
        Layout::c_contiguous([]).expect("scalar layout"),
        &[2.0_f32],
        &[0.5_f32],
        &[0.25_f32],
        SgdParameterPair::new(
            SgdParameters::new(0.1, 0.5).expect("SGD parameters"),
            CpuSgdParameters::new(0.1, 0.5).expect("Leto SGD parameters"),
        ),
    );
    assert_sgd_layout(
        device,
        operations,
        Layout::c_contiguous([1, 1, 1, 0, 1, 1, 1, 1]).expect("empty rank-eight layout"),
        &[9.0_f32],
        &[8.0_f32],
        &[7.0_f32],
        SgdParameterPair::new(
            SgdParameters::new(0.1, 0.5).expect("SGD parameters"),
            CpuSgdParameters::new(0.1, 0.5).expect("Leto SGD parameters"),
        ),
    );
    assert_rejections_are_atomic::<D, O, f32>(
        device,
        operations,
        SgdParameters::new(0.1, 0.0).expect("SGD parameters"),
    );
}

/// Run all stateful-update value and rejection clauses at f64 against one backend.
///
/// Mirrors [`assert_stateful_update_contract`] with the `ParametersF64` twins
/// and the f64 Leto oracle; every clause body is shared through [`Width`].
///
/// # Panics
///
/// Panics with the backend and violated differential, repeated-dispatch,
/// striding, guard-storage, rank-boundary, alias, or failure-atomicity clause
/// when the provider diverges from the Leto CPU contract.
pub fn assert_stateful_update_contract_f64<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Sgd: StatefulUpdateRule<O::Dialect, ParametersF64 = SgdParametersF64>,
    Adam: StatefulUpdateRule<O::Dialect, ParametersF64 = AdamParametersF64>,
    AdamW: StatefulUpdateRule<O::Dialect, ParametersF64 = AdamWParametersF64>,
    RmsProp: StatefulUpdateRule<O::Dialect, ParametersF64 = RmsPropParametersF64>,
    AdaGrad: StatefulUpdateRule<O::Dialect, ParametersF64 = AdaGradParametersF64>,
{
    assert_rule::<D, O, Sgd, CpuSgd, f64>(
        device,
        operations,
        SgdParametersF64::new(0.05, 0.9).expect("SGD parameters"),
        CpuSgdParameters::new(0.05, 0.9).expect("Leto SGD parameters"),
    );
    assert_rule::<D, O, Adam, CpuAdam, f64>(
        device,
        operations,
        AdamParametersF64::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Adam parameters"),
        CpuAdamParameters::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Leto Adam parameters"),
    );
    assert_rule::<D, O, AdamW, CpuAdamW, f64>(
        device,
        operations,
        AdamWParametersF64::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("AdamW parameters"),
        CpuAdamWParameters::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("Leto AdamW parameters"),
    );
    assert_rule::<D, O, RmsProp, CpuRmsProp, f64>(
        device,
        operations,
        RmsPropParametersF64::new(0.05, 0.9, 1.0e-6).expect("RMSProp parameters"),
        CpuRmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("Leto RMSProp parameters"),
    );
    assert_rule::<D, O, AdaGrad, CpuAdaGrad, f64>(
        device,
        operations,
        AdaGradParametersF64::new(0.05, 1.0e-6).expect("AdaGrad parameters"),
        CpuAdaGradParameters::new(0.05, 1.0e-6).expect("Leto AdaGrad parameters"),
    );

    assert_sgd_layout(
        device,
        operations,
        Layout::c_contiguous([]).expect("scalar layout"),
        &[2.0_f64],
        &[0.5_f64],
        &[0.25_f64],
        SgdParameterPair::new(
            SgdParametersF64::new(0.1, 0.5).expect("SGD parameters"),
            CpuSgdParameters::new(0.1, 0.5).expect("Leto SGD parameters"),
        ),
    );
    assert_sgd_layout(
        device,
        operations,
        Layout::c_contiguous([1, 1, 1, 0, 1, 1, 1, 1]).expect("empty rank-eight layout"),
        &[9.0_f64],
        &[8.0_f64],
        &[7.0_f64],
        SgdParameterPair::new(
            SgdParametersF64::new(0.1, 0.5).expect("SGD parameters"),
            CpuSgdParameters::new(0.1, 0.5).expect("Leto SGD parameters"),
        ),
    );
    assert_rejections_are_atomic::<D, O, f64>(
        device,
        operations,
        SgdParametersF64::new(0.1, 0.0).expect("SGD parameters"),
    );
}
