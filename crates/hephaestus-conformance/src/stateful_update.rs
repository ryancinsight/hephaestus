//! Contract clauses for provider-owned stateful parameter updates.

use hephaestus_core::{
    AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW, AdamWParameters, ComputeDevice,
    HephaestusError, RmsProp, RmsPropParameters, Sgd, SgdParameters, StatefulUpdateOperands,
    StatefulUpdateOps, StatefulUpdateRule, StridedView,
};
use leto::Layout;

const LOGICAL: [usize; 4] = [1, 2, 4, 5];

fn logical_values(storage: &[f32; 6]) -> [f32; 4] {
    LOGICAL.map(|index| storage[index])
}

fn assert_close(name: &str, actual: [f32; 4], expected: [f32; 4]) {
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        // Each rule uses at most 24 rounded f32 operations plus one correctly
        // rounded square root. The factor 32 covers that straight-line depth
        // and the final comparison rounding without masking percent-scale
        // formula or state-order defects.
        let tolerance = 32.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}[{index}]: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn run_rule<D, O, Rule>(
    device: &D,
    operations: &O,
    parameters: Rule::Parameters,
    state_count: usize,
) -> ([f32; 4], [f32; 4], Option<[f32; 4]>)
where
    D: ComputeDevice,
    O: StatefulUpdateOps<D>,
    Rule: StatefulUpdateRule<O::Dialect>,
{
    let layout = Layout::new([2, 2], [3, 1], 1);
    let parameter = device
        .upload(&[91.0, 1.0, 2.0, 92.0, 3.0, 4.0])
        .expect("parameter upload");
    let gradient = device
        .upload(&[81.0, 0.1, 0.2, 82.0, 0.3, 0.4])
        .expect("gradient upload");
    let state_zero = device
        .upload(&[71.0, 0.5, 0.6, 72.0, 0.7, 0.8])
        .expect("state-zero upload");
    let state_one = device
        .upload(&[61.0, 0.25, 0.36, 62.0, 0.49, 0.64])
        .expect("state-one upload");
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
    let mut parameter_storage = [0.0; 6];
    let mut state_zero_storage = [0.0; 6];
    let mut state_one_storage = [0.0; 6];
    device
        .download(&parameter, &mut parameter_storage)
        .expect("parameter download");
    device
        .download(&state_zero, &mut state_zero_storage)
        .expect("state-zero download");
    let state_one_values = if state_count == 2 {
        device
            .download(&state_one, &mut state_one_storage)
            .expect("state-one download");
        Some(logical_values(&state_one_storage))
    } else {
        None
    };
    (
        logical_values(&parameter_storage),
        logical_values(&state_zero_storage),
        state_one_values,
    )
}

/// Run all stateful-update value and rejection clauses against one backend.
///
/// # Panics
///
/// Panics with the backend and violated rule, striding, alias, or validation
/// clause when the provider diverges from the contract.
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
    let name = device.backend_name();
    let parameter = [1.0_f32, 2.0, 3.0, 4.0];
    let gradient = [0.1_f32, 0.2, 0.3, 0.4];
    let state_zero = [0.5_f32, 0.6, 0.7, 0.8];
    let state_one = [0.25_f32, 0.36, 0.49, 0.64];

    let (actual_parameter, actual_state, _) = run_rule::<D, O, Sgd>(
        device,
        operations,
        SgdParameters::new(0.05, 0.9).expect("SGD parameters"),
        1,
    );
    let expected_state = core::array::from_fn(|i| state_zero[i] * 0.9 + gradient[i]);
    let expected_parameter = core::array::from_fn(|i| parameter[i] - 0.05 * expected_state[i]);
    assert_close(
        &format!("{name} SGD parameter"),
        actual_parameter,
        expected_parameter,
    );
    assert_close(&format!("{name} SGD state"), actual_state, expected_state);

    let adam_parameters = AdamParameters::new(0.01, 0.9, 0.99, 1.0e-6, 3).expect("Adam parameters");
    let (actual_parameter, actual_moment, actual_variance) =
        run_rule::<D, O, Adam>(device, operations, adam_parameters, 2);
    let moment = core::array::from_fn(|i| state_zero[i] * 0.9 + 0.1 * gradient[i]);
    let variance = core::array::from_fn(|i| state_one[i] * 0.99 + 0.01 * gradient[i] * gradient[i]);
    let bias_one = 1.0 - 0.9_f32.powi(3);
    let bias_two = 1.0 - 0.99_f32.powi(3);
    let expected_parameter = core::array::from_fn(|i| {
        parameter[i] - 0.01 * (moment[i] / bias_one) / ((variance[i] / bias_two).sqrt() + 1.0e-6)
    });
    assert_close(
        &format!("{name} Adam parameter"),
        actual_parameter,
        expected_parameter,
    );
    assert_close(&format!("{name} Adam first moment"), actual_moment, moment);
    assert_close(
        &format!("{name} Adam second moment"),
        actual_variance.expect("second state"),
        variance,
    );

    let (actual_parameter, _, _) = run_rule::<D, O, AdamW>(
        device,
        operations,
        AdamWParameters::new(0.01, 0.9, 0.99, 1.0e-6, 0.1, 3).expect("AdamW parameters"),
        2,
    );
    let expected_parameter = core::array::from_fn(|i| {
        parameter[i] * (1.0 - 0.01 * 0.1)
            - 0.01 * (moment[i] / bias_one) / ((variance[i] / bias_two).sqrt() + 1.0e-6)
    });
    assert_close(
        &format!("{name} AdamW parameter"),
        actual_parameter,
        expected_parameter,
    );

    let (actual_parameter, actual_state, _) = run_rule::<D, O, RmsProp>(
        device,
        operations,
        RmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("RMSProp parameters"),
        1,
    );
    let expected_state =
        core::array::from_fn(|i| state_zero[i] * 0.9 + 0.1 * gradient[i] * gradient[i]);
    let expected_parameter = core::array::from_fn(|i| {
        parameter[i] - 0.05 * gradient[i] / (expected_state[i].sqrt() + 1.0e-6)
    });
    assert_close(
        &format!("{name} RMSProp parameter"),
        actual_parameter,
        expected_parameter,
    );
    assert_close(
        &format!("{name} RMSProp state"),
        actual_state,
        expected_state,
    );

    let (actual_parameter, actual_state, _) = run_rule::<D, O, AdaGrad>(
        device,
        operations,
        AdaGradParameters::new(0.05, 1.0e-6).expect("AdaGrad parameters"),
        1,
    );
    let expected_state = core::array::from_fn(|i| state_zero[i] + gradient[i] * gradient[i]);
    let expected_parameter = core::array::from_fn(|i| {
        parameter[i] - 0.05 * gradient[i] / (expected_state[i].sqrt() + 1.0e-6)
    });
    assert_close(
        &format!("{name} AdaGrad parameter"),
        actual_parameter,
        expected_parameter,
    );
    assert_close(
        &format!("{name} AdaGrad state"),
        actual_state,
        expected_state,
    );

    let layout = Layout::c_contiguous([2]).expect("layout");
    let parameter_buffer = device.upload(&[1.0_f32, 2.0]).expect("parameter upload");
    let gradient_buffer = device.upload(&[0.1_f32, 0.2]).expect("gradient upload");
    let states = [StridedView::new(&parameter_buffer, &layout)];
    let error = operations
        .stateful_update::<Sgd, 1>(
            device,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter_buffer, &layout),
                gradient: StridedView::new(&gradient_buffer, &layout),
                states: &states,
            },
            SgdParameters::new(0.1, 0.0).expect("SGD parameters"),
        )
        .expect_err("aliased state must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "{name}: expected alias dispatch failure, got {error}"
    );
}
