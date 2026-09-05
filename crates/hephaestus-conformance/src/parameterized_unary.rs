//! Contract clauses for runtime-parameter unary dispatch.

use hephaestus_core::{
    CeluGradOp, CeluOp, ComputeDevice, HardshrinkGradOp, HardshrinkOp, HardtanhGradOp, HardtanhOp,
    HephaestusError, LeakyReluGradOp, LeakyReluOp, ParameterizedUnaryOps, SoftshrinkGradOp,
    SoftshrinkOp, StridedView, ThresholdGradOp, ThresholdOp,
};
use leto::Layout;

fn logical_values(storage: &[f32; 12]) -> [f32; 6] {
    [
        storage[0],
        storage[2],
        storage[4],
        storage[6],
        storage[8],
        storage[10],
    ]
}

fn dispatch<D, O, Op>(
    device: &D,
    operations: &O,
    input: &D::Buffer<f32>,
    input_layout: &Layout<2>,
    parameters: [f32; 2],
    output_layout: &Layout<2>,
) -> [f32; 6]
where
    D: ComputeDevice,
    O: ParameterizedUnaryOps<D>,
    Op: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
{
    let output = device.alloc_zeroed::<f32>(12).expect("output allocation");
    operations
        .parameterized_unary_into::<Op, 2>(
            device,
            StridedView::new(input, input_layout),
            parameters,
            StridedView::new(&output, output_layout),
        )
        .expect("parameterized unary dispatch");
    let mut storage = [0.0_f32; 12];
    device.download(&output, &mut storage).expect("download");
    logical_values(&storage)
}

fn assert_close(actual: [f32; 6], expected: [f32; 6], name: &str, clause: &str) {
    // The non-linear clauses contain one elementary-function evaluation and
    // at most three native-f32 arithmetic operations. The bound allows 64
    // unit roundoffs at the largest operand scale for backend math-library
    // variation without masking a branch or parameter defect.
    let tolerance = 64.0 * f32::EPSILON;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "{name}: {clause} index {index}: actual {actual} expected {expected} bound {}",
            tolerance * scale
        );
    }
}

/// Run every runtime-parameter unary clause against one backend.
///
/// # Panics
///
/// Panics with the violated value, parameter-sensitivity, striding, or alias
/// clause when the backend diverges from the contract.
pub fn assert_parameterized_unary_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ParameterizedUnaryOps<D>,
    HardtanhOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    HardtanhGradOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    ThresholdOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    ThresholdGradOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    LeakyReluOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    LeakyReluGradOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    HardshrinkOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    HardshrinkGradOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    SoftshrinkOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    SoftshrinkGradOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    CeluOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
    CeluGradOp: hephaestus_core::ParameterizedUnaryExpr<O::Dialect>,
{
    let physical = [
        99.0_f32, -2.0, 98.0, -0.75, 97.0, -0.25, 96.0, 0.5, 95.0, 1.25, 94.0, 2.0,
    ];
    let logical_input = [-2.0_f32, -0.75, -0.25, 0.5, 1.25, 2.0];
    let input = device.upload(&physical).expect("input upload");
    let input_layout =
        Layout::try_new([2, 3], [6, 2], 1).expect("valid conformance fixture layout");
    let output_layout =
        Layout::try_new([2, 3], [6, 2], 0).expect("valid conformance fixture layout");
    let name = device.backend_name();

    assert_eq!(
        dispatch::<D, O, HardtanhOp>(
            device,
            operations,
            &input,
            &input_layout,
            [-0.75, 1.25],
            &output_layout,
        ),
        [-0.75, -0.75, -0.25, 0.5, 1.25, 1.25],
        "{name}: Hardtanh values and boundaries"
    );
    assert_eq!(
        dispatch::<D, O, HardtanhGradOp>(
            device,
            operations,
            &input,
            &input_layout,
            [-0.75, 1.25],
            &output_layout,
        ),
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "{name}: Hardtanh open-interval gradient"
    );
    assert_eq!(
        dispatch::<D, O, ThresholdOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, -3.25],
            &output_layout,
        ),
        [-3.25, -3.25, -3.25, -3.25, 1.25, 2.0],
        "{name}: Threshold values and equality boundary"
    );
    assert_eq!(
        dispatch::<D, O, ThresholdGradOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, -3.25],
            &output_layout,
        ),
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "{name}: Threshold strict-greater-than gradient"
    );
    assert_eq!(
        dispatch::<D, O, LeakyReluOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.25, 0.0],
            &output_layout,
        ),
        [-0.5, -0.1875, -0.0625, 0.5, 1.25, 2.0],
        "{name}: LeakyReLU parameter-sensitive values"
    );
    assert_eq!(
        dispatch::<D, O, LeakyReluGradOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.25, 0.0],
            &output_layout,
        ),
        [0.25, 0.25, 0.25, 1.0, 1.0, 1.0],
        "{name}: LeakyReLU gradient boundary"
    );
    assert_eq!(
        dispatch::<D, O, HardshrinkOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, 0.0],
            &output_layout,
        ),
        [-2.0, -0.75, 0.0, 0.0, 1.25, 2.0],
        "{name}: Hardshrink strict threshold"
    );
    assert_eq!(
        dispatch::<D, O, HardshrinkGradOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, 0.0],
            &output_layout,
        ),
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "{name}: Hardshrink gradient boundary"
    );
    assert_eq!(
        dispatch::<D, O, SoftshrinkOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, 0.0],
            &output_layout,
        ),
        [-1.5, -0.25, 0.0, 0.0, 0.75, 1.5],
        "{name}: Softshrink threshold"
    );
    assert_eq!(
        dispatch::<D, O, SoftshrinkGradOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, 0.0],
            &output_layout,
        ),
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "{name}: Softshrink gradient boundary"
    );
    let celu_expected = logical_input.map(|x| {
        if x >= 0.0 {
            x
        } else {
            0.5 * (f32::exp(x / 0.5) - 1.0)
        }
    });
    assert_close(
        dispatch::<D, O, CeluOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, 0.0],
            &output_layout,
        ),
        celu_expected,
        name,
        "CELU values",
    );
    let celu_grad_expected = logical_input.map(|x| if x >= 0.0 { 1.0 } else { f32::exp(x / 0.5) });
    assert_close(
        dispatch::<D, O, CeluGradOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, 0.0],
            &output_layout,
        ),
        celu_grad_expected,
        name,
        "CELU gradient values",
    );
    assert_eq!(
        dispatch::<D, O, HardtanhOp>(
            device,
            operations,
            &input,
            &input_layout,
            [-0.25, 0.5],
            &output_layout,
        ),
        [-0.25, -0.25, -0.25, 0.5, 0.5, 0.5],
        "{name}: changed runtime parameters alter cached-kernel output"
    );
    assert_eq!(
        dispatch::<D, O, HardtanhOp>(
            device,
            operations,
            &input,
            &input_layout,
            [0.5, -0.25],
            &output_layout,
        ),
        [0.5, 0.5, 0.5, -0.25, -0.25, -0.25],
        "{name}: reversed Hardtanh bounds retain cross-dialect comparison semantics"
    );

    let overlapping_output = device.alloc_zeroed::<f32>(4).expect("output allocation");
    let overlapping_layout =
        Layout::try_new([2, 2], [1, 1], 0).expect("valid conformance fixture layout");
    let overlapping_input_layout =
        Layout::try_new([2, 2], [6, 2], 1).expect("valid conformance fixture layout");
    let error = operations
        .parameterized_unary_into::<HardtanhOp, 2>(
            device,
            StridedView::new(&input, &overlapping_input_layout),
            [-0.75, 1.25],
            StridedView::new(&overlapping_output, &overlapping_layout),
        )
        .expect_err("overlapping parameterized output must be rejected");
    match error {
        HephaestusError::DispatchFailed { message } => {
            assert_eq!(message, "output layout must be non-overlapping")
        }
        other => panic!("{name}: expected overlap DispatchFailed, got {other}"),
    }

    let error = operations
        .parameterized_unary_into::<HardtanhOp, 2>(
            device,
            StridedView::new(&input, &input_layout),
            [-0.75, 1.25],
            StridedView::new(&input, &input_layout),
        )
        .expect_err("aliased parameterized output must be rejected");
    match error {
        HephaestusError::DispatchFailed { message } => {
            assert_eq!(message, "output buffer must not alias input buffer")
        }
        other => panic!("{name}: expected alias DispatchFailed, got {other}"),
    }
}
