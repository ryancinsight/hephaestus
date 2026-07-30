use super::*;
use crate::domain::view::StridedView;
use themis::MemoryTier;

struct Buffer {
    len: usize,
}

impl<T> DeviceBuffer<T> for Buffer {
    fn len(&self) -> usize {
        self.len
    }

    fn tier(&self) -> MemoryTier {
        MemoryTier::Dram
    }
}

fn layout<const R: usize>(shape: [usize; R]) -> Layout<R> {
    Layout::c_contiguous(shape).expect("valid test layout")
}

#[test]
fn plans_regular_two_dimensional_forward() {
    let input_buffer = Buffer { len: 25 };
    let weight_buffer = Buffer { len: 9 };
    let output_buffer = Buffer { len: 9 };
    let input = layout([1, 1, 5, 5]);
    let weight = layout([1, 1, 3, 3]);
    let output = layout([1, 1, 3, 3]);
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        bias: None,
        output: StridedView::new(&output_buffer, &output),
    };
    let parameters = ConvolutionParameters::new([1, 1], [0, 0], [1, 1]).expect("valid parameters");

    let plan =
        plan_convolution_forward::<f32, _, 4, 2>(&operands, parameters, false).expect("valid plan");

    assert_eq!(plan.input_spatial, [5, 5]);
    assert_eq!(plan.kernel_spatial, [3, 3]);
    assert_eq!(plan.output_spatial, [3, 3]);
    assert_eq!(plan.output_elements, 9);
    assert_eq!(plan.max_physical_offset, 24);
}

#[test]
fn plans_transposed_three_dimensional_forward() {
    let input_buffer = Buffer { len: 8 };
    let weight_buffer = Buffer { len: 8 };
    let output_buffer = Buffer { len: 125 };
    let input = layout([1, 1, 2, 2, 2]);
    let weight = layout([1, 1, 2, 2, 2]);
    let output = layout([1, 1, 5, 5, 5]);
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        bias: None,
        output: StridedView::new(&output_buffer, &output),
    };
    let parameters =
        TransposedConvolutionParameters::new([2, 2, 2], [0, 0, 0], [1, 1, 1], [1, 1, 1])
            .expect("valid parameters");

    let plan = plan_transposed_convolution_forward::<f32, _, 5, 3>(&operands, parameters, false)
        .expect("valid plan");

    assert_eq!(plan.output_spatial, [5, 5, 5]);
    assert_eq!(plan.output_elements, 125);
}

#[test]
fn rejects_rank_shape_storage_and_alias_violations() {
    let input_buffer = Buffer { len: 9 };
    let weight_buffer = Buffer { len: 4 };
    let output_buffer = Buffer { len: 4 };
    let input = layout([1, 1, 3, 3]);
    let weight = layout([1, 1, 2, 2]);
    let output = layout([1, 1, 2, 2]);
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        bias: None,
        output: StridedView::new(&output_buffer, &output),
    };
    let parameters = ConvolutionParameters::new([1, 1], [0, 0], [1, 1]).expect("valid parameters");

    let rank_error = plan_convolution_forward::<f32, _, 4, 1>(
        &operands,
        ConvolutionParameters::new([1], [0], [1]).expect("valid parameters"),
        false,
    )
    .expect_err("rank mismatch");
    assert_eq!(
        rank_error.to_string(),
        "invalid configuration: convolution tensor rank 4 must equal spatial rank 1 plus batch/channel axes"
    );

    let alias_error = plan_convolution_forward::<f32, _, 4, 2>(&operands, parameters, true)
        .expect_err("aliasing must fail");
    assert_eq!(
        alias_error.to_string(),
        "invalid configuration: convolution writable buffers must not alias readable operands or each other"
    );

    let short_output = Buffer { len: 3 };
    let short_operands = ConvolutionForwardOperands {
        input: operands.input,
        weight: operands.weight,
        bias: None,
        output: StridedView::new(&short_output, &output),
    };
    let storage_error =
        plan_convolution_forward::<f32, _, 4, 2>(&short_operands, parameters, false)
            .expect_err("short output storage");
    assert_eq!(
        storage_error.to_string(),
        "kernel dispatch failed: layout rejected: Storage error: storage length 3 does not cover layout physical offsets 0..=3"
    );
}

#[test]
fn backward_requires_targets_and_validates_all_before_dispatch() {
    let input_buffer = Buffer { len: 9 };
    let weight_buffer = Buffer { len: 4 };
    let grad_output_buffer = Buffer { len: 4 };
    let input = layout([1, 1, 3, 3]);
    let weight = layout([1, 1, 2, 2]);
    let grad_output = layout([1, 1, 2, 2]);
    let parameters = ConvolutionParameters::new([1, 1], [0, 0], [1, 1]).expect("valid parameters");
    let operands = ConvolutionBackwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        grad_output: StridedView::new(&grad_output_buffer, &grad_output),
        gradients: super::super::ConvolutionGradientViews {
            input: None,
            weight: None,
            bias: None,
        },
    };

    let error = plan_convolution_backward::<f32, _, 4, 2>(&operands, parameters, false)
        .expect_err("empty target set");
    assert_eq!(
        error.to_string(),
        "invalid configuration: convolution backward requires at least one gradient target"
    );
}

#[test]
fn address_limit_covers_physical_offsets_and_parameters() {
    let input_buffer = Buffer { len: 9 };
    let weight_buffer = Buffer { len: 4 };
    let output_buffer = Buffer { len: 16 };
    let input = layout([1, 1, 3, 3]);
    let weight = layout([1, 1, 2, 2]);
    let output = Layout::new([1, 1, 2, 2], [4, 4, 2, 1], 12);
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        bias: None,
        output: StridedView::new(&output_buffer, &output),
    };
    let parameters =
        ConvolutionParameters::new([1, 1], [0, 0], [1, 1]).expect("valid convolution parameters");
    let plan =
        plan_convolution_forward::<f32, _, 4, 2>(&operands, parameters, false).expect("valid plan");

    plan.validate_address_limit(15)
        .expect("largest physical offset is addressable");
    let error = plan
        .validate_address_limit(14)
        .expect_err("largest physical offset exceeds the backend contract");
    assert_eq!(
        error.to_string(),
        "invalid configuration: convolution plan exceeds backend address limit 14"
    );
}

#[test]
fn writable_layout_must_prove_nonoverlap() {
    let input_buffer = Buffer { len: 4 };
    let weight_buffer = Buffer { len: 1 };
    let output_buffer = Buffer { len: 3 };
    let input = layout([1, 1, 2, 2]);
    let weight = layout([1, 1, 1, 1]);
    let overlapping_output = Layout::new([1, 1, 2, 2], [4, 4, 1, 1], 0);
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        bias: None,
        output: StridedView::new(&output_buffer, &overlapping_output),
    };
    let parameters =
        ConvolutionParameters::new([1, 1], [0, 0], [1, 1]).expect("valid convolution parameters");

    let error = plan_convolution_forward::<f32, _, 4, 2>(&operands, parameters, false)
        .expect_err("overlapping writable layout");
    assert_eq!(
        error.to_string(),
        "invalid configuration: convolution output layout must be non-overlapping"
    );

    let transposed_output = Layout::new([1, 1, 2, 2], [4, 4, 1, 2], 0);
    let safe_output_buffer = Buffer { len: 4 };
    let safe_operands = ConvolutionForwardOperands {
        input: operands.input,
        weight: operands.weight,
        bias: None,
        output: StridedView::new(&safe_output_buffer, &transposed_output),
    };
    let safe_plan = plan_convolution_forward::<f32, _, 4, 2>(&safe_operands, parameters, false)
        .expect("transposed output layout is injective");
    assert_eq!(safe_plan.output_elements, 4);
}

#[test]
fn address_limit_covers_projection_intermediates() {
    let input_buffer = Buffer { len: 4 };
    let weight_buffer = Buffer { len: 3 };
    let output_buffer = Buffer { len: 3 };
    let input = layout([1, 1, 4]);
    let weight = layout([1, 1, 3]);
    let output = layout([1, 1, 3]);
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&input_buffer, &input),
        weight: StridedView::new(&weight_buffer, &weight),
        bias: None,
        output: StridedView::new(&output_buffer, &output),
    };
    let parameters =
        ConvolutionParameters::new([3], [4], [2]).expect("valid convolution parameters");
    let plan = plan_convolution_forward::<f32, _, 3, 1>(&operands, parameters, false)
        .expect("valid convolution");

    plan.validate_address_limit(10)
        .expect("projection maximum equals address limit");
    let error = plan
        .validate_address_limit(9)
        .expect_err("projection exceeds address limit");
    assert_eq!(
        error.to_string(),
        "invalid configuration: convolution projection exceeds backend address limit 9"
    );
}
