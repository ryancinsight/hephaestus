use leto::Layout;
use themis::MemoryTier;

use super::*;
use crate::domain::buffer::DeviceBuffer;
use crate::domain::view::StridedView;

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

fn operands<'a, const R: usize>(
    real_buffer: &'a Buffer,
    imaginary_buffer: &'a Buffer,
    real_layout: &'a Layout<R>,
    imaginary_layout: &'a Layout<R>,
) -> FftOperands<'a, Buffer, R> {
    FftOperands {
        real: StridedView::new(real_buffer, real_layout),
        imaginary: StridedView::new(imaginary_buffer, imaginary_layout),
    }
}

fn plan_shape<const R: usize>(shape: [usize; R]) -> FftPlan<R> {
    let real_buffer = Buffer {
        len: shape.iter().product(),
    };
    let imaginary_buffer = Buffer {
        len: shape.iter().product(),
    };
    let layout = Layout::c_contiguous(shape).expect("valid test layout");
    plan_fft::<f32, _, R>(
        &operands(&real_buffer, &imaginary_buffer, &layout, &layout),
        FftDirection::Forward,
        false,
    )
    .expect("valid FFT plan")
}

#[test]
fn plans_one_two_and_three_dimensional_transforms() {
    assert_eq!(plan_shape([7]).elements, 7);
    assert_eq!(plan_shape([3, 5]).elements, 15);
    let plan = plan_shape([2, 3, 5]);
    assert_eq!(plan.shape, [2, 3, 5]);
    assert_eq!(plan.elements, 30);
    assert_eq!(plan.max_physical_offset, 29);
}

#[test]
fn rejects_ranks_outside_the_contract() {
    let empty = Buffer { len: 0 };
    let rank_zero = Layout::c_contiguous([]).expect("rank-zero layout");
    let error = plan_fft::<f32, _, 0>(
        &operands(&empty, &empty, &rank_zero, &rank_zero),
        FftDirection::Forward,
        true,
    )
    .expect_err("rank zero must fail before alias validation");
    assert_eq!(
        error.to_string(),
        "invalid configuration: FFT rank 0 is unsupported; expected a rank from 1 through 3"
    );

    let rank_four = Layout::c_contiguous([1, 1, 1, 1]).expect("rank-four layout");
    let singleton = Buffer { len: 1 };
    assert_eq!(
        plan_fft::<f32, _, 4>(
            &operands(&singleton, &singleton, &rank_four, &rank_four),
            FftDirection::Forward,
            false,
        )
        .expect_err("rank four must fail")
        .to_string(),
        "invalid configuration: FFT rank 4 is unsupported; expected a rank from 1 through 3"
    );
}

#[test]
fn rejects_empty_mismatched_and_aliased_components() {
    let empty_layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let empty = Buffer { len: 0 };
    assert!(
        plan_fft::<f32, _, 2>(
            &operands(&empty, &empty, &empty_layout, &empty_layout),
            FftDirection::Forward,
            false,
        )
        .expect_err("empty axis must fail")
        .to_string()
        .contains("nonzero")
    );

    let real_layout = Layout::c_contiguous([2, 3]).expect("real layout");
    let imaginary_layout = Layout::c_contiguous([3, 2]).expect("imaginary layout");
    let buffer = Buffer { len: 6 };
    assert!(
        plan_fft::<f32, _, 2>(
            &operands(&buffer, &buffer, &real_layout, &imaginary_layout),
            FftDirection::Forward,
            false,
        )
        .expect_err("mismatched layouts must fail")
        .to_string()
        .contains("layouts must match")
    );
    assert!(
        plan_fft::<f32, _, 2>(
            &operands(&buffer, &buffer, &real_layout, &real_layout),
            FftDirection::Forward,
            true,
        )
        .expect_err("aliased buffers must fail")
        .to_string()
        .contains("distinct buffers")
    );
}

#[test]
fn rejects_strided_offset_and_partial_buffer_views() {
    let real_buffer = Buffer { len: 6 };
    let imaginary_buffer = Buffer { len: 6 };
    let transposed = Layout::try_new([2, 3], [1, 2], 0).expect("transposed layout");
    let error = plan_fft::<f32, _, 2>(
        &operands(&real_buffer, &imaginary_buffer, &transposed, &transposed),
        FftDirection::Forward,
        false,
    )
    .expect_err("strided view must fail");
    assert!(error.to_string().contains("dense C-order"));

    let offset = Layout::try_new([2, 2], [2, 1], 1).expect("offset layout");
    let offset_buffer = Buffer { len: 5 };
    assert!(
        plan_fft::<f32, _, 2>(
            &operands(&offset_buffer, &offset_buffer, &offset, &offset),
            FftDirection::Forward,
            false,
        )
        .expect_err("offset view must fail")
        .to_string()
        .contains("zero-offset")
    );

    let full = Layout::c_contiguous([2, 2]).expect("full layout");
    let oversized = Buffer { len: 5 };
    assert!(
        plan_fft::<f32, _, 2>(
            &operands(&oversized, &oversized, &full, &full),
            FftDirection::Forward,
            false,
        )
        .expect_err("partial buffer view must fail")
        .to_string()
        .contains("exactly 4 elements")
    );
}

#[test]
fn address_limit_covers_extents_count_and_offset() {
    let plan = plan_shape([3, 5]);
    plan.validate_address_limit(15).expect("inclusive bound");
    assert_eq!(
        plan.validate_address_limit(14)
            .expect_err("element count exceeds bound")
            .to_string(),
        "invalid configuration: FFT plan exceeds backend address limit 14"
    );
}

#[test]
fn inverse_direction_is_preserved_by_the_plan() {
    let real_buffer = Buffer { len: 8 };
    let imaginary_buffer = Buffer { len: 8 };
    let layout = Layout::c_contiguous([8]).expect("rank-one layout");
    let plan = plan_fft::<f32, _, 1>(
        &operands(&real_buffer, &imaginary_buffer, &layout, &layout),
        FftDirection::Inverse,
        false,
    )
    .expect("valid inverse plan");
    assert_eq!(plan.direction, FftDirection::Inverse);
}
