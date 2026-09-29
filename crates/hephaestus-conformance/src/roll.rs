//! Contract clauses for the device-neutral [`hephaestus_core::RollOps`] seam.

use core::fmt::Debug;

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, RollOps, StridedView};
use leto::Layout;

fn assert_case<D, R, T>(
    device: &D,
    ops: &R,
    input: &[T],
    input_layout: &Layout<2>,
    axis: usize,
    shift: i64,
    output_layout: &Layout<2>,
    expected: &[T],
    label: &str,
) where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: Pod + Copy + Debug + PartialEq,
{
    let input_buffer = device.upload(input).expect("fixture upload");
    let output_buffer = device
        .alloc_zeroed::<T>(expected.len())
        .expect("output allocation");
    ops.roll_axis_into(
        device,
        StridedView::new(&input_buffer, input_layout),
        axis,
        shift,
        StridedView::new(&output_buffer, output_layout),
    )
    .expect("roll dispatch");
    let got = device
        .download_owned(&output_buffer)
        .expect("output download");
    assert_eq!(got, expected, "{label}: roll output mismatch");
}

/// Runs the roll contract for one shipped scalar type on a backend.
pub fn assert_roll_contract_for_scalar<D, R, T>(
    device: &D,
    ops: &R,
    input: &[T],
    shift_two: &[T],
    shift_negative_one: &[T],
    axis_zero: &[T],
    non_contiguous_input: &[T],
    non_contiguous_shift: &[T],
) where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: Pod + Copy + Debug + PartialEq,
{
    let contiguous = Layout::c_contiguous([2, 5]).expect("input layout");
    assert_case(
        device,
        ops,
        input,
        &contiguous,
        1,
        2,
        &contiguous,
        shift_two,
        "axis 1 shift 2",
    );
    assert_case(
        device,
        ops,
        input,
        &contiguous,
        1,
        -1,
        &contiguous,
        shift_negative_one,
        "axis 1 shift -1",
    );
    assert_case(
        device,
        ops,
        input,
        &contiguous,
        1,
        7,
        &contiguous,
        shift_two,
        "axis 1 shift 7",
    );
    assert_case(
        device,
        ops,
        input,
        &contiguous,
        1,
        0,
        &contiguous,
        input,
        "axis 1 shift 0",
    );
    assert_case(
        device,
        ops,
        input,
        &contiguous,
        1,
        i64::MIN,
        &contiguous,
        shift_two,
        "axis 1 shift i64::MIN",
    );
    assert_case(
        device,
        ops,
        input,
        &contiguous,
        1,
        i64::MAX,
        &contiguous,
        shift_two,
        "axis 1 shift i64::MAX",
    );

    assert_case(
        device,
        ops,
        input,
        &contiguous,
        0,
        1,
        &contiguous,
        axis_zero,
        "axis 0 shift 1",
    );

    let non_contiguous = Layout::try_new([2, 3], [1, 3], 1).expect("non-contiguous layout");
    assert_case(
        device,
        ops,
        non_contiguous_input,
        &non_contiguous,
        1,
        1,
        &non_contiguous,
        non_contiguous_shift,
        "non-contiguous axis 1 shift 1",
    );
}

/// Runs the shared roll contract with the canonical signed 32-bit fixture.
pub fn assert_roll_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, i32>,
{
    assert_roll_contract_for_scalar(
        device,
        ops,
        &[1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        &[4, 5, 1, 2, 3, 40, 50, 10, 20, 30],
        &[2, 3, 4, 5, 1, 20, 30, 40, 50, 10],
        &[10, 20, 30, 40, 50, 1, 2, 3, 4, 5],
        &[0, 10, 20, 0, 30, 40, 0, 50, 60],
        &[0, 50, 60, 0, 10, 20, 0, 30, 40],
    );
}
