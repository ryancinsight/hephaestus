//! Contract clauses for the device-neutral [`hephaestus_core::RollOps`] seam.

use core::fmt::Debug;

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, RollOps, StridedView};
use leto::Layout;

struct RollCase<'a, T> {
    input: &'a [T],
    input_layout: &'a Layout<2>,
    axis: usize,
    shift: i64,
    output_layout: &'a Layout<2>,
    expected: &'a [T],
    label: &'a str,
}

fn assert_case<D, R, T>(device: &D, ops: &R, case: RollCase<'_, T>)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: Pod + Copy + Debug + PartialEq,
{
    let input_buffer = device.upload(case.input).expect("fixture upload");
    let output_buffer = device
        .alloc_zeroed::<T>(case.expected.len())
        .expect("output allocation");
    ops.roll_axis_into(
        device,
        StridedView::new(&input_buffer, case.input_layout),
        case.axis,
        case.shift,
        StridedView::new(&output_buffer, case.output_layout),
    )
    .expect("roll dispatch");
    let got = device
        .download_owned(&output_buffer)
        .expect("output download");
    assert_eq!(got, case.expected, "{}: roll output mismatch", case.label);
}

/// Fixtures used by the generic roll contract.
pub struct RollContractFixtures<'a, T> {
    /// Contiguous input values with shape `[2, 5]`.
    pub input: &'a [T],
    /// Expected contiguous output for shift `2` along axis `1`.
    pub shift_two: &'a [T],
    /// Expected contiguous output for shift `-1` along axis `1`.
    pub shift_negative_one: &'a [T],
    /// Expected contiguous output for shift `1` along axis `0`.
    pub axis_zero: &'a [T],
    /// Offset/stride input storage for shape `[2, 3]`.
    pub non_contiguous_input: &'a [T],
    /// Expected offset/stride output for shift `1` along axis `1`.
    pub non_contiguous_shift: &'a [T],
}

/// Runs the roll contract for one shipped scalar type on a backend.
pub fn assert_roll_contract_for_scalar<D, R, T>(
    device: &D,
    ops: &R,
    fixtures: RollContractFixtures<'_, T>,
) where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: Pod + Copy + Debug + PartialEq,
{
    let contiguous = Layout::c_contiguous([2, 5]).expect("input layout");
    let case = |axis, shift, expected, label| RollCase {
        input: fixtures.input,
        input_layout: &contiguous,
        axis,
        shift,
        output_layout: &contiguous,
        expected,
        label,
    };
    assert_case(
        device,
        ops,
        case(1, 2, fixtures.shift_two, "axis 1 shift 2"),
    );
    assert_case(
        device,
        ops,
        case(1, -1, fixtures.shift_negative_one, "axis 1 shift -1"),
    );
    assert_case(
        device,
        ops,
        case(1, 7, fixtures.shift_two, "axis 1 shift 7"),
    );
    assert_case(device, ops, case(1, 0, fixtures.input, "axis 1 shift 0"));
    assert_case(
        device,
        ops,
        case(1, i64::MIN, fixtures.shift_two, "axis 1 shift i64::MIN"),
    );
    assert_case(
        device,
        ops,
        case(1, i64::MAX, fixtures.shift_two, "axis 1 shift i64::MAX"),
    );

    assert_case(
        device,
        ops,
        RollCase {
            input: fixtures.input,
            input_layout: &contiguous,
            axis: 0,
            shift: 1,
            output_layout: &contiguous,
            expected: fixtures.axis_zero,
            label: "axis 0 shift 1",
        },
    );

    let non_contiguous = Layout::try_new([2, 3], [1, 3], 1).expect("non-contiguous layout");
    assert_case(
        device,
        ops,
        RollCase {
            input: fixtures.non_contiguous_input,
            input_layout: &non_contiguous,
            axis: 1,
            shift: 1,
            output_layout: &non_contiguous,
            expected: fixtures.non_contiguous_shift,
            label: "non-contiguous axis 1 shift 1",
        },
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
        RollContractFixtures {
            input: &[1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            shift_two: &[4, 5, 1, 2, 3, 40, 50, 10, 20, 30],
            shift_negative_one: &[2, 3, 4, 5, 1, 20, 30, 40, 50, 10],
            axis_zero: &[10, 20, 30, 40, 50, 1, 2, 3, 4, 5],
            non_contiguous_input: &[0, 10, 20, 0, 30, 40, 0, 50, 60],
            non_contiguous_shift: &[0, 50, 60, 0, 10, 20, 0, 30, 40],
        },
    );
}
