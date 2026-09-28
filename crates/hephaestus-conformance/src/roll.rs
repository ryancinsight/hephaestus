//! Contract clauses for the device-neutral [`hephaestus_core::RollOps`]
//! seam.
//!
//! Fixture `[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]` (2x5, axis 1 rolled)
//! exercises a positive in-range shift, a negative shift, a shift exceeding
//! the axis length (multiple wraps), and a zero shift (identity) — each
//! checked against hand-derived values.

use hephaestus_core::{ComputeDevice, RollOps, StridedView};
use leto::Layout;

/// Positive, negative, over-magnitude, and zero shifts along axis 1,
/// checked against hand-derived values.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// expected values.
pub fn assert_roll_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, i32>,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 5]).expect("input layout");
    let input = device
        .upload(&[1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
        .expect("fixture upload");
    let out_layout = Layout::c_contiguous([2, 5]).expect("output layout");

    // shift=2: dst reads src (dst-2).rem_euclid(5), i.e. the last two
    // elements move to the front.
    let shifted = device.alloc_zeroed::<i32>(10).expect("shifted alloc");
    ops.roll_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        2,
        StridedView::new(&shifted, &out_layout),
    )
    .expect("roll(+2) dispatch");
    let mut got_shifted = vec![0i32; 10];
    device
        .download(&shifted, &mut got_shifted)
        .expect("download");
    assert_eq!(
        got_shifted,
        vec![4, 5, 1, 2, 3, 40, 50, 10, 20, 30],
        "{name}: roll(axis=1, shift=2) mismatch"
    );

    // shift=-1: the first element moves to the back.
    let neg = device.alloc_zeroed::<i32>(10).expect("neg alloc");
    ops.roll_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        -1,
        StridedView::new(&neg, &out_layout),
    )
    .expect("roll(-1) dispatch");
    let mut got_neg = vec![0i32; 10];
    device.download(&neg, &mut got_neg).expect("download");
    assert_eq!(
        got_neg,
        vec![2, 3, 4, 5, 1, 20, 30, 40, 50, 10],
        "{name}: roll(axis=1, shift=-1) mismatch"
    );

    // shift=7 on axis_len=5 wraps to the same result as shift=2.
    let over = device.alloc_zeroed::<i32>(10).expect("over alloc");
    ops.roll_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        7,
        StridedView::new(&over, &out_layout),
    )
    .expect("roll(+7) dispatch");
    let mut got_over = vec![0i32; 10];
    device.download(&over, &mut got_over).expect("download");
    assert_eq!(
        got_over, got_shifted,
        "{name}: roll(axis=1, shift=7) must match roll(shift=2) (5 == axis_len)"
    );

    // shift=0 is an identity pass.
    let identity = device.alloc_zeroed::<i32>(10).expect("identity alloc");
    ops.roll_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        0,
        StridedView::new(&identity, &out_layout),
    )
    .expect("roll(0) dispatch");
    let mut got_identity = vec![0i32; 10];
    device
        .download(&identity, &mut got_identity)
        .expect("download");
    assert_eq!(
        got_identity,
        vec![1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        "{name}: roll(axis=1, shift=0) must be an identity pass"
    );
}
