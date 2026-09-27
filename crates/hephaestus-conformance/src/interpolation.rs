//! Contract clauses for the device-neutral
//! [`hephaestus_core::InterpolationOps`] seam.
//!
//! Fixture `[[0, 10], [100, 200]]` (2x2, axis 1 resized) exercises the
//! align-corners convention with exact rational fractions (`out_len = 5`
//! over `in_len = 2` gives quarter steps: `0, 1/4, 1/2, 3/4, 1`), a
//! single-input broadcast (`in_len = 1`), and a single-output degenerate
//! resize (`out_len = 1`), each checked against hand-derived values so the
//! backend's kernel and the host reference agree on the exact mapping, not
//! only on each other.

use hephaestus_core::{ComputeDevice, InterpolationMode, InterpolationOps, StridedView};
use leto::Layout;

/// Linear and nearest resampling along axis 1, plus the `in_len == 1` and
/// `out_len == 1` degenerate cases, checked against hand-derived values.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// expected values.
pub fn assert_interpolation_contract<D, I>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: InterpolationOps<D, f32>,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 2]).expect("input layout");
    let input = device
        .upload(&[0.0f32, 10.0, 100.0, 200.0])
        .expect("fixture upload");

    // Axis 1, resized 2 -> 5: row 0 [0, 10] -> quarter steps of 10; row 1
    // [100, 200] -> quarter steps of 100.
    let out_layout = Layout::c_contiguous([2, 5]).expect("output layout");
    let linear = device.alloc_zeroed::<f32>(10).expect("linear alloc");
    ops.interpolate_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        InterpolationMode::Linear,
        StridedView::new(&linear, &out_layout),
    )
    .expect("linear dispatch");
    let mut got_linear = vec![0.0f32; 10];
    device.download(&linear, &mut got_linear).expect("download");
    assert_eq!(
        got_linear,
        vec![0.0, 2.5, 5.0, 7.5, 10.0, 100.0, 125.0, 150.0, 175.0, 200.0],
        "{name}: linear interpolate(axis=1) mismatch"
    );

    // Nearest at the same fractions: 1/4 and 3/4 round toward the nearer
    // endpoint (frac < 0.5 keeps lower, frac >= 0.5 advances).
    let nearest = device.alloc_zeroed::<f32>(10).expect("nearest alloc");
    ops.interpolate_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        InterpolationMode::Nearest,
        StridedView::new(&nearest, &out_layout),
    )
    .expect("nearest dispatch");
    let mut got_nearest = vec![0.0f32; 10];
    device
        .download(&nearest, &mut got_nearest)
        .expect("download");
    assert_eq!(
        got_nearest,
        vec![
            0.0, 0.0, 10.0, 10.0, 10.0, 100.0, 100.0, 200.0, 200.0, 200.0
        ],
        "{name}: nearest interpolate(axis=1) mismatch"
    );

    // `in_len == 1`: every output sample reads the sole input element.
    let single_in_layout = Layout::c_contiguous([1, 1]).expect("single-input layout");
    let single_in = device.upload(&[7.0f32]).expect("single-input upload");
    let broadcast_out_layout = Layout::c_contiguous([1, 4]).expect("broadcast output layout");
    let broadcast = device.alloc_zeroed::<f32>(4).expect("broadcast alloc");
    ops.interpolate_axis_into(
        device,
        StridedView::new(&single_in, &single_in_layout),
        1,
        InterpolationMode::Linear,
        StridedView::new(&broadcast, &broadcast_out_layout),
    )
    .expect("broadcast dispatch");
    let mut got_broadcast = vec![0.0f32; 4];
    device
        .download(&broadcast, &mut got_broadcast)
        .expect("download");
    assert_eq!(
        got_broadcast,
        vec![7.0, 7.0, 7.0, 7.0],
        "{name}: in_len == 1 must broadcast the sole input element"
    );

    // `out_len == 1`: the single output sample reads input index 0.
    let three_in_layout = Layout::c_contiguous([1, 3]).expect("three-input layout");
    let three_in = device
        .upload(&[1.0f32, 2.0, 3.0])
        .expect("three-input upload");
    let single_out_layout = Layout::c_contiguous([1, 1]).expect("single-output layout");
    let single_out = device.alloc_zeroed::<f32>(1).expect("single-output alloc");
    ops.interpolate_axis_into(
        device,
        StridedView::new(&three_in, &three_in_layout),
        1,
        InterpolationMode::Linear,
        StridedView::new(&single_out, &single_out_layout),
    )
    .expect("single-output dispatch");
    let mut got_single = vec![0.0f32; 1];
    device
        .download(&single_out, &mut got_single)
        .expect("download");
    assert_eq!(
        got_single,
        vec![1.0],
        "{name}: out_len == 1 must read input index 0"
    );
}
