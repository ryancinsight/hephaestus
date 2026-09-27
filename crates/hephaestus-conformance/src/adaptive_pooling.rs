//! Contract clauses for the device-neutral
//! [`hephaestus_core::AdaptivePoolingOps`] seam.
//!
//! Fixture `[[1, 2, 3, 4, 5, 6, 7], [10, 20, 30, 40, 50, 60, 70]]` (2x7,
//! axis 1 resized to 3) exercises the overlapping windows the
//! adaptive-pooling formula produces for a non-exact `7 / 3` split
//! (`[0,3)`, `[2,5)`, `[4,7)` — each size `ceil(7/3) = 3`, sharing one
//! index with its neighbor), plus the two degenerate cases every
//! extent-remap seam must handle: `out_len == 1` (one window spanning the
//! whole axis) and `out_len == in_len` (every window has exactly one
//! element, an identity pass for both average and maximum).

use hephaestus_core::{AdaptivePoolingMode, AdaptivePoolingOps, ComputeDevice, StridedView};
use leto::Layout;

/// Average and maximum pooling along axis 1, plus the `out_len == 1` and
/// `out_len == in_len` degenerate cases, checked against hand-derived
/// values.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// expected values.
pub fn assert_adaptive_pooling_contract<D, A>(device: &D, ops: &A)
where
    D: ComputeDevice,
    A: AdaptivePoolingOps<D, f32>,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 7]).expect("input layout");
    let input = device
        .upload(&[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
        ])
        .expect("fixture upload");

    // Axis 1, resized 7 -> 3: windows [0,3), [2,5), [4,7) (overlapping,
    // each size 3 = ceil(7/3)).
    // Row 0 [1..7]: avg(1,2,3)=2, avg(3,4,5)=4, avg(5,6,7)=6;
    //               max(1,2,3)=3, max(3,4,5)=5, max(5,6,7)=7.
    // Row 1 [10..70] scales row 0 by 10: avg 20/40/60, max 30/50/70.
    let out_layout = Layout::c_contiguous([2, 3]).expect("output layout");
    let avg = device.alloc_zeroed::<f32>(6).expect("avg alloc");
    ops.adaptive_pool_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        AdaptivePoolingMode::Average,
        StridedView::new(&avg, &out_layout),
    )
    .expect("average dispatch");
    let mut got_avg = vec![0.0f32; 6];
    device.download(&avg, &mut got_avg).expect("download");
    assert_eq!(
        got_avg,
        vec![2.0, 4.0, 6.0, 20.0, 40.0, 60.0],
        "{name}: adaptive average pool(axis=1) mismatch"
    );

    let max = device.alloc_zeroed::<f32>(6).expect("max alloc");
    ops.adaptive_pool_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        AdaptivePoolingMode::Maximum,
        StridedView::new(&max, &out_layout),
    )
    .expect("maximum dispatch");
    let mut got_max = vec![0.0f32; 6];
    device.download(&max, &mut got_max).expect("download");
    assert_eq!(
        got_max,
        vec![3.0, 5.0, 7.0, 30.0, 50.0, 70.0],
        "{name}: adaptive maximum pool(axis=1) mismatch"
    );

    // `out_len == 1`: one window spans the whole axis.
    let single_out_layout = Layout::c_contiguous([2, 1]).expect("single-output layout");
    let single_avg = device.alloc_zeroed::<f32>(2).expect("single-avg alloc");
    ops.adaptive_pool_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        AdaptivePoolingMode::Average,
        StridedView::new(&single_avg, &single_out_layout),
    )
    .expect("single-output average dispatch");
    let mut got_single_avg = vec![0.0f32; 2];
    device
        .download(&single_avg, &mut got_single_avg)
        .expect("download");
    assert_eq!(
        got_single_avg,
        vec![4.0, 40.0],
        "{name}: out_len == 1 must average the whole axis"
    );

    // `out_len == in_len`: every window is one element (identity).
    let identity_layout = Layout::c_contiguous([2, 7]).expect("identity output layout");
    let identity = device.alloc_zeroed::<f32>(14).expect("identity alloc");
    ops.adaptive_pool_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        AdaptivePoolingMode::Maximum,
        StridedView::new(&identity, &identity_layout),
    )
    .expect("identity dispatch");
    let mut got_identity = vec![0.0f32; 14];
    device
        .download(&identity, &mut got_identity)
        .expect("download");
    assert_eq!(
        got_identity,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0
        ],
        "{name}: out_len == in_len must be an identity pass"
    );
}
