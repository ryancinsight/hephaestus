//! Contract clauses for the device-neutral [`hephaestus_core::TopKOps`] seam.
//!
//! Fixture rows: `[5,1,9,3]` (distinct), `[2,2,2,2]` (all tied — the first
//! two occurrences must win), `[7,8,6,4]` (distinct), each reduced to its
//! top 2 along axis 1.

use hephaestus_core::{ComputeDevice, StridedView, TopKOps};
use leto::Layout;

fn fixture() -> Vec<i32> {
    vec![5, 1, 9, 3, 2, 2, 2, 2, 7, 8, 6, 4]
}

/// Top-2 selection along axis 1 of the 3x4 fixture, including an all-tied
/// row.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees.
pub fn assert_topk_contract<D, K>(device: &D, ops: &K)
where
    D: ComputeDevice,
    K: TopKOps<D, i32>,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([3, 4]).expect("input layout");
    let out_layout = Layout::c_contiguous([3, 2]).expect("output layout");
    let input = device.upload(&fixture()).expect("fixture upload");
    let values = device.alloc_zeroed::<i32>(6).expect("values alloc");
    let indices = device.alloc_zeroed::<u32>(6).expect("indices alloc");

    ops.topk_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        2,
        StridedView::new(&values, &out_layout),
        StridedView::new(&indices, &out_layout),
    )
    .expect("topk dispatch");

    let mut got_vals = vec![0i32; 6];
    let mut got_idxs = vec![0u32; 6];
    device
        .download(&values, &mut got_vals)
        .expect("download values");
    device
        .download(&indices, &mut got_idxs)
        .expect("download indices");

    assert_eq!(
        got_vals,
        vec![9, 5, 2, 2, 8, 7],
        "{name}: top-2 values must match, descending, per row"
    );
    assert_eq!(
        got_idxs,
        vec![2, 0, 0, 1, 1, 0],
        "{name}: top-2 indices must match, including the tied row's first-two-occurrences rule"
    );
}
