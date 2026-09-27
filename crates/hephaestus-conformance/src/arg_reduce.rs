//! Contract clauses for the device-neutral [`hephaestus_core::ArgReduceOps`]
//! seam.
//!
//! The 2x3 fixture `[[3,1,2],[1,4,4]]` gives every lane a distinct extremum
//! except row 1's max, which ties at columns 1 and 2 — column 1 must win
//! (first strict improvement wins, matching `leto::argmax`/`argmin`).
//!
//! Axis 0 is the *non-last*, non-contiguous axis of this C-contiguous
//! fixture (stride 3, vs axis 1's stride 1) — [`assert_arg_reduce_contract`]
//! exercises both. [`assert_arg_reduce_transposed_view_contract`] goes
//! further: it reduces the same data through a *transposed* view (swapped
//! strides, neither axis contiguous in the caller's own indexing), the
//! mechanism an N-D-tensor caller relies on to reduce an interior axis —
//! merge the dimensions outside the target axis into one `StridedView` axis
//! via their existing strides (valid whenever those dimensions are
//! stride-mergeable, i.e. contiguous with each other in memory) and pass the
//! target axis as the seam's other rank-2 axis, in either position.

use hephaestus_core::{ArgReduceOps, ComputeDevice, StridedView};
use leto::Layout;

fn fixture() -> Vec<i32> {
    vec![3, 1, 2, 1, 4, 4]
}

/// Argmax/argmin along axis 1 (within each row) and axis 0 (within each
/// column), checked against hand-derived expected indices.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// expected indices.
pub fn assert_arg_reduce_contract<D, A>(device: &D, ops: &A)
where
    D: ComputeDevice,
    A: ArgReduceOps<D, i32>,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 3]).expect("input layout");
    let input = device.upload(&fixture()).expect("fixture upload");

    // Axis 1 (within each row): row 0 = [3,1,2] -> max@0, min@1.
    // Row 1 = [1,4,4] -> max@1 (tie broken to the lower index), min@0.
    let axis1_layout = Layout::c_contiguous([2, 1]).expect("axis-1 output layout");
    let max1 = device.alloc_zeroed::<u32>(2).expect("max1 alloc");
    ops.argmax_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        StridedView::new(&max1, &axis1_layout),
    )
    .expect("argmax axis 1 dispatch");
    let mut got_max1 = vec![0u32; 2];
    device.download(&max1, &mut got_max1).expect("download");
    assert_eq!(got_max1, vec![0, 1], "{name}: argmax(axis=1) mismatch");

    let min1 = device.alloc_zeroed::<u32>(2).expect("min1 alloc");
    ops.argmin_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        1,
        StridedView::new(&min1, &axis1_layout),
    )
    .expect("argmin axis 1 dispatch");
    let mut got_min1 = vec![0u32; 2];
    device.download(&min1, &mut got_min1).expect("download");
    assert_eq!(got_min1, vec![1, 0], "{name}: argmin(axis=1) mismatch");

    // Axis 0 (within each column): col 0 = [3,1] -> max@0, min@1.
    // Col 1 = [1,4] -> max@1, min@0. Col 2 = [2,4] -> max@1, min@0.
    let axis0_layout = Layout::c_contiguous([1, 3]).expect("axis-0 output layout");
    let max0 = device.alloc_zeroed::<u32>(3).expect("max0 alloc");
    ops.argmax_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        0,
        StridedView::new(&max0, &axis0_layout),
    )
    .expect("argmax axis 0 dispatch");
    let mut got_max0 = vec![0u32; 3];
    device.download(&max0, &mut got_max0).expect("download");
    assert_eq!(got_max0, vec![0, 1, 1], "{name}: argmax(axis=0) mismatch");

    let min0 = device.alloc_zeroed::<u32>(3).expect("min0 alloc");
    ops.argmin_axis_into(
        device,
        StridedView::new(&input, &in_layout),
        0,
        StridedView::new(&min0, &axis0_layout),
    )
    .expect("argmin axis 0 dispatch");
    let mut got_min0 = vec![0u32; 3];
    device.download(&min0, &mut got_min0).expect("download");
    assert_eq!(got_min0, vec![1, 0, 0], "{name}: argmin(axis=0) mismatch");
}

/// Argmax over a transposed (swapped-stride) view of the same fixture: the
/// same physical buffer read as logical shape `[3, 2]` with strides `[1, 3]`
/// instead of the natural `[2, 3]` / `[3, 1]`. Reducing this view's axis 0
/// (stride 1, i.e. physically non-contiguous *between* lanes even though
/// contiguous *within* one) must match the original layout's axis-0 result —
/// the seam's output depends only on the logical view, never on which axis
/// happens to be memory-contiguous.
///
/// # Panics
///
/// Panics when the transposed-view result disagrees with the natural-layout
/// result for the same logical reduction.
pub fn assert_arg_reduce_transposed_view_contract<D, A>(device: &D, ops: &A)
where
    D: ComputeDevice,
    A: ArgReduceOps<D, i32>,
{
    let name = device.backend_name();
    let input = device.upload(&fixture()).expect("fixture upload");

    // Transposed: logical [row, col] -> physical offset row*3 + col becomes
    // logical [col, row] -> physical offset col*1 + row*3, i.e. shape [3, 2],
    // strides [1, 3]. Reducing along axis 1 here (the `row` index) is the
    // same computation as axis 0 on the natural [2, 3] layout: column 0
    // wins for row 0's max, column 1 wins for rows 1 and 2 (matching the
    // natural-layout `argmax(axis=0) == [0, 1, 1]` computed above).
    let transposed = Layout::try_new([3, 2], [1, 3], 0).expect("transposed layout");
    let out_layout = Layout::c_contiguous([3, 1]).expect("transposed output layout");
    let max = device.alloc_zeroed::<u32>(3).expect("max alloc");
    ops.argmax_axis_into(
        device,
        StridedView::new(&input, &transposed),
        1,
        StridedView::new(&max, &out_layout),
    )
    .expect("argmax over transposed view dispatch");
    let mut got = vec![0u32; 3];
    device.download(&max, &mut got).expect("download");
    assert_eq!(
        got,
        vec![0, 1, 1],
        "{name}: argmax over a transposed view must match the natural-layout axis-0 result"
    );
}
