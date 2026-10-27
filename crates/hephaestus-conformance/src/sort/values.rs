//! Value and tie-breaking stability clauses for the [`SortOps`] seam.

use hephaestus_core::{ComputeDevice, SortOps};
use leto::Layout;

/// Sort along one axis in both directions against hand-computed
/// fixtures: a 1-D input sorted ascending and descending, a 2-D input
/// sorted along axis 0, and a 2-D input sorted along axis 1 in both
/// directions. Every fixture asserts both the sorted values and the
/// `u32` argsort positions exactly.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with any
/// fixture.
pub fn assert_sort_value_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // 1-D: [3, 1, 2] sorts to [1, 2, 3] / [1, 2, 0] ascending and
    // [3, 2, 1] / [0, 2, 1] descending.
    let lane_host: Vec<T> = [3i8, 1, 2].map(T::from).to_vec();
    let lane_layout = Layout::c_contiguous([3]).expect("lane layout");
    let lane_input = device.upload(&lane_host).expect("lane upload");
    let lane = || hephaestus_core::StridedView::new(&lane_input, &lane_layout);
    let lane_values = device.alloc_zeroed::<T>(3).expect("lane values alloc");
    let lane_indices = device.alloc_zeroed::<u32>(3).expect("lane indices alloc");
    ops.sort_into::<1>(
        device,
        lane(),
        0,
        false,
        hephaestus_core::StridedView::new(&lane_values, &lane_layout),
        hephaestus_core::StridedView::new(&lane_indices, &lane_layout),
    )
    .expect("ascending sort dispatch");
    let mut lane_got = vec![T::from(0i8); 3];
    let mut lane_idx = vec![0u32; 3];
    device
        .download(&lane_values, &mut lane_got)
        .expect("download");
    device
        .download(&lane_indices, &mut lane_idx)
        .expect("download");
    let lane_expected: Vec<T> = [1i8, 2, 3].map(T::from).to_vec();
    assert_eq!(
        lane_got, lane_expected,
        "{name}: 1-D ascending sort must match the hand-computed values exactly"
    );
    assert_eq!(
        lane_idx,
        vec![1u32, 2, 0],
        "{name}: 1-D ascending sort must match the hand-computed indices exactly"
    );

    let lane_down_values = device.alloc_zeroed::<T>(3).expect("lane values alloc");
    let lane_down_indices = device.alloc_zeroed::<u32>(3).expect("lane indices alloc");
    ops.sort_into::<1>(
        device,
        lane(),
        0,
        true,
        hephaestus_core::StridedView::new(&lane_down_values, &lane_layout),
        hephaestus_core::StridedView::new(&lane_down_indices, &lane_layout),
    )
    .expect("descending sort dispatch");
    let mut lane_down_got = vec![T::from(0i8); 3];
    let mut lane_down_idx = vec![0u32; 3];
    device
        .download(&lane_down_values, &mut lane_down_got)
        .expect("download");
    device
        .download(&lane_down_indices, &mut lane_down_idx)
        .expect("download");
    let lane_down_expected: Vec<T> = [3i8, 2, 1].map(T::from).to_vec();
    assert_eq!(
        lane_down_got, lane_down_expected,
        "{name}: 1-D descending sort must match the hand-computed values exactly"
    );
    assert_eq!(
        lane_down_idx,
        vec![0u32, 2, 1],
        "{name}: 1-D descending sort must match the hand-computed indices exactly"
    );

    // 2-D: [[3, 1], [2, 0]] along axis 0 ascending sorts each column:
    // values [[2, 0], [3, 1]], indices [[1, 1], [0, 0]].
    let square_host: Vec<T> = [3i8, 1, 2, 0].map(T::from).to_vec();
    let square_layout = Layout::c_contiguous([2, 2]).expect("square layout");
    let square_input = device.upload(&square_host).expect("square upload");
    let square = || hephaestus_core::StridedView::new(&square_input, &square_layout);
    let square_values = device.alloc_zeroed::<T>(4).expect("square values alloc");
    let square_indices = device.alloc_zeroed::<u32>(4).expect("square indices alloc");
    ops.sort_into::<2>(
        device,
        square(),
        0,
        false,
        hephaestus_core::StridedView::new(&square_values, &square_layout),
        hephaestus_core::StridedView::new(&square_indices, &square_layout),
    )
    .expect("axis-0 sort dispatch");
    let mut square_got = vec![T::from(0i8); 4];
    let mut square_idx = vec![0u32; 4];
    device
        .download(&square_values, &mut square_got)
        .expect("download");
    device
        .download(&square_indices, &mut square_idx)
        .expect("download");
    let square_expected: Vec<T> = [2i8, 0, 3, 1].map(T::from).to_vec();
    assert_eq!(
        square_got, square_expected,
        "{name}: axis-0 ascending sort must match the hand-computed values exactly"
    );
    assert_eq!(
        square_idx,
        vec![1u32, 1, 0, 0],
        "{name}: axis-0 ascending sort must match the hand-computed indices exactly"
    );

    // 2-D: [[1, 3], [0, 2]] along axis 1 sorts each row: ascending
    // values [[1, 3], [0, 2]] with identity indices is vacuous, so the
    // ascending case uses the reversed square [[3, 1], [2, 0]] —
    // values [[1, 3], [0, 2]], indices [[1, 0], [1, 0]] — and the
    // descending case uses [[1, 3], [0, 2]] — values [[3, 1], [2, 0]],
    // indices [[1, 0], [1, 0]].
    ops.sort_into::<2>(
        device,
        square(),
        1,
        false,
        hephaestus_core::StridedView::new(&square_values, &square_layout),
        hephaestus_core::StridedView::new(&square_indices, &square_layout),
    )
    .expect("axis-1 ascending sort dispatch");
    device
        .download(&square_values, &mut square_got)
        .expect("download");
    device
        .download(&square_indices, &mut square_idx)
        .expect("download");
    let row_expected: Vec<T> = [1i8, 3, 0, 2].map(T::from).to_vec();
    assert_eq!(
        square_got, row_expected,
        "{name}: axis-1 ascending sort must match the hand-computed values exactly"
    );
    assert_eq!(
        square_idx,
        vec![1u32, 0, 1, 0],
        "{name}: axis-1 ascending sort must match the hand-computed indices exactly"
    );

    let down_host: Vec<T> = [1i8, 3, 0, 2].map(T::from).to_vec();
    let down_input = device.upload(&down_host).expect("down upload");
    let down_values = device.alloc_zeroed::<T>(4).expect("down values alloc");
    let down_indices = device.alloc_zeroed::<u32>(4).expect("down indices alloc");
    ops.sort_into::<2>(
        device,
        hephaestus_core::StridedView::new(&down_input, &square_layout),
        1,
        true,
        hephaestus_core::StridedView::new(&down_values, &square_layout),
        hephaestus_core::StridedView::new(&down_indices, &square_layout),
    )
    .expect("axis-1 descending sort dispatch");
    let mut down_got = vec![T::from(0i8); 4];
    let mut down_idx = vec![0u32; 4];
    device
        .download(&down_values, &mut down_got)
        .expect("download");
    device
        .download(&down_indices, &mut down_idx)
        .expect("download");
    let down_expected: Vec<T> = [3i8, 1, 2, 0].map(T::from).to_vec();
    assert_eq!(
        down_got, down_expected,
        "{name}: axis-1 descending sort must match the hand-computed values exactly"
    );
    assert_eq!(
        down_idx,
        vec![1u32, 0, 1, 0],
        "{name}: axis-1 descending sort must match the hand-computed indices exactly"
    );
}

/// Equal values keep their input order in both directions, observably
/// through the indices: `[2, 0, 2, 1, 2]` sorts to values
/// `[0, 1, 2, 2, 2]` with indices `[1, 3, 0, 2, 4]` ascending, and to
/// `[2, 2, 2, 1, 0]` with indices `[0, 2, 4, 3, 1]` descending. An
/// unstable network that swaps equal cells fails the descending case in
/// particular.
///
/// # Panics
///
/// Panics when equal values reorder in either direction.
pub fn assert_sort_stability<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let tied_host: Vec<T> = [2i8, 0, 2, 1, 2].map(T::from).to_vec();
    let layout = Layout::c_contiguous([5]).expect("tied layout");
    let input = device.upload(&tied_host).expect("tied upload");
    let view = || hephaestus_core::StridedView::new(&input, &layout);

    let up_values = device.alloc_zeroed::<T>(5).expect("up values alloc");
    let up_indices = device.alloc_zeroed::<u32>(5).expect("up indices alloc");
    ops.sort_into::<1>(
        device,
        view(),
        0,
        false,
        hephaestus_core::StridedView::new(&up_values, &layout),
        hephaestus_core::StridedView::new(&up_indices, &layout),
    )
    .expect("ascending stability dispatch");
    let mut up_got = vec![T::from(0i8); 5];
    let mut up_idx = vec![0u32; 5];
    device.download(&up_values, &mut up_got).expect("download");
    device.download(&up_indices, &mut up_idx).expect("download");
    let up_expected: Vec<T> = [0i8, 1, 2, 2, 2].map(T::from).to_vec();
    assert_eq!(
        up_got, up_expected,
        "{name}: ascending tied values must match exactly"
    );
    assert_eq!(
        up_idx,
        vec![1u32, 3, 0, 2, 4],
        "{name}: ascending ties must keep input order"
    );

    let down_values = device.alloc_zeroed::<T>(5).expect("down values alloc");
    let down_indices = device.alloc_zeroed::<u32>(5).expect("down indices alloc");
    ops.sort_into::<1>(
        device,
        view(),
        0,
        true,
        hephaestus_core::StridedView::new(&down_values, &layout),
        hephaestus_core::StridedView::new(&down_indices, &layout),
    )
    .expect("descending stability dispatch");
    let mut down_got = vec![T::from(0i8); 5];
    let mut down_idx = vec![0u32; 5];
    device
        .download(&down_values, &mut down_got)
        .expect("download");
    device
        .download(&down_indices, &mut down_idx)
        .expect("download");
    let down_expected: Vec<T> = [2i8, 2, 2, 1, 0].map(T::from).to_vec();
    assert_eq!(
        down_got, down_expected,
        "{name}: descending tied values must match exactly"
    );
    assert_eq!(
        down_idx,
        vec![0u32, 2, 4, 3, 1],
        "{name}: descending ties must keep input order"
    );
}
