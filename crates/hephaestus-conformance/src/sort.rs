//! Contract clauses for the device-neutral [`hephaestus_core::SortOps`] seam.
//!
//! There is no `leto::sort` to delegate to, so the host reference carries
//! the direct implementation and these clauses pin its value semantics
//! with hand-computed fixtures: stably sort along one axis into the
//! sorted values plus the `u32` argsort positions — the `torch.sort`
//! semantic `coeus_ops::sort` documents, except the indices are `u32`
//! rather than `T`-encoded. Fixtures use small signed integers so every
//! comparison is an exact equality, never a tolerance-bounded one.
//!
//! Stability is pinned observably through the indices: equal values keep
//! their input order in both directions, which a `(value, position)`
//! total order guarantees but an unstable network/net cannot. `NaN`
//! sorts as greater than every other value — last ascending, first
//! descending — except where the dialect opts out: on a non-advertising
//! dialect an unordered `NaN` comparison can corrupt the whole lane's
//! order, so the `NaN`-mixed fixtures skip wholesale by the ADR 0043
//! const branch, while the `-0.0`/`-inf` fixtures assert on every
//! backend.

use hephaestus_core::{ComputeDevice, HephaestusError, SortOps};
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

/// The float order edges are pinned by IEEE comparison in two layers.
/// Every backend, on every dialect, must sort `-0.0` as equal to `0.0`
/// (stable by input order) and `-inf` below everything finite:
/// `[-0.0, 1.0, 0.0, -inf]` sorts to `[-inf, -0.0, 0.0, 1.0]` with
/// indices `[3, 0, 2, 1]` ascending, and to `[1.0, -0.0, 0.0, -inf]`
/// with indices `[1, 0, 2, 3]` descending — a naive bitwise comparator
/// fails the signed-zero cells.
///
/// The `NaN` fixtures assert only where the dialect advertises IEEE
/// special values (`KernelDialect::IEEE_SPECIAL_VALUES`, ADR 0043), and
/// they skip wholesale rather than per cell: one unordered `NaN`
/// comparison can corrupt a whole lane's order on a non-advertising
/// dialect, so no assertion about a `NaN`-mixed lane can hold there.
/// Where they run, `NaN` sorts as greater than everything: ascending,
/// `[-0.0, NaN, 1.0, -inf, 0.0]` sorts to
/// `[-inf, -0.0, 0.0, 1.0, NaN]` with indices `[3, 0, 4, 2, 1]` and
/// `[NaN, 1.0, NaN]` to `[1.0, NaN, NaN]` with indices `[1, 0, 2]`
/// (`NaN`s stable among themselves); descending, the first lane sorts to
/// `[NaN, 1.0, -0.0, 0.0, -inf]` with indices `[1, 2, 0, 4, 3]`.
///
/// # Panics
///
/// Panics when the backend misorders a special-values fixture.
pub fn assert_sort_special_float_values<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, f32>,
{
    let name = device.backend_name();

    // No NaN: every backend asserts.
    let clean_host: Vec<f32> = vec![-0.0, 1.0, 0.0, f32::NEG_INFINITY];
    let clean_layout = Layout::c_contiguous([4]).expect("clean layout");
    let clean_input = device.upload(&clean_host).expect("clean upload");
    let clean = || hephaestus_core::StridedView::new(&clean_input, &clean_layout);
    let clean_values = device.alloc_zeroed::<f32>(4).expect("clean values alloc");
    let clean_indices = device.alloc_zeroed::<u32>(4).expect("clean indices alloc");
    ops.sort_into::<1>(
        device,
        clean(),
        0,
        false,
        hephaestus_core::StridedView::new(&clean_values, &clean_layout),
        hephaestus_core::StridedView::new(&clean_indices, &clean_layout),
    )
    .expect("clean ascending dispatch");
    let mut clean_got = vec![0.0f32; 4];
    let mut clean_idx = vec![0u32; 4];
    device
        .download(&clean_values, &mut clean_got)
        .expect("download");
    device
        .download(&clean_indices, &mut clean_idx)
        .expect("download");
    assert_eq!(
        clean_got,
        vec![f32::NEG_INFINITY, -0.0, 0.0, 1.0],
        "{name}: ascending -0.0/-inf values must match exactly"
    );
    assert_eq!(
        clean_idx,
        vec![3u32, 0, 2, 1],
        "{name}: ascending -0.0 must stay ahead of +0.0 by input order"
    );

    let clean_down_values = device.alloc_zeroed::<f32>(4).expect("clean values alloc");
    let clean_down_indices = device.alloc_zeroed::<u32>(4).expect("clean indices alloc");
    ops.sort_into::<1>(
        device,
        clean(),
        0,
        true,
        hephaestus_core::StridedView::new(&clean_down_values, &clean_layout),
        hephaestus_core::StridedView::new(&clean_down_indices, &clean_layout),
    )
    .expect("clean descending dispatch");
    let mut clean_down_got = vec![0.0f32; 4];
    let mut clean_down_idx = vec![0u32; 4];
    device
        .download(&clean_down_values, &mut clean_down_got)
        .expect("download");
    device
        .download(&clean_down_indices, &mut clean_down_idx)
        .expect("download");
    assert_eq!(
        clean_down_got,
        vec![1.0, -0.0, 0.0, f32::NEG_INFINITY],
        "{name}: descending -0.0/-inf values must match exactly"
    );
    assert_eq!(
        clean_down_idx,
        vec![1u32, 0, 2, 3],
        "{name}: descending signed zeros must keep input order"
    );

    // NaN-mixed: only IEEE-advertising dialects assert, wholesale.
    if <S::Dialect as hephaestus_core::KernelDialect>::IEEE_SPECIAL_VALUES {
        let nan_host: Vec<f32> = vec![-0.0, f32::NAN, 1.0, f32::NEG_INFINITY, 0.0];
        let nan_layout = Layout::c_contiguous([5]).expect("nan layout");
        let nan_input = device.upload(&nan_host).expect("nan upload");
        let nan = || hephaestus_core::StridedView::new(&nan_input, &nan_layout);
        let nan_values = device.alloc_zeroed::<f32>(5).expect("nan values alloc");
        let nan_indices = device.alloc_zeroed::<u32>(5).expect("nan indices alloc");
        ops.sort_into::<1>(
            device,
            nan(),
            0,
            false,
            hephaestus_core::StridedView::new(&nan_values, &nan_layout),
            hephaestus_core::StridedView::new(&nan_indices, &nan_layout),
        )
        .expect("nan ascending dispatch");
        let mut nan_got = vec![0.0f32; 5];
        let mut nan_idx = vec![0u32; 5];
        device
            .download(&nan_values, &mut nan_got)
            .expect("download");
        device
            .download(&nan_indices, &mut nan_idx)
            .expect("download");
        assert_eq!(
            nan_idx,
            vec![3u32, 0, 4, 2, 1],
            "{name}: ascending NaN lane indices must match exactly"
        );
        assert!(
            nan_got[4].is_nan(),
            "{name}: ascending NaN must sort last, got {nan_got:?}"
        );
        assert_eq!(
            &nan_got[..4],
            &[f32::NEG_INFINITY, -0.0, 0.0, 1.0],
            "{name}: ascending NaN lane finite values must match exactly"
        );

        let nan_down_values = device.alloc_zeroed::<f32>(5).expect("nan values alloc");
        let nan_down_indices = device.alloc_zeroed::<u32>(5).expect("nan indices alloc");
        ops.sort_into::<1>(
            device,
            nan(),
            0,
            true,
            hephaestus_core::StridedView::new(&nan_down_values, &nan_layout),
            hephaestus_core::StridedView::new(&nan_down_indices, &nan_layout),
        )
        .expect("nan descending dispatch");
        let mut nan_down_got = vec![0.0f32; 5];
        let mut nan_down_idx = vec![0u32; 5];
        device
            .download(&nan_down_values, &mut nan_down_got)
            .expect("download");
        device
            .download(&nan_down_indices, &mut nan_down_idx)
            .expect("download");
        assert_eq!(
            nan_down_idx,
            vec![1u32, 2, 0, 4, 3],
            "{name}: descending NaN lane indices must match exactly"
        );
        assert!(
            nan_down_got[0].is_nan(),
            "{name}: descending NaN must sort first, got {nan_down_got:?}"
        );
        assert_eq!(
            &nan_down_got[1..],
            &[1.0, -0.0, 0.0, f32::NEG_INFINITY],
            "{name}: descending NaN lane finite values must match exactly"
        );

        // NaNs stable among themselves: [NaN, 1.0, NaN] -> [1.0, NaN, NaN].
        let twins_host: Vec<f32> = vec![f32::NAN, 1.0, f32::NAN];
        let twins_layout = Layout::c_contiguous([3]).expect("twins layout");
        let twins_input = device.upload(&twins_host).expect("twins upload");
        let twins_values = device.alloc_zeroed::<f32>(3).expect("twins values alloc");
        let twins_indices = device.alloc_zeroed::<u32>(3).expect("twins indices alloc");
        ops.sort_into::<1>(
            device,
            hephaestus_core::StridedView::new(&twins_input, &twins_layout),
            0,
            false,
            hephaestus_core::StridedView::new(&twins_values, &twins_layout),
            hephaestus_core::StridedView::new(&twins_indices, &twins_layout),
        )
        .expect("nan twins dispatch");
        let mut twins_got = vec![0.0f32; 3];
        let mut twins_idx = vec![0u32; 3];
        device
            .download(&twins_values, &mut twins_got)
            .expect("download");
        device
            .download(&twins_indices, &mut twins_idx)
            .expect("download");
        assert_eq!(
            twins_idx,
            vec![1u32, 0, 2],
            "{name}: NaNs must stay stable among themselves"
        );
        assert_eq!(twins_got[0], 1.0, "{name}: NaN twins finite cell mismatch");
        assert!(
            twins_got[1].is_nan() && twins_got[2].is_nan(),
            "{name}: NaN twins must both be NaN, got {twins_got:?}"
        );
    }
}

/// An empty shape sorts vacuously: matching `[2, 0]` shapes dispatch
/// successfully and write zero cells.
///
/// # Panics
///
/// Panics when the vacuous sort fails to dispatch.
pub fn assert_sort_of_empty_is_vacuous<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let input = device.alloc_zeroed::<T>(0).expect("input alloc");
    let values = device.alloc_zeroed::<T>(0).expect("values alloc");
    let indices = device.alloc_zeroed::<u32>(0).expect("indices alloc");

    ops.sort_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        1,
        false,
        hephaestus_core::StridedView::new(&values, &layout),
        hephaestus_core::StridedView::new(&indices, &layout),
    )
    .expect("vacuous sort must dispatch");
}

/// Invalid geometry must fail typed: an out-of-range `dim`, a values
/// shape mismatch, and an indices shape mismatch each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_sort_rejects_invalid_geometry<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let short_layout = Layout::c_contiguous([2, 3]).expect("short layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let values = device.alloc_zeroed::<T>(8).expect("values alloc");
    let indices = device.alloc_zeroed::<u32>(8).expect("indices alloc");
    let short_values = device.alloc_zeroed::<T>(6).expect("short values alloc");
    let short_indices = device.alloc_zeroed::<u32>(6).expect("short indices alloc");
    let inp = || hephaestus_core::StridedView::new(&input, &layout);
    let val = || hephaestus_core::StridedView::new(&values, &layout);
    let idx = || hephaestus_core::StridedView::new(&indices, &layout);
    let shr_val = || hephaestus_core::StridedView::new(&short_values, &short_layout);
    let shr_idx = || hephaestus_core::StridedView::new(&short_indices, &short_layout);

    let err = ops
        .sort_into::<2>(device, inp(), 2, false, val(), idx())
        .expect_err("out-of-range dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range dim must fail typed, got {err:?}"
    );

    let err = ops
        .sort_into::<2>(device, inp(), 1, false, shr_val(), idx())
        .expect_err("values mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: values mismatch must fail typed, got {err:?}"
    );

    let err = ops
        .sort_into::<2>(device, inp(), 1, false, val(), shr_idx())
        .expect_err("indices mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: indices mismatch must fail typed, got {err:?}"
    );
}

/// Values aliasing the input must fail before mutating anything: every
/// thread reads the input while other threads write the values, so
/// sharing that storage is a dispatch error on every backend, not just
/// the device ones. The indices buffer is `u32` against the input's `T`
/// and can never name the same storage, so only the values alias is
/// tested.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_sort_rejects_aliased_output<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = [3i8, 1, 2, 0, 5, 4, 7, 6].map(T::from).to_vec();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let shared = device.upload(&host).expect("shared upload");
    let indices = device.alloc_zeroed::<u32>(8).expect("indices alloc");
    let shared_view = || hephaestus_core::StridedView::new(&shared, &layout);

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .sort_into::<2>(
            device,
            shared_view(),
            1,
            false,
            shared_view(),
            hephaestus_core::StridedView::new(&indices, &layout),
        )
        .expect_err("sort dispatch must reject values aliasing the input");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(
        after, before,
        "{name}: alias rejection mutated the shared buffer"
    );
}

/// Run every generic clause in this module against one backend.
///
/// The `f32`-only [`assert_sort_special_float_values`] clause runs
/// separately from each backend's `f32` instantiation, the way the
/// masked-fill and where special-values clauses do.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_sort_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_sort_value_contract(device, ops);
    assert_sort_stability(device, ops);
    assert_sort_of_empty_is_vacuous(device, ops);
    assert_sort_rejects_invalid_geometry(device, ops);
    assert_sort_rejects_aliased_output(device, ops);
}
