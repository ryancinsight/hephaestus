//! Contract clauses for the device-neutral [`hephaestus_core::NonzeroOps`] seam.
//!
//! There is no `leto::nonzero` to delegate to, so the host reference
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures: scan the input in logical
//! row-major order, write one `u32` coordinate row per nonzero cell
//! into the worst-case `[numel, N]` coordinates, and write the nonzero
//! count `N` into the single count cell — the `torch.nonzero` semantic
//! `coeus_ops::nonzero` documents, except the coordinates are `u32`
//! rather than `T`-encoded and the count crosses explicitly. Fixtures
//! use small signed integers so every coordinate is an exact equality,
//! never a tolerance-bounded comparison.
//!
//! The tail rows (`N..numel`) are pinned untouched: every coordinates
//! buffer uploads pre-filled with a sentinel, and the clauses assert
//! the sentinel survives past row `N`, so a backend that fills or
//! overruns the tail fails. `NaN` counts as nonzero (unordered
//! compares unequal to zero) except where the dialect opts out: on a
//! non-advertising dialect the `NaN`-mixed fixtures skip wholesale by
//! the ADR 0043 const branch, since compaction packs rows densely and
//! one dropped `NaN` row shifts every later row.

use hephaestus_core::{ComputeDevice, HephaestusError, NonzeroOps};
use leto::Layout;

/// Sentinel pre-filling every coordinates buffer: tail rows must still
/// hold it after dispatch. No valid coordinate reaches it — extents
/// beyond `u32` are rejected before dispatch.
const SENTINEL: u32 = u32::MAX;

/// Scan a 1-D and a 2-D input against hand-computed fixtures: `[0, 2,
/// 0, 3]` yields the count `2` with rows `[1]`, `[3]`, and `[[0, 5,
/// 0], [6, 7, 0]]` yields the count `3` with rows `(0, 1)`, `(1, 0)`,
/// `(1, 1)`. Every fixture asserts the count cell, the leading rows,
/// and the sentinel surviving in every tail row.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with any
/// fixture.
pub fn assert_nonzero_value_contract<D, Z, T>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // 1-D: [0, 2, 0, 3] has nonzeros at flat 1 and 3.
    let lane_host: Vec<T> = [0i8, 2, 0, 3].map(T::from).to_vec();
    let lane_layout = Layout::c_contiguous([4]).expect("lane layout");
    let lane_input = device.upload(&lane_host).expect("lane upload");
    let lane_coords_layout = Layout::c_contiguous([4, 1]).expect("lane coords layout");
    let lane_coords = device.upload(&[SENTINEL; 4]).expect("lane coords upload");
    let count_layout = Layout::c_contiguous([1]).expect("count layout");
    let lane_count = device.alloc_zeroed::<u32>(1).expect("lane count alloc");
    ops.nonzero_into::<1>(
        device,
        hephaestus_core::StridedView::new(&lane_input, &lane_layout),
        hephaestus_core::StridedView::new(&lane_coords, &lane_coords_layout),
        hephaestus_core::StridedView::new(&lane_count, &count_layout),
    )
    .expect("1-D nonzero dispatch");
    let mut lane_got = vec![0u32; 4];
    device
        .download(&lane_coords, &mut lane_got)
        .expect("download");
    let mut lane_n = vec![0u32; 1];
    device.download(&lane_count, &mut lane_n).expect("download");
    assert_eq!(lane_n, vec![2u32], "{name}: 1-D count must be 2");
    assert_eq!(
        &lane_got[..2],
        &[1u32, 3],
        "{name}: 1-D rows must match the hand-computed coordinates exactly"
    );
    assert_eq!(
        &lane_got[2..],
        &[SENTINEL, SENTINEL],
        "{name}: 1-D tail rows must stay untouched"
    );

    // 2-D: [[0, 5, 0], [6, 7, 0]] has nonzeros at (0,1), (1,0), (1,1)
    // in row-major order.
    let square_host: Vec<T> = [0i8, 5, 0, 6, 7, 0].map(T::from).to_vec();
    let square_layout = Layout::c_contiguous([2, 3]).expect("square layout");
    let square_input = device.upload(&square_host).expect("square upload");
    let square_coords_layout = Layout::c_contiguous([6, 2]).expect("square coords layout");
    let square_coords = device
        .upload(&[SENTINEL; 12])
        .expect("square coords upload");
    let square_count = device.alloc_zeroed::<u32>(1).expect("square count alloc");
    ops.nonzero_into::<2>(
        device,
        hephaestus_core::StridedView::new(&square_input, &square_layout),
        hephaestus_core::StridedView::new(&square_coords, &square_coords_layout),
        hephaestus_core::StridedView::new(&square_count, &count_layout),
    )
    .expect("2-D nonzero dispatch");
    let mut square_got = vec![0u32; 12];
    device
        .download(&square_coords, &mut square_got)
        .expect("download");
    let mut square_n = vec![0u32; 1];
    device
        .download(&square_count, &mut square_n)
        .expect("download");
    assert_eq!(square_n, vec![3u32], "{name}: 2-D count must be 3");
    assert_eq!(
        &square_got[..6],
        &[0u32, 1, 1, 0, 1, 1],
        "{name}: 2-D rows must match the hand-computed coordinates exactly"
    );
    assert_eq!(
        &square_got[6..],
        &[SENTINEL; 6],
        "{name}: 2-D tail rows must stay untouched"
    );
}

/// An all-zero input yields the count `0` and writes no coordinate
/// row: the whole sentinel-filled coordinates buffer must survive.
///
/// # Panics
///
/// Panics when the count is nonzero or any coordinates cell changes.
pub fn assert_nonzero_all_zero_yields_empty<D, Z, T>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 2]).expect("layout");
    let input = device.alloc_zeroed::<T>(4).expect("input alloc");
    let coords_layout = Layout::c_contiguous([4, 2]).expect("coords layout");
    let coords = device.upload(&[SENTINEL; 8]).expect("coords upload");
    let count_layout = Layout::c_contiguous([1]).expect("count layout");
    let count = device.alloc_zeroed::<u32>(1).expect("count alloc");

    ops.nonzero_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&coords, &coords_layout),
        hephaestus_core::StridedView::new(&count, &count_layout),
    )
    .expect("all-zero nonzero dispatch");

    let mut got = vec![0u32; 8];
    device.download(&coords, &mut got).expect("download");
    let mut n = vec![0u32; 1];
    device.download(&count, &mut n).expect("download");
    assert_eq!(n, vec![0u32], "{name}: all-zero count must be 0");
    assert_eq!(
        got,
        vec![SENTINEL; 8],
        "{name}: all-zero input must write no coordinate row"
    );
}

/// The float zero test is pinned by IEEE equality in two layers. Every
/// backend, on every dialect, must treat `-0.0` as zero: `[-0.0, 1.0,
/// 0.0, -2.0]` yields the count `2` with rows `[1]`, `[3]` — a naive
/// bitwise nonzero test fails the signed-zero cell.
///
/// The `NaN` fixture asserts only where the dialect advertises IEEE
/// special values (`KernelDialect::IEEE_SPECIAL_VALUES`, ADR 0043), and
/// it skips wholesale rather than per cell: compaction packs rows
/// densely, so one unordered `NaN` comparison shifts every later row on
/// a non-advertising dialect. Where it runs, `NaN` counts as nonzero:
/// `[NaN, 0.0, 1.0]` yields the count `2` with rows `[0]`, `[2]`.
///
/// # Panics
///
/// Panics when the backend miscounts a special-values fixture.
pub fn assert_nonzero_special_float_values<D, Z>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, f32>,
{
    let name = device.backend_name();
    let count_layout = Layout::c_contiguous([1]).expect("count layout");

    // No NaN: every backend asserts.
    let clean_host: Vec<f32> = vec![-0.0, 1.0, 0.0, -2.0];
    let clean_layout = Layout::c_contiguous([4]).expect("clean layout");
    let clean_input = device.upload(&clean_host).expect("clean upload");
    let clean_coords_layout = Layout::c_contiguous([4, 1]).expect("clean coords layout");
    let clean_coords = device.upload(&[SENTINEL; 4]).expect("clean coords upload");
    let clean_count = device.alloc_zeroed::<u32>(1).expect("clean count alloc");
    ops.nonzero_into::<1>(
        device,
        hephaestus_core::StridedView::new(&clean_input, &clean_layout),
        hephaestus_core::StridedView::new(&clean_coords, &clean_coords_layout),
        hephaestus_core::StridedView::new(&clean_count, &count_layout),
    )
    .expect("clean nonzero dispatch");
    let mut clean_got = vec![0u32; 4];
    device
        .download(&clean_coords, &mut clean_got)
        .expect("download");
    let mut clean_n = vec![0u32; 1];
    device
        .download(&clean_count, &mut clean_n)
        .expect("download");
    assert_eq!(clean_n, vec![2u32], "{name}: clean count must be 2");
    assert_eq!(
        &clean_got[..2],
        &[1u32, 3],
        "{name}: -0.0 must count as zero"
    );
    assert_eq!(
        &clean_got[2..],
        &[SENTINEL, SENTINEL],
        "{name}: clean tail rows must stay untouched"
    );

    // NaN-mixed: only IEEE-advertising dialects assert, wholesale.
    if <Z::Dialect as hephaestus_core::KernelDialect>::IEEE_SPECIAL_VALUES {
        let nan_host: Vec<f32> = vec![f32::NAN, 0.0, 1.0];
        let nan_layout = Layout::c_contiguous([3]).expect("nan layout");
        let nan_input = device.upload(&nan_host).expect("nan upload");
        let nan_coords_layout = Layout::c_contiguous([3, 1]).expect("nan coords layout");
        let nan_coords = device.upload(&[SENTINEL; 3]).expect("nan coords upload");
        let nan_count = device.alloc_zeroed::<u32>(1).expect("nan count alloc");
        ops.nonzero_into::<1>(
            device,
            hephaestus_core::StridedView::new(&nan_input, &nan_layout),
            hephaestus_core::StridedView::new(&nan_coords, &nan_coords_layout),
            hephaestus_core::StridedView::new(&nan_count, &count_layout),
        )
        .expect("nan nonzero dispatch");
        let mut nan_got = vec![0u32; 3];
        device
            .download(&nan_coords, &mut nan_got)
            .expect("download");
        let mut nan_n = vec![0u32; 1];
        device.download(&nan_count, &mut nan_n).expect("download");
        assert_eq!(nan_n, vec![2u32], "{name}: NaN must count as nonzero");
        assert_eq!(
            &nan_got[..2],
            &[0u32, 2],
            "{name}: NaN row must lead in row-major order"
        );
        assert_eq!(
            &nan_got[2..],
            &[SENTINEL],
            "{name}: NaN tail row must stay untouched"
        );
    }
}

/// An empty shape scans vacuously: matching `[2, 0]` input, `[0, 2]`
/// coordinates, and `[1]` count dispatch successfully, write zero
/// coordinate rows, and report the count `0`.
///
/// # Panics
///
/// Panics when the vacuous scan fails to dispatch or miscounts.
pub fn assert_nonzero_of_empty_is_vacuous<D, Z, T>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let input = device.alloc_zeroed::<T>(0).expect("input alloc");
    let coords_layout = Layout::c_contiguous([0, 2]).expect("empty coords layout");
    let coords = device.alloc_zeroed::<u32>(0).expect("coords alloc");
    let count_layout = Layout::c_contiguous([1]).expect("count layout");
    let count = device.alloc_zeroed::<u32>(1).expect("count alloc");

    ops.nonzero_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&coords, &coords_layout),
        hephaestus_core::StridedView::new(&count, &count_layout),
    )
    .expect("vacuous nonzero must dispatch");

    let mut n = vec![0u32; 1];
    device.download(&count, &mut n).expect("download");
    assert_eq!(n, vec![0u32], "{name}: vacuous count must be 0");
}

/// Invalid geometry must fail typed: short coordinates rows, wrong
/// coordinates columns, and a non-scalar count each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_nonzero_rejects_invalid_geometry<D, Z, T>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let coords_layout = Layout::c_contiguous([8, 2]).expect("coords layout");
    let short_rows_layout = Layout::c_contiguous([7, 2]).expect("short rows layout");
    let wide_cols_layout = Layout::c_contiguous([8, 3]).expect("wide cols layout");
    let count_layout = Layout::c_contiguous([1]).expect("count layout");
    let long_count_layout = Layout::c_contiguous([2]).expect("long count layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let coords = device.alloc_zeroed::<u32>(16).expect("coords alloc");
    let short_rows = device.alloc_zeroed::<u32>(14).expect("short rows alloc");
    let wide_cols = device.alloc_zeroed::<u32>(24).expect("wide cols alloc");
    let count = device.alloc_zeroed::<u32>(1).expect("count alloc");
    let long_count = device.alloc_zeroed::<u32>(2).expect("long count alloc");
    let inp = || hephaestus_core::StridedView::new(&input, &layout);
    let crd = || hephaestus_core::StridedView::new(&coords, &coords_layout);
    let cnt = || hephaestus_core::StridedView::new(&count, &count_layout);

    let err = ops
        .nonzero_into::<2>(
            device,
            inp(),
            hephaestus_core::StridedView::new(&short_rows, &short_rows_layout),
            cnt(),
        )
        .expect_err("short coords rows must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short coords rows must fail typed, got {err:?}"
    );

    let err = ops
        .nonzero_into::<2>(
            device,
            inp(),
            hephaestus_core::StridedView::new(&wide_cols, &wide_cols_layout),
            cnt(),
        )
        .expect_err("wide coords columns must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: wide coords columns must fail typed, got {err:?}"
    );

    let err = ops
        .nonzero_into::<2>(
            device,
            inp(),
            crd(),
            hephaestus_core::StridedView::new(&long_count, &long_count_layout),
        )
        .expect_err("non-scalar count must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: non-scalar count must fail typed, got {err:?}"
    );
}

/// Every aliased pair must fail before mutating anything: coordinates
/// sharing the input, the count sharing the input, and the coordinates
/// sharing the count are each a dispatch error on every backend, not
/// just the device ones.
///
/// The clause runs concrete at `T = u32` so all three buffers share one
/// element type: same-type aliasing is the only aliasing constructible
/// through the safe buffer API, and it is exactly what each backend's
/// `aliases` check observes.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_nonzero_rejects_aliased_output<D, Z>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, u32>,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let coords_layout = Layout::c_contiguous([8, 2]).expect("coords layout");
    let count_layout = Layout::c_contiguous([1]).expect("count layout");

    // `which` selects the aliased pair: 0 = coords/input, 1 =
    // count/input, 2 = coords/count. The shared buffer holds the
    // larger view's cells, so every view fits its storage: the
    // geometry is fully valid and only the aliasing fails.
    for which in 0..3 {
        let host: Vec<u32> = (0..16u32).collect();
        let shared = device.upload(&host).expect("shared upload");
        let fresh_input = device.upload(&host[..8]).expect("fresh input upload");
        let fresh_coords = device.upload(&host).expect("fresh coords upload");
        let fresh_count = device.upload(&host[..1]).expect("fresh count upload");
        let shared_in = || hephaestus_core::StridedView::new(&shared, &layout);
        let shared_crd = || hephaestus_core::StridedView::new(&shared, &coords_layout);
        let shared_cnt = || hephaestus_core::StridedView::new(&shared, &count_layout);
        let fresh_in = || hephaestus_core::StridedView::new(&fresh_input, &layout);
        let fresh_crd = || hephaestus_core::StridedView::new(&fresh_coords, &coords_layout);
        let fresh_cnt = || hephaestus_core::StridedView::new(&fresh_count, &count_layout);
        let (input, coords, count) = match which {
            0 => (shared_in(), shared_crd(), fresh_cnt()),
            1 => (shared_in(), fresh_crd(), shared_cnt()),
            _ => (fresh_in(), shared_crd(), shared_cnt()),
        };

        let before = device
            .download_owned(&shared)
            .expect("alias fixture download");
        let error = ops
            .nonzero_into::<2>(device, input, coords, count)
            .expect_err("nonzero dispatch must reject an aliased pair");
        assert!(
            error.to_string().contains("must not alias"),
            "{name}: unexpected alias error for pair {which}: {error}"
        );
        let after = device
            .download_owned(&shared)
            .expect("alias fixture download");
        assert_eq!(
            after, before,
            "{name}: alias rejection for pair {which} mutated the shared buffer"
        );
    }
}

/// Run every generic clause in this module against one backend.
///
/// The `f32`-only [`assert_nonzero_special_float_values`] clause and
/// the `u32`-only [`assert_nonzero_rejects_aliased_output`] clause run
/// separately from each backend's matching instantiation, the way the
/// sort special-values clause does.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_nonzero_contract<D, Z, T>(device: &D, ops: &Z)
where
    D: ComputeDevice,
    Z: NonzeroOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_nonzero_value_contract(device, ops);
    assert_nonzero_all_zero_yields_empty(device, ops);
    assert_nonzero_of_empty_is_vacuous(device, ops);
    assert_nonzero_rejects_invalid_geometry(device, ops);
}
