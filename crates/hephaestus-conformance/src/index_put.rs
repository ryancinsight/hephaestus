//! Contract clauses for the device-neutral [`hephaestus_core::IndexPutOps`] seam.
//!
//! There is no `leto::index_put` to delegate to, so the host reference
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures: the output starts as a copy of
//! the input, then each selection writes its values slice at the indexed
//! position along `dim`, replacing or accumulating — the
//! `torch.index_put` contract `coeus_ops::index_put` documents, lifted from
//! its dim-0-only entry point to any axis. Fixtures use small signed
//! integers so every copy and sum is an exact equality, never a
//! tolerance-bounded comparison. Indices are `u32`, matching
//! [`hephaestus_core::EmbeddingOps`].
//!
//! Duplicate indices in replace mode are deliberately unpinned, exactly as
//! `torch.index_put` leaves them undefined: no clause below selects the
//! same target twice without `accumulate`.

use hephaestus_core::{ComputeDevice, HephaestusError, IndexPutOps};
use leto::Layout;

/// Scatter-assign along both a middle and a leading dimension against
/// hand-computed fixtures: a 2x4 base with three length-3 updates along
/// dim 1, and a 4x2 base with two updates along dim 0. Untouched cells
/// keep their base values.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_index_put_replace_contract<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // Along dim 1: out[i, index[s]] = values[i, s], other cells kept.
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let base_layout = Layout::c_contiguous([2, 4]).expect("base layout");
    let vals_layout = Layout::c_contiguous([2, 3]).expect("values layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");
    let input = device.upload(&host).expect("input upload");
    let index = device.upload(&[3u32, 0, 1]).expect("index upload");
    let vals_host: Vec<T> = (10..16i8).map(T::from).collect();
    let values = device.upload(&vals_host).expect("values upload");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");

    ops.index_put_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &base_layout),
        1,
        hephaestus_core::StridedView::new(&index, &idx_layout),
        hephaestus_core::StridedView::new(&values, &vals_layout),
        false,
        hephaestus_core::StridedView::new(&output, &base_layout),
    )
    .expect("index_put dispatch");

    // Row 0: col 3 <- 10, col 0 <- 11, col 1 <- 12, col 2 kept.
    // Row 1: col 3 <- 13, col 0 <- 14, col 1 <- 15, col 2 kept.
    let expected: Vec<T> = [11i8, 12, 2, 10, 14, 15, 6, 13].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 8];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: dim-1 index_put must match the hand-computed fixture exactly"
    );

    // Along dim 0: out[index[s], j] = values[s, j], other rows kept.
    let tall: Vec<T> = (0..8i8).map(T::from).collect();
    let tall_layout = Layout::c_contiguous([4, 2]).expect("tall layout");
    let tall_vals = Layout::c_contiguous([2, 2]).expect("tall values layout");
    let short_idx = Layout::c_contiguous([2]).expect("short index layout");
    let tall_input = device.upload(&tall).expect("tall upload");
    let tall_index = device.upload(&[3u32, 1]).expect("tall index upload");
    let tall_values_host: Vec<T> = [20i8, 21, 22, 23].map(T::from).to_vec();
    let tall_values = device
        .upload(&tall_values_host)
        .expect("tall values upload");
    let tall_output = device.alloc_zeroed::<T>(8).expect("tall output alloc");

    ops.index_put_into::<2>(
        device,
        hephaestus_core::StridedView::new(&tall_input, &tall_layout),
        0,
        hephaestus_core::StridedView::new(&tall_index, &short_idx),
        hephaestus_core::StridedView::new(&tall_values, &tall_vals),
        false,
        hephaestus_core::StridedView::new(&tall_output, &tall_layout),
    )
    .expect("index_put dispatch");

    // Row 3 <- [20, 21], row 1 <- [22, 23], rows 0 and 2 kept.
    let tall_expected: Vec<T> = [0i8, 1, 22, 23, 4, 5, 20, 21].map(T::from).to_vec();
    let mut tall_got = vec![T::from(0i8); 8];
    device
        .download(&tall_output, &mut tall_got)
        .expect("download");
    assert_eq!(
        tall_got, tall_expected,
        "{name}: dim-0 index_put must match the hand-computed fixture exactly"
    );
}

/// Accumulated updates add onto the base in ascending selection order:
/// repeated indices sum every contribution, and the base cell seeds the
/// total.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_index_put_accumulate_contract<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = [1i8, 2, 3, 4].map(T::from).to_vec();
    let base_layout = Layout::c_contiguous([4]).expect("base layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");

    let input = device.upload(&host).expect("input upload");
    let index = device.upload(&[0u32, 0, 2]).expect("index upload");
    let values_host: Vec<T> = [5i8, 3, 10].map(T::from).to_vec();
    let values = device.upload(&values_host).expect("values upload");
    let output = device.alloc_zeroed::<T>(4).expect("output alloc");

    ops.index_put_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &base_layout),
        0,
        hephaestus_core::StridedView::new(&index, &idx_layout),
        hephaestus_core::StridedView::new(&values, &idx_layout),
        true,
        hephaestus_core::StridedView::new(&output, &base_layout),
    )
    .expect("index_put dispatch");

    // Cell 0 seeds from 1 and adds 5 then 3; cell 2 seeds from 3, adds 10.
    let expected: Vec<T> = [9i8, 2, 13, 4].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 4];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: accumulated index_put must match the hand-computed fixture exactly"
    );
}

/// An empty index updates vacuously: a length-0 index over a `[2, 4]`
/// base dispatches successfully and reproduces the base exactly.
///
/// # Panics
///
/// Panics when the vacuous update fails to dispatch or changes any cell.
pub fn assert_index_put_of_empty_index_is_vacuous<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let base_layout = Layout::c_contiguous([2, 4]).expect("base layout");
    let vals_layout = Layout::c_contiguous([2, 0]).expect("values layout");
    let idx_layout = Layout::c_contiguous([0]).expect("index layout");

    let input = device.upload(&host).expect("input upload");
    let index = device.alloc_zeroed::<u32>(0).expect("index alloc");
    let values = device.alloc_zeroed::<T>(0).expect("values alloc");

    for accumulate in [false, true] {
        let output = device.alloc_zeroed::<T>(8).expect("output alloc");
        ops.index_put_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &base_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&values, &vals_layout),
            accumulate,
            hephaestus_core::StridedView::new(&output, &base_layout),
        )
        .expect("vacuous index_put must dispatch");

        let mut got = vec![T::from(0i8); 8];
        device.download(&output, &mut got).expect("download");
        assert_eq!(
            got, host,
            "{name}: vacuous index_put (accumulate={accumulate}) must reproduce the base"
        );
    }
}

/// Invalid geometry must fail typed: an out-of-range dim, an output shape
/// that is not the input shape, a values non-dim mismatch, and a values
/// dim extent missing the index length each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_index_put_rejects_invalid_geometry<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let base_layout = Layout::c_contiguous([2, 4]).expect("base layout");
    let vals_layout = Layout::c_contiguous([2, 3]).expect("values layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let index = device.alloc_zeroed::<u32>(3).expect("index alloc");
    let values = device.alloc_zeroed::<T>(6).expect("values alloc");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &base_layout);
    let idx = || hephaestus_core::StridedView::new(&index, &idx_layout);
    let vals = || hephaestus_core::StridedView::new(&values, &vals_layout);
    let out = || hephaestus_core::StridedView::new(&output, &base_layout);

    let err = ops
        .index_put_into::<2>(device, view(), 2, idx(), vals(), false, out())
        .expect_err("out-of-range dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range dim must fail typed, got {err:?}"
    );

    let wide_layout = Layout::c_contiguous([5, 4]).expect("wide layout");
    let wide = device.alloc_zeroed::<T>(20).expect("wide alloc");
    let err = ops
        .index_put_into::<2>(
            device,
            view(),
            1,
            idx(),
            vals(),
            false,
            hephaestus_core::StridedView::new(&wide, &wide_layout),
        )
        .expect_err("output shape mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: output shape mismatch must fail typed, got {err:?}"
    );

    let skewed_layout = Layout::c_contiguous([5, 3]).expect("skewed layout");
    let skewed = device.alloc_zeroed::<T>(15).expect("skewed alloc");
    let err = ops
        .index_put_into::<2>(
            device,
            view(),
            1,
            idx(),
            hephaestus_core::StridedView::new(&skewed, &skewed_layout),
            false,
            out(),
        )
        .expect_err("values non-dim mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: values non-dim mismatch must fail typed, got {err:?}"
    );

    let short_layout = Layout::c_contiguous([2, 2]).expect("short layout");
    let short = device.alloc_zeroed::<T>(4).expect("short alloc");
    let err = ops
        .index_put_into::<2>(
            device,
            view(),
            1,
            idx(),
            hephaestus_core::StridedView::new(&short, &short_layout),
            false,
            out(),
        )
        .expect_err("short values dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short values dim must fail typed, got {err:?}"
    );
}

/// An output aliasing either source must fail before mutating anything:
/// reads and writes race across threads, so sharing storage is a dispatch
/// error on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_index_put_rejects_aliased_output<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let base_layout = Layout::c_contiguous([2, 4]).expect("base layout");
    let vals_layout = Layout::c_contiguous([2, 3]).expect("values layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");

    // The shared buffer holds all eight base cells, so the base, values,
    // and output views all fit its storage: the geometry is fully valid
    // and only the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");
    let index = device.upload(&[0u32, 1, 2]).expect("index upload");
    let values = device.alloc_zeroed::<T>(6).expect("values alloc");

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .index_put_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &base_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&values, &vals_layout),
            false,
            hephaestus_core::StridedView::new(&shared, &base_layout),
        )
        .expect_err("index_put dispatch must reject aliased input/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected input-alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: input-alias rejection mutated input");

    let input = device.upload(&host).expect("input upload");
    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .index_put_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &base_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&shared, &vals_layout),
            false,
            hephaestus_core::StridedView::new(&shared, &base_layout),
        )
        .expect_err("index_put dispatch must reject aliased values/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected values-alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(
        after, before,
        "{name}: values-alias rejection mutated values"
    );
}

/// An index value at or past the input's `dim` extent must fail typed with
/// an out-of-range message in both modes: a data-dependent fault is
/// surfaced, never clamped or wrapped, on every backend.
///
/// # Panics
///
/// Panics when an out-of-range index dispatches or fails with another error.
pub fn assert_index_put_rejects_out_of_range_index<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let base_layout = Layout::c_contiguous([2, 4]).expect("base layout");
    let vals_layout = Layout::c_contiguous([2, 3]).expect("values layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");

    let input = device.upload(&host).expect("input upload");
    // Index 4 is out of range for the dim-1 extent of 4.
    let index = device.upload(&[0u32, 4, 1]).expect("index upload");
    let values = device.alloc_zeroed::<T>(6).expect("values alloc");

    for accumulate in [false, true] {
        let output = device.alloc_zeroed::<T>(8).expect("output alloc");
        let error = ops
            .index_put_into::<2>(
                device,
                hephaestus_core::StridedView::new(&input, &base_layout),
                1,
                hephaestus_core::StridedView::new(&index, &idx_layout),
                hephaestus_core::StridedView::new(&values, &vals_layout),
                accumulate,
                hephaestus_core::StridedView::new(&output, &base_layout),
            )
            .expect_err("out-of-range index must fail");
        assert!(
            matches!(error, HephaestusError::InvalidConfiguration { .. }),
            "{name}: out-of-range index (accumulate={accumulate}) must fail typed, got {error:?}"
        );
        assert!(
            error.to_string().contains("out of range"),
            "{name}: unexpected out-of-range error: {error}"
        );
    }
}

/// Run every clause in this module against one backend.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_index_put_contract<D, I, T>(device: &D, ops: &I)
where
    D: ComputeDevice,
    I: IndexPutOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_index_put_replace_contract(device, ops);
    assert_index_put_accumulate_contract(device, ops);
    assert_index_put_of_empty_index_is_vacuous(device, ops);
    assert_index_put_rejects_invalid_geometry(device, ops);
    assert_index_put_rejects_aliased_output(device, ops);
    assert_index_put_rejects_out_of_range_index(device, ops);
}
