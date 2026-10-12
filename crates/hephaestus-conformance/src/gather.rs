//! Contract clauses for the device-neutral [`hephaestus_core::GatherOps`] seam.
//!
//! There is no `leto::gather` to delegate to, so the host reference carries
//! the direct implementation and these clauses pin its value semantics with
//! hand-computed fixtures: `output[i] = input[i with dim := index[i]]` on
//! every backend, the `torch.gather` contract `coeus_ops::gather`
//! documents. Fixtures use small signed integers so every copy is an exact
//! equality, never a tolerance-bounded comparison. Indices are `u32`,
//! matching [`hephaestus_core::EmbeddingOps`].

use hephaestus_core::{ComputeDevice, GatherOps, HephaestusError};
use leto::Layout;

/// Gather along both a middle and a leading dimension against hand-computed
/// fixtures: a 2x4 input with a 2x3 index along dim 1, and a 3x2 input with
/// a 2x2 index along dim 0.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_gather_index_contract<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // Along dim 1: out[i, j] = input[i, index[i, j]].
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 3]).expect("index layout");
    let input = device.upload(&host).expect("input upload");
    let index = device.upload(&[3u32, 0, 1, 2, 2, 0]).expect("index upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.gather_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        1,
        hephaestus_core::StridedView::new(&index, &idx_layout),
        hephaestus_core::StridedView::new(&output, &idx_layout),
    )
    .expect("gather dispatch");

    // Row 0 reads input cells 3, 0, 1; row 1 reads cells 6, 6, 4.
    let expected: Vec<T> = [3i8, 0, 1, 6, 6, 4].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: dim-1 gather must match the hand-computed fixture exactly"
    );

    // Along dim 0: out[i, j] = input[index[i, j], j].
    let tall: Vec<T> = (0..6i8).map(T::from).collect();
    let tall_layout = Layout::c_contiguous([3, 2]).expect("tall layout");
    let short_layout = Layout::c_contiguous([2, 2]).expect("short layout");
    let tall_input = device.upload(&tall).expect("tall upload");
    let tall_index = device.upload(&[2u32, 0, 1, 1]).expect("tall index upload");
    let tall_output = device.alloc_zeroed::<T>(4).expect("tall output alloc");

    ops.gather_into::<2>(
        device,
        hephaestus_core::StridedView::new(&tall_input, &tall_layout),
        0,
        hephaestus_core::StridedView::new(&tall_index, &short_layout),
        hephaestus_core::StridedView::new(&tall_output, &short_layout),
    )
    .expect("gather dispatch");

    // Cells read: input[2, 0], input[0, 1], input[1, 0], input[1, 1].
    let tall_expected: Vec<T> = [4i8, 1, 2, 3].map(T::from).to_vec();
    let mut tall_got = vec![T::from(0i8); 4];
    device
        .download(&tall_output, &mut tall_got)
        .expect("download");
    assert_eq!(
        tall_got, tall_expected,
        "{name}: dim-0 gather must match the hand-computed fixture exactly"
    );
}

/// An identity index must reproduce the input exactly: the seam copies, and
/// any zero-copy fast path lives with the caller, as `coeus_ops::gather`
/// documents.
///
/// # Panics
///
/// Panics when an identity-index gather changes any element.
pub fn assert_gather_identity_index_is_identity<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");

    let input = device.upload(&host).expect("input upload");
    let index = device.upload(&[0u32, 1, 2, 0, 1, 2]).expect("index upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.gather_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        1,
        hephaestus_core::StridedView::new(&index, &layout),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("gather dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: identity-index gather must reproduce the input exactly"
    );
}

/// An empty index gathers vacuously: a `[2, 0]` index over a `[2, 4]` input
/// dispatches successfully and writes zero cells.
///
/// # Panics
///
/// Panics when the vacuous gather fails to dispatch.
pub fn assert_gather_of_empty_index_is_vacuous<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let empty_layout = Layout::c_contiguous([2, 0]).expect("empty layout");

    let input = device.upload(&host).expect("input upload");
    let index = device.alloc_zeroed::<u32>(0).expect("index alloc");
    let output = device.alloc_zeroed::<T>(0).expect("output alloc");

    ops.gather_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        1,
        hephaestus_core::StridedView::new(&index, &empty_layout),
        hephaestus_core::StridedView::new(&output, &empty_layout),
    )
    .expect("vacuous gather must dispatch");
}

/// Invalid geometry must fail typed: an out-of-range dim, an input/index
/// non-dim mismatch, and an output shape missing the index shape (on both
/// the gather dim and a non-gather dim) each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_gather_rejects_invalid_geometry<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 3]).expect("index layout");
    let other = Layout::c_contiguous([5, 3]).expect("mismatched layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let index = device.alloc_zeroed::<u32>(6).expect("index alloc");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &in_layout);
    let idx = || hephaestus_core::StridedView::new(&index, &idx_layout);
    let out = || hephaestus_core::StridedView::new(&output, &idx_layout);

    let err = ops
        .gather_into::<2>(device, view(), 2, idx(), out())
        .expect_err("out-of-range dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range dim must fail typed, got {err:?}"
    );

    let mistyped = device.alloc_zeroed::<u32>(15).expect("mismatched alloc");
    let err = ops
        .gather_into::<2>(
            device,
            view(),
            1,
            hephaestus_core::StridedView::new(&mistyped, &other),
            out(),
        )
        .expect_err("shape mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: shape mismatch must fail typed, got {err:?}"
    );

    let short_layout = Layout::c_contiguous([2, 2]).expect("short layout");
    let short = device.alloc_zeroed::<T>(4).expect("short alloc");
    let err = ops
        .gather_into::<2>(
            device,
            view(),
            1,
            idx(),
            hephaestus_core::StridedView::new(&short, &short_layout),
        )
        .expect_err("short output dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short output dim must fail typed, got {err:?}"
    );

    let wide_layout = Layout::c_contiguous([5, 3]).expect("wide layout");
    let wide = device.alloc_zeroed::<T>(15).expect("wide alloc");
    let err = ops
        .gather_into::<2>(
            device,
            view(),
            1,
            idx(),
            hephaestus_core::StridedView::new(&wide, &wide_layout),
        )
        .expect_err("wide output non-dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: wide output non-dim must fail typed, got {err:?}"
    );
}

/// An output aliasing the input must fail before mutating anything: reads
/// and writes race across threads, so sharing storage is a dispatch error
/// on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_gather_rejects_aliased_output<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 3]).expect("index layout");

    // The shared buffer holds all eight input cells, so both the input view
    // and the smaller output view fit its storage: the geometry is fully
    // valid and only the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");
    let index = device.upload(&[0u32, 1, 2, 0, 1, 2]).expect("index upload");

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .gather_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &in_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&shared, &idx_layout),
        )
        .expect_err("gather dispatch must reject aliased input/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: alias rejection mutated input");
}

/// An index value at or past the input's `dim` extent must fail typed with
/// an out-of-range message: a data-dependent fault is surfaced, never
/// clamped or wrapped, on every backend.
///
/// # Panics
///
/// Panics when an out-of-range index dispatches or fails with another error.
pub fn assert_gather_rejects_out_of_range_index<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 3]).expect("index layout");

    let input = device.upload(&host).expect("input upload");
    // Index 4 is out of range for the dim-1 extent of 4.
    let index = device.upload(&[0u32, 1, 2, 3, 4, 0]).expect("index upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    let error = ops
        .gather_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &in_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&output, &idx_layout),
        )
        .expect_err("out-of-range index must fail");
    assert!(
        matches!(error, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range index must fail typed, got {error:?}"
    );
    assert!(
        error.to_string().contains("out of range"),
        "{name}: unexpected out-of-range error: {error}"
    );
}

/// Run every clause in this module against one backend.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_gather_contract<D, G, T>(device: &D, ops: &G)
where
    D: ComputeDevice,
    G: GatherOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_gather_index_contract(device, ops);
    assert_gather_identity_index_is_identity(device, ops);
    assert_gather_of_empty_index_is_vacuous(device, ops);
    assert_gather_rejects_invalid_geometry(device, ops);
    assert_gather_rejects_aliased_output(device, ops);
    assert_gather_rejects_out_of_range_index(device, ops);
}
