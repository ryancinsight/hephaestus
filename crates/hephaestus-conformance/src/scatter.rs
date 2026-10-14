//! Contract clauses for the device-neutral [`hephaestus_core::ScatterOps`] seam.
//!
//! There is no `leto::scatter_add` to delegate to, so the host reference
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures: copy `input` to the output, then
//! accumulate each `src` element into the output position with the `dim`
//! lane replaced by the index there, in ascending source order on every
//! backend, the `torch.scatter_add` contract
//! `coeus_ops::scatter_add` documents. Integer fixtures use small signed
//! values so every copy is an exact equality; the accumulation-order clause
//! uses `f32` values whose sum provably differs under any other order.
//! Indices are `u32`, matching [`hephaestus_core::EmbeddingOps`].

use hephaestus_core::{ComputeDevice, HephaestusError, ScatterOps};
use leto::Layout;

/// Scatter-add along dim 1 against a hand-computed fixture: a nonzero 2x4
/// base, a 2x2 index, and a 2x2 source.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_scatter_index_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (10..18i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 2]).expect("index layout");
    let src: Vec<T> = [1i8, 2, 4, 8].map(T::from).to_vec();

    let input = device.upload(&host).expect("input upload");
    let index = device.upload(&[3u32, 0, 1, 2]).expect("index upload");
    let src_buf = device.upload(&src).expect("src upload");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");

    ops.scatter_add_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        1,
        hephaestus_core::StridedView::new(&index, &idx_layout),
        hephaestus_core::StridedView::new(&src_buf, &idx_layout),
        hephaestus_core::StridedView::new(&output, &in_layout),
    )
    .expect("scatter dispatch");

    // (0,0): out[0,3] += 1 -> 14; (0,1): out[0,0] += 2 -> 12;
    // (1,0): out[1,1] += 4 -> 19; (1,1): out[1,2] += 8 -> 24.
    let expected: Vec<T> = [12i8, 11, 12, 14, 14, 19, 24, 17].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 8];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: scatter-add must match the hand-computed fixture exactly"
    );
}

/// Repeated indices accumulate every contribution: three updates into one
/// cell sum all three sources.
///
/// # Panics
///
/// Panics when a repeated index drops or duplicates any contribution.
pub fn assert_scatter_repeated_indices_accumulate<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = vec![T::from(0i8); 3];
    let in_layout = Layout::c_contiguous([3]).expect("input layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");
    let src: Vec<T> = [1i8, 2, 4].map(T::from).to_vec();

    let input = device.upload(&host).expect("input upload");
    let index = device.upload(&[2u32, 0, 2]).expect("index upload");
    let src_buf = device.upload(&src).expect("src upload");
    let output = device.alloc_zeroed::<T>(3).expect("output alloc");

    ops.scatter_add_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        0,
        hephaestus_core::StridedView::new(&index, &idx_layout),
        hephaestus_core::StridedView::new(&src_buf, &idx_layout),
        hephaestus_core::StridedView::new(&output, &in_layout),
    )
    .expect("scatter dispatch");

    // Cell 2 takes sources 1 and 4; cell 0 takes source 2.
    let expected: Vec<T> = [2i8, 0, 5].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 3];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: repeated indices must accumulate every contribution"
    );
}

/// Accumulation runs in ascending source order, observably: summing
/// `1e16`, `-1e16`, `1.0` in `f32` is exactly `1.0` ascending, but `0.0`
/// when the `1.0` folds into the large magnitude first, so any other order
/// fails this clause.
///
/// # Panics
///
/// Panics when the backend accumulates in any other order.
pub fn assert_scatter_accumulates_in_ascending_source_order<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, f32>,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([1]).expect("input layout");
    let idx_layout = Layout::c_contiguous([3]).expect("index layout");

    let input = device.upload(&[0.0f32]).expect("input upload");
    let index = device.upload(&[0u32, 0, 0]).expect("index upload");
    let src = device
        .upload(&[1e16f32, -1e16f32, 1.0])
        .expect("src upload");
    let output = device.alloc_zeroed::<f32>(1).expect("output alloc");

    ops.scatter_add_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        0,
        hephaestus_core::StridedView::new(&index, &idx_layout),
        hephaestus_core::StridedView::new(&src, &idx_layout),
        hephaestus_core::StridedView::new(&output, &in_layout),
    )
    .expect("scatter dispatch");

    let mut got = [0.0f32; 1];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got,
        [1.0f32],
        "{name}: scatter-add must accumulate in ascending source order"
    );
}

/// An empty index scatters vacuously: the output reproduces the input with
/// no updates applied.
///
/// # Panics
///
/// Panics when the vacuous scatter fails or changes any element.
pub fn assert_scatter_of_empty_index_is_vacuous<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let empty_layout = Layout::c_contiguous([2, 0]).expect("empty layout");

    let input = device.upload(&host).expect("input upload");
    let index = device.alloc_zeroed::<u32>(0).expect("index alloc");
    let src = device.alloc_zeroed::<T>(0).expect("src alloc");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");

    ops.scatter_add_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        1,
        hephaestus_core::StridedView::new(&index, &empty_layout),
        hephaestus_core::StridedView::new(&src, &empty_layout),
        hephaestus_core::StridedView::new(&output, &in_layout),
    )
    .expect("vacuous scatter must dispatch");

    let mut got = vec![T::from(0i8); 8];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: vacuous scatter must reproduce the input exactly"
    );
}

/// Invalid geometry must fail typed: an out-of-range dim, an index/src
/// shape mismatch, a non-dim index extent outside the input, and an output
/// shape missing the input shape each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_scatter_rejects_invalid_geometry<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 3]).expect("index layout");
    let other = Layout::c_contiguous([2, 2]).expect("mismatched layout");
    let wide = Layout::c_contiguous([5, 3]).expect("wide layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let index = device.alloc_zeroed::<u32>(6).expect("index alloc");
    let src = device.alloc_zeroed::<T>(6).expect("src alloc");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &in_layout);
    let idx = || hephaestus_core::StridedView::new(&index, &idx_layout);
    let src_view = || hephaestus_core::StridedView::new(&src, &idx_layout);
    let out = || hephaestus_core::StridedView::new(&output, &in_layout);

    let err = ops
        .scatter_add_into::<2>(device, view(), 2, idx(), src_view(), out())
        .expect_err("out-of-range dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range dim must fail typed, got {err:?}"
    );

    let short_src = device.alloc_zeroed::<T>(4).expect("short src alloc");
    let err = ops
        .scatter_add_into::<2>(
            device,
            view(),
            1,
            idx(),
            hephaestus_core::StridedView::new(&short_src, &other),
            out(),
        )
        .expect_err("index/src mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: index/src mismatch must fail typed, got {err:?}"
    );

    let wide_index = device.alloc_zeroed::<u32>(15).expect("wide index alloc");
    let wide_src = device.alloc_zeroed::<T>(15).expect("wide src alloc");
    let err = ops
        .scatter_add_into::<2>(
            device,
            view(),
            1,
            hephaestus_core::StridedView::new(&wide_index, &wide),
            hephaestus_core::StridedView::new(&wide_src, &wide),
            out(),
        )
        .expect_err("index outside input must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: index outside input must fail typed, got {err:?}"
    );

    let short_layout = Layout::c_contiguous([2, 3]).expect("short layout");
    let short = device.alloc_zeroed::<T>(6).expect("short alloc");
    let err = ops
        .scatter_add_into::<2>(
            device,
            view(),
            1,
            idx(),
            src_view(),
            hephaestus_core::StridedView::new(&short, &short_layout),
        )
        .expect_err("short output must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short output must fail typed, got {err:?}"
    );
}

/// An output aliasing the input or the source must fail before mutating
/// anything: accumulation reads every operand while writing the output, so
/// sharing storage is a dispatch error on every backend, not just the
/// device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates a shared buffer.
pub fn assert_scatter_rejects_aliased_output<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 2]).expect("index layout");
    let src: Vec<T> = [1i8, 1, 1, 1].map(T::from).to_vec();

    // An output sharing the input's buffer: the geometry is fully valid
    // (shapes agree, every extent fits) and only the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");
    let index = device.upload(&[0u32, 1, 2, 3]).expect("index upload");
    let src_buf = device.upload(&src).expect("src upload");
    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .scatter_add_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &in_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&src_buf, &idx_layout),
            hephaestus_core::StridedView::new(&shared, &in_layout),
        )
        .expect_err("scatter dispatch must reject an output aliasing the input");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected input-alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: input-alias rejection mutated input");

    // An output sharing the source's buffer races accumulation reads
    // against output writes even though the input is untouched. The shared
    // buffer holds all eight output cells, so both the source view and the
    // wider output view fit its storage.
    let peers_host: Vec<T> = [1i8, 1, 1, 1, 0, 0, 0, 0].map(T::from).to_vec();
    let input = device.upload(&host).expect("input upload");
    let peers = device.upload(&peers_host).expect("peers upload");
    let peers_before = device
        .download_owned(&peers)
        .expect("peers fixture download");
    let error = ops
        .scatter_add_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &in_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&peers, &idx_layout),
            hephaestus_core::StridedView::new(&peers, &in_layout),
        )
        .expect_err("scatter dispatch must reject an output aliasing the source");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected source-alias error: {error}"
    );
    let peers_after = device
        .download_owned(&peers)
        .expect("peers fixture download");
    assert_eq!(
        peers_after, peers_before,
        "{name}: source-alias rejection mutated a shared source"
    );
}

/// An index value at or past the input's `dim` extent must fail typed with
/// an out-of-range message: a data-dependent fault is surfaced, never
/// clamped or wrapped, on every backend.
///
/// # Panics
///
/// Panics when an out-of-range index dispatches or fails with another error.
pub fn assert_scatter_rejects_out_of_range_index<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let idx_layout = Layout::c_contiguous([2, 3]).expect("index layout");
    let src: Vec<T> = [1i8, 1, 1, 1, 1, 1].map(T::from).to_vec();

    let input = device.upload(&host).expect("input upload");
    // Index 4 is out of range for the dim-1 extent of 4.
    let index = device.upload(&[0u32, 1, 2, 3, 4, 0]).expect("index upload");
    let src_buf = device.upload(&src).expect("src upload");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");

    let error = ops
        .scatter_add_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &in_layout),
            1,
            hephaestus_core::StridedView::new(&index, &idx_layout),
            hephaestus_core::StridedView::new(&src_buf, &idx_layout),
            hephaestus_core::StridedView::new(&output, &in_layout),
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
pub fn assert_scatter_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_scatter_index_contract(device, ops);
    assert_scatter_repeated_indices_accumulate(device, ops);
    assert_scatter_of_empty_index_is_vacuous(device, ops);
    assert_scatter_rejects_invalid_geometry(device, ops);
    assert_scatter_rejects_aliased_output(device, ops);
    assert_scatter_rejects_out_of_range_index(device, ops);
}

/// Run the order-pinned clause, which needs `f32` to observe accumulation
/// order through rounding.
///
/// # Panics
///
/// Panics when the backend accumulates in any other order.
pub fn assert_scatter_order_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: ScatterOps<D, f32>,
{
    assert_scatter_accumulates_in_ascending_source_order(device, ops);
}
