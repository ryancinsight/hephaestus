//! Contract clauses for the device-neutral [`hephaestus_core::CatOps`] seam.
//!
//! The oracle is [`leto::concat`] itself — the same function the host reference
//! device delegates to — so every backend's differential compares against
//! the definition of concatenation, not a second implementation of it.
//! Fixtures use small signed integers so every copy is an exact equality,
//! never a tolerance-bounded comparison.

use hephaestus_core::{CatOps, ComputeDevice, HephaestusError};
use leto::Layout;

/// Run one clause backend against Leto's `concat` reference: 2x3, 2x1, and
/// 2x4 fixtures along axis 1 with asymmetric segment lengths.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with Leto.
pub fn assert_cat_leto_contract<D, C, T>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: CatOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let first: Vec<T> = (0..6i8).map(T::from).collect();
    let second: Vec<T> = (6..8i8).map(T::from).collect();
    let third: Vec<T> = (8..16i8).map(T::from).collect();
    let first_layout = Layout::c_contiguous([2, 3]).expect("first layout");
    let second_layout = Layout::c_contiguous([2, 1]).expect("second layout");
    let third_layout = Layout::c_contiguous([2, 4]).expect("third layout");
    let out_layout = Layout::c_contiguous([2, 8]).expect("output layout");

    let first_buf = device.upload(&first).expect("first upload");
    let second_buf = device.upload(&second).expect("second upload");
    let third_buf = device.upload(&third).expect("third upload");
    let output = device.alloc_zeroed::<T>(16).expect("output alloc");

    ops.cat_into::<2>(
        device,
        &[
            hephaestus_core::StridedView::new(&first_buf, &first_layout),
            hephaestus_core::StridedView::new(&second_buf, &second_layout),
            hephaestus_core::StridedView::new(&third_buf, &third_layout),
        ],
        1,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("cat dispatch");

    let leto_first = leto::Array::from_shape_vec([2, 3], first).expect("leto first");
    let leto_second = leto::Array::from_shape_vec([2, 1], second).expect("leto second");
    let leto_third = leto::Array::from_shape_vec([2, 4], third).expect("leto third");
    let expected = leto::concat(
        &[leto_first.view(), leto_second.view(), leto_third.view()],
        1,
    )
    .expect("leto concat")
    .into_vec();

    let mut got = vec![T::from(0i8); 16];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: concatenated output must match Leto's reference exactly"
    );
}

/// A single input must reproduce exactly (the seam copies; any zero-copy
/// fast path lives with the caller, as `coeus_ops::cat` documents).
///
/// # Panics
///
/// Panics when a single-input concatenation changes any element.
pub fn assert_cat_single_input_is_identity<D, C, T>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: CatOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");

    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.cat_into::<2>(
        device,
        &[hephaestus_core::StridedView::new(&input, &layout)],
        0,
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("cat dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: single-input cat must reproduce the input exactly"
    );
}

/// Zero-length segments contribute nothing: concatenating a `[2, 0]` view
/// between two `[2, 2]` views yields the `[2, 4]` the neighbors imply.
///
/// # Panics
///
/// Panics when the empty segment shifts, drops, or duplicates any element.
pub fn assert_cat_of_empty_segments_skips_them<D, C, T>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: CatOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let left: Vec<T> = (0..4i8).map(T::from).collect();
    let right: Vec<T> = (4..8i8).map(T::from).collect();
    let side_layout = Layout::c_contiguous([2, 2]).expect("side layout");
    let empty_layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let out_layout = Layout::c_contiguous([2, 4]).expect("output layout");

    let left_buf = device.upload(&left).expect("left upload");
    let empty_buf = device.alloc_zeroed::<T>(0).expect("empty alloc");
    let right_buf = device.upload(&right).expect("right upload");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");

    ops.cat_into::<2>(
        device,
        &[
            hephaestus_core::StridedView::new(&left_buf, &side_layout),
            hephaestus_core::StridedView::new(&empty_buf, &empty_layout),
            hephaestus_core::StridedView::new(&right_buf, &side_layout),
        ],
        1,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("cat dispatch");

    // The oracle stays `leto::concat` (as the first clause does): a flat
    // append of the inputs is only correct along axis 0, while row-major
    // output interleaves rows for any later axis.
    let leto_left = leto::Array::from_shape_vec([2, 2], left).expect("leto left");
    let leto_empty = leto::Array::from_shape_vec([2, 0], Vec::<T>::new()).expect("leto empty");
    let leto_right = leto::Array::from_shape_vec([2, 2], right).expect("leto right");
    let expected = leto::concat(&[leto_left.view(), leto_empty.view(), leto_right.view()], 1)
        .expect("leto concat")
        .into_vec();

    let mut got = vec![T::from(0i8); 8];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: empty segments must contribute nothing to the concatenation"
    );
}

/// Invalid geometry must fail typed: an empty input list, an out-of-range
/// axis, a non-axis shape mismatch, and a short output axis each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_cat_rejects_invalid_geometry<D, C, T>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: CatOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");
    let other = Layout::c_contiguous([4, 3]).expect("mismatched layout");
    let input = device.alloc_zeroed::<T>(6).expect("input alloc");
    let mistyped = device.alloc_zeroed::<T>(12).expect("mismatched alloc");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &layout);
    let out = || hephaestus_core::StridedView::new(&output, &layout);

    let err = ops
        .cat_into::<2>(device, &[], 0, out())
        .expect_err("empty input list must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: empty input list must fail typed, got {err:?}"
    );

    let err = ops
        .cat_into::<2>(device, &[view()], 2, out())
        .expect_err("out-of-range axis must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range axis must fail typed, got {err:?}"
    );

    let err = ops
        .cat_into::<2>(
            device,
            &[view(), hephaestus_core::StridedView::new(&mistyped, &other)],
            1,
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
        .cat_into::<2>(
            device,
            &[view(), view()],
            1,
            hephaestus_core::StridedView::new(&short, &short_layout),
        )
        .expect_err("short output axis must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short output axis must fail typed, got {err:?}"
    );
}

/// An output aliasing any input must fail before mutating anything: reads
/// and writes race across passes, so sharing storage is a dispatch error
/// on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_cat_rejects_aliased_output<D, C, T>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: CatOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..12i8).map(T::from).collect();
    let other: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");
    let wide_layout = Layout::c_contiguous([2, 6]).expect("wide layout");

    // The shared buffer holds all twelve output cells, so the wide output
    // view fits its storage: the geometry is fully valid and only the
    // aliasing fails.
    let shared = device.upload(&host).expect("shared upload");
    let second = device.upload(&other).expect("second upload");

    // The same buffer serves as the first input and the output:
    // `StridedView` borrows, so no `Clone` bound is needed, and owning
    // buffers (CUDA/ROCm free on drop) alias identically to handle-backed
    // ones. The output view is wider than the input view, so the geometry
    // is otherwise valid and only the aliasing fails.
    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .cat_into::<2>(
            device,
            &[
                hephaestus_core::StridedView::new(&shared, &layout),
                hephaestus_core::StridedView::new(&second, &layout),
            ],
            1,
            hephaestus_core::StridedView::new(&shared, &wide_layout),
        )
        .expect_err("cat dispatch must reject aliased input/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: alias rejection mutated input");
}

/// Run every clause in this module against one backend.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_cat_contract<D, C, T>(device: &D, ops: &C)
where
    D: ComputeDevice,
    C: CatOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_cat_leto_contract(device, ops);
    assert_cat_single_input_is_identity(device, ops);
    assert_cat_of_empty_segments_skips_them(device, ops);
    assert_cat_rejects_invalid_geometry(device, ops);
    assert_cat_rejects_aliased_output(device, ops);
}
