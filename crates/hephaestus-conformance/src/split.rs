//! Contract clauses for the device-neutral [`hephaestus_core::SplitOps`] seam.
//!
//! The oracle is [`leto::split`] itself — the same function the host reference
//! device delegates to — so every backend's differential compares against
//! the definition of splitting, not a second implementation of it.
//! Fixtures use small signed integers so every copy is an exact equality,
//! never a tolerance-bounded comparison.

use hephaestus_core::{ComputeDevice, HephaestusError, SplitOps};
use leto::Layout;

/// Run one split backend against Leto's `split` reference: a 2x8 fixture
/// along axis 1 with asymmetric segment lengths 3, 1, and 4.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with Leto.
pub fn assert_split_leto_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SplitOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..16i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 8]).expect("input layout");
    let first_layout = Layout::c_contiguous([2, 3]).expect("first layout");
    let second_layout = Layout::c_contiguous([2, 1]).expect("second layout");
    let third_layout = Layout::c_contiguous([2, 4]).expect("third layout");

    let input = device.upload(&host).expect("input upload");
    let first = device.alloc_zeroed::<T>(6).expect("first alloc");
    let second = device.alloc_zeroed::<T>(2).expect("second alloc");
    let third = device.alloc_zeroed::<T>(8).expect("third alloc");

    ops.split_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        1,
        &[3, 1, 4],
        &[
            hephaestus_core::StridedView::new(&first, &first_layout),
            hephaestus_core::StridedView::new(&second, &second_layout),
            hephaestus_core::StridedView::new(&third, &third_layout),
        ],
    )
    .expect("split dispatch");

    let leto_input = leto::Array::from_shape_vec([2, 8], host).expect("leto input");
    let expected = leto::split(&leto_input.view(), 1, &[3, 1, 4]).expect("leto split");

    let mut got_first = vec![T::from(0i8); 6];
    let mut got_second = vec![T::from(0i8); 2];
    let mut got_third = vec![T::from(0i8); 8];
    device
        .download(&first, &mut got_first)
        .expect("first download");
    device
        .download(&second, &mut got_second)
        .expect("second download");
    device
        .download(&third, &mut got_third)
        .expect("third download");
    assert_eq!(
        got_first,
        expected[0].to_contiguous().into_vec(),
        "{name}: first split segment must match Leto's reference exactly"
    );
    assert_eq!(
        got_second,
        expected[1].to_contiguous().into_vec(),
        "{name}: second split segment must match Leto's reference exactly"
    );
    assert_eq!(
        got_third,
        expected[2].to_contiguous().into_vec(),
        "{name}: third split segment must match Leto's reference exactly"
    );
}

/// A single full-length segment must reproduce the input exactly.
///
/// # Panics
///
/// Panics when a single-segment split changes any element.
pub fn assert_split_single_full_segment_is_identity<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SplitOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");

    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.split_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        0,
        &[2],
        &[hephaestus_core::StridedView::new(&output, &layout)],
    )
    .expect("split dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: single-segment split must reproduce the input exactly"
    );
}

/// Zero-length segments contribute nothing: splitting `[2, 4]` into
/// `[2, 2]`, `[2, 0]`, `[2, 2]` yields the neighbors `leto::split` implies.
///
/// # Panics
///
/// Panics when the empty segment shifts, drops, or duplicates any element.
pub fn assert_split_of_empty_segments_skips_them<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SplitOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 4]).expect("input layout");
    let side_layout = Layout::c_contiguous([2, 2]).expect("side layout");
    let empty_layout = Layout::c_contiguous([2, 0]).expect("empty layout");

    let input = device.upload(&host).expect("input upload");
    let left = device.alloc_zeroed::<T>(4).expect("left alloc");
    let empty = device.alloc_zeroed::<T>(0).expect("empty alloc");
    let right = device.alloc_zeroed::<T>(4).expect("right alloc");

    ops.split_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        1,
        &[2, 0, 2],
        &[
            hephaestus_core::StridedView::new(&left, &side_layout),
            hephaestus_core::StridedView::new(&empty, &empty_layout),
            hephaestus_core::StridedView::new(&right, &side_layout),
        ],
    )
    .expect("split dispatch");

    let leto_input = leto::Array::from_shape_vec([2, 4], host).expect("leto input");
    let expected = leto::split(&leto_input.view(), 1, &[2, 0, 2]).expect("leto split");

    let mut got_left = vec![T::from(0i8); 4];
    let mut got_right = vec![T::from(0i8); 4];
    device
        .download(&left, &mut got_left)
        .expect("left download");
    device
        .download(&right, &mut got_right)
        .expect("right download");
    assert_eq!(
        got_left,
        expected[0].to_contiguous().into_vec(),
        "{name}: the segment before the empty one must match Leto exactly"
    );
    assert_eq!(
        got_right,
        expected[2].to_contiguous().into_vec(),
        "{name}: the segment after the empty one must match Leto exactly"
    );
}

/// Invalid geometry must fail typed: a sizes/outputs arity mismatch, an
/// out-of-range axis, a non-axis shape mismatch, an output extent missing
/// its size, and a sizes sum missing the axis length each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_split_rejects_invalid_geometry<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SplitOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");
    let other = Layout::c_contiguous([4, 3]).expect("mismatched layout");
    let input = device.alloc_zeroed::<T>(6).expect("input alloc");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &layout);
    let out = || hephaestus_core::StridedView::new(&output, &layout);

    let err = ops
        .split_into::<2>(device, view(), 0, &[1, 1], &[out()])
        .expect_err("arity mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: arity mismatch must fail typed, got {err:?}"
    );

    let err = ops
        .split_into::<2>(device, view(), 2, &[2], &[out()])
        .expect_err("out-of-range axis must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range axis must fail typed, got {err:?}"
    );

    let mistyped = device.alloc_zeroed::<T>(12).expect("mismatched alloc");
    let err = ops
        .split_into::<2>(
            device,
            view(),
            0,
            &[2],
            &[hephaestus_core::StridedView::new(&mistyped, &other)],
        )
        .expect_err("shape mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: shape mismatch must fail typed, got {err:?}"
    );

    let short_layout = Layout::c_contiguous([2, 2]).expect("short layout");
    let short = device.alloc_zeroed::<T>(4).expect("short alloc");
    let err = ops
        .split_into::<2>(
            device,
            view(),
            1,
            &[3],
            &[hephaestus_core::StridedView::new(&short, &short_layout)],
        )
        .expect_err("short output axis must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short output axis must fail typed, got {err:?}"
    );

    let err = ops
        .split_into::<2>(device, view(), 1, &[2], &[out()])
        .expect_err("short sizes sum must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: short sizes sum must fail typed, got {err:?}"
    );
}

/// An output aliasing the input or another output must fail before mutating
/// anything: reads and writes race across passes, so sharing storage is a
/// dispatch error on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates a shared buffer.
pub fn assert_split_rejects_aliased_output<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SplitOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..12i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 6]).expect("input layout");
    let side_layout = Layout::c_contiguous([2, 3]).expect("side layout");

    // An output sharing the input's buffer: the geometry is fully valid
    // (shapes agree, sizes sum to the axis length) and only the aliasing
    // fails.
    let shared = device.upload(&host).expect("shared upload");
    let separate = device.alloc_zeroed::<T>(6).expect("separate alloc");
    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .split_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &in_layout),
            1,
            &[3, 3],
            &[
                hephaestus_core::StridedView::new(&shared, &side_layout),
                hephaestus_core::StridedView::new(&separate, &side_layout),
            ],
        )
        .expect_err("split dispatch must reject an output aliasing the input");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected input-alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: input-alias rejection mutated input");

    // Two outputs sharing one buffer race segment writes against each other,
    // so the pair fails even though neither aliases the input.
    let input = device.upload(&host).expect("input upload");
    let peers = device.alloc_zeroed::<T>(6).expect("peers alloc");
    let peers_before = device
        .download_owned(&peers)
        .expect("peers fixture download");
    let error = ops
        .split_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &in_layout),
            1,
            &[3, 3],
            &[
                hephaestus_core::StridedView::new(&peers, &side_layout),
                hephaestus_core::StridedView::new(&peers, &side_layout),
            ],
        )
        .expect_err("split dispatch must reject outputs aliasing each other");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected output-alias error: {error}"
    );
    let peers_after = device
        .download_owned(&peers)
        .expect("peers fixture download");
    assert_eq!(
        peers_after, peers_before,
        "{name}: output-alias rejection mutated a shared output"
    );
}

/// Run every clause in this module against one backend.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_split_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SplitOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_split_leto_contract(device, ops);
    assert_split_single_full_segment_is_identity(device, ops);
    assert_split_of_empty_segments_skips_them(device, ops);
    assert_split_rejects_invalid_geometry(device, ops);
    assert_split_rejects_aliased_output(device, ops);
}
