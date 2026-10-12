//! Contract clauses for the device-neutral [`hephaestus_core::RollOps`] seam.
//!
//! There is no `leto::roll` to delegate to, so the host reference carries
//! the direct implementation and these clauses pin its value semantics
//! with hand-computed fixtures: `output[i] = input[i with the `dim` lane
//! moved back by `shift` modulo the extent]` per `(shift, dim)` pair —
//! the `torch.roll` contract `coeus_ops::roll` documents. Fixtures use
//! small signed integers so every copy is an exact equality, never a
//! tolerance-bounded comparison. Rolls along distinct dims commute and
//! rolls along one dim add, so duplicate dims compose by summation and
//! the pair order is unobservable.

use hephaestus_core::{ComputeDevice, HephaestusError, RollOps};
use leto::Layout;

/// Shift against hand-computed fixtures: a 1-D roll by `+1`, `-1`, and a
/// larger-than-extent `+5`, plus a 2-D roll along both axes at once.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_roll_value_contract<D, R, T>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([4]).expect("layout");

    // 1-D rolls: +1, -1, and +5 (which reduces to +1 mod 4).
    for (shift, expected) in [
        (1isize, [3i8, 0, 1, 2]),
        (-1, [1i8, 2, 3, 0]),
        (5, [3i8, 0, 1, 2]),
    ] {
        let host: Vec<T> = (0..4i8).map(T::from).collect();
        let input = device.upload(&host).expect("input upload");
        let output = device.alloc_zeroed::<T>(4).expect("output alloc");
        ops.roll_into::<1>(
            device,
            hephaestus_core::StridedView::new(&input, &layout),
            &[shift],
            &[0],
            hephaestus_core::StridedView::new(&output, &layout),
        )
        .expect("roll dispatch");

        let expected: Vec<T> = expected.map(T::from).to_vec();
        let mut got = vec![T::from(0i8); 4];
        device.download(&output, &mut got).expect("download");
        assert_eq!(
            got, expected,
            "{name}: 1-D roll by {shift} must match the hand-computed fixture exactly"
        );
    }

    // 2-D roll along both axes: rows shift by 1, then columns by 1.
    let square: Vec<T> = [1i8, 2, 3, 4, 5, 6].map(T::from).to_vec();
    let square_layout = Layout::c_contiguous([2, 3]).expect("square layout");
    let square_input = device.upload(&square).expect("square upload");
    let square_output = device.alloc_zeroed::<T>(6).expect("square output alloc");
    ops.roll_into::<2>(
        device,
        hephaestus_core::StridedView::new(&square_input, &square_layout),
        &[1, 1],
        &[0, 1],
        hephaestus_core::StridedView::new(&square_output, &square_layout),
    )
    .expect("roll dispatch");

    // Rows become [[4, 5, 6], [1, 2, 3]], then columns shift to
    // [[6, 4, 5], [3, 1, 2]].
    let square_expected: Vec<T> = [6i8, 4, 5, 3, 1, 2].map(T::from).to_vec();
    let mut square_got = vec![T::from(0i8); 6];
    device
        .download(&square_output, &mut square_got)
        .expect("download");
    assert_eq!(
        square_got, square_expected,
        "{name}: 2-D roll must match the hand-computed fixture exactly"
    );
}

/// The identity rolls reproduce the input exactly: an empty shift list, a
/// zero shift, a whole-extent shift, cancelling duplicate dims, and a
/// shift along a size-1 dim.
///
/// # Panics
///
/// Panics when any identity roll changes any element.
pub fn assert_roll_identity_cases<D, R, T>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..4i8).map(T::from).collect();
    let layout = Layout::c_contiguous([4]).expect("layout");
    let input = device.upload(&host).expect("input upload");

    // (shifts, dims) pairs that must all act as the identity.
    let cases: [(&[isize], &[usize]); 4] =
        [(&[], &[]), (&[0], &[0]), (&[4], &[0]), (&[1, -1], &[0, 0])];
    for (shifts, dims) in cases {
        let output = device.alloc_zeroed::<T>(4).expect("output alloc");
        ops.roll_into::<1>(
            device,
            hephaestus_core::StridedView::new(&input, &layout),
            shifts,
            dims,
            hephaestus_core::StridedView::new(&output, &layout),
        )
        .expect("identity roll must dispatch");
        let mut got = vec![T::from(0i8); 4];
        device.download(&output, &mut got).expect("download");
        assert_eq!(
            got, host,
            "{name}: identity roll {shifts:?}/{dims:?} must reproduce the input exactly"
        );
    }

    // A shift along a size-1 dim is an identity however large.
    let single: Vec<T> = [1i8, 2, 3].map(T::from).to_vec();
    let single_layout = Layout::c_contiguous([1, 3]).expect("single layout");
    let single_input = device.upload(&single).expect("single upload");
    let single_output = device.alloc_zeroed::<T>(3).expect("single output alloc");
    ops.roll_into::<2>(
        device,
        hephaestus_core::StridedView::new(&single_input, &single_layout),
        &[7],
        &[0],
        hephaestus_core::StridedView::new(&single_output, &single_layout),
    )
    .expect("size-1 roll must dispatch");
    let mut single_got = vec![T::from(0i8); 3];
    device
        .download(&single_output, &mut single_got)
        .expect("download");
    assert_eq!(
        single_got, single,
        "{name}: a roll along a size-1 dim must reproduce the input exactly"
    );
}

/// An empty shape rolls vacuously: matching `[2, 0]` shapes dispatch
/// successfully and write zero cells. No shift normalizes against a zero
/// extent — there is no modulo by zero on any backend.
///
/// # Panics
///
/// Panics when the vacuous roll fails to dispatch.
pub fn assert_roll_of_empty_is_vacuous<D, R, T>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let input = device.alloc_zeroed::<T>(0).expect("input alloc");
    let output = device.alloc_zeroed::<T>(0).expect("output alloc");

    ops.roll_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        &[3, -2],
        &[0, 1],
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("vacuous roll must dispatch");
}

/// Invalid geometry must fail typed: a shift/dim length mismatch, an
/// out-of-range dim, and an output shape that is not the input shape each
/// return [`HephaestusError::InvalidConfiguration`] rather than
/// dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_roll_rejects_invalid_geometry<D, R, T>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &layout);
    let out = || hephaestus_core::StridedView::new(&output, &layout);

    let err = ops
        .roll_into::<2>(device, view(), &[1, 2], &[1], out())
        .expect_err("length mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: length mismatch must fail typed, got {err:?}"
    );

    let err = ops
        .roll_into::<2>(device, view(), &[1], &[2], out())
        .expect_err("out-of-range dim must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range dim must fail typed, got {err:?}"
    );

    let wide_layout = Layout::c_contiguous([2, 5]).expect("wide layout");
    let wide = device.alloc_zeroed::<T>(10).expect("wide alloc");
    let err = ops
        .roll_into::<2>(
            device,
            view(),
            &[1],
            &[1],
            hephaestus_core::StridedView::new(&wide, &wide_layout),
        )
        .expect_err("output mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: output mismatch must fail typed, got {err:?}"
    );
}

/// An output aliasing the input must fail before mutating anything: reads
/// and writes race across threads, so sharing storage is a dispatch error
/// on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_roll_rejects_aliased_output<D, R, T>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");

    // Both views fit the shared buffer's storage: the geometry is fully
    // valid and only the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .roll_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &layout),
            &[1],
            &[1],
            hephaestus_core::StridedView::new(&shared, &layout),
        )
        .expect_err("roll dispatch must reject aliased input/output");
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
pub fn assert_roll_contract<D, R, T>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RollOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_roll_value_contract(device, ops);
    assert_roll_identity_cases(device, ops);
    assert_roll_of_empty_is_vacuous(device, ops);
    assert_roll_rejects_invalid_geometry(device, ops);
    assert_roll_rejects_aliased_output(device, ops);
}
