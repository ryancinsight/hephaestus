//! Contract clauses for the device-neutral [`hephaestus_core::StackOps`] seam.
//!
//! The oracle is [`leto::stack`] itself — the same function the host reference
//! device delegates to — so every backend's differential compares against
//! the definition of stacking, not a second implementation of it.
//! Fixtures use small signed integers so every copy is an exact equality,
//! never a tolerance-bounded comparison.

use hephaestus_core::{ComputeDevice, HephaestusError, StackOps};
use leto::Layout;

/// Run one stack backend against Leto's `stack` reference: three 2x3
/// fixtures along a new middle axis 1, producing a 2x3x3 output.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with Leto.
pub fn assert_stack_leto_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StackOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let first: Vec<T> = (0..6i8).map(T::from).collect();
    let second: Vec<T> = (6..12i8).map(T::from).collect();
    let third: Vec<T> = (12..18i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 3]).expect("input layout");
    let out_layout = Layout::c_contiguous([2, 3, 3]).expect("output layout");

    let first_buf = device.upload(&first).expect("first upload");
    let second_buf = device.upload(&second).expect("second upload");
    let third_buf = device.upload(&third).expect("third upload");
    let output = device.alloc_zeroed::<T>(18).expect("output alloc");

    ops.stack_into::<2, 3>(
        device,
        &[
            hephaestus_core::StridedView::new(&first_buf, &in_layout),
            hephaestus_core::StridedView::new(&second_buf, &in_layout),
            hephaestus_core::StridedView::new(&third_buf, &in_layout),
        ],
        1,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("stack dispatch");

    let leto_first = leto::Array::from_shape_vec([2, 3], first).expect("leto first");
    let leto_second = leto::Array::from_shape_vec([2, 3], second).expect("leto second");
    let leto_third = leto::Array::from_shape_vec([2, 3], third).expect("leto third");
    let expected = leto::stack::<T, 2, 3>(
        &[leto_first.view(), leto_second.view(), leto_third.view()],
        1,
    )
    .expect("leto stack")
    .into_vec();

    let mut got = vec![T::from(0i8); 18];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: stacked output must match Leto's reference exactly"
    );
}

/// A single input must reproduce exactly behind a length-one new axis.
///
/// # Panics
///
/// Panics when a single-input stack changes any element.
pub fn assert_stack_single_input_inserts_a_length_one_axis<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StackOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 3]).expect("input layout");
    let out_layout = Layout::c_contiguous([1, 2, 3]).expect("output layout");

    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.stack_into::<2, 3>(
        device,
        &[hephaestus_core::StridedView::new(&input, &in_layout)],
        0,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("stack dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: single-input stack must reproduce the input exactly"
    );
}

/// Zero-sized inputs stack vacuously: two `[2, 0]` views along a new axis 1
/// dispatch successfully and produce the zero-cell `[2, 2, 0]` the geometry
/// implies.
///
/// # Panics
///
/// Panics when the vacuous stack fails to dispatch.
pub fn assert_stack_of_empty_inputs_is_vacuous<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StackOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 0]).expect("input layout");
    let out_layout = Layout::c_contiguous([2, 2, 0]).expect("output layout");

    let first = device.alloc_zeroed::<T>(0).expect("first alloc");
    let second = device.alloc_zeroed::<T>(0).expect("second alloc");
    let output = device.alloc_zeroed::<T>(0).expect("output alloc");

    ops.stack_into::<2, 3>(
        device,
        &[
            hephaestus_core::StridedView::new(&first, &in_layout),
            hephaestus_core::StridedView::new(&second, &in_layout),
        ],
        1,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("vacuous stack must dispatch");

    let leto_first = leto::Array::from_shape_vec([2, 0], Vec::<T>::new()).expect("leto first");
    let leto_second = leto::Array::from_shape_vec([2, 0], Vec::<T>::new()).expect("leto second");
    let expected = leto::stack::<T, 2, 3>(&[leto_first.view(), leto_second.view()], 1)
        .expect("leto stack")
        .into_vec();
    assert!(
        expected.is_empty(),
        "{name}: vacuous stack oracle must hold zero cells"
    );
}

/// Invalid geometry must fail typed: an axis past the new trailing lane,
/// an empty input list, an input shape mismatch, an output axis extent
/// missing the input count, and a shared-dimension mismatch each return
/// [`HephaestusError::InvalidConfiguration`] rather than dispatching. A
/// wrong output rank needs no runtime case: the seam's `InsertAxis` bound
/// makes `M != N + 1` ill-typed at the call site.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_stack_rejects_invalid_geometry<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StackOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");
    let other = Layout::c_contiguous([2, 4]).expect("mismatched layout");
    let out_layout = Layout::c_contiguous([2, 1, 3]).expect("output layout");
    let input = device.alloc_zeroed::<T>(6).expect("input alloc");
    let mistyped = device.alloc_zeroed::<T>(8).expect("mismatched alloc");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &layout);
    let out = || hephaestus_core::StridedView::new(&output, &out_layout);

    let err = ops
        .stack_into::<2, 3>(device, &[view()], 3, out())
        .expect_err("out-of-range axis must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: out-of-range axis must fail typed, got {err:?}"
    );

    let err = ops
        .stack_into::<2, 3>(device, &[], 0, out())
        .expect_err("empty input list must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: empty input list must fail typed, got {err:?}"
    );

    let err = ops
        .stack_into::<2, 3>(
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

    let wide_layout = Layout::c_contiguous([2, 2, 3]).expect("wide layout");
    let wide = device.alloc_zeroed::<T>(12).expect("wide alloc");
    let err = ops
        .stack_into::<2, 3>(
            device,
            &[view()],
            1,
            hephaestus_core::StridedView::new(&wide, &wide_layout),
        )
        .expect_err("wrong axis extent must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: wrong axis extent must fail typed, got {err:?}"
    );

    let short_layout = Layout::c_contiguous([2, 1, 4]).expect("short layout");
    let short = device.alloc_zeroed::<T>(8).expect("short alloc");
    let err = ops
        .stack_into::<2, 3>(
            device,
            &[view()],
            1,
            hephaestus_core::StridedView::new(&short, &short_layout),
        )
        .expect_err("shared-dimension mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: shared-dimension mismatch must fail typed, got {err:?}"
    );
}

/// An output aliasing any input must fail before mutating anything: reads
/// and writes race across passes, so sharing storage is a dispatch error
/// on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_stack_rejects_aliased_output<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StackOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..12i8).map(T::from).collect();
    let other: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");
    let wide_layout = Layout::c_contiguous([2, 2, 3]).expect("wide layout");

    // The shared buffer holds all twelve output cells, so the stacked
    // output view fits its storage: the geometry is fully valid and only
    // the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");
    let second = device.upload(&other).expect("second upload");

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .stack_into::<2, 3>(
            device,
            &[
                hephaestus_core::StridedView::new(&shared, &layout),
                hephaestus_core::StridedView::new(&second, &layout),
            ],
            1,
            hephaestus_core::StridedView::new(&shared, &wide_layout),
        )
        .expect_err("stack dispatch must reject aliased input/output");
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
pub fn assert_stack_contract<D, S, T>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: StackOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_stack_leto_contract(device, ops);
    assert_stack_single_input_inserts_a_length_one_axis(device, ops);
    assert_stack_of_empty_inputs_is_vacuous(device, ops);
    assert_stack_rejects_invalid_geometry(device, ops);
    assert_stack_rejects_aliased_output(device, ops);
}
