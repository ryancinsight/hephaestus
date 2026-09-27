//! Contract clauses for the device-neutral [`hephaestus_core::PadOps`] seam.
//!
//! The oracle is [`leto::pad`] itself — the same function the host reference
//! device delegates to — so every backend's differential compares against
//! the definition of pad, not a second implementation of it. Fixtures use
//! small signed integers so every fill/copy is an exact equality, never a
//! tolerance-bounded comparison.

use hephaestus_core::{ComputeDevice, PadOps};
use leto::Layout;

/// Run one clause backend against Leto's `pad` reference: a 3x4 fixture with
/// asymmetric before/after widths on both axes and a nonzero fill.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with Leto.
pub fn assert_pad_leto_contract<D, P, T>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: PadOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..12i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([3, 4]).expect("input layout");
    let width = [(1usize, 2usize), (0usize, 1usize)];
    let fill = T::from(-1i8);

    let input = device.upload(&host).expect("input upload");
    let out_shape = [3 + 1 + 2, 4 + 1];
    let out_layout = Layout::c_contiguous(out_shape).expect("output layout");
    let out_len = out_shape[0] * out_shape[1];
    let output = device.alloc_zeroed::<T>(out_len).expect("output alloc");

    ops.pad_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        width,
        fill,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("pad dispatch");

    let leto_input = leto::Array::from_shape_vec([3, 4], host).expect("leto input");
    let expected = leto::pad(&leto_input.view(), width, fill)
        .expect("leto pad")
        .into_vec();

    let mut got = vec![T::from(0i8); out_len];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: padded output must match Leto's reference exactly"
    );
}

/// Zero padding on every axis must be an exact copy.
///
/// # Panics
///
/// Panics when a zero-width pad changes any element.
pub fn assert_pad_zero_width_is_identity<D, P, T>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: PadOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");

    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.pad_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        [(0, 0), (0, 0)],
        T::from(0i8),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("pad dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: zero-width pad must reproduce the input exactly"
    );
}

/// Padding an empty (zero-length) axis must fill the whole output.
///
/// # Panics
///
/// Panics when any output cell is not the fill value.
pub fn assert_pad_of_empty_input_fills_entirely<D, P, T>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: PadOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([0, 3]).expect("empty input layout");
    let out_layout = Layout::c_contiguous([2, 3]).expect("output layout");
    let fill = T::from(7i8);

    let input = device.alloc_zeroed::<T>(0).expect("empty input alloc");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.pad_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        [(1, 1), (0, 0)],
        fill,
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("pad dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got,
        vec![fill; 6],
        "{name}: padding an empty axis must fill the whole output with `fill`"
    );
}

/// Run every clause in this module against one backend.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_pad_contract<D, P, T>(device: &D, ops: &P)
where
    D: ComputeDevice,
    P: PadOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_pad_leto_contract(device, ops);
    assert_pad_zero_width_is_identity(device, ops);
    assert_pad_of_empty_input_fills_entirely(device, ops);
}
