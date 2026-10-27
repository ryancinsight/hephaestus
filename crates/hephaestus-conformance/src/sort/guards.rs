//! Guard clauses for the [`SortOps`] seam: vacuous shapes, invalid
//! geometry, alias rejection, and the aggregate entry point.
//!
//! These clauses are about *not dispatching* rather than about
//! dispatched results: they pin the typed-error contract, so a backend
//! that quietly dispatches an invalid shape fails here rather than
//! corrupting memory later.

use hephaestus_core::{ComputeDevice, HephaestusError, SortOps};
use leto::Layout;

use super::values::{assert_sort_stability, assert_sort_value_contract};

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
/// The `f32`-only [`assert_sort_special_float_values`](super::special::assert_sort_special_float_values)
/// clause runs separately from each backend's `f32` instantiation, the
/// way the masked-fill and where special-values clauses do.
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
