//! Contract clauses for the device-neutral [`hephaestus_core::MaskedFillOps`] seam.
//!
//! There is no `leto::masked_fill` to delegate to, so the host reference
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures: `output[i] = value` wherever
//! the mask cell is nonzero, `input[i]` elsewhere — the same-shape
//! contract `coeus_ops::masked_fill` documents. Fixtures use small signed
//! integers so every copy is an exact equality, never a tolerance-bounded
//! comparison. A cell fills exactly when its mask value compares unequal
//! to zero under the scalar's equality, which pins the float edges (a
//! `NaN` mask fills, a negative-zero mask keeps) in a dedicated `f32`
//! clause; backends must not reinterpret the mask bitwise nor threshold
//! it.

use hephaestus_core::{ComputeDevice, HephaestusError, MaskedFillOps};
use leto::Layout;

/// Fill under a mixed mask against hand-computed fixtures: a 1-D input
/// with positive, zero, and negative mask cells, and a 2-D input with a
/// diagonal-ish mask. Any nonzero magnitude, however small or negative,
/// fills; only exact zero keeps.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_masked_fill_value_contract<D, M, T>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // 1-D: mask [0, 1, 0, -2] fills cells 1 and 3, keeps 0 and 2.
    let host: Vec<T> = [1i8, 2, 3, 4].map(T::from).to_vec();
    let mask_host: Vec<T> = [0i8, 1, 0, -2].map(T::from).to_vec();
    let layout = Layout::c_contiguous([4]).expect("layout");
    let input = device.upload(&host).expect("input upload");
    let mask = device.upload(&mask_host).expect("mask upload");
    let output = device.alloc_zeroed::<T>(4).expect("output alloc");

    ops.masked_fill_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&mask, &layout),
        T::from(9i8),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("masked_fill dispatch");

    let expected: Vec<T> = [1i8, 9, 3, 9].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 4];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: 1-D masked_fill must match the hand-computed fixture exactly"
    );

    // 2-D: mask [[1, 0], [0, -1]] fills [0, 0] and [1, 1].
    let square: Vec<T> = [1i8, 2, 3, 4].map(T::from).to_vec();
    let square_mask: Vec<T> = [1i8, 0, 0, -1].map(T::from).to_vec();
    let square_layout = Layout::c_contiguous([2, 2]).expect("square layout");
    let square_input = device.upload(&square).expect("square upload");
    let square_mask_buf = device.upload(&square_mask).expect("square mask upload");
    let square_output = device.alloc_zeroed::<T>(4).expect("square output alloc");

    ops.masked_fill_into::<2>(
        device,
        hephaestus_core::StridedView::new(&square_input, &square_layout),
        hephaestus_core::StridedView::new(&square_mask_buf, &square_layout),
        T::from(-7i8),
        hephaestus_core::StridedView::new(&square_output, &square_layout),
    )
    .expect("masked_fill dispatch");

    let square_expected: Vec<T> = [-7i8, 2, 3, -7].map(T::from).to_vec();
    let mut square_got = vec![T::from(0i8); 4];
    device
        .download(&square_output, &mut square_got)
        .expect("download");
    assert_eq!(
        square_got, square_expected,
        "{name}: 2-D masked_fill must match the hand-computed fixture exactly"
    );
}

/// An all-zero mask is the identity and an all-nonzero mask fills every
/// cell: the two degenerate masks every backend must get exactly right.
///
/// # Panics
///
/// Panics when either degenerate mask changes a cell it must not, or keeps
/// one it must not.
pub fn assert_masked_fill_degenerate_masks<D, M, T>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = [5i8, 6, 7].map(T::from).to_vec();
    let layout = Layout::c_contiguous([3]).expect("layout");
    let input = device.upload(&host).expect("input upload");

    let zero_mask = device.alloc_zeroed::<T>(3).expect("zero mask alloc");
    let kept = device.alloc_zeroed::<T>(3).expect("kept alloc");
    ops.masked_fill_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&zero_mask, &layout),
        T::from(42i8),
        hephaestus_core::StridedView::new(&kept, &layout),
    )
    .expect("masked_fill dispatch");
    let mut kept_got = vec![T::from(0i8); 3];
    device.download(&kept, &mut kept_got).expect("download");
    assert_eq!(
        kept_got, host,
        "{name}: an all-zero mask must reproduce the input exactly"
    );

    let ones_host: Vec<T> = [1i8, 1, 1].map(T::from).to_vec();
    let ones_mask = device.upload(&ones_host).expect("ones mask upload");
    let filled = device.alloc_zeroed::<T>(3).expect("filled alloc");
    ops.masked_fill_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&ones_mask, &layout),
        T::from(8i8),
        hephaestus_core::StridedView::new(&filled, &layout),
    )
    .expect("masked_fill dispatch");
    let filled_expected: Vec<T> = [8i8, 8, 8].map(T::from).to_vec();
    let mut filled_got = vec![T::from(0i8); 3];
    device.download(&filled, &mut filled_got).expect("download");
    assert_eq!(
        filled_got, filled_expected,
        "{name}: an all-nonzero mask must fill every cell"
    );
}

/// The float mask edges are pinned by IEEE equality, observably, except
/// where the dialect opts out: a negative-zero mask keeps (`-0.0 == 0.0`)
/// on every backend, so a naive bitwise mask test fails this clause, while
/// a `NaN` mask fills (unordered compares unequal to zero) only where the
/// dialect advertises IEEE special values (`KernelDialect::
/// IEEE_SPECIAL_VALUES`, ADR 0043) — a non-advertising dialect (WGSL)
/// skips that cell by construction through the const branch below, since
/// its specification permits treating NaN as absent.
///
/// # Panics
///
/// Panics when the backend fills under `-0.0`, or keeps under `NaN` on an
/// advertising dialect.
pub fn assert_masked_fill_special_mask_values<D, M>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, f32>,
{
    let name = device.backend_name();
    let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mask_host: Vec<f32> = vec![-0.0, f32::NAN, 0.0, -5.0];
    let layout = Layout::c_contiguous([4]).expect("layout");
    let input = device.upload(&host).expect("input upload");
    let mask = device.upload(&mask_host).expect("mask upload");
    let output = device.alloc_zeroed::<f32>(4).expect("output alloc");

    ops.masked_fill_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&mask, &layout),
        9.0,
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("masked_fill dispatch");

    let mut got = vec![0.0f32; 4];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        [got[0], got[2], got[3]],
        [1.0, 3.0, 9.0],
        "{name}: -0.0 must keep and finite nonzero must fill under IEEE mask equality"
    );
    if <M::Dialect as hephaestus_core::KernelDialect>::IEEE_SPECIAL_VALUES {
        assert_eq!(
            got[1], 9.0,
            "{name}: NaN must fill on an IEEE-advertising dialect"
        );
    }
}

/// An empty shape fills vacuously: matching `[2, 0]` shapes dispatch
/// successfully and write zero cells.
///
/// # Panics
///
/// Panics when the vacuous fill fails to dispatch.
pub fn assert_masked_fill_of_empty_is_vacuous<D, M, T>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let input = device.alloc_zeroed::<T>(0).expect("input alloc");
    let mask = device.alloc_zeroed::<T>(0).expect("mask alloc");
    let output = device.alloc_zeroed::<T>(0).expect("output alloc");

    ops.masked_fill_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        hephaestus_core::StridedView::new(&mask, &layout),
        T::from(9i8),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("vacuous masked_fill must dispatch");
}

/// Invalid geometry must fail typed: a mask shape mismatch and an output
/// shape mismatch each return [`HephaestusError::InvalidConfiguration`]
/// rather than dispatching. There is no broadcasting.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_masked_fill_rejects_invalid_geometry<D, M, T>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let input = device.alloc_zeroed::<T>(8).expect("input alloc");
    let mask = device.alloc_zeroed::<T>(8).expect("mask alloc");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");
    let view = || hephaestus_core::StridedView::new(&input, &layout);
    let msk = || hephaestus_core::StridedView::new(&mask, &layout);
    let out = || hephaestus_core::StridedView::new(&output, &layout);

    let short_layout = Layout::c_contiguous([2, 3]).expect("short layout");
    let short_mask = device.alloc_zeroed::<T>(6).expect("short mask alloc");
    let err = ops
        .masked_fill_into::<2>(
            device,
            view(),
            hephaestus_core::StridedView::new(&short_mask, &short_layout),
            T::from(9i8),
            out(),
        )
        .expect_err("mask mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: mask mismatch must fail typed, got {err:?}"
    );

    let short_output = device.alloc_zeroed::<T>(6).expect("short output alloc");
    let err = ops
        .masked_fill_into::<2>(
            device,
            view(),
            msk(),
            T::from(9i8),
            hephaestus_core::StridedView::new(&short_output, &short_layout),
        )
        .expect_err("output mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: output mismatch must fail typed, got {err:?}"
    );
}

/// An output aliasing either source must fail before mutating anything:
/// reads and writes race across threads, so sharing storage is a dispatch
/// error on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_masked_fill_rejects_aliased_output<D, M, T>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");

    // The shared buffer holds all eight cells, so both views fit its
    // storage: the geometry is fully valid and only the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");
    let mask = device.alloc_zeroed::<T>(8).expect("mask alloc");

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .masked_fill_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &layout),
            hephaestus_core::StridedView::new(&mask, &layout),
            T::from(9i8),
            hephaestus_core::StridedView::new(&shared, &layout),
        )
        .expect_err("masked_fill dispatch must reject aliased input/output");
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
        .masked_fill_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &layout),
            hephaestus_core::StridedView::new(&shared, &layout),
            T::from(9i8),
            hephaestus_core::StridedView::new(&shared, &layout),
        )
        .expect_err("masked_fill dispatch must reject aliased mask/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected mask-alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: mask-alias rejection mutated mask");
}

/// Run every generic clause in this module against one backend.
///
/// The `f32`-only [`assert_masked_fill_special_mask_values`] clause runs
/// separately from each backend's `f32` instantiation, the way the scatter
/// order clause does.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_masked_fill_contract<D, M, T>(device: &D, ops: &M)
where
    D: ComputeDevice,
    M: MaskedFillOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_masked_fill_value_contract(device, ops);
    assert_masked_fill_degenerate_masks(device, ops);
    assert_masked_fill_of_empty_is_vacuous(device, ops);
    assert_masked_fill_rejects_invalid_geometry(device, ops);
    assert_masked_fill_rejects_aliased_output(device, ops);
}
