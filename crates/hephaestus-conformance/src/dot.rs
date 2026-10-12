//! Contract clauses for the device-neutral [`hephaestus_core::DotOps`] seam.
//!
//! There is no strided `leto::dot` to delegate to — leto's `dot`
//! methods live on fixed-size vector types, not strided views — so the
//! host reference carries the direct implementation and these clauses
//! pin its value semantics with hand-computed fixtures: fold the
//! elementwise products of two equal-numel inputs in ascending flat
//! order into a caller-owned `[1]` cell, the `torch.dot` semantic
//! `coeus_ops::dot` documents. Fixtures use small integers so every
//! scalar is an exact equality, never a tolerance-bounded comparison.
//!
//! The kernel is pure arithmetic with no value comparison, so unlike
//! the ordering and masking seams there is no dialect branch:
//! non-finite propagation follows the scalar arithmetic on every
//! backend, and the special-values fixtures assert everywhere.

use hephaestus_core::{ComputeDevice, DotOps, HephaestusError};
use leto::Layout;

/// Fold hand-computed fixtures into the scalar cell: 1-D `[1, 2, 3]`
/// against `[4, 5, 6]` yields `32`; mixed-rank `[[1, 2], [3, 4]]`
/// against `[5, 6, 7, 8]` yields `70`, pinning the dual-rank shapes and
/// the shared flat order; and a self-dot of `[1, 2, 3]` against the
/// same buffer yields `14`, pinning input-input aliasing as legal —
/// both inputs are only read, so `dot(x, x)` dispatches.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with any
/// fixture.
pub fn assert_dot_value_contract<D, O, T>(device: &D, ops: &O)
where
    D: ComputeDevice,
    O: DotOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let out_layout = Layout::c_contiguous([1]).expect("out layout");

    // 1-D: 1*4 + 2*5 + 3*6 = 32.
    let lane_layout = Layout::c_contiguous([3]).expect("lane layout");
    let lane_a = device
        .upload(&[1i8, 2, 3].map(T::from))
        .expect("lane a upload");
    let lane_b = device
        .upload(&[4i8, 5, 6].map(T::from))
        .expect("lane b upload");
    let lane_out = device.alloc_zeroed::<T>(1).expect("lane out alloc");
    ops.dot_into::<1, 1>(
        device,
        hephaestus_core::StridedView::new(&lane_a, &lane_layout),
        hephaestus_core::StridedView::new(&lane_b, &lane_layout),
        hephaestus_core::StridedView::new(&lane_out, &out_layout),
    )
    .expect("1-D dot dispatch");
    let mut lane_got = vec![T::from(0i8); 1];
    device.download(&lane_out, &mut lane_got).expect("download");
    assert_eq!(
        lane_got,
        vec![T::from(32i8)],
        "{name}: 1-D dot must match the hand-computed scalar exactly"
    );

    // Mixed rank: flat [1, 2, 3, 4] against [5, 6, 7, 8] —
    // 5 + 12 + 21 + 32 = 70.
    let square_layout = Layout::c_contiguous([2, 2]).expect("square layout");
    let row_layout = Layout::c_contiguous([4]).expect("row layout");
    let square_a = device
        .upload(&[1i8, 2, 3, 4].map(T::from))
        .expect("square a upload");
    let row_b = device
        .upload(&[5i8, 6, 7, 8].map(T::from))
        .expect("row b upload");
    let mixed_out = device.alloc_zeroed::<T>(1).expect("mixed out alloc");
    ops.dot_into::<2, 1>(
        device,
        hephaestus_core::StridedView::new(&square_a, &square_layout),
        hephaestus_core::StridedView::new(&row_b, &row_layout),
        hephaestus_core::StridedView::new(&mixed_out, &out_layout),
    )
    .expect("mixed-rank dot dispatch");
    let mut mixed_got = vec![T::from(0i8); 1];
    device
        .download(&mixed_out, &mut mixed_got)
        .expect("download");
    assert_eq!(
        mixed_got,
        vec![T::from(70i8)],
        "{name}: mixed-rank dot must match the hand-computed scalar exactly"
    );

    // Self-dot over one shared buffer: 1 + 4 + 9 = 14. Inputs may alias
    // each other; only the output aliasing an input is rejected.
    let self_out = device.alloc_zeroed::<T>(1).expect("self out alloc");
    let shared = || hephaestus_core::StridedView::new(&lane_a, &lane_layout);
    ops.dot_into::<1, 1>(
        device,
        shared(),
        shared(),
        hephaestus_core::StridedView::new(&self_out, &out_layout),
    )
    .expect("self-dot must dispatch: inputs may alias each other");
    let mut self_got = vec![T::from(0i8); 1];
    device.download(&self_out, &mut self_got).expect("download");
    assert_eq!(
        self_got,
        vec![T::from(14i8)],
        "{name}: self-dot must match the hand-computed scalar exactly"
    );
}

/// The float fixtures every backend asserts: exactly-representable
/// fractions, the empty sum, and non-finite propagation. `[0.5, 1.5]`
/// against `[2.0, 4.0]` yields exactly `7.0`; `[inf]` against `[2.0]`
/// yields `inf`; `[NaN]` against `[1.0]` yields `NaN` — multiplication
/// and addition propagate `NaN` on every dialect, since the kernel
/// never compares a value.
///
/// # Panics
///
/// Panics when the backend miscomputes a special-values fixture.
pub fn assert_dot_special_float_values<D, O>(device: &D, ops: &O)
where
    D: ComputeDevice,
    O: DotOps<D, f32>,
{
    let name = device.backend_name();
    let out_layout = Layout::c_contiguous([1]).expect("out layout");

    let frac_layout = Layout::c_contiguous([2]).expect("frac layout");
    let frac_a = device.upload(&[0.5f32, 1.5]).expect("frac a upload");
    let frac_b = device.upload(&[2.0f32, 4.0]).expect("frac b upload");
    let frac_out = device.alloc_zeroed::<f32>(1).expect("frac out alloc");
    ops.dot_into::<1, 1>(
        device,
        hephaestus_core::StridedView::new(&frac_a, &frac_layout),
        hephaestus_core::StridedView::new(&frac_b, &frac_layout),
        hephaestus_core::StridedView::new(&frac_out, &out_layout),
    )
    .expect("fraction dot dispatch");
    let mut frac_got = vec![0.0f32; 1];
    device.download(&frac_out, &mut frac_got).expect("download");
    assert_eq!(
        frac_got,
        vec![7.0f32],
        "{name}: fraction dot must match exactly"
    );

    let single_layout = Layout::c_contiguous([1]).expect("single layout");
    let inf_a = device.upload(&[f32::INFINITY]).expect("inf a upload");
    let inf_b = device.upload(&[2.0f32]).expect("inf b upload");
    let inf_out = device.alloc_zeroed::<f32>(1).expect("inf out alloc");
    ops.dot_into::<1, 1>(
        device,
        hephaestus_core::StridedView::new(&inf_a, &single_layout),
        hephaestus_core::StridedView::new(&inf_b, &single_layout),
        hephaestus_core::StridedView::new(&inf_out, &out_layout),
    )
    .expect("inf dot dispatch");
    let mut inf_got = vec![0.0f32; 1];
    device.download(&inf_out, &mut inf_got).expect("download");
    assert_eq!(
        inf_got,
        vec![f32::INFINITY],
        "{name}: inf dot must yield inf"
    );

    let nan_a = device.upload(&[f32::NAN]).expect("nan a upload");
    let nan_b = device.upload(&[1.0f32]).expect("nan b upload");
    let nan_out = device.alloc_zeroed::<f32>(1).expect("nan out alloc");
    ops.dot_into::<1, 1>(
        device,
        hephaestus_core::StridedView::new(&nan_a, &single_layout),
        hephaestus_core::StridedView::new(&nan_b, &single_layout),
        hephaestus_core::StridedView::new(&nan_out, &out_layout),
    )
    .expect("nan dot dispatch");
    let mut nan_got = vec![0.0f32; 1];
    device.download(&nan_out, &mut nan_got).expect("download");
    assert!(
        nan_got[0].is_nan(),
        "{name}: NaN dot must propagate NaN, got {:?}",
        nan_got[0]
    );
}

/// An empty fold yields zero without reading either input: matching
/// `[2, 0]` and `[0]` inputs dispatch successfully and write `T::zero()`
/// into the scalar cell.
///
/// # Panics
///
/// Panics when the vacuous fold fails to dispatch or misreports.
pub fn assert_dot_of_empty_is_vacuous<D, O, T>(device: &D, ops: &O)
where
    D: ComputeDevice,
    O: DotOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let a_layout = Layout::c_contiguous([2, 0]).expect("empty a layout");
    let b_layout = Layout::c_contiguous([0]).expect("empty b layout");
    let out_layout = Layout::c_contiguous([1]).expect("out layout");
    let a = device.alloc_zeroed::<T>(0).expect("a alloc");
    let b = device.alloc_zeroed::<T>(0).expect("b alloc");
    let out = device.alloc_zeroed::<T>(1).expect("out alloc");

    ops.dot_into::<2, 1>(
        device,
        hephaestus_core::StridedView::new(&a, &a_layout),
        hephaestus_core::StridedView::new(&b, &b_layout),
        hephaestus_core::StridedView::new(&out, &out_layout),
    )
    .expect("vacuous dot must dispatch");

    let mut got = vec![T::from(0i8); 1];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        vec![T::from(0i8)],
        "{name}: vacuous dot must yield zero"
    );
}

/// Invalid geometry must fail typed: a numel mismatch and a non-scalar
/// output each return [`HephaestusError::InvalidConfiguration`] rather
/// than dispatching.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_dot_rejects_invalid_geometry<D, O, T>(device: &D, ops: &O)
where
    D: ComputeDevice,
    O: DotOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let a_layout = Layout::c_contiguous([2, 4]).expect("a layout");
    let b_layout = Layout::c_contiguous([8]).expect("b layout");
    let short_layout = Layout::c_contiguous([7]).expect("short layout");
    let out_layout = Layout::c_contiguous([1]).expect("out layout");
    let long_layout = Layout::c_contiguous([2]).expect("long layout");
    let a = device.alloc_zeroed::<T>(8).expect("a alloc");
    let b = device.alloc_zeroed::<T>(8).expect("b alloc");
    let short = device.alloc_zeroed::<T>(7).expect("short alloc");
    let out = device.alloc_zeroed::<T>(1).expect("out alloc");
    let long = device.alloc_zeroed::<T>(2).expect("long alloc");
    let va = || hephaestus_core::StridedView::new(&a, &a_layout);
    let vb = || hephaestus_core::StridedView::new(&b, &b_layout);
    let vo = || hephaestus_core::StridedView::new(&out, &out_layout);

    let err = ops
        .dot_into::<2, 1>(
            device,
            va(),
            hephaestus_core::StridedView::new(&short, &short_layout),
            vo(),
        )
        .expect_err("numel mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: numel mismatch must fail typed, got {err:?}"
    );

    let err = ops
        .dot_into::<2, 1>(
            device,
            va(),
            vb(),
            hephaestus_core::StridedView::new(&long, &long_layout),
        )
        .expect_err("non-scalar output must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: non-scalar output must fail typed, got {err:?}"
    );
}

/// The output aliasing either input must fail before mutating anything:
/// the single write races the input reads it folds, so sharing that
/// storage is a dispatch error on every backend, not just the device
/// ones. The two inputs aliasing each other stays legal — see the
/// self-dot in [`assert_dot_value_contract`] — so only the two
/// output pairs are tested.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_dot_rejects_aliased_output<D, O, T>(device: &D, ops: &O)
where
    D: ComputeDevice,
    O: DotOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let lane_layout = Layout::c_contiguous([4]).expect("lane layout");
    let out_layout = Layout::c_contiguous([1]).expect("out layout");

    // `which` selects the aliased pair: 0 = out/a, 1 = out/b. The
    // shared buffer holds the input's 4 cells with the scalar view at
    // offset 0, so the geometry is fully valid and only the aliasing
    // fails.
    for which in 0..2 {
        let host: Vec<T> = [3i8, 1, 2, 0].map(T::from).to_vec();
        let shared = device.upload(&host).expect("shared upload");
        let fresh = device.upload(&host).expect("fresh upload");
        let shared_lane = || hephaestus_core::StridedView::new(&shared, &lane_layout);
        let shared_out = || hephaestus_core::StridedView::new(&shared, &out_layout);
        let fresh_lane = || hephaestus_core::StridedView::new(&fresh, &lane_layout);
        let (a, b, out) = match which {
            0 => (shared_lane(), fresh_lane(), shared_out()),
            _ => (fresh_lane(), shared_lane(), shared_out()),
        };

        let before = device
            .download_owned(&shared)
            .expect("alias fixture download");
        let error = ops
            .dot_into::<1, 1>(device, a, b, out)
            .expect_err("dot dispatch must reject an aliased output");
        assert!(
            error.to_string().contains("must not alias"),
            "{name}: unexpected alias error for pair {which}: {error}"
        );
        let after = device
            .download_owned(&shared)
            .expect("alias fixture download");
        assert_eq!(
            after, before,
            "{name}: alias rejection for pair {which} mutated the shared buffer"
        );
    }
}

/// Run every generic clause in this module against one backend.
///
/// The `f32`-only [`assert_dot_special_float_values`] clause runs
/// separately from each backend's `f32` instantiation, the way the
/// sort and nonzero special-values clauses do.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_dot_contract<D, O, T>(device: &D, ops: &O)
where
    D: ComputeDevice,
    O: DotOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_dot_value_contract(device, ops);
    assert_dot_of_empty_is_vacuous(device, ops);
    assert_dot_rejects_invalid_geometry(device, ops);
    assert_dot_rejects_aliased_output(device, ops);
}
