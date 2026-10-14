//! Contract clauses for the device-neutral [`hephaestus_core::WhereOps`] seam.
//!
//! There is no `leto::where_cond` to delegate to, so the host reference
//! carries the direct implementation and these clauses pin its value
//! semantics with hand-computed fixtures: `output[i] = on_true[i]`
//! wherever the condition cell is nonzero, `on_false[i]` elsewhere — the
//! same-shape contract `coeus_ops::where_cond` documents. Fixtures use
//! small signed integers so every copy is an exact equality, never a
//! tolerance-bounded comparison. A cell selects `on_true` exactly when
//! its condition value compares unequal to zero under the scalar's
//! equality, which pins the float edges (a `NaN` condition selects
//! `on_true`, a negative-zero condition selects `on_false`) in a
//! dedicated `f32` clause; backends must not reinterpret the condition
//! bitwise nor threshold it.

use hephaestus_core::{ComputeDevice, HephaestusError, WhereOps};
use leto::Layout;

/// Select under a mixed condition against hand-computed fixtures: a 1-D
/// input with positive, zero, and negative condition cells, and a 2-D
/// input with a diagonal-ish condition. Any nonzero magnitude, however
/// small or negative, selects `on_true`; only exact zero selects
/// `on_false`.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_where_value_contract<D, W, T>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // 1-D: cond [0, 1, 0, -2] selects [F, T, F, T] per cell.
    let cond_host: Vec<T> = [0i8, 1, 0, -2].map(T::from).to_vec();
    let true_host: Vec<T> = [10i8, 11, 12, 13].map(T::from).to_vec();
    let false_host: Vec<T> = [20i8, 21, 22, 23].map(T::from).to_vec();
    let layout = Layout::c_contiguous([4]).expect("layout");
    let cond = device.upload(&cond_host).expect("cond upload");
    let on_true = device.upload(&true_host).expect("on_true upload");
    let on_false = device.upload(&false_host).expect("on_false upload");
    let output = device.alloc_zeroed::<T>(4).expect("output alloc");

    ops.where_into::<1>(
        device,
        hephaestus_core::StridedView::new(&cond, &layout),
        hephaestus_core::StridedView::new(&on_true, &layout),
        hephaestus_core::StridedView::new(&on_false, &layout),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("where dispatch");

    let expected: Vec<T> = [20i8, 11, 22, 13].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 4];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: 1-D where must match the hand-computed fixture exactly"
    );

    // 2-D: cond [[1, 0], [0, -1]] selects the corners from on_true.
    let square_cond: Vec<T> = [1i8, 0, 0, -1].map(T::from).to_vec();
    let square_true: Vec<T> = [1i8, 2, 3, 4].map(T::from).to_vec();
    let square_false: Vec<T> = [5i8, 6, 7, 8].map(T::from).to_vec();
    let square_layout = Layout::c_contiguous([2, 2]).expect("square layout");
    let square_c = device.upload(&square_cond).expect("square cond upload");
    let square_t = device.upload(&square_true).expect("square on_true upload");
    let square_f = device
        .upload(&square_false)
        .expect("square on_false upload");
    let square_output = device.alloc_zeroed::<T>(4).expect("square output alloc");

    ops.where_into::<2>(
        device,
        hephaestus_core::StridedView::new(&square_c, &square_layout),
        hephaestus_core::StridedView::new(&square_t, &square_layout),
        hephaestus_core::StridedView::new(&square_f, &square_layout),
        hephaestus_core::StridedView::new(&square_output, &square_layout),
    )
    .expect("where dispatch");

    let square_expected: Vec<T> = [1i8, 6, 7, 4].map(T::from).to_vec();
    let mut square_got = vec![T::from(0i8); 4];
    device
        .download(&square_output, &mut square_got)
        .expect("download");
    assert_eq!(
        square_got, square_expected,
        "{name}: 2-D where must match the hand-computed fixture exactly"
    );
}

/// An all-zero condition copies `on_false` and an all-nonzero condition
/// copies `on_true`: the two degenerate conditions every backend must get
/// exactly right.
///
/// # Panics
///
/// Panics when either degenerate condition selects a cell from the wrong
/// branch.
pub fn assert_where_degenerate_conds<D, W, T>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let true_host: Vec<T> = [1i8, 2, 3].map(T::from).to_vec();
    let false_host: Vec<T> = [4i8, 5, 6].map(T::from).to_vec();
    let layout = Layout::c_contiguous([3]).expect("layout");
    let on_true = device.upload(&true_host).expect("on_true upload");
    let on_false = device.upload(&false_host).expect("on_false upload");

    let zero_cond = device.alloc_zeroed::<T>(3).expect("zero cond alloc");
    let kept = device.alloc_zeroed::<T>(3).expect("kept alloc");
    ops.where_into::<1>(
        device,
        hephaestus_core::StridedView::new(&zero_cond, &layout),
        hephaestus_core::StridedView::new(&on_true, &layout),
        hephaestus_core::StridedView::new(&on_false, &layout),
        hephaestus_core::StridedView::new(&kept, &layout),
    )
    .expect("where dispatch");
    let mut kept_got = vec![T::from(0i8); 3];
    device.download(&kept, &mut kept_got).expect("download");
    assert_eq!(
        kept_got, false_host,
        "{name}: an all-zero condition must copy on_false exactly"
    );

    let ones_host: Vec<T> = [1i8, 1, 1].map(T::from).to_vec();
    let ones_cond = device.upload(&ones_host).expect("ones cond upload");
    let taken = device.alloc_zeroed::<T>(3).expect("taken alloc");
    ops.where_into::<1>(
        device,
        hephaestus_core::StridedView::new(&ones_cond, &layout),
        hephaestus_core::StridedView::new(&on_true, &layout),
        hephaestus_core::StridedView::new(&on_false, &layout),
        hephaestus_core::StridedView::new(&taken, &layout),
    )
    .expect("where dispatch");
    let mut taken_got = vec![T::from(0i8); 3];
    device.download(&taken, &mut taken_got).expect("download");
    assert_eq!(
        taken_got, true_host,
        "{name}: an all-nonzero condition must copy on_true exactly"
    );
}

/// The float condition edges are pinned by IEEE equality, observably,
/// except where the dialect opts out: a negative-zero condition selects
/// `on_false` (`-0.0 == 0.0`) on every backend, so a naive bitwise
/// condition test fails this clause, while a `NaN` condition selects
/// `on_true` (unordered compares unequal to zero) only where the dialect
/// advertises IEEE special values (`KernelDialect::IEEE_SPECIAL_VALUES`,
/// ADR 0043) — a non-advertising dialect (WGSL) skips that cell by
/// construction through the const branch below, since its specification
/// permits treating NaN as absent.
///
/// # Panics
///
/// Panics when the backend selects `on_true` under `-0.0`, or `on_false`
/// under `NaN` on an advertising dialect.
pub fn assert_where_special_cond_values<D, W>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, f32>,
{
    let name = device.backend_name();
    let cond_host: Vec<f32> = vec![-0.0, f32::NAN, 0.0, -5.0];
    let true_host: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0];
    let false_host: Vec<f32> = vec![20.0, 21.0, 22.0, 23.0];
    let layout = Layout::c_contiguous([4]).expect("layout");
    let cond = device.upload(&cond_host).expect("cond upload");
    let on_true = device.upload(&true_host).expect("on_true upload");
    let on_false = device.upload(&false_host).expect("on_false upload");
    let output = device.alloc_zeroed::<f32>(4).expect("output alloc");

    ops.where_into::<1>(
        device,
        hephaestus_core::StridedView::new(&cond, &layout),
        hephaestus_core::StridedView::new(&on_true, &layout),
        hephaestus_core::StridedView::new(&on_false, &layout),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("where dispatch");

    let mut got = vec![0.0f32; 4];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        [got[0], got[2], got[3]],
        [20.0, 22.0, 13.0],
        "{name}: -0.0 must select on_false and finite nonzero must select on_true"
    );
    if <W::Dialect as hephaestus_core::KernelDialect>::IEEE_SPECIAL_VALUES {
        assert_eq!(
            got[1], 11.0,
            "{name}: NaN must select on_true on an IEEE-advertising dialect"
        );
    }
}

/// An empty shape selects vacuously: matching `[2, 0]` shapes dispatch
/// successfully and write zero cells.
///
/// # Panics
///
/// Panics when the vacuous selection fails to dispatch.
pub fn assert_where_of_empty_is_vacuous<D, W, T>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let layout = Layout::c_contiguous([2, 0]).expect("empty layout");
    let cond = device.alloc_zeroed::<T>(0).expect("cond alloc");
    let on_true = device.alloc_zeroed::<T>(0).expect("on_true alloc");
    let on_false = device.alloc_zeroed::<T>(0).expect("on_false alloc");
    let output = device.alloc_zeroed::<T>(0).expect("output alloc");

    ops.where_into::<2>(
        device,
        hephaestus_core::StridedView::new(&cond, &layout),
        hephaestus_core::StridedView::new(&on_true, &layout),
        hephaestus_core::StridedView::new(&on_false, &layout),
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("vacuous where must dispatch");
}

/// Invalid geometry must fail typed: an `on_true`, `on_false`, or output
/// shape mismatch each return [`HephaestusError::InvalidConfiguration`]
/// rather than dispatching. There is no broadcasting.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_where_rejects_invalid_geometry<D, W, T>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");
    let short_layout = Layout::c_contiguous([2, 3]).expect("short layout");
    let cond = device.alloc_zeroed::<T>(8).expect("cond alloc");
    let on_true = device.alloc_zeroed::<T>(8).expect("on_true alloc");
    let on_false = device.alloc_zeroed::<T>(8).expect("on_false alloc");
    let output = device.alloc_zeroed::<T>(8).expect("output alloc");
    let short = device.alloc_zeroed::<T>(6).expect("short alloc");
    let cnd = || hephaestus_core::StridedView::new(&cond, &layout);
    let tru = || hephaestus_core::StridedView::new(&on_true, &layout);
    let fls = || hephaestus_core::StridedView::new(&on_false, &layout);
    let out = || hephaestus_core::StridedView::new(&output, &layout);
    let shr = || hephaestus_core::StridedView::new(&short, &short_layout);

    let err = ops
        .where_into::<2>(device, cnd(), shr(), fls(), out())
        .expect_err("on_true mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: on_true mismatch must fail typed, got {err:?}"
    );

    let err = ops
        .where_into::<2>(device, cnd(), tru(), shr(), out())
        .expect_err("on_false mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: on_false mismatch must fail typed, got {err:?}"
    );

    let err = ops
        .where_into::<2>(device, cnd(), tru(), fls(), shr())
        .expect_err("output mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: output mismatch must fail typed, got {err:?}"
    );
}

/// An output aliasing any source must fail before mutating anything:
/// reads and writes race across threads, so sharing storage is a dispatch
/// error on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_where_rejects_aliased_output<D, W, T>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..8i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 4]).expect("layout");

    // `which` selects the aliased source: 0 = cond, 1 = on_true,
    // 2 = on_false. The shared buffer holds all eight cells, so every
    // view fits its storage: the geometry is fully valid and only the
    // aliasing fails.
    for which in 0..3 {
        let shared = device.upload(&host).expect("shared upload");
        let fresh = device.upload(&host).expect("fresh upload");
        let shared_view = || hephaestus_core::StridedView::new(&shared, &layout);
        let fresh_view = || hephaestus_core::StridedView::new(&fresh, &layout);
        let (cond, on_true, on_false) = match which {
            0 => (shared_view(), fresh_view(), fresh_view()),
            1 => (fresh_view(), shared_view(), fresh_view()),
            _ => (fresh_view(), fresh_view(), shared_view()),
        };

        let before = device
            .download_owned(&shared)
            .expect("alias fixture download");
        let error = ops
            .where_into::<2>(device, cond, on_true, on_false, shared_view())
            .expect_err("where dispatch must reject an aliased output");
        assert!(
            error.to_string().contains("must not alias"),
            "{name}: unexpected alias error for source {which}: {error}"
        );
        let after = device
            .download_owned(&shared)
            .expect("alias fixture download");
        assert_eq!(
            after, before,
            "{name}: alias rejection for source {which} mutated the shared buffer"
        );
    }
}

/// Run every generic clause in this module against one backend.
///
/// The `f32`-only [`assert_where_special_cond_values`] clause runs
/// separately from each backend's `f32` instantiation, the way the
/// masked-fill special-values clause does.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_where_contract<D, W, T>(device: &D, ops: &W)
where
    D: ComputeDevice,
    W: WhereOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_where_value_contract(device, ops);
    assert_where_degenerate_conds(device, ops);
    assert_where_of_empty_is_vacuous(device, ops);
    assert_where_rejects_invalid_geometry(device, ops);
    assert_where_rejects_aliased_output(device, ops);
}
