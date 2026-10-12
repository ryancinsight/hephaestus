//! Contract clauses for untyped unary and binary elementwise dispatch.
//!
//! Complements [`crate::typed_elementwise`], which owns the scalar-aware
//! comparison paths: these clauses cover the plain arithmetic surface —
//! `unary_into`, `binary_into`, their prepared forms, and strided traversal.
//!
//! Every fixture is dyadic (representable as `m·2^e`) and every operation
//! either preserves dyadic values exactly (negation, absolute value,
//! addition, subtraction, multiplication, division by powers of two) or is
//! IEEE-754 correctly rounded onto an exactly representable result
//! (`sqrt` of a perfect square). All oracles are therefore exact
//! equalities; no tolerance is involved.

use eunomia::Pod;
use hephaestus_core::{
    AbsOp, AddOp, BinaryExpr, ComputeDevice, DialectScalar, DivOp, ElementwiseOps, Expm1Op, J0Op,
    J1Op, K0Op, Log1pOp, MulOp, NegOp, SincOp, SoftplusOp, SqrtOp, StridedView, SubOp, UnaryExpr,
    UnaryValue,
};
use leto::Layout;

/// Run every untyped elementwise clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_elementwise_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    NegOp: UnaryExpr<E::Dialect>,
    AbsOp: UnaryExpr<E::Dialect>,
    SqrtOp: UnaryExpr<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
    SubOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
    DivOp: BinaryExpr<E::Dialect>,
{
    unary_ops_compute_exact_values(device, ops);
    binary_ops_compute_exact_values(device, ops);
    strided_operands_are_read_through_their_layouts(device, ops);
    prepared_unary_rebinds_bound_operands(device, ops);
    prepared_binary_rebinds_bound_operands(device, ops);
    scalar_ops_compute_exact_values(device, ops);
    prepared_scalar_rebinds_bound_operands(device, ops);
    shape_mismatch_is_rejected_before_mutation(device, ops);
}

/// Run the `sinc` clause against one backend.
///
/// `sinc` resolves its removable singularity to exactly 1 at `+-0` on every
/// backend. Nonzero lanes exercise each dialect's `sin`, whose rounding is
/// the driver's declared capability (ADR 0043), so the device clause pins
/// only the exact branch; value accuracy against `sin(x)/x` is covered by
/// the core `UnaryValue` oracles.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_sinc_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    SincOp: UnaryExpr<E::Dialect>,
{
    sinc_zero_is_exactly_one(device, ops);
}

/// Absolute tolerance for the Bessel clauses, derived below.
const BESSEL_BOUND: f32 = 1e-5;

/// Run the J0/J1 clauses against one backend.
///
/// Unlike the exact-oracle clauses above, Bessel values are transcendental:
/// the rational branch rounds f32 coefficients (~1e-7) through a six-level
/// Horner evaluation plus one division, and the Hankel branch rounds the
/// device `sin`/`cos`/`sqrt` (~1-2 ULP each) through a dozen flops. Both
/// branches land within ~2e-6 absolute on O(1) values; the assertions use
/// 1e-5, headroom that still fails a wrong coefficient or branch by orders
/// of magnitude. Oracles are leto's scalar `j0`/`j1` (the 1:1 reference);
/// test points avoid Bessel zeros so absolute error is the honest metric.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_bessel_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    J0Op: UnaryExpr<E::Dialect>,
    J1Op: UnaryExpr<E::Dialect>,
{
    bessel_branches_match_oracle(device, ops);
    bessel_zeros_are_exact(device, ops);
}

/// Run the K0 clauses against one backend.
///
/// Shares the `BESSEL_BOUND` derivation: the 9.8.5 branch rounds f32
/// coefficients through two six-level Horners plus one `log`, and the 9.8.6
/// branch rounds one `exp`/`sqrt` plus a six-level series, each landing
/// within ~2e-6 absolute on O(1) values. Oracles are leto's scalar
/// `bessel_k0` (the 1:1 reference). The domain guard is pinned exactly:
/// non-positive and non-finite lanes must read back NaN.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_k0_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    K0Op: UnaryExpr<E::Dialect>,
{
    k0_branches_match_oracle(device, ops);
    k0_guard_yields_nan(device, ops);
}

/// Relative tolerance shared by the KERNEL-TAIL clauses, derived below.
/// A correct f32 rendering combines ppm-level polynomial arms (dense
/// f32 sweeps of the exact op sequences peak at ~3e-7) with native
/// transcendentals — and the transcendentals are the binding
/// constraint: a dense Vulkan sweep on the reference adapter (RTX 5080)
/// measures `exp` up to 32 ULP data-dependently (3.8e-6 relative worst
/// over `[-90, 88]`, 15.5 ULP worst over the `[-40, -10]` tail), and
/// `log` carries ~1 ULP of unit magnitude in absolute error near 1. The
/// assertions admit `1e-5`, 2.6x headroom over the weakest accurate
/// platform link, with room for cross-driver variance; the naive
/// transcriptions (collapse to `0`, overflow to `inf`) still fail by
/// five orders of magnitude. Oracles are f64 value functions, accurate
/// to a few ULP.
const TAIL_REL_BOUND: f64 = 1e-5;

/// Run the expm1/log1p tail clauses against one backend.
///
/// `expm1`/`log1p` near zero are the KERNEL-TAIL defect ADR 0061 Decision 6
/// reserves for the unfused GPU renderings: the naive `(exp(x) - 1.0)` and
/// `log(1.0 + x)` transcriptions cancel to exactly `0` once `|x|` drops to
/// f32's `ulp(1)` — 100% relative error against an oracle of magnitude `|x|`.
/// Shares the `TAIL_REL_BOUND` derivation (native `expm1f`/`log1pf` on
/// the C dialects, Taylor/Chebyshev select arms on WGSL). Oracles are f64
/// `exp_m1`/`ln_1p`, accurate to a few ULP; every probe stays inside the
/// overflow limit (`expm1` finite below 88.7, `log1p` on `x > -1`), so
/// relative error is the honest metric. Control points at `+-0.5`/`+-1` pin
/// the already-correct range; only the tail probes exercise the defect.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_expm1_log1p_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    Expm1Op: UnaryExpr<E::Dialect>,
    Log1pOp: UnaryExpr<E::Dialect>,
{
    expm1_log1p_tails_match_oracle(device, ops);
}

/// Run the softplus tail clauses against one backend.
///
/// Shares the `TAIL_REL_BOUND` derivation: `softplus` is defective at
/// both tails under the naive `log(1.0 + exp(x))` transcription — the
/// negative tail collapses to exactly `0` once `exp(x)` drops below
/// f32's `ulp(1)` (100% relative error against an oracle of magnitude
/// `exp(x)`), and the positive tail overflows to `inf` past `x = 88.7`
/// where the true value is `x` itself. Oracles are the f64
/// [`UnaryValue`] stable form `max(x, 0) + ln_1p(exp(-|x|))`. Control
/// points at `0`/`+-0.5`/`+-1`/`5` pin the already-correct range; only
/// the tail probes exercise the defects.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_softplus_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    SoftplusOp: UnaryExpr<E::Dialect>,
{
    softplus_tails_match_oracle(device, ops);
}

/// Dispatch one rank-1 unary op and return the downloaded results.
fn unary<D, E, Op, const LEN: usize>(device: &D, ops: &E, input: &[f32]) -> [f32; LEN]
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Op: UnaryExpr<E::Dialect>,
{
    let a = device.upload(input).expect("input upload");
    let out = device.alloc_zeroed::<f32>(LEN).expect("output alloc");
    let layout = Layout::c_contiguous([LEN]).expect("rank-1 layout");
    ops.unary_into::<Op, 1>(
        device,
        StridedView::new(&a, &layout),
        StridedView::new(&out, &layout),
    )
    .expect("unary dispatch");
    let mut got = [0.0f32; LEN];
    device.download(&out, &mut got).expect("download");
    got
}

/// Dispatch one rank-1 binary op and return the downloaded results.
fn binary<D, E, Op, const LEN: usize>(device: &D, ops: &E, lhs: &[f32], rhs: &[f32]) -> [f32; LEN]
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Op: BinaryExpr<E::Dialect>,
{
    let a = device.upload(lhs).expect("lhs upload");
    let b = device.upload(rhs).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(LEN).expect("output alloc");
    let layout = Layout::c_contiguous([LEN]).expect("rank-1 layout");
    ops.binary_into::<Op, 1>(
        device,
        StridedView::new(&a, &layout),
        StridedView::new(&b, &layout),
        StridedView::new(&out, &layout),
    )
    .expect("binary dispatch");
    let mut got = [0.0f32; LEN];
    device.download(&out, &mut got).expect("download");
    got
}

/// Negation and absolute value are sign flips (exact); square roots of
/// perfect squares are correctly rounded onto exact dyadic results.
fn unary_ops_compute_exact_values<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    NegOp: UnaryExpr<E::Dialect>,
    AbsOp: UnaryExpr<E::Dialect>,
    SqrtOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let signed = [1.5f32, -2.0, 0.0, -0.25];
    assert_eq!(
        unary::<_, _, NegOp, 4>(device, ops, &signed),
        [-1.5, 2.0, 0.0, 0.25],
        "{name}: negation"
    );
    assert_eq!(
        unary::<_, _, AbsOp, 4>(device, ops, &signed),
        [1.5, 2.0, 0.0, 0.25],
        "{name}: absolute value"
    );
    assert_eq!(
        unary::<_, _, SqrtOp, 4>(device, ops, &[4.0, 9.0, 0.25, 1.0]),
        [2.0, 3.0, 0.5, 1.0],
        "{name}: square root of perfect squares"
    );
}

/// The removable singularity resolves to exactly 1 at `+-0`, exercising
/// each dialect's branch spelling (`select` vs ternary) on device.
fn sinc_zero_is_exactly_one<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SincOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    assert_eq!(
        unary::<_, _, SincOp, 2>(device, ops, &[0.0, -0.0]),
        [1.0, 1.0],
        "{name}: sinc resolves +-0 to exactly 1"
    );
}

/// Each lane takes its own branch (`|x| < 8` rational vs Hankel), so one
/// mixed dispatch proves per-lane branch selection plus both formulations.
fn bessel_branches_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    J0Op: UnaryExpr<E::Dialect>,
    J1Op: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points_j0 = [0.5f32, 1.0, 4.0, 7.0, 9.0, 12.0];
    let got_j0 = unary::<_, _, J0Op, 6>(device, ops, &points_j0);
    for (lane, (&got, &x)) in got_j0.iter().zip(&points_j0).enumerate() {
        let expected = leto_ops::j0(f64::from(x)) as f32;
        assert!(
            (got - expected).abs() < BESSEL_BOUND,
            "{name}: j0 lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
    let points_j1 = [0.5f32, 2.0, 5.0, -3.0, 10.0, -9.0];
    let got_j1 = unary::<_, _, J1Op, 6>(device, ops, &points_j1);
    for (lane, (&got, &x)) in got_j1.iter().zip(&points_j1).enumerate() {
        let expected = leto_ops::j1(f64::from(x)) as f32;
        assert!(
            (got - expected).abs() < BESSEL_BOUND,
            "{name}: j1 lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// `j0(0)` is exactly 1 through the zero guard; `j1(0)` is exactly 0 through
/// the rational form (`0 * finite / finite`).
fn bessel_zeros_are_exact<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    J0Op: UnaryExpr<E::Dialect>,
    J1Op: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    assert_eq!(
        unary::<_, _, J0Op, 1>(device, ops, &[0.0]),
        [1.0],
        "{name}: j0 resolves 0 to exactly 1"
    );
    assert_eq!(
        unary::<_, _, J1Op, 2>(device, ops, &[0.0, -0.0]),
        [0.0, 0.0],
        "{name}: j1 resolves +-0 to 0"
    );
}

/// Each lane takes its own branch (`x <= 2` polynomial vs scaled
/// exponential series), so one mixed dispatch proves per-lane branch
/// selection plus both formulations.
fn k0_branches_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    K0Op: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [0.1f32, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0];
    let got = unary::<_, _, K0Op, 7>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = leto_ops::bessel_k0(f64::from(x)) as f32;
        assert!(
            (got - expected).abs() < BESSEL_BOUND,
            "{name}: k0 lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// `K0` diverges at the origin and is undefined for `x <= 0`: the guard
/// reads back NaN on every invalid lane, exactly as the scalar helper.
fn k0_guard_yields_nan<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    K0Op: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [
        0.0f32,
        -0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let got = unary::<_, _, K0Op, 6>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        assert!(
            got.is_nan(),
            "{name}: k0 guard lane {lane} (x = {x}): got {got}, expected NaN"
        );
    }
}

/// Tail probes at `+-1e-10`/`+-1e-7`/`+-1e-5` collapse to exactly `0` under
/// the naive transcriptions (`1.0 + x` rounds to `1.0` at f32's `ulp(1)`,
/// and `exp(x) - 1` loses every significant bit the same way); the control
/// points at `+-0.5`/`+-1` already pass and pin the correct range.
fn expm1_log1p_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    Expm1Op: UnaryExpr<E::Dialect>,
    Log1pOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [
        1e-10f32, -1e-10, 1e-7, -1e-7, 1e-5, -1e-5, 0.5, -0.5, 1.0, -1.0,
    ];
    let got_expm1 = unary::<_, _, Expm1Op, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got_expm1.iter().zip(&points).enumerate() {
        let expected = f64::from(x).exp_m1();
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: expm1 lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
    let points_log1p = [
        1e-10f32, -1e-10, 1e-7, -1e-7, 1e-5, -1e-5, 0.5, -0.5, 1.0, 2.0,
    ];
    let got_log1p = unary::<_, _, Log1pOp, 10>(device, ops, &points_log1p);
    for (lane, (&got, &x)) in got_log1p.iter().zip(&points_log1p).enumerate() {
        let expected = f64::from(x).ln_1p();
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: log1p lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-20`/`-30` collapse to exactly `0` under the
/// naive transcription (`exp(x)` drops below f32's `ulp(1)`, so `1.0 +
/// exp(x)` rounds to `1.0`); positive-tail probes at `90`/`100` overflow
/// to `inf` (`exp(x)` overflows, and `log(inf)` stays `inf`). The
/// controls pin the already-correct middle range.
fn softplus_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SoftplusOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [-30.0f32, -20.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 90.0, 100.0];
    let got = unary::<_, _, SoftplusOp, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <SoftplusOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: softplus lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Dyadic binary arithmetic: every result is exactly representable.
fn binary_ops_compute_exact_values<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
    SubOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
    DivOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let lhs = [1.5f32, -2.0, 8.0, 0.25];
    let rhs = [0.5f32, 4.0, -2.0, 2.0];
    assert_eq!(
        binary::<_, _, AddOp, 4>(device, ops, &lhs, &rhs),
        [2.0, 2.0, 6.0, 2.25],
        "{name}: addition"
    );
    assert_eq!(
        binary::<_, _, SubOp, 4>(device, ops, &lhs, &rhs),
        [1.0, -6.0, 10.0, -1.75],
        "{name}: subtraction"
    );
    assert_eq!(
        binary::<_, _, MulOp, 4>(device, ops, &lhs, &rhs),
        [0.75, -8.0, -16.0, 0.5],
        "{name}: multiplication"
    );
    assert_eq!(
        binary::<_, _, DivOp, 4>(device, ops, &lhs, &rhs),
        [3.0, -0.5, -4.0, 0.125],
        "{name}: division by powers of two"
    );
}

/// A transposed input view is traversed through its strides, not its
/// storage order.
fn strided_operands_are_read_through_their_layouts<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    NegOp: UnaryExpr<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    // Storage [1,2,3,4] viewed as the transpose of the row-major 2x2, so the
    // logical matrix is [[1,3],[2,4]].
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("upload");
    let transposed = Layout::try_new([2, 2], [1, 2], 0).expect("valid conformance fixture layout");
    let dense = Layout::c_contiguous([2, 2]).expect("dense layout");

    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    ops.unary_into::<NegOp, 2>(
        device,
        StridedView::new(&a, &transposed),
        StridedView::new(&out, &dense),
    )
    .expect("strided unary dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [-1.0, -3.0, -2.0, -4.0],
        "{name}: unary must traverse the transposed layout"
    );

    let b = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    ops.binary_into::<AddOp, 2>(
        device,
        StridedView::new(&a, &transposed),
        StridedView::new(&b, &dense),
        StridedView::new(&out, &dense),
    )
    .expect("strided binary dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [11.0, 23.0, 32.0, 44.0],
        "{name}: binary must traverse the transposed lhs layout"
    );
}

/// A prepared unary dispatch re-runs over its bound operands: idempotent
/// over unchanged inputs, and writes to a bound input are observed.
fn prepared_unary_rebinds_bound_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    NegOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, -2.0, 3.0, -4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let prepared = ops
        .prepare_unary_into::<NegOp, 1>(
            device,
            StridedView::new(&a, &layout),
            StridedView::new(&out, &layout),
        )
        .expect("prepare unary");

    let expected = [-1.0f32, 2.0, -3.0, 4.0];
    ops.dispatch_unary::<1>(device, &prepared)
        .expect("dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, expected, "{name}: prepared negation");

    ops.dispatch_unary::<1>(device, &prepared)
        .expect("re-dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: prepared unary must be idempotent over unchanged operands"
    );

    device
        .write_buffer(&a, &[8.0f32, -8.0, 0.5, -0.5])
        .expect("input rewrite");
    ops.dispatch_unary::<1>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [-8.0, 8.0, -0.5, 0.5],
        "{name}: prepared unary must observe writes to its bound operand"
    );
}

/// A prepared binary dispatch re-runs over its bound operands likewise.
fn prepared_binary_rebinds_bound_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let b = device.upload(&[0.5f32, 0.5, 0.5, 0.5]).expect("rhs upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let prepared = ops
        .prepare_binary_into::<AddOp, 1>(
            device,
            StridedView::new(&a, &layout),
            StridedView::new(&b, &layout),
            StridedView::new(&out, &layout),
        )
        .expect("prepare binary");

    ops.dispatch_binary::<1>(device, &prepared)
        .expect("dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [1.5, 2.5, 3.5, 4.5], "{name}: prepared addition");

    device
        .write_buffer(&b, &[2.0f32, 2.0, 2.0, 2.0])
        .expect("rhs rewrite");
    ops.dispatch_binary::<1>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [3.0, 4.0, 5.0, 6.0],
        "{name}: prepared binary must observe writes to its bound operands"
    );
}

/// A shape mismatch is rejected before any output element is written.
fn shape_mismatch_is_rejected_before_mutation<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("lhs upload");
    let b = device.upload(&[1.0f32, 2.0]).expect("rhs upload");
    let out = device.upload(&[7.0f32, 7.0, 7.0, 7.0]).expect("sentinel");
    let four = Layout::c_contiguous([4]).expect("layout");
    let two = Layout::c_contiguous([2]).expect("layout");
    assert!(
        ops.binary_into::<AddOp, 1>(
            device,
            StridedView::new(&a, &four),
            StridedView::new(&b, &two),
            StridedView::new(&out, &four),
        )
        .is_err(),
        "{name}: shape mismatch must be rejected"
    );
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got, [7.0; 4],
        "{name}: rejected dispatch must not mutate the output"
    );
}

/// Broadcast-scalar arithmetic: dyadic operands and scalars, exact results.
fn scalar_ops_compute_exact_values<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    SubOp: BinaryExpr<E::Dialect>,
    MulOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let input = [3.0f32, 0.5, -1.0, 8.0];
    let a = device.upload(&input).expect("input upload");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");

    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    ops.scalar_into::<SubOp, 1>(
        device,
        StridedView::new(&a, &layout),
        2.0,
        StridedView::new(&out, &layout),
    )
    .expect("scalar sub dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [1.0, -1.5, -3.0, 6.0],
        "{name}: broadcast-scalar subtraction"
    );

    ops.scalar_into::<MulOp, 1>(
        device,
        StridedView::new(&a, &layout),
        0.5,
        StridedView::new(&out, &layout),
    )
    .expect("scalar mul dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [1.5, 0.25, -0.5, 4.0],
        "{name}: broadcast-scalar multiplication"
    );
}

/// A prepared broadcast-scalar dispatch re-runs over its bound buffers; the
/// scalar itself is fixed at preparation.
fn prepared_scalar_rebinds_bound_operands<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    AddOp: BinaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).expect("upload");
    let out = device.alloc_zeroed::<f32>(4).expect("output alloc");
    let layout = Layout::c_contiguous([4]).expect("rank-1 layout");
    let prepared = ops
        .prepare_scalar_into::<AddOp, 1>(
            device,
            StridedView::new(&a, &layout),
            1.0,
            StridedView::new(&out, &layout),
        )
        .expect("prepare scalar");

    ops.dispatch_scalar::<1>(device, &prepared)
        .expect("dispatch");
    let mut got = [0.0f32; 4];
    device.download(&out, &mut got).expect("download");
    assert_eq!(got, [2.0, 3.0, 4.0, 5.0], "{name}: prepared scalar add");

    device
        .write_buffer(&a, &[10.0f32, 20.0, 30.0, 40.0])
        .expect("input rewrite");
    ops.dispatch_scalar::<1>(device, &prepared)
        .expect("rebind dispatch");
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        [11.0, 21.0, 31.0, 41.0],
        "{name}: prepared scalar must observe buffer writes while keeping its captured scalar"
    );
}
