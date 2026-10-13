//! Special-function clauses: sinc, Bessel J0/J1, K0, and expm1/log1p tails.

use eunomia::Pod;
use hephaestus_core::{
    ComputeDevice, DialectScalar, ElementwiseOps, Expm1Op, J0Op, J1Op, K0Op, Log1pOp, SincOp,
    UnaryExpr,
};

use super::core::{TAIL_REL_BOUND, unary};

/// Absolute tolerance for the Bessel clauses, derived below.
const BESSEL_BOUND: f32 = 1e-5;

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
