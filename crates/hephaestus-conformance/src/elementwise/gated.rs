//! Gated-activation tail clauses: softplus, elu, mish, gelu, and gradients.

use eunomia::Pod;
use hephaestus_core::{
    ComputeDevice, DialectScalar, ElementwiseOps, EluOp, GeluGradOp, GeluOp, GeluTanhGradOp,
    GeluTanhOp, MishGradOp, MishOp, SoftplusOp, UnaryExpr, UnaryValue,
};

use super::core::{TAIL_REL_BOUND, unary};

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

/// Run the elu tail clauses against one backend.
///
/// Shares the `TAIL_REL_BOUND` derivation: `elu` is defective just below
/// zero under the naive `exp(x) - 1` transcription — the subtraction
/// cancels every significant bit once `|x|` drops to f32's `ulp(1)`
/// (100% relative error against an oracle of magnitude `|x|`). The
/// negative arm is exactly `expm1`, so the fix reuses its spelling
/// (native `expm1` on the C dialects, the Taylor select arm on WGSL).
/// Oracles are the f64 [`UnaryValue`] form `x` at `x >= 0`, `exp_m1`
/// below. Control points at `-20`/`-1`/`-0.5`/`0`/`0.5`/`1`/`5` pin the
/// already-correct arms on both sides of the `-0.5` crossover; only the
/// `0-` probes exercise the defect.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_elu_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    EluOp: UnaryExpr<E::Dialect>,
{
    elu_tails_match_oracle(device, ops);
}

/// Run the mish tail clauses against one backend.
///
/// Shares the `TAIL_REL_BOUND` derivation: `mish` inherits softplus's
/// negative-tail defect through the naive `tanh(log(1.0 + exp(x)))`
/// inner transcription — `1.0 + exp(x)` rounds to `1.0`, so the inner
/// value collapses to exactly `0` and the product to `0` (100%
/// relative error against an oracle of magnitude `|x| * exp(x)`). The
/// positive tail is already correct (`inf` flows through `tanh` to
/// `1`, and the output is `x` itself). Oracles are the f64
/// [`UnaryValue`] form `x * tanh(softplus(x))`. Control points at
/// `-1`/`-0.5`/`0`/`0.5`/`1`/`5`/`20`/`90` pin the already-correct
/// range; only the negative-tail probes exercise the defect.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_mish_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    MishOp: UnaryExpr<E::Dialect>,
{
    mish_tails_match_oracle(device, ops);
}

/// Run the gelu/gelu-tanh tail clauses against one backend.
///
/// Shares the `TAIL_REL_BOUND` derivation. Both forms are defective on
/// the negative tail, where `0.5·x·(1 + F)` collapses to exactly `0`
/// once `F` rounds to `-1`: the exact form through the naive
/// `1 + erf(x/√2)` transcription (the `erfc` value below `x = -5` is a
/// normal float the subtraction cannot see), the tanh approximation
/// through `1 + tanh(√(2/π)·(x + 0.044715·x³))` the same way. Oracles
/// are the f64 [`UnaryValue`] forms `0.5·x·erfc(-x/√2)` and
/// `x·sigmoid(2z)`. Probes stay where the truths are normal f32 (the
/// exact form below `-12`, the tanh form below `-10`, are already
/// f32-subnormal); controls pin the already-correct range.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_gelu_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    GeluOp: UnaryExpr<E::Dialect>,
    GeluTanhOp: UnaryExpr<E::Dialect>,
{
    gelu_tails_match_oracle(device, ops);
    gelu_tanh_tails_match_oracle(device, ops);
}

/// Run the gradient tail clauses against one backend.
///
/// Shares the `TAIL_REL_BOUND` derivation. The three gradient forms are
/// defective on the negative tail: `mish-grad` through the naive
/// `tanh(log(1 + exp(x)))` lead term (reads `0`, missing `~3%` of the
/// total at `x = -30`); `gelu-grad` through the naive `0.5·(1 + erf)`
/// term (reads `0`, missing the `Φ` term, `~1%` on the probes);
/// `gelu-tanh-grad` through both `1 ± tanh` factors (both collapse, and
/// the total reads `-0.0`, `100%` off). Oracles are the f64
/// [`UnaryValue`] forms. Probes stay where the truths are normal f32
/// (and, for `gelu-tanh-grad`, where the `sigmoid` intermediate is
/// normal too, so no `FTZ` driver can diverge); controls pin the
/// already-correct range.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_grad_tails_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect> + Pod,
    MishGradOp: UnaryExpr<E::Dialect>,
    GeluGradOp: UnaryExpr<E::Dialect>,
    GeluTanhGradOp: UnaryExpr<E::Dialect>,
{
    mish_grad_tails_match_oracle(device, ops);
    gelu_grad_tails_match_oracle(device, ops);
    gelu_tanh_grad_tails_match_oracle(device, ops);
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

/// `0-` probes at `-1e-10`/`-1e-7`/`-1e-5` collapse to exactly `0` under
/// the naive transcription (`exp(x)` rounds to `1.0` at f32's `ulp(1)`,
/// so `exp(x) - 1` loses every significant bit); the controls pin the
/// already-correct arms on both sides of the `-0.5` crossover.
fn elu_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    EluOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [
        -1e-10f32, -1e-7, -1e-5, -0.5, -1.0, -20.0, 0.0, 0.5, 1.0, 5.0,
    ];
    let got = unary::<_, _, EluOp, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <EluOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: elu lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-30`/`-20` collapse to exactly `0` under
/// the naive transcription (`1.0 + exp(x)` rounds to `1.0`, so the
/// `tanh(log(...))` inner factor is `tanh(0) = 0`); the controls pin
/// the already-correct middle and positive ranges.
fn mish_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    MishOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [-30.0f32, -20.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 20.0, 90.0];
    let got = unary::<_, _, MishOp, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <MishOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: mish lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-12`/`-10`/`-8`/`-6` collapse to exactly
/// `0` under the naive transcription (`erf(x/√2)` rounds to `-1.0`,
/// so `1 + erf` loses the `erfc` value, a normal float down to
/// `x = -12`); the controls pin the already-correct range.
fn gelu_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [
        -12.0f32, -10.0, -8.0, -6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0,
    ];
    let got = unary::<_, _, GeluOp, 12>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <GeluOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: gelu lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-10`/`-8`/`-6` collapse to exactly `0`
/// under the naive transcription (`tanh` rounds to `-1.0`, so
/// `1 + tanh` loses the true factor, normal down to `x = -10`); the
/// controls pin the already-correct range.
fn gelu_tanh_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluTanhOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [-10.0f32, -8.0, -6.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0];
    let got = unary::<_, _, GeluTanhOp, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <GeluTanhOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: gelu-tanh lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-80`/`-60`/`-30` miss the `tanh(sp)` lead
/// term under the naive transcription (the inner `log(1 + exp(x))`
/// reads `0`, so the lead term is `0` instead of `sp` — `~3%` of the
/// total at `x = -30`); `-14.5`/`-7` pin the spelling's arm boundaries
/// and the rest are already-correct controls.
fn mish_grad_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    MishGradOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [
        -80.0f32, -60.0, -30.0, -14.5, -7.0, -1.0, 0.0, 1.0, 5.0, 20.0,
    ];
    let got = unary::<_, _, MishGradOp, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <MishGradOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: mish-grad lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-12`/`-10`/`-8`/`-6` miss the `Φ` term
/// under the naive transcription (`0.5·(1 + erf)` reads `0` — `~1%` of
/// the total on these lanes); `-4`/`-2` pin the `erfc` spelling's arm
/// boundaries and the rest are already-correct controls.
fn gelu_grad_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluGradOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [
        -12.0f32, -10.0, -8.0, -6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0,
    ];
    let got = unary::<_, _, GeluGradOp, 12>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <GeluGradOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: gelu-grad lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}

/// Negative-tail probes at `-9.5`/`-8`/`-6` collapse to exactly `0`
/// under the naive transcription (both `1 ± tanh` factors lose their
/// true values — `100%` off); the deepest probe stops where the
/// `sigmoid` intermediate is still a normal float, and the rest are
/// already-correct controls.
fn gelu_tanh_grad_tails_match_oracle<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: ElementwiseOps<D, f32>,
    f32: DialectScalar<E::Dialect>,
    GeluTanhGradOp: UnaryExpr<E::Dialect>,
{
    let name = device.backend_name();
    let points = [-9.5f32, -8.0, -6.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0];
    let got = unary::<_, _, GeluTanhGradOp, 10>(device, ops, &points);
    for (lane, (&got, &x)) in got.iter().zip(&points).enumerate() {
        let expected = <GeluTanhGradOp as UnaryValue>::apply(f64::from(x));
        let error = (f64::from(got) - expected).abs();
        assert!(
            error <= TAIL_REL_BOUND * expected.abs(),
            "{name}: gelu-tanh-grad lane {lane} (x = {x}): got {got}, oracle {expected}"
        );
    }
}
