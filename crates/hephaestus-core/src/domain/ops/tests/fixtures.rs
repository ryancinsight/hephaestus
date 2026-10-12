use super::*;

/// The value-function cancellation and rounding oracles ADR 0061 names,
/// exercised once for every shipped real scalar (`f32`, `f64`) rather than
/// per-type copies (standards: Generic Instantiation Coverage).
///
/// Tolerances are relative and derived from each function's own error
/// budget: `expm1`/`log1p`/`erfc` are libm-backed to a few ULP, so `1e-3`
/// relative is generous headroom above rounding noise while still catching a
/// cancellation collapse to zero (a collapse fails by 100%, not by ULPs).
pub(crate) fn assert_value_oracles_hold<T>()
where
    T: RealField + core::fmt::Debug,
{
    let tiny = T::from_f64(1e-10);
    let relative = |got: T, reference: f64| (got.to_f64() - reference).abs() / reference.abs();

    // exp(1e-10) - 1 would cancel to 0 at f32's ~1.2e-7 machine epsilon;
    // exp_m1 keeps the full relative value.
    assert!(
        relative(<Expm1Op as UnaryValue>::apply(tiny), 1e-10) < 1e-3,
        "expm1(1e-10) must retain relative precision"
    );
    // ln(1 + 1e-10) suffers the same cancellation; ln_1p does not.
    assert!(
        relative(<Log1pOp as UnaryValue>::apply(tiny), 1e-10) < 1e-3,
        "log1p(1e-10) must retain relative precision"
    );

    // softplus(-20) = ln(1+exp(-20)); the naive rendering underflows
    // ln(1+0) = 0 (ADR 0061 Decision 6), the value function must not.
    assert!(
        relative(
            <SoftplusOp as UnaryValue>::apply(T::from_f64(-20.0)),
            2.061_153_622e-9
        ) < 1e-3,
        "softplus(-20) must match the reference tail value"
    );
    // softplus(100) = 100 + a term far below 100's ULP; must not overflow
    // exp(100) into the sum.
    assert_eq!(
        <SoftplusOp as UnaryValue>::apply(T::from_f64(100.0)),
        T::from_f64(100.0),
        "softplus(100) must equal 100 exactly"
    );

    // gelu(-10) is a tiny but normal float; a naive 1+erf(-10/sqrt2)
    // cancellation would give exactly 0 (ADR 0061 Decision 6).
    let gelu_neg10 = <GeluOp as UnaryValue>::apply(T::from_f64(-10.0));
    assert!(
        gelu_neg10 != T::ZERO && gelu_neg10 < T::ZERO,
        "gelu(-10) must be a nonzero, negative-signed normal float, got {gelu_neg10:?}"
    );

    // silu(-89) sits in the WGSL rendering's measured underflow band
    // [-91.86, -88.72] (ADR 0061 Decision 6); the value function must not
    // reproduce it.
    let silu_neg89 = <SiluOp as UnaryValue>::apply(T::from_f64(-89.0));
    assert!(
        silu_neg89 != T::ZERO,
        "silu(-89) must not underflow to zero, got {silu_neg89:?}"
    );

    // RoundOp rounds ties to even, not away from zero.
    assert_eq!(
        <RoundOp as UnaryValue>::apply(T::from_f64(2.5)),
        T::from_f64(2.0)
    );
    assert_eq!(
        <RoundOp as UnaryValue>::apply(T::from_f64(3.5)),
        T::from_f64(4.0)
    );

    // SignOp is 0 at +-0 and NaN, not eunomia's signed `signum`.
    assert_eq!(<SignOp as UnaryValue>::apply(T::ZERO), T::ZERO);
    assert_eq!(<SignOp as UnaryValue>::apply(-T::ZERO), T::ZERO);
    assert_eq!(<SignOp as UnaryValue>::apply(T::NAN), T::ZERO);

    // SincOp resolves the removable singularity to exactly 1 at +-0 and
    // propagates NaN; elsewhere it is sin(x)/x to libm rounding.
    assert_eq!(<SincOp as UnaryValue>::apply(T::ZERO), T::ONE);
    assert_eq!(<SincOp as UnaryValue>::apply(-T::ZERO), T::ONE);
    assert!(<SincOp as UnaryValue>::apply(T::NAN).is_nan());
    assert!(
        relative(
            <SincOp as UnaryValue>::apply(T::from_f64(1.0)),
            0.8414709848078965
        ) < 1e-6,
        "sinc(1) must match sin(1)/1 to libm rounding"
    );

    // J0/J1 resolve their x = 0 values exactly, match the classic J(1)
    // constants through the rational branch, and carry even/odd symmetry
    // bit-exactly (negation is exact, and J0's branch sees only |x|).
    assert_eq!(<J0Op as UnaryValue>::apply(T::ZERO), T::ONE);
    assert_eq!(<J1Op as UnaryValue>::apply(T::ZERO), T::ZERO);
    assert!(
        relative(
            <J0Op as UnaryValue>::apply(T::from_f64(1.0)),
            0.7651976865579666
        ) < 1e-6,
        "j0(1) must match the classic constant"
    );
    assert!(
        relative(
            <J1Op as UnaryValue>::apply(T::from_f64(1.0)),
            0.4400505857449335
        ) < 1e-6,
        "j1(1) must match the classic constant"
    );
    assert_eq!(
        <J0Op as UnaryValue>::apply(T::from_f64(-2.0)),
        <J0Op as UnaryValue>::apply(T::from_f64(2.0))
    );
    assert_eq!(
        <J1Op as UnaryValue>::apply(T::from_f64(-2.0)),
        -<J1Op as UnaryValue>::apply(T::from_f64(2.0))
    );

    // K0 matches the DLMF/A&S constants through both branches and yields
    // NaN outside (0, +inf), as the scalar helper does.
    assert!(
        relative(<K0Op as UnaryValue>::apply(T::from_f64(1.0)), 0.4210244382) < 1e-6,
        "k0(1) must match the classic constant"
    );
    assert!(
        relative(<K0Op as UnaryValue>::apply(T::from_f64(5.0)), 0.0036910985) < 1e-6,
        "k0(5) must match the classic constant"
    );
    assert!(<K0Op as UnaryValue>::apply(T::ZERO).is_nan());
    assert!(<K0Op as UnaryValue>::apply(-T::from_f64(1.0)).is_nan());
    assert!(<K0Op as UnaryValue>::apply(T::NAN).is_nan());
    assert!(<K0Op as UnaryValue>::apply(T::from_f64(f64::INFINITY)).is_nan());
}

/// The infinite-input regression set (ADR 0061 Decision 6). Every listed
/// formula forms `0 · ∞` or `∞ / ∞` at `x = ±∞` because its sigmoid/tanh
/// term saturates to exactly `0` or `1` there — `stable_sigmoid(-∞) == 0`
/// and `stable_sigmoid(+∞) == 1` bit-exactly, so e.g. `x * stable_sigmoid(x)`
/// at `x = -∞` is `-∞ · 0`. The value function must resolve each case to
/// its analytic limit instead of propagating `NaN`, for every shipped real
/// scalar (Generic Instantiation Coverage). Every assertion here fails
/// against the pre-fix code at commit ca3c28c, which returns `NaN` in every
/// case (confirmed by running this test before applying the fix).
pub(crate) fn assert_infinite_inputs_resolve_to_the_analytic_limit<T>()
where
    T: RealField + core::fmt::Debug,
{
    let neg_inf = T::neg_infinity();
    let pos_inf = T::infinity();

    // Silu, Mish, Gelu, GeluTanh: x -> x as x -> +-inf; only the negative
    // branch cancels (the naive product forms `-inf * 0`).
    let got = SiluOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "Silu(-inf) must be -0, got {got:?}"
    );
    assert_eq!(SiluOp::apply(pos_inf), pos_inf, "Silu(+inf) must be +inf");

    let got = MishOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "Mish(-inf) must be -0, got {got:?}"
    );
    assert_eq!(MishOp::apply(pos_inf), pos_inf, "Mish(+inf) must be +inf");

    let got = GeluOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "Gelu(-inf) must be -0, got {got:?}"
    );
    assert_eq!(GeluOp::apply(pos_inf), pos_inf, "Gelu(+inf) must be +inf");

    let got = GeluTanhOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "GeluTanh(-inf) must be -0, got {got:?}"
    );
    assert_eq!(
        GeluTanhOp::apply(pos_inf),
        pos_inf,
        "GeluTanh(+inf) must be +inf"
    );

    // Their gradients and GeluGrad: the analytic derivative limit is 1 at
    // +inf, 0 at -inf.
    assert_eq!(
        SiluGradOp::apply(pos_inf),
        T::ONE,
        "SiluGrad(+inf) must be 1"
    );
    assert_eq!(
        SiluGradOp::apply(neg_inf),
        T::ZERO,
        "SiluGrad(-inf) must be 0"
    );
    assert_eq!(
        MishGradOp::apply(pos_inf),
        T::ONE,
        "MishGrad(+inf) must be 1"
    );
    assert_eq!(
        MishGradOp::apply(neg_inf),
        T::ZERO,
        "MishGrad(-inf) must be 0"
    );
    assert_eq!(
        GeluGradOp::apply(pos_inf),
        T::ONE,
        "GeluGrad(+inf) must be 1"
    );
    assert_eq!(
        GeluGradOp::apply(neg_inf),
        T::ZERO,
        "GeluGrad(-inf) must be 0"
    );
    assert_eq!(
        GeluTanhGradOp::apply(pos_inf),
        T::ONE,
        "GeluTanhGrad(+inf) must be 1"
    );
    assert_eq!(
        GeluTanhGradOp::apply(neg_inf),
        T::ZERO,
        "GeluTanhGrad(-inf) must be 0"
    );

    // Softsign: x / (1 + |x|) forms inf/inf at +-inf; the limit is +-1.
    assert_eq!(
        SoftsignOp::apply(pos_inf),
        T::ONE,
        "Softsign(+inf) must be 1"
    );
    assert_eq!(
        SoftsignOp::apply(neg_inf),
        -T::ONE,
        "Softsign(-inf) must be -1"
    );
}

/// The NaN-propagation regression set: `ReluOp`/`ReluGradOp`,
/// `HardsigmoidOp`/`HardsigmoidGradOp`, and `HardswishGradOp` each branched
/// on a comparison (`x.max(0)`'s ignored-NaN contract, `x > 0`, `clamp`'s
/// same ignored-NaN contract, `x > -3 && x < 3`, a three-arm range test) —
/// every comparison against `NaN` is `false` under IEEE 754's unordered
/// semantics, so each fell through to an arm that returns a concrete
/// number (`0`, a clamp bound) instead of propagating `NaN`. An explicit
/// `is_nan` check now comes first in all five. Every assertion here fails
/// against the pre-fix code (confirmed by running this test before
/// applying the fix): all five silently returned `0` for a `NaN` input.
/// `HardswishOp` propagates `NaN` through its arithmetic because its
/// `x <= -3` short-circuit is `false` for `NaN`; the equivalent-looking
/// `!(x > -3)` guard is `true` for `NaN` and would return `-0` instead.
pub(crate) fn assert_nan_propagates_through_every_comparison_based_value_function<T>()
where
    T: RealField + core::fmt::Debug,
{
    let nan = T::NAN;
    assert!(ReluOp::apply(nan).is_nan(), "Relu(NaN) must be NaN");
    assert!(ReluGradOp::apply(nan).is_nan(), "ReluGrad(NaN) must be NaN");
    assert!(
        HardsigmoidOp::apply(nan).is_nan(),
        "Hardsigmoid(NaN) must be NaN"
    );
    assert!(
        HardsigmoidGradOp::apply(nan).is_nan(),
        "HardsigmoidGrad(NaN) must be NaN"
    );
    assert!(
        HardswishOp::apply(nan).is_nan(),
        "Hardswish(NaN) must be NaN"
    );
    assert!(
        HardswishGradOp::apply(nan).is_nan(),
        "HardswishGrad(NaN) must be NaN"
    );
}

/// One accuracy case: an input, its reference value as the unevaluated
/// double-double sum `hi + lo`, and the largest admitted error in ULP of the
/// reference.
///
/// Every reference below is a double-double (about 106-bit) evaluation of the
/// operator's defining formula at the exact input value (an `f32` input is
/// widened exactly): `SiluGrad(x) = σ(x)·(1 + x·σ(−x))`;
/// `MishGrad(x) = tanh(sp) + x·sech²(sp)·σ(x)` with `sp = softplus(x)`;
/// `GeluTanhGrad(x) = σ(w) + k·σ(w)·σ(−w)` with `w = 2·c0·(x + c1·x³)` and
/// `k = 2·c0·x·(1 + 3·c1·x²)`, `c0 = 0.7978845608028654`, `c1 = 0.044715`;
/// `TanhGrad(y) = 1 − y²` with `y²` formed exactly by an error-free product.
/// Exponentials use a range-reduced Taylor series and `log1p` a Newton step
/// in double-double, so no reference shares a code path with eunomia's
/// `exp`/`ln_1p`/`tanh` or with the implementation's region choice.
///
/// Each case sits where the implementation's two forms (direct subtraction
/// from `1`, and the independent form) differ by at least one ULP, so it
/// fails when the switch routes the input to the wrong form. Each bound lies
/// strictly between the measured error of the form the implementation
/// selects and the measured error of the other form; both are recorded on
/// the case as `selected / other`.
pub(crate) struct AccuracyCase<T> {
    pub(crate) input: T,
    pub(crate) reference: (f64, f64),
    pub(crate) bound_ulp: f64,
}

/// Distance of `got` from `reference`, in ULP of the reference's leading
/// component for a format with `significand_bits` bits of precision (24 for
/// `f32`, 53 for `f64`).
pub(crate) fn ulps_from_reference(got: f64, reference: (f64, f64), significand_bits: u32) -> f64 {
    let (hi, lo) = reference;
    ((got - hi) - lo).abs() / ulp_of(hi, significand_bits)
}

/// ULP of the normal value `value` in a format with `significand_bits` bits
/// of precision: `2^(exponent(value) - significand_bits + 1)`.
pub(crate) fn ulp_of(value: f64, significand_bits: u32) -> f64 {
    let biased_exponent = i32::try_from((value.to_bits() >> 52) & 0x7ff)
        .expect("invariant: an 11-bit exponent field fits in i32");
    let precision =
        i32::try_from(significand_bits).expect("invariant: a significand width fits in i32");
    2f64.powi(biased_exponent - 1023 - (precision - 1))
}

pub(crate) fn assert_accuracy_cases<T>(
    operator: &str,
    apply: impl Fn(T) -> T,
    significand_bits: u32,
    cases: &[AccuracyCase<T>],
) where
    T: RealField + core::fmt::Debug,
{
    for case in cases {
        let got = apply(case.input);
        let error = ulps_from_reference(got.to_f64(), case.reference, significand_bits);
        assert!(
            error <= case.bound_ulp,
            "{operator}({:?}) = {got:?} is {error} ULP from the double-double \
             reference {:e} + {:e}, above the bound {}",
            case.input,
            case.reference.0,
            case.reference.1,
            case.bound_ulp
        );
    }
}

/// The root of `GeluTanhGrad` near `x = -0.7524614`.
pub(crate) const GELU_TANH_GRAD_ROOT: f64 = -0.752_461_4;

/// `GeluTanhGrad(x)` and its dominant term `s = σ(w)` in double-double,
/// from `σ(w) + k·σ(w)·σ(−w)` with `w = 2·c0·(x + c1·x³)` and
/// `k = 2·c0·x·(1 + 3·c1·x²)`.
pub(crate) fn gelu_tanh_grad_double_double(x: f64) -> (DoubleDouble, DoubleDouble) {
    let c0 = 0.797_884_560_802_865_4_f64;
    let c1 = 0.044715_f64;
    let one = DoubleDouble::from_f64(1.0);
    let xd = DoubleDouble::from_f64(x);
    let x2 = xd.mul(xd);
    let w =
        DoubleDouble::from_f64(2.0 * c0).mul(xd.add(DoubleDouble::from_f64(c1).mul(x2).mul(xd)));
    let e = w.neg().exp();
    let s = one.div(one.add(e));
    let one_minus_s = e.div(one.add(e));
    let k = DoubleDouble::from_f64(2.0 * c0)
        .mul(xd)
        .mul(one.add(DoubleDouble::from_f64(3.0 * c1).mul(x2)));
    (s.add(k.mul(s).mul(one_minus_s)), s)
}

/// Error of `got` from the double-double reference, in ULP of the dominant
/// term `s` for a format with `significand_bits` bits of precision.
pub(crate) fn gelu_tanh_grad_dominant_term_ulps(x: f64, got: f64, significand_bits: u32) -> f64 {
    let (reference, dominant) = gelu_tanh_grad_double_double(x);
    ((got - reference.hi) - reference.lo).abs() / ulp_of(dominant.hi, significand_bits)
}

/// `GeluTanhGrad` by the original tanh form
/// `½·(1 + tanh z) + ½·x·(1 − tanh² z)·z′(x)`, evaluated in `f64` with
/// `std`'s `tanh` — a different algebraic form and library from the
/// implementation. Over the windows below `tanh z` stays below `0.93`, so
/// `1 − tanh² z` loses at most four bits of `f64`, far below an `f32` ULP.
pub(crate) fn gelu_tanh_grad_tanh_form(x: f64) -> f64 {
    let c0 = 0.797_884_560_802_865_4_f64;
    let c1 = 0.044715_f64;
    let z = c0 * (x + c1 * x * x * x);
    let t = z.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c0 * (1.0 + 3.0 * c1 * x * x)
}

/// Largest error, in ULP of the reference, of `GeluTanhGrad` over every
/// `f32` input in `[low, high]` (both positive).
pub(crate) fn gelu_tanh_grad_f32_window_maximum(low: f32, high: f32) -> f64 {
    let mut worst = 0.0_f64;
    for bits in low.to_bits()..=high.to_bits() {
        let x = f32::from_bits(bits);
        let reference = gelu_tanh_grad_tanh_form(f64::from(x));
        let got = f64::from(GeluTanhGradOp::apply(x));
        worst = worst.max(ulps_from_reference(
            got,
            (reference, 0.0),
            f32::MANTISSA_DIGITS,
        ));
    }
    worst
}
