//! Value functions for the special-function markers: `erf`, `erfc`, `lgamma`,
//! `sinc`, `j0`, `j1`.

use super::super::{ErfOp, ErfcOp, J0Op, J1Op, LgammaOp, SincOp, UnaryValue};
use eunomia::RealField;

impl UnaryValue for ErfOp {
    fn apply<T: RealField>(x: T) -> T {
        x.erf()
    }
}

impl UnaryValue for SincOp {
    fn apply<T: RealField>(x: T) -> T {
        if x == T::ZERO { T::ONE } else { x.sin() / x }
    }
}

/// Bessel J0 over any real scalar: the Numerical-Recipes rational
/// approximation for `|x| < 8` and Hankel's asymptotic expansion otherwise,
/// the same nested program as leto's scalar `j0` and elementwise `J0Op`.
fn bessel_j0_value<T: RealField>(x: T) -> T {
    use core::f64::consts::{FRAC_PI_4, PI};
    if x == T::ZERO {
        return T::ONE;
    }
    let ax = x.abs();
    if ax < T::from_f64(8.0) {
        let y = x * x;
        let num = T::from_f64(57568490574.0)
            + y * (T::from_f64(-13362590354.0)
                + y * (T::from_f64(651619640.7)
                    + y * (T::from_f64(-11214424.18)
                        + y * (T::from_f64(77392.33017) + y * T::from_f64(-184.9052456)))));
        let den = T::from_f64(57568490411.0)
            + y * (T::from_f64(1029532985.0)
                + y * (T::from_f64(9494680.718)
                    + y * (T::from_f64(59272.64853) + y * (T::from_f64(267.8532712) + y))));
        num / den
    } else {
        let z = T::from_f64(8.0) / ax;
        let y = z * z;
        let xx = ax - T::from_f64(FRAC_PI_4);
        let p = T::ONE
            + y * (T::from_f64(-0.001098628627)
                + y * (T::from_f64(0.000002734510407)
                    + y * (T::from_f64(-2.073370639e-6) + y * T::from_f64(2.093887211e-7))));
        let q = T::from_f64(-0.01562499995)
            + y * (T::from_f64(0.0001430488765)
                + y * (T::from_f64(-6.911147651e-5)
                    + y * (T::from_f64(7.621095161e-5) - y * T::from_f64(9.34935152e-7))));
        (T::from_f64(2.0) / (T::from_f64(PI) * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    }
}

/// Bessel J1 over any real scalar: the same Numerical-Recipes/Hankel program
/// as leto's scalar `j1` and elementwise `J1Op`. `j1(-0.0)` is `-0.0` by
/// oddness (the scalar helper returns `+0.0`; the two compare equal).
fn bessel_j1_value<T: RealField>(x: T) -> T {
    use core::f64::consts::PI;
    if x == T::ZERO {
        return T::ZERO;
    }
    let ax = x.abs();
    if ax < T::from_f64(8.0) {
        let y = x * x;
        let num = x
            * (T::from_f64(72362614232.0)
                + y * (T::from_f64(-7895059235.0)
                    + y * (T::from_f64(242396853.1)
                        + y * (T::from_f64(-2972611.439)
                            + y * (T::from_f64(15704.48260) + y * T::from_f64(-30.16036606))))));
        let den = T::from_f64(144725228442.0)
            + y * (T::from_f64(2300535178.0)
                + y * (T::from_f64(18583304.74)
                    + y * (T::from_f64(99447.43394) + y * (T::from_f64(376.9991397) + y))));
        num / den
    } else {
        let z = T::from_f64(8.0) / ax;
        let y = z * z;
        let xx = ax - T::from_f64(3.0 * PI / 4.0);
        let p = T::ONE
            + y * (T::from_f64(0.183105e-2)
                + y * (T::from_f64(-3.516396496e-5)
                    + y * (T::from_f64(2.457520174e-5) - y * T::from_f64(2.400505341e-7))));
        let q = T::from_f64(0.04687499995)
            + y * (T::from_f64(-0.2002690873e-3)
                + y * (T::from_f64(8.449199096e-5)
                    + y * (T::from_f64(-8.8228987e-5) + y * T::from_f64(1.050343160e-6))));
        let r =
            (T::from_f64(2.0) / (T::from_f64(PI) * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin());
        if x < T::ZERO { -r } else { r }
    }
}

impl UnaryValue for J0Op {
    fn apply<T: RealField>(x: T) -> T {
        bessel_j0_value(x)
    }
}

impl UnaryValue for J1Op {
    fn apply<T: RealField>(x: T) -> T {
        bessel_j1_value(x)
    }
}

impl UnaryValue for ErfcOp {
    fn apply<T: RealField>(x: T) -> T {
        x.erfc()
    }
}

impl UnaryValue for LgammaOp {
    fn apply<T: RealField>(x: T) -> T {
        x.lgamma()
    }
}
