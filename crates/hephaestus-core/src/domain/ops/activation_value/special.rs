//! Value functions for the special-function markers: `erf`, `erfc`, `lgamma`,
//! `sinc`, `j0`, `j1`, `k0`.
//!
//! The Bessel markers delegate to the one generic core on
//! [`eunomia::RealField`]: `j0`/`j1` route through `libm` (native precision
//! at `f64`), `k0` evaluates the Abramowitz & Stegun 9.8.5/9.8.6 program in
//! lane precision. The GPU dialect macros keep their own coefficients —
//! shader code cannot call Rust — and the conformance suite checks them
//! against this host core.

use super::super::{ErfOp, ErfcOp, J0Op, J1Op, K0Op, LgammaOp, SincOp, UnaryValue};
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

impl UnaryValue for J0Op {
    fn apply<T: RealField>(x: T) -> T {
        x.bessel_j0()
    }
}

impl UnaryValue for J1Op {
    fn apply<T: RealField>(x: T) -> T {
        x.bessel_j1()
    }
}

impl UnaryValue for K0Op {
    fn apply<T: RealField>(x: T) -> T {
        x.bessel_k0()
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
