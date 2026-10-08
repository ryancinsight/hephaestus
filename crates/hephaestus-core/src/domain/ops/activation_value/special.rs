//! Value functions for the special-function markers: `erf`, `erfc`, `lgamma`,
//! `sinc`.

use super::super::{ErfOp, ErfcOp, LgammaOp, SincOp, UnaryValue};
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
