//! Value functions for the special-function markers: `erf`, `erfc`, `lgamma`.

use super::super::{ErfOp, ErfcOp, LgammaOp, UnaryValue};
use eunomia::RealField;

impl UnaryValue for ErfOp {
    fn apply<T: RealField>(x: T) -> T {
        x.erf()
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
