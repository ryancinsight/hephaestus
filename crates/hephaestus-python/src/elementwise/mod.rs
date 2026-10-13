//! Elementwise operations on `Array`: unary transcendentals, binary
//! arithmetic dunders (array-array and array-scalar), and their
//! module-level function forms.

mod binary;
mod functions;
mod unary;

pub(crate) use functions::{abs, add, cos, div, exp, log, mul, neg, pow, sin, sqrt, sub};
