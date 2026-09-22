//! Runtime-parameter unary expressions and their device-neutral dispatch seam.
//!
//! Parameter values remain dispatch data. They never enter generated source or
//! pipeline-cache keys, so changing activation bounds reuses the compiled
//! kernel and changes only the two scalar arguments.

mod expr;
mod layout;
mod marker;
mod ops;
mod shader;
mod value;

#[cfg(test)]
mod tests;

pub use expr::{ParameterizedUnaryExpr, ParameterizedUnaryValue};
pub use layout::validate_parameterized_output;
pub use marker::{
    CeluGradOp, CeluOp, HardshrinkGradOp, HardshrinkOp, HardtanhGradOp, HardtanhOp,
    LeakyReluGradOp, LeakyReluOp, SoftshrinkGradOp, SoftshrinkOp, ThresholdGradOp, ThresholdOp,
};
pub use ops::ParameterizedUnaryOps;
