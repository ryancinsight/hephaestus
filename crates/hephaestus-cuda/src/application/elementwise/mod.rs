//! Elementwise CUDA compute operations.

use crate::infrastructure::buffer::CudaBuffer;
use hephaestus_core::{HephaestusError, Result};

/// Binary elementwise operations.
pub mod binary;
/// Scalar elementwise operations.
pub mod scalar;
/// Unary elementwise operations.
pub mod unary;

pub use binary::{
    AddOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, PowOp, SubOp, binary_elementwise,
    binary_elementwise_into, binary_elementwise_typed, binary_elementwise_typed_into,
};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, Exp2Op,
    ExpNegOp, ExpOp, Expm1Op, FloorOp, GeluTanhGradOp, GeluTanhOp, IdentityOp, LnOp, Log1pOp,
    Log2Op, Log10Op, NegOp, RecipOp, ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp,
    SiluGradOp, SiluOp, SinOp, SinhOp, SoftplusGradOp, SoftplusOp, SqrtOp, TanOp, TanhGradOp,
    TanhOp, TruncOp, unary_elementwise, unary_elementwise_into,
};

fn reject_output_alias<T, U>(
    input_label: &'static str,
    input: &CudaBuffer<T>,
    out: &CudaBuffer<U>,
) -> Result<()> {
    if input.aliases(out) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("output buffer must not alias {input_label} input"),
        });
    }
    Ok(())
}
