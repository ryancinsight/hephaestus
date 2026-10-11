//! ROCm elementwise operations over typed device buffers.

use guard::{checked_work_items, reject_output_alias};

/// Binary elementwise operations.
pub mod binary;
/// Output-alias and work-item bound checks shared by every dispatch.
mod guard;
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
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, EluGradOp,
    EluOp, ErfOp, ErfcOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp, GeluGradOp, GeluOp,
    GeluTanhGradOp, GeluTanhOp, HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp,
    IdentityOp, J0Op, J1Op, K0Op, LgammaOp, LnOp, Log1pOp, Log2Op, Log10Op, MishGradOp, MishOp,
    NegOp, RecipOp, ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp, SiluGradOp,
    SiluOp, SinOp, SincOp, SinhOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, SqrtOp,
    TanOp, TanhGradOp, TanhOp, TruncOp, unary_elementwise, unary_elementwise_into,
};
