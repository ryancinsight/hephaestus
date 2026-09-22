//! Per-scalar resolution of the operator value functions (ADR 0061
//! Decision 5).

use eunomia::NumericElement;
use hephaestus_core::{BinaryExpr, DialectScalar, Host, TypedBinaryExpr, UnaryExpr};

/// Host-local per-scalar dispatch (ADR 0061 Decision 5): bridges the two
/// value-method bounds the source traits carry behind one interface the
/// host's generic `T: Pod` seam can call uniformly.
///
/// `f32`/`f64` resolve `unary_fn`/`binary_fn` through the `RealField`-bound
/// methods (`UnaryExpr::value`, `BinaryExpr::real_value`); `u32`/`i32`/`F16`/
/// `Bf16` resolve `binary_fn` through the `NumericElement`-bound
/// `BinaryExpr::value` (valid for every admitting operator, `None` for the
/// real-only [`hephaestus_core::PowOp`]) and `unary_fn` to a function with no
/// application at all, since every unary operator is real-only (ADR 0061
/// Decision 2). Both branches funnel into the same `None` ->
/// [`unsupported_operator`] translation at the call site, so "operator has no
/// host value function" and "scalar admits no real application" are one
/// diagnosable error.
pub(super) trait ElementwiseDispatch: NumericElement + DialectScalar<Host> + Sized {
    /// Resolve `Op`'s host unary value function for this scalar.
    fn unary_fn<Op: UnaryExpr<Host>>() -> fn(Self) -> Option<Self>;
    /// Resolve `Op`'s host binary value function for this scalar.
    fn binary_fn<Op: BinaryExpr<Host>>() -> fn(Self, Self) -> Option<Self>;
    /// Resolve `Op`'s host typed-comparison value function for this scalar.
    fn typed_binary_fn<Op: TypedBinaryExpr<Host, Self>>() -> fn(Self, Self) -> Option<Self>;
}

macro_rules! impl_real_elementwise_dispatch {
    ($($t:ty),+ $(,)?) => {
        $(
            impl ElementwiseDispatch for $t {
                fn unary_fn<Op: UnaryExpr<Host>>() -> fn(Self) -> Option<Self> {
                    <Op as UnaryExpr<Host>>::value::<Self>
                }
                fn binary_fn<Op: BinaryExpr<Host>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as BinaryExpr<Host>>::real_value::<Self>
                }
                fn typed_binary_fn<Op: TypedBinaryExpr<Host, Self>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as TypedBinaryExpr<Host, Self>>::value
                }
            }
        )+
    };
}

impl_real_elementwise_dispatch!(f32, f64);

macro_rules! impl_numeric_only_elementwise_dispatch {
    ($($t:ty),+ $(,)?) => {
        $(
            impl ElementwiseDispatch for $t {
                fn unary_fn<Op: UnaryExpr<Host>>() -> fn(Self) -> Option<Self> {
                    |_| None
                }
                fn binary_fn<Op: BinaryExpr<Host>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as BinaryExpr<Host>>::value::<Self>
                }
                fn typed_binary_fn<Op: TypedBinaryExpr<Host, Self>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as TypedBinaryExpr<Host, Self>>::value
                }
            }
        )+
    };
}

impl_numeric_only_elementwise_dispatch!(u32, i32, eunomia::F16, eunomia::Bf16);
