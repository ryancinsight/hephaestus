use hephaestus_core::{
    ComputeDevice, DialectScalar, ElementwiseOps, EqOp, GeOp, GtOp, LeOp, LtOp, NeOp,
    TypedBinaryExpr,
};

/// A backend able to run every clause in this module.
///
/// Bundles the scalar instantiations and the comparison-operator expressions
/// each dialect must define, so a backend satisfies one bound instead of the
/// full operator × scalar matrix at every clause.
pub trait TypedElementwiseBackend<D: ComputeDevice>:
    ElementwiseOps<D, u32> + ElementwiseOps<D, i32> + ElementwiseOps<D, f32>
where
    u32: DialectScalar<<Self as ElementwiseOps<D, u32>>::Dialect>,
    i32: DialectScalar<<Self as ElementwiseOps<D, i32>>::Dialect>,
    f32: DialectScalar<<Self as ElementwiseOps<D, f32>>::Dialect>,
    EqOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
    NeOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
    LtOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
    LeOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>,
    GtOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, i32>>::Dialect, i32>,
    GeOp: TypedBinaryExpr<<Self as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Self as ElementwiseOps<D, f32>>::Dialect, f32>,
{
}

impl<D, E> TypedElementwiseBackend<D> for E
where
    D: ComputeDevice,
    E: ElementwiseOps<D, u32> + ElementwiseOps<D, i32> + ElementwiseOps<D, f32>,
    u32: DialectScalar<<E as ElementwiseOps<D, u32>>::Dialect>,
    i32: DialectScalar<<E as ElementwiseOps<D, i32>>::Dialect>,
    f32: DialectScalar<<E as ElementwiseOps<D, f32>>::Dialect>,
    EqOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    NeOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    LtOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
    LeOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>,
    GtOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, i32>>::Dialect, i32>,
    GeOp: TypedBinaryExpr<<E as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect, f32>,
{
}
