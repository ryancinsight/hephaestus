use std::marker::PhantomData;

use hephaestus_core::{BinaryExpr, DialectScalar, TypedBinaryExpr, Wgsl};

/// Adapter so a `TypedBinaryExpr` can be used as a plain `BinaryExpr`
/// (the expression string is already dialect-specific, and the kernel
/// binds no additional scalar parameters).
#[derive(Clone, Copy)]
pub(crate) struct SeamTypedExpr<Op, T: Send + Sync + 'static>(PhantomData<(Op, T)>);

impl<
    Op: TypedBinaryExpr<Wgsl, T> + Send + Sync + 'static,
    T: DialectScalar<Wgsl> + Send + Sync + 'static,
> BinaryExpr<Wgsl> for SeamTypedExpr<Op, T>
{
    const EXPR: &'static str = <Op as TypedBinaryExpr<Wgsl, T>>::EXPR;
}
