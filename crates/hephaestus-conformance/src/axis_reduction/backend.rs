use hephaestus_core::{
    AxisReductionOps, CombineExpr, ComputeDevice, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp,
    SumOp,
};

/// A backend able to run every clause in this module.
///
/// The clauses need four facts that the seam alone does not state: that the
/// backend's dialect defines the sum and product reduction expressions, and that
/// `f32` carries an identity and identity token for each. Restating those at
/// every clause would be generic-parameter chaining; bundling them here keeps
/// each clause's signature to the two parameters it actually varies over, and
/// gives a backend one bound to satisfy.
pub trait ReductionBackend<D: ComputeDevice>: AxisReductionOps<D, f32>
where
    ProdOp: CombineExpr<Self::Dialect>,
    SumOp: CombineExpr<Self::Dialect>,
    MinOp: CombineExpr<Self::Dialect>,
    MaxOp: CombineExpr<Self::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, Self::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, Self::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, Self::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Self::Dialect>,
{
}

impl<D, R> ReductionBackend<D> for R
where
    D: ComputeDevice,
    R: AxisReductionOps<D, f32>,
    ProdOp: CombineExpr<R::Dialect>,
    SumOp: CombineExpr<R::Dialect>,
    MinOp: CombineExpr<R::Dialect>,
    MaxOp: CombineExpr<R::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, R::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, R::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, R::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, R::Dialect>,
{
}
