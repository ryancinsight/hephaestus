use hephaestus_core::{
    CombineExpr, ComputeDevice, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, SumOp,
};

use super::backend::ReductionBackend;
use super::mean::mean_axis_divides_by_the_reduced_length;
use super::minmax::min_and_max_axis_reduce_exactly;
use super::prepared::{
    prepared_reduction_is_parameterized_by_operator,
    prepared_reduction_is_reusable_and_observes_writes,
    prepared_reduction_rejects_mismatched_output_shape,
};
use super::prod::{
    prod_axis_follows_strided_views, prod_axis_reduces_both_axes,
    prod_axis_rejects_out_of_range_axis,
};
use super::sum::axis_sum_matches_the_leto_reference_beyond_the_block_width;

/// Run every axis-reduction clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a device.
pub fn assert_axis_reduction_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: ReductionBackend<D>,
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
    prod_axis_reduces_both_axes(device, ops);
    min_and_max_axis_reduce_exactly(device, ops);
    mean_axis_divides_by_the_reduced_length(device, ops);
    prod_axis_follows_strided_views(device, ops);
    prod_axis_rejects_out_of_range_axis(device, ops);
    prepared_reduction_is_reusable_and_observes_writes(device, ops);
    prepared_reduction_is_parameterized_by_operator(device, ops);
    prepared_reduction_rejects_mismatched_output_shape(device, ops);
    axis_sum_matches_the_leto_reference_beyond_the_block_width(device, ops);
}
