mod api;
mod dispatch;
mod prepared;
mod shaders;

pub use api::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, prod_axis,
    prod_axis_into, reduce_axis, reduce_axis_into, sum_axis, sum_axis_into,
};
pub use api::{
    prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into, prepare_reduce_axis_into,
    prepare_sum_axis_into,
};
pub use api::{reduction, reduction_with_width};
pub use dispatch::{PreparedAxisReduction, submit_prepared_axis_reduction_batch};
pub use hephaestus_core::{MaxOp, MinOp, ProdOp, SumOp};
pub use prepared::{
    PreparedReduction, prepare_reduction, prepare_reduction_with_width,
    submit_prepared_mixed_reduction_batch, submit_prepared_reduction_batch,
};

#[cfg(test)]
mod tests;
