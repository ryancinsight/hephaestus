mod axis;
mod entry;
mod prepared;
mod prepared_axis;
mod shaders;
#[cfg(test)]
mod tests;

pub use axis::{
    prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into, prepare_reduce_axis_into,
    prepare_sum_axis_into, reduce_axis, reduce_axis_into,
};
pub use entry::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, prod_axis,
    prod_axis_into, reduction, reduction_with_width, sum_axis, sum_axis_into,
};
pub use hephaestus_core::{MaxOp, MinOp, ProdOp, SumOp};
pub use prepared::{
    PreparedReduction, prepare_reduction, prepare_reduction_with_width,
    submit_prepared_mixed_reduction_batch, submit_prepared_reduction_batch,
};
pub use prepared_axis::{PreparedAxisReduction, submit_prepared_axis_reduction_batch};
