//! Reduction operations.

mod axis;
mod entry;
mod scalar;

use hephaestus_wgpu as wgpu_backend;

pub use axis::{
    PreparedAxisReduction, prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into,
    prepare_reduce_axis_into, prepare_sum_axis_into, reduce_axis, reduce_axis_into,
    submit_prepared_axis_reduction_batch,
};
pub use entry::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, prod_axis,
    prod_axis_into, sum_axis, sum_axis_into,
};
pub use scalar::{
    PreparedReduction, prepare_reduction, prepare_reduction_with_width, reduction,
    reduction_with_width, submit_prepared_reduction_batch,
};
pub use wgpu_backend::{MaxOp, MinOp, ProdOp, SumOp};
