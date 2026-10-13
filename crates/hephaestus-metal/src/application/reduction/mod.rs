//! Reduction operations.

use hephaestus_wgpu as wgpu_backend;

pub use wgpu_backend::{MaxOp, MinOp, ProdOp, SumOp};

mod axis;
mod full;
mod prepare;
mod prepared;

pub use axis::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, prod_axis,
    prod_axis_into, reduce_axis, reduce_axis_into, sum_axis, sum_axis_into,
};
pub use full::{reduction, reduction_with_width};
pub use prepare::{
    prepare_max_axis_into, prepare_mean_axis_into, prepare_min_axis_into, prepare_reduce_axis_into,
    prepare_reduction, prepare_reduction_with_width, prepare_sum_axis_into,
};
pub use prepared::{
    PreparedAxisReduction, PreparedReduction, submit_prepared_axis_reduction_batch,
    submit_prepared_reduction_batch,
};
