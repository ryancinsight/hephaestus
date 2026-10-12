mod axis;
mod entry;
mod scalar;
mod shaders;
#[cfg(test)]
mod tests;

pub(crate) use axis::{
    axis_len, launch_axis_dispatch, plan_axis_reduction_dispatch, reject_empty_axis,
};
pub use axis::{mean_axis, mean_axis_into, reduce_axis, reduce_axis_into};
pub use entry::{
    max_axis, max_axis_into, min_axis, min_axis_into, prod_axis, prod_axis_into, sum_axis,
    sum_axis_into,
};
pub use hephaestus_core::{MaxOp, MinOp, ProdOp, SumOp};
pub use scalar::{reduction, reduction_with_width};
pub(crate) use shaders::{axis_reduction_shader_source, mean_axis_shader_source, shader_source};
