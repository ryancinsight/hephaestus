//! HIP reductions over strided rank-2 views.
//!
//! When a dense full-buffer reduction collapses to a scalar, it uses
//! the [contiguous reduction](super::reduction) workgroup tree.

pub use hephaestus_core::{MaxOp, MinOp, ProdOp, SumOp};

mod axis;
mod mean;
mod named;
mod plan;
mod shaders;

pub use axis::{reduce_axis, reduce_axis_into};
pub use mean::{mean_axis, mean_axis_into};
pub use named::{
    max_axis, max_axis_into, min_axis, min_axis_into, prod_axis, prod_axis_into, sum_axis,
    sum_axis_into,
};
pub(crate) use plan::{
    axis_len, launch_axis_dispatch, plan_axis_reduction_dispatch, reject_empty_axis,
};
pub(crate) use shaders::{axis_reduction_shader_source, mean_axis_shader_source};
