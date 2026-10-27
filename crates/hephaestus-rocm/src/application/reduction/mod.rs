//! ROCm contiguous and rank-2 axis reductions.
//!
//! The contiguous path runs a multi-pass HIP workgroup tree over a dense
//! buffer; the axis path runs one thread per output over a strided view.
//! Both honour the same block-width, shared-memory, and launch contracts, and
//! the axis path reuses the contiguous tree whenever a reduction collapses to
//! a scalar over a dense full-buffer input.

pub use hephaestus_core::{MaxOp, MinOp, ProdOp, SumOp};

mod axis;
mod full;
mod mean;
mod named;
mod plan;
mod shaders;

pub use axis::{reduce_axis, reduce_axis_into};
pub use full::{reduction, reduction_with_width};
pub use mean::{mean_axis, mean_axis_into};
pub use named::{
    max_axis, max_axis_into, min_axis, min_axis_into, prod_axis, prod_axis_into, sum_axis,
    sum_axis_into,
};
pub(crate) use plan::{
    axis_len, launch_axis_dispatch, plan_axis_reduction_dispatch, reject_empty_axis,
};
pub(crate) use shaders::{axis_reduction_shader_source, mean_axis_shader_source, shader_source};
