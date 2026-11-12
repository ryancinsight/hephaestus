//! Multi-pass HIP workgroup-tree reductions over contiguous buffers.

pub use hephaestus_core::{MaxOp, MinOp, ProdOp, SumOp};

mod full;
mod shaders;

pub use full::{reduction, reduction_with_width};
pub(crate) use shaders::shader_source;
