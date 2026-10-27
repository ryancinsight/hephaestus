//! Strided dot products, traces, and norms on ROCm.
//!
//! One HIP workgroup maps each logical input element and reduces its mapped
//! value in shared memory. The host packs leto::Layout metadata into a
//! rank-four representation, so transposed, sliced, and diagonal views use
//! the same device path as contiguous inputs.

mod ops;
mod plan;
mod shader;
#[cfg(test)]
#[path = "../tests_norms.rs"]
mod tests;

pub use ops::{L2NormScalar, dot, norm_l1, norm_l2, norm_max, trace};
pub(crate) use plan::{
    PreparedMapReduction, RetainedMapReduction, prepare_map_reduction_with_layouts,
};
#[cfg(test)]
use shader::shader_source;
pub(crate) use shader::{DotMap, SquareMap};
