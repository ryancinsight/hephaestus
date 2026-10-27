//! Vector/matrix reductions on the CUDA device: dot product, trace, and norms.
//!
//! Each operation uses one fused strided map-reduction kernel for its first
//! reduction pass. Only the workgroup partials remain as a temporary, so the
//! operation does not materialize an elementwise result of the logical input
//! size before reducing it. Prepared plans use the same first-pass object and
//! retain its partials across dispatches.

mod ops;
mod plan;
mod shader;
#[cfg(test)]
mod tests;

pub use ops::{dot, norm_l1, norm_l2, norm_max, trace};
pub(crate) use plan::{
    PreparedMapReduction, RetainedMapReduction, prepare_map_reduction_with_layouts,
};
#[cfg(test)]
pub(crate) use shader::shader_source;
pub(crate) use shader::{DotMap, SquareMap};
