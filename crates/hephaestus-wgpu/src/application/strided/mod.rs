//! Strided-layout-aware dispatch over leto host-side layout metadata.
//!
//! Consumers describe operands with [`leto::Layout`] (shape/strides/offset) so
//! transposed, sliced, and broadcast (zero-stride) views dispatch directly —
//! no host-side materialization into contiguous staging copies. Inputs
//! broadcast to the output shape with leto's own broadcast rules, keeping
//! device semantics identical to leto's CPU `binary_map`.
//!
//! One shared metadata/pipeline/encode core serves the binary, unary, and
//! scalar op families. The scalar family has a dedicated kernel that reads the
//! broadcast scalar from a small pooled `uniform` (like the contiguous
//! `scalar_elementwise_into` path) rather than uploading a one-element storage
//! operand per call, so a strided scalar op allocates no per-call device buffer.

pub use hephaestus_core::{
    MAX_STRIDED_RANK, StridedMeta, map_layout_err, pad_shape, pad_shape_dyn, pad_strides, to_u32,
};

mod binary;
mod encode;
mod scalar;
mod shaders;
mod types;
mod unary;
mod validate;

pub use binary::{
    binary_elementwise_strided, binary_elementwise_strided_into, binary_elementwise_strided_typed,
    binary_elementwise_strided_typed_into,
};
pub use scalar::{scalar_elementwise_strided, scalar_elementwise_strided_into};
pub(crate) use shaders::{WGSL_DECODE, WGSL_META};
pub use types::StridedOperand;
pub use unary::{unary_elementwise_strided, unary_elementwise_strided_into};
pub(crate) use validate::{to_i32, validate_out};
