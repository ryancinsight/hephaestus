pub use hephaestus_core::{
    MAX_STRIDED_RANK, StridedMeta, map_layout_err, pad_shape, pad_shape_dyn, pad_strides,
    pad_usize_strides_dyn, to_u32,
};

mod dynamic;
mod launch;
mod meta;
mod ranked;
mod scalar;
mod shaders;
mod types;
mod validate;

pub use dynamic::{
    binary_elementwise_strided_dyn_into, binary_elementwise_strided_dyn_typed_into,
    unary_elementwise_strided_dyn_into,
};
pub(crate) use meta::{binary_strided_meta, scalar_strided_meta, unary_strided_meta};
pub use ranked::{
    binary_elementwise_strided, binary_elementwise_strided_into, binary_elementwise_strided_typed,
    binary_elementwise_strided_typed_into, unary_elementwise_strided,
    unary_elementwise_strided_into,
};
pub use scalar::{scalar_elementwise_strided, scalar_elementwise_strided_into};
pub(crate) use shaders::{binary_shader, scalar_shader, unary_shader};
pub(crate) use types::{CUDA_DECODE, CUDA_META};
pub use types::{StridedLayout, StridedOperand, StridedOperandDyn};
