//! Source-level contract for the CUDA strided map-reduction shader.

use super::{DotMap, shader_source};
use hephaestus_core::BlockWidth;

#[test]
fn source_declares_strided_map_reduction_contract() {
    let source = shader_source::<DotMap, f32>(BlockWidth::DEFAULT);
    assert!(source.contains("shape[8]"));
    assert!(source.contains("a_strides[8]"));
    assert!(source.contains("b_strides[8]"));
    assert!(source.contains("shared_data"));
    assert!(source.contains("lhs * rhs"));
    assert!(source.contains("__syncthreads();"));
}
