//! Test module for this operation family, split out of the parent
//! source file to keep it under the 500-line conformance budget.

    use super::{DotMap, shader_source};
    use hephaestus_core::BlockWidth;

    #[test]
    fn source_declares_strided_map_reduction_contract() {
        let source = shader_source::<DotMap, i32>(BlockWidth::DEFAULT);
        assert!(source.contains("shape[4]"));
        assert!(source.contains("a_strides[4]"));
        assert!(source.contains("b_strides[4]"));
        assert!(source.contains("shared_data"));
        assert!(source.contains("lhs * rhs"));
        assert!(source.contains("__syncthreads();"));
    }
