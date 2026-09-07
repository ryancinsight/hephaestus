use super::{binary_shader, hip_decode, hip_meta, scalar_shader, unary_shader};

#[test]
fn sources_share_the_metadata_abi_and_operation_expressions() {
    assert_eq!(
        hip_meta(),
        "\nstruct Meta {\n    unsigned int shape[8];\n    int a_strides[8];\n    int b_strides[8];\n    int out_strides[8];\n    unsigned int offsets[4];\n};\n"
    );
    let binary = binary_shader::<i32>("lhs + rhs");
    let unary = unary_shader::<hephaestus_core::IdentityOp, i32>();
    let scalar = scalar_shader::<hephaestus_core::MulOp, i32>();
    for source in [&binary, &unary, &scalar] {
        assert!(source.contains(&hip_meta()));
        assert!(source.contains(&hip_decode()));
        assert!(source.contains("for (int dimension = 7; dimension >= 0; dimension--)"));
        assert!(source.contains("long long index = (long long)(rem % extent)"));
        for (operand, lane) in [("a", 0), ("b", 1), ("out", 2)] {
            assert!(source.contains(&format!(
                "long long {operand}_offset = (long long)lmeta.offsets[{lane}]"
            )));
            assert!(source.contains(&format!(
                "{operand}_offset += index * (long long)lmeta.{operand}_strides[dimension]"
            )));
        }
    }
    assert!(binary.contains("out[out_offset] = lhs + rhs"));
    assert!(unary.contains("out[out_offset] = x"));
    assert!(scalar.contains("out[out_offset] = lhs * rhs"));
}
