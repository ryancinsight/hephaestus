use hephaestus_core::AttentionSemanticStatus;

use super::*;

#[test]
fn f64_attention_shaders_use_native_double_tokens() {
    let weights = forward_shader::<f64>(ForwardStage::Weights, 64);
    assert!(weights.contains("array<f64>"));
    assert!(weights.contains("scale_and_padding: vec2<f64>,"));
    assert!(weights.contains("-> f64 {"));
    assert!(weights.contains("-1.7976931348623157e+308"));
    assert!(!weights.contains("f32"));

    let query = backward_shader::<f64>(BackwardStage::Query, 64);
    assert!(query.contains("array<f64>"));
    assert!(query.contains("parameters.scale_and_padding.x * accumulated"));
    assert!(!query.contains("f32"));

    let finite =
        finite_preflight_shader::<f64>("query", 3, AttentionSemanticStatus::NonFiniteQuery, 64);
    assert!(finite.contains("fn finite(value: f64) -> bool"));
    assert!(finite.contains("1.7976931348623157e+308"));

    let probability = backward_probability_preflight_shader::<f64>(64);
    assert!(probability.contains("2.220446049250313e-16"));
    assert!(probability.contains("f64(parameters.dimensions.z)"));
}

#[test]
fn f32_attention_shaders_keep_their_rendering() {
    let weights = forward_shader::<f32>(ForwardStage::Weights, 64);
    assert!(weights.contains("array<f32>"));
    assert!(weights.contains("scale_and_padding: vec4<f32>,"));
    assert!(weights.contains("-> f32 {"));
    assert!(weights.contains("-3.402823466e+38"));
    assert!(!weights.contains("f64"));

    let probability = backward_probability_preflight_shader::<f32>(64);
    assert!(probability.contains("1.192092896e-7"));
    assert!(probability.contains("f32(parameters.dimensions.z)"));
}
