use super::{MatmulZero, MatrixIdentityScalar, identity_buffer_layout, identity_shader_source};
use hephaestus_core::{DialectScalar, Wgsl};

#[repr(transparent)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorIdentity([i32; 2]);

impl DialectScalar<Wgsl> for VectorIdentity {
    const TYPE_TOKEN: &'static str = "vec2<i32>";
}

impl MatmulZero for VectorIdentity {
    const WGSL_ZERO: &'static str = "vec2<i32>(0)";
}

impl MatrixIdentityScalar for VectorIdentity {
    const ZERO: Self = Self([-7, -5]);
    const ONE: Self = Self([9, 11]);
}

#[test]
fn identity_shader_preserves_separate_typed_bindings() {
    let source = identity_shader_source::<f32>();

    assert!(source.contains("@group(0) @binding(2) var<uniform> zero_value: f32;"));
    assert!(source.contains("@group(0) @binding(3) var<uniform> one_value: f32;"));
    assert_eq!(source.matches("@binding(").count(), 4);
    assert!(source.contains("select(zero_value, one_value, id.x == id.y)"));
}

#[test]
fn identity_buffer_aligns_three_lane_values_without_host_padding() {
    let (one_offset, buffer_size) = identity_buffer_layout::<[i32; 3]>(256).unwrap();

    assert_eq!(one_offset, 256);
    assert_eq!(buffer_size, 268);
}

#[test]
fn identity_shader_preserves_admitted_vector_tokens() {
    let source = identity_shader_source::<VectorIdentity>();

    assert!(source.contains("zero_value: vec2<i32>"));
    assert!(source.contains("one_value: vec2<i32>"));
    assert!(!source.contains("vec2<vec2<i32>>"));
}
