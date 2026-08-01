use super::{MatmulZero, MatrixIdentityScalar, identity_shader_source};
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
fn identity_shader_packs_scalar_identities_in_one_binding() {
    let source = identity_shader_source::<f32>();

    assert!(source.contains("@group(0) @binding(2) var<uniform> identity_values: IdentityValues;"));
    assert_eq!(source.matches("@binding(").count(), 3);
    assert!(!source.contains("@binding(3)"));
    assert!(source.contains("identity_values.zero_value"));
    assert!(source.contains("identity_values.one_value"));
}

#[test]
fn identity_shader_preserves_admitted_vector_tokens() {
    let source = identity_shader_source::<VectorIdentity>();

    assert!(source.contains("zero_value: vec2<i32>"));
    assert!(source.contains("one_value: vec2<i32>"));
    assert!(!source.contains("vec2<vec2<i32>>"));
}
