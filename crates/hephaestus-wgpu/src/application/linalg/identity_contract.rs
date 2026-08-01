use super::identity_shader_source;

#[test]
fn identity_shader_packs_scalar_identities_in_one_binding() {
    let source = identity_shader_source::<f32>();

    assert!(source.contains("@group(0) @binding(2) var<uniform> identity_values: vec2<f32>;"));
    assert_eq!(source.matches("@binding(").count(), 3);
    assert!(!source.contains("@binding(3)"));
    assert!(source.contains("select(identity_values.x, identity_values.y, id.x == id.y)"));
}
