use super::*;

#[test]
fn discriminants_match_the_shader_branch() {
    for (scheme, discriminant) in scheme_discriminants() {
        assert_eq!(scheme as u32, discriminant, "{scheme:?}");
    }
}
