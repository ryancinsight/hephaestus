use hephaestus_core::FixedFd3DScheme;

/// Scheme discriminants for cross-checking the uniform branch.
fn scheme_discriminants() -> [(FixedFd3DScheme, u32); 5] {
    use FixedFd3DScheme::*;
    [
        (CentralSecondOrder, 0),
        (CentralFourthOrder, 1),
        (CentralSixthOrder, 2),
        (StaggeredForward, 3),
        (StaggeredBackward, 4),
    ]
}

#[test]
fn discriminants_match_the_shader_branch() {
    for (scheme, discriminant) in scheme_discriminants() {
        assert_eq!(scheme as u32, discriminant, "{scheme:?}");
    }
}
