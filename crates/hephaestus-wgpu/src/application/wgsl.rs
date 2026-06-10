use bytemuck::Pod;

/// Maps a host scalar type to its WGSL type token at compile time.
///
/// This keeps kernels generic over `T` (one `binary_elementwise::<Op, T>`
/// entry point monomorphized per scalar) while the shader text substitutes
/// the matching WGSL type — no type names appear in function identifiers.
pub trait WgslScalar: Pod {
    /// The WGSL scalar type token (`"f32"`, `"u32"`, `"i32"`).
    const WGSL_TYPE: &'static str;
}

impl WgslScalar for f32 {
    const WGSL_TYPE: &'static str = "f32";
}

impl WgslScalar for u32 {
    const WGSL_TYPE: &'static str = "u32";
}

impl WgslScalar for i32 {
    const WGSL_TYPE: &'static str = "i32";
}
