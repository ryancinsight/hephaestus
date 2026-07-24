//! Kernel-dialect vocabulary: sealed dialect markers and per-dialect scalar
//! type tokens.
//!
//! A [`KernelDialect`](crate::KernelDialect) is the shading/kernel language a backend compiles —
//! WGSL for wgpu (native and Metal-pinned), CUDA C++ for the NVRTC-compiled
//! CUDA backend, and HIP C++ for the hipRTC-compiled ROCm backend. Operation
//! markers ([`ops`](crate::domain::ops)) and scalar types carry their shader
//! tokens per dialect through these traits, so one op vocabulary serves every
//! backend and a kernel authored for one dialect simply does not implement the
//! others — dispatching it on the wrong backend is a compile error, not a
//! runtime failure.
//!
//! The trait is sealed: a new dialect is a deliberate substrate extension
//! (new backend crate), not a consumer-side extension point.

use bytemuck::Pod;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Wgsl {}
    impl Sealed for super::CudaC {}
    impl Sealed for super::HipC {}
}

/// A kernel-source dialect a backend can compile.
pub trait KernelDialect:
    sealed::Sealed + Copy + Clone + core::fmt::Debug + Default + Send + Sync + 'static
{
    /// Human-readable dialect name for diagnostics.
    const NAME: &'static str;
}

/// WGSL dialect marker (wgpu backends).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Wgsl;

/// CUDA C++ dialect marker (NVRTC runtime compilation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaC;

/// HIP C++ dialect marker (hipRTC runtime compilation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HipC;

impl KernelDialect for Wgsl {
    const NAME: &'static str = "wgsl";
}

impl KernelDialect for CudaC {
    const NAME: &'static str = "cuda-c";
}

impl KernelDialect for HipC {
    const NAME: &'static str = "hip-c";
}

/// Maps a host scalar type to its shader type token in dialect `L` at compile
/// time.
///
/// Unifies the previous per-backend `WgslScalar::WGSL_TYPE` /
/// `CudaScalar::CUDA_TYPE` pair: one bound (`T: DialectScalar<L>`) expresses
/// "this scalar is representable in this dialect", and generated sources
/// substitute [`TYPE_TOKEN`](Self::TYPE_TOKEN) — type names never appear in
/// API identifiers.
pub trait DialectScalar<L: KernelDialect>: Pod {
    /// The dialect's scalar type token (e.g. `"f32"` in WGSL, `"float"` in
    /// CUDA/HIP C++).
    const TYPE_TOKEN: &'static str;
}

impl DialectScalar<Wgsl> for f32 {
    const TYPE_TOKEN: &'static str = "f32";
}

impl DialectScalar<Wgsl> for u32 {
    const TYPE_TOKEN: &'static str = "u32";
}

impl DialectScalar<Wgsl> for i32 {
    const TYPE_TOKEN: &'static str = "i32";
}

impl DialectScalar<CudaC> for f32 {
    const TYPE_TOKEN: &'static str = "float";
}

impl DialectScalar<CudaC> for u32 {
    const TYPE_TOKEN: &'static str = "unsigned int";
}

impl DialectScalar<CudaC> for i32 {
    const TYPE_TOKEN: &'static str = "int";
}

impl DialectScalar<HipC> for f32 {
    const TYPE_TOKEN: &'static str = "float";
}

impl DialectScalar<HipC> for u32 {
    const TYPE_TOKEN: &'static str = "unsigned int";
}

impl DialectScalar<HipC> for i32 {
    const TYPE_TOKEN: &'static str = "int";
}

// ── f64 ──────────────────────────────────────────────────────────────────

impl DialectScalar<Wgsl> for f64 {
    const TYPE_TOKEN: &'static str = "f64";
}

impl DialectScalar<CudaC> for f64 {
    const TYPE_TOKEN: &'static str = "double";
}

impl DialectScalar<HipC> for f64 {
    const TYPE_TOKEN: &'static str = "double";
}

// ── GPU vector types (fixed-size arrays of scalar elements) ──────────────

macro_rules! impl_dialect_vector {
    ($ty:ty, $wgsl_token:expr, $cuda_token:expr, $hip_token:expr) => {
        impl DialectScalar<Wgsl> for $ty {
            const TYPE_TOKEN: &'static str = $wgsl_token;
        }
        impl DialectScalar<CudaC> for $ty {
            const TYPE_TOKEN: &'static str = $cuda_token;
        }
        impl DialectScalar<HipC> for $ty {
            const TYPE_TOKEN: &'static str = $hip_token;
        }
    };
}

// f32 vectors
impl_dialect_vector!([f32; 2], "vec2<f32>", "float2", "float2");
impl_dialect_vector!([f32; 3], "vec3<f32>", "float3", "float3");
impl_dialect_vector!([f32; 4], "vec4<f32>", "float4", "float4");
// f64 vectors
impl_dialect_vector!([f64; 2], "vec2<f64>", "double2", "double2");
impl_dialect_vector!([f64; 3], "vec3<f64>", "double3", "double3");
impl_dialect_vector!([f64; 4], "vec4<f64>", "double4", "double4");
// i32 vectors
impl_dialect_vector!([i32; 2], "vec2<i32>", "int2", "int2");
impl_dialect_vector!([i32; 3], "vec3<i32>", "int3", "int3");
impl_dialect_vector!([i32; 4], "vec4<i32>", "int4", "int4");
// u32 vectors
impl_dialect_vector!([u32; 2], "vec2<u32>", "uint2", "uint2");
impl_dialect_vector!([u32; 3], "vec3<u32>", "uint3", "uint3");
impl_dialect_vector!([u32; 4], "vec4<u32>", "uint4", "uint4");

#[cfg(test)]
mod tests {
    use super::*;

    fn token_of<T: DialectScalar<L>, L: KernelDialect>() -> &'static str {
        T::TYPE_TOKEN
    }

    #[test]
    fn scalar_tokens_are_dialect_specific() {
        assert_eq!(token_of::<f32, Wgsl>(), "f32");
        assert_eq!(token_of::<f32, CudaC>(), "float");
        assert_eq!(token_of::<f32, HipC>(), "float");
        assert_eq!(token_of::<u32, Wgsl>(), "u32");
        assert_eq!(token_of::<u32, CudaC>(), "unsigned int");
        assert_eq!(token_of::<u32, HipC>(), "unsigned int");
        assert_eq!(token_of::<i32, Wgsl>(), "i32");
        assert_eq!(token_of::<i32, CudaC>(), "int");
        assert_eq!(token_of::<i32, HipC>(), "int");
        // f64
        assert_eq!(token_of::<f64, Wgsl>(), "f64");
        assert_eq!(token_of::<f64, CudaC>(), "double");
        assert_eq!(token_of::<f64, HipC>(), "double");
        // GPU vector types — f32
        assert_eq!(token_of::<[f32; 2], Wgsl>(), "vec2<f32>");
        assert_eq!(token_of::<[f32; 2], CudaC>(), "float2");
        assert_eq!(token_of::<[f32; 2], HipC>(), "float2");
        assert_eq!(token_of::<[f32; 3], Wgsl>(), "vec3<f32>");
        assert_eq!(token_of::<[f32; 3], CudaC>(), "float3");
        assert_eq!(token_of::<[f32; 4], Wgsl>(), "vec4<f32>");
        assert_eq!(token_of::<[f32; 4], CudaC>(), "float4");
        // GPU vector types — f64
        assert_eq!(token_of::<[f64; 2], Wgsl>(), "vec2<f64>");
        assert_eq!(token_of::<[f64; 2], CudaC>(), "double2");
        assert_eq!(token_of::<[f64; 3], Wgsl>(), "vec3<f64>");
        assert_eq!(token_of::<[f64; 3], CudaC>(), "double3");
        assert_eq!(token_of::<[f64; 4], Wgsl>(), "vec4<f64>");
        assert_eq!(token_of::<[f64; 4], CudaC>(), "double4");
        // GPU vector types — i32
        assert_eq!(token_of::<[i32; 2], Wgsl>(), "vec2<i32>");
        assert_eq!(token_of::<[i32; 2], CudaC>(), "int2");
        assert_eq!(token_of::<[i32; 3], Wgsl>(), "vec3<i32>");
        assert_eq!(token_of::<[i32; 3], CudaC>(), "int3");
        assert_eq!(token_of::<[i32; 4], Wgsl>(), "vec4<i32>");
        assert_eq!(token_of::<[i32; 4], CudaC>(), "int4");
        // GPU vector types — u32
        assert_eq!(token_of::<[u32; 2], Wgsl>(), "vec2<u32>");
        assert_eq!(token_of::<[u32; 2], CudaC>(), "uint2");
        assert_eq!(token_of::<[u32; 3], Wgsl>(), "vec3<u32>");
        assert_eq!(token_of::<[u32; 3], CudaC>(), "uint3");
        assert_eq!(token_of::<[u32; 4], Wgsl>(), "vec4<u32>");
        assert_eq!(token_of::<[u32; 4], CudaC>(), "uint4");
    }
}
