//! Kernel-dialect vocabulary: sealed dialect markers and per-dialect scalar
//! type tokens.
//!
//! A [`KernelDialect`] is the shading/kernel language a backend compiles —
//! WGSL for wgpu (native and Metal-pinned), CUDA C++ for the NVRTC-compiled
//! CUDA backend. Operation markers ([`crate::domain::ops`]) and scalar types
//! carry their shader tokens per dialect through these traits, so one op
//! vocabulary serves every backend and a kernel authored for one dialect
//! simply does not implement the others — dispatching it on the wrong
//! backend is a compile error, not a runtime failure.
//!
//! The trait is sealed: a new dialect is a deliberate substrate extension
//! (new backend crate), not a consumer-side extension point.

use bytemuck::Pod;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Wgsl {}
    impl Sealed for super::CudaC {}
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

impl KernelDialect for Wgsl {
    const NAME: &'static str = "wgsl";
}

impl KernelDialect for CudaC {
    const NAME: &'static str = "cuda-c";
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
    /// CUDA C++).
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
        assert_eq!(token_of::<u32, Wgsl>(), "u32");
        assert_eq!(token_of::<u32, CudaC>(), "unsigned int");
        assert_eq!(token_of::<i32, Wgsl>(), "i32");
        assert_eq!(token_of::<i32, CudaC>(), "int");
    }
}
