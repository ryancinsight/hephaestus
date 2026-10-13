//! Scalar-seam traits for GPU matrix kernels.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Wgsl};

/// Helper trait to substitute the correct zero literal in WGSL for different scalar types.
pub trait MatmulZero: DialectScalar<Wgsl> {
    /// The WGSL zero literal (e.g. `"0.0"`, `"0u"`, `"0"`).
    const WGSL_ZERO: &'static str;
}

/// WGPU scalar supporting matrix identity initialization.
pub trait MatrixIdentityScalar: MatmulZero + Pod {
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
}

impl MatmulZero for f32 {
    const WGSL_ZERO: &'static str = "0.0";
}

impl MatrixIdentityScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl MatmulZero for u32 {
    const WGSL_ZERO: &'static str = "0u";
}

impl MatrixIdentityScalar for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl MatmulZero for i32 {
    const WGSL_ZERO: &'static str = "0";
}

impl MatmulZero for f64 {
    const WGSL_ZERO: &'static str = "0.0";
}

impl MatrixIdentityScalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

/// WGPU scalar whose shader type supports row-reduction rank estimation.
///
/// Bundles the arithmetic and ordering bounds the host-delegated Gaussian
/// elimination behind [`matrix_rank`](super::properties::matrix_rank),
/// [`matrix_rank_with_tolerance`](super::properties::matrix_rank_with_tolerance), and
/// [`det`](super::properties::det) requires, so callers need only state `T: MatrixRankScalar` (SSOT
/// for the rank/determinant scalar contract).
pub trait MatrixRankScalar:
    DialectScalar<Wgsl>
    + Pod
    + PartialOrd
    + core::ops::Neg<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + From<f32>
{
}

impl MatrixRankScalar for f32 {}
