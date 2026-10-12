//! WGPU provider implementation for scaled dot-product attention.

mod metadata;
mod preflight;
mod prepared;
mod resources;
mod seam;
mod shader;

pub use seam::WgpuAttentionOps;

#[cfg(test)]
mod tests;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{AttentionScalar, DialectScalar, Wgsl};

/// Closed scalar set the WGSL attention kernels serve, with per-type shader
/// tokens and the native uniform representation of the runtime score scale.
///
/// The uniform tail is 16 bytes for both members (`vec4<f32>` /
/// `vec2<f64>`), so the metadata ABI keeps one shape and size; only the
/// scale declaration token differs. Reduced-precision types stay out until
/// every advertised provider has native arithmetic and shared conformance
/// coverage (mirroring the `AttentionScalar` contract and CUDA's
/// `CudaAttentionScalar` seal).
pub(in crate::application::attention) trait WgslAttentionScalar:
    AttentionScalar + DialectScalar<Wgsl>
{
    /// Native uniform tail holding the score scale (16 bytes).
    type ScaleRepr: Pod + Zeroable + Copy;
    /// Largest finite literal for `finite` preflight bounds.
    const FINITE_MAX: &'static str;
    /// Most negative finite literal initializing softmax maxima.
    const NEG_MAX: &'static str;
    /// Machine epsilon literal for the probability-sum tolerance.
    const EPSILON: &'static str;
    /// Uniform struct field declaring the scale tail
    /// (e.g. `"scale_and_padding: vec4<f32>,"`).
    const SCALE_DECL: &'static str;
    /// Pack the runtime scale into its uniform representation.
    fn scale_repr(scale: Self) -> Self::ScaleRepr;
}

impl WgslAttentionScalar for f32 {
    type ScaleRepr = [f32; 4];
    const FINITE_MAX: &'static str = "3.402823466e+38";
    const NEG_MAX: &'static str = "-3.402823466e+38";
    const EPSILON: &'static str = "1.192092896e-7";
    const SCALE_DECL: &'static str = "scale_and_padding: vec4<f32>,";
    fn scale_repr(scale: Self) -> Self::ScaleRepr {
        [scale, 0.0, 0.0, 0.0]
    }
}

impl WgslAttentionScalar for f64 {
    type ScaleRepr = [f64; 2];
    const FINITE_MAX: &'static str = "1.7976931348623157e+308";
    const NEG_MAX: &'static str = "-1.7976931348623157e+308";
    const EPSILON: &'static str = "2.220446049250313e-16";
    const SCALE_DECL: &'static str = "scale_and_padding: vec2<f64>,";
    fn scale_repr(scale: Self) -> Self::ScaleRepr {
        [scale, 0.0]
    }
}
