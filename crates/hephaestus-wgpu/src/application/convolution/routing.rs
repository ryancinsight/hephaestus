use core::any::TypeId;

use hephaestus_core::{HephaestusError, Result};

use super::shader::{BiasMode, ConvolutionDirection, GradientTarget};

struct RegularForwardKernel<const S: usize, const BIAS: bool>;
struct TransposedForwardKernel<const S: usize, const BIAS: bool>;
struct RegularInputGradientKernel<const S: usize>;
struct RegularWeightGradientKernel<const S: usize>;
struct RegularBiasGradientKernel<const S: usize>;
struct TransposedInputGradientKernel<const S: usize>;
struct TransposedWeightGradientKernel<const S: usize>;
struct TransposedBiasGradientKernel<const S: usize>;

pub(super) fn validate_spatial_rank<const S: usize>() -> Result<()> {
    if !(1..=3).contains(&S) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("WGPU convolution supports spatial ranks 1 through 3, got {S}"),
        });
    }
    Ok(())
}

pub(super) fn forward_pipeline_key<const S: usize>(
    direction: ConvolutionDirection,
    bias: BiasMode,
) -> TypeId {
    match (direction, bias) {
        (ConvolutionDirection::Regular, BiasMode::Absent) => {
            TypeId::of::<RegularForwardKernel<S, false>>()
        }
        (ConvolutionDirection::Regular, BiasMode::Present) => {
            TypeId::of::<RegularForwardKernel<S, true>>()
        }
        (ConvolutionDirection::Transposed, BiasMode::Absent) => {
            TypeId::of::<TransposedForwardKernel<S, false>>()
        }
        (ConvolutionDirection::Transposed, BiasMode::Present) => {
            TypeId::of::<TransposedForwardKernel<S, true>>()
        }
    }
}

pub(super) fn gradient_pipeline_key<const S: usize>(
    direction: ConvolutionDirection,
    target: GradientTarget,
) -> TypeId {
    match (direction, target) {
        (ConvolutionDirection::Regular, GradientTarget::Input) => {
            TypeId::of::<RegularInputGradientKernel<S>>()
        }
        (ConvolutionDirection::Regular, GradientTarget::Weight) => {
            TypeId::of::<RegularWeightGradientKernel<S>>()
        }
        (ConvolutionDirection::Regular, GradientTarget::Bias) => {
            TypeId::of::<RegularBiasGradientKernel<S>>()
        }
        (ConvolutionDirection::Transposed, GradientTarget::Input) => {
            TypeId::of::<TransposedInputGradientKernel<S>>()
        }
        (ConvolutionDirection::Transposed, GradientTarget::Weight) => {
            TypeId::of::<TransposedWeightGradientKernel<S>>()
        }
        (ConvolutionDirection::Transposed, GradientTarget::Bias) => {
            TypeId::of::<TransposedBiasGradientKernel<S>>()
        }
    }
}

pub(super) fn gradient_label(
    direction: ConvolutionDirection,
    target: GradientTarget,
) -> &'static str {
    match (direction, target) {
        (ConvolutionDirection::Regular, GradientTarget::Input) => {
            "hephaestus-convolution-input-gradient"
        }
        (ConvolutionDirection::Regular, GradientTarget::Weight) => {
            "hephaestus-convolution-weight-gradient"
        }
        (ConvolutionDirection::Regular, GradientTarget::Bias) => {
            "hephaestus-convolution-bias-gradient"
        }
        (ConvolutionDirection::Transposed, GradientTarget::Input) => {
            "hephaestus-convolution-transposed-input-gradient"
        }
        (ConvolutionDirection::Transposed, GradientTarget::Weight) => {
            "hephaestus-convolution-transposed-weight-gradient"
        }
        (ConvolutionDirection::Transposed, GradientTarget::Bias) => {
            "hephaestus-convolution-transposed-bias-gradient"
        }
    }
}
