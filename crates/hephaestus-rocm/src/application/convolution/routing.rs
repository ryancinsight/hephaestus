use hephaestus_core::{HephaestusError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConvolutionDirection {
    Regular,
    Transposed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BiasMode {
    Absent,
    Present,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GradientTarget {
    Input,
    Weight,
    Bias,
}

pub(super) fn validate_spatial_rank<const S: usize>() -> Result<()> {
    if !(1..=3).contains(&S) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("ROCm convolution supports spatial ranks 1 through 3, got {S}"),
        });
    }
    Ok(())
}

pub(super) const fn forward_label(direction: ConvolutionDirection) -> (&'static str, &'static str) {
    match direction {
        ConvolutionDirection::Regular => (
            "hephaestus-convolution-forward",
            "hephaestus_convolution_forward",
        ),
        ConvolutionDirection::Transposed => (
            "hephaestus-convolution-transposed-forward",
            "hephaestus_convolution_transposed_forward",
        ),
    }
}

pub(super) const fn gradient_label(
    direction: ConvolutionDirection,
    target: GradientTarget,
) -> (&'static str, &'static str) {
    match (direction, target) {
        (ConvolutionDirection::Regular, GradientTarget::Input) => (
            "hephaestus-convolution-input-gradient",
            "hephaestus_convolution_input_gradient",
        ),
        (ConvolutionDirection::Regular, GradientTarget::Weight) => (
            "hephaestus-convolution-weight-gradient",
            "hephaestus_convolution_weight_gradient",
        ),
        (ConvolutionDirection::Regular, GradientTarget::Bias) => (
            "hephaestus-convolution-bias-gradient",
            "hephaestus_convolution_bias_gradient",
        ),
        (ConvolutionDirection::Transposed, GradientTarget::Input) => (
            "hephaestus-convolution-transposed-input-gradient",
            "hephaestus_convolution_transposed_input_gradient",
        ),
        (ConvolutionDirection::Transposed, GradientTarget::Weight) => (
            "hephaestus-convolution-transposed-weight-gradient",
            "hephaestus_convolution_transposed_weight_gradient",
        ),
        (ConvolutionDirection::Transposed, GradientTarget::Bias) => (
            "hephaestus-convolution-transposed-bias-gradient",
            "hephaestus_convolution_transposed_bias_gradient",
        ),
    }
}
