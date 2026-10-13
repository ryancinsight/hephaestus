use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use crate::domain::error::Result;

use super::limits::validate_plan_address_limit;
use super::limits::validate_regular_projection_limit;
use super::limits::validate_transposed_projection_limit;

/// Validated regular-convolution dimensions and launch bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvolutionPlan<const S: usize> {
    /// Batch extent.
    pub batch: usize,
    /// Input channel extent.
    pub input_channels: usize,
    /// Output channel extent.
    pub output_channels: usize,
    /// Input spatial extents.
    pub input_spatial: [usize; S],
    /// Kernel spatial extents.
    pub kernel_spatial: [usize; S],
    /// Output spatial extents.
    pub output_spatial: [usize; S],
    /// Validated convolution parameters.
    pub parameters: ConvolutionParameters<S>,
    /// Logical output element count.
    pub output_elements: usize,
    /// Largest physical element offset touched by any operand.
    pub max_physical_offset: usize,
}

impl<const S: usize> ConvolutionPlan<S> {
    /// Validate every value narrowed by a backend kernel's address contract.
    ///
    /// Backends pass the inclusive limit of every integer representation used
    /// by their address and projection arithmetic. CUDA and the current WGSL
    /// implementation both use signed 32-bit projection arithmetic and
    /// therefore pass `i32::MAX`.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error when any extent, parameter, logical
    /// element count, or physical offset exceeds `max_inclusive`.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        validate_plan_address_limit(
            max_inclusive,
            [
                self.batch,
                self.input_channels,
                self.output_channels,
                self.output_elements,
                self.max_physical_offset,
            ],
            [
                &self.input_spatial,
                &self.kernel_spatial,
                &self.output_spatial,
                self.parameters.stride(),
                self.parameters.padding(),
                self.parameters.dilation(),
            ],
        )?;
        validate_regular_projection_limit(self, max_inclusive)
    }
}

/// Validated transposed-convolution dimensions and launch bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransposedConvolutionPlan<const S: usize> {
    /// Batch extent.
    pub batch: usize,
    /// Input channel extent.
    pub input_channels: usize,
    /// Output channel extent.
    pub output_channels: usize,
    /// Input spatial extents.
    pub input_spatial: [usize; S],
    /// Kernel spatial extents.
    pub kernel_spatial: [usize; S],
    /// Output spatial extents.
    pub output_spatial: [usize; S],
    /// Validated transposed-convolution parameters.
    pub parameters: TransposedConvolutionParameters<S>,
    /// Logical output element count.
    pub output_elements: usize,
    /// Largest physical element offset touched by any operand.
    pub max_physical_offset: usize,
}

impl<const S: usize> TransposedConvolutionPlan<S> {
    /// Validate every value narrowed by a backend kernel's address contract.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error when any extent, parameter, logical
    /// element count, or physical offset exceeds `max_inclusive`.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        validate_plan_address_limit(
            max_inclusive,
            [
                self.batch,
                self.input_channels,
                self.output_channels,
                self.output_elements,
                self.max_physical_offset,
            ],
            [
                &self.input_spatial,
                &self.kernel_spatial,
                &self.output_spatial,
                self.parameters.stride(),
                self.parameters.padding(),
                self.parameters.output_padding(),
                self.parameters.dilation(),
            ],
        )?;
        validate_transposed_projection_limit(self, max_inclusive)
    }
}
