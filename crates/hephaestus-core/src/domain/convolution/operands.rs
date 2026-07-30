use crate::domain::view::StridedView;

/// Borrowed operands for a regular or transposed convolution forward pass.
pub struct ConvolutionForwardOperands<'a, B, const R: usize> {
    /// Input tensor.
    pub input: StridedView<'a, B, R>,
    /// Kernel weights.
    pub weight: StridedView<'a, B, R>,
    /// Optional channel bias.
    pub bias: Option<StridedView<'a, B, 1>>,
    /// Caller-owned output tensor.
    pub output: StridedView<'a, B, R>,
}

/// Optional caller-owned targets for an additive convolution backward pass.
pub struct ConvolutionGradientViews<'a, B, const R: usize> {
    /// Input gradient target.
    pub input: Option<StridedView<'a, B, R>>,
    /// Weight gradient target.
    pub weight: Option<StridedView<'a, B, R>>,
    /// Bias gradient target.
    pub bias: Option<StridedView<'a, B, 1>>,
}

impl<B, const R: usize> ConvolutionGradientViews<'_, B, R> {
    /// Returns true when no gradient target was selected.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.input.is_none() && self.weight.is_none() && self.bias.is_none()
    }
}

/// Borrowed operands for a regular or transposed additive backward pass.
pub struct ConvolutionBackwardOperands<'a, B, const R: usize> {
    /// Forward input tensor.
    pub input: StridedView<'a, B, R>,
    /// Forward kernel weights.
    pub weight: StridedView<'a, B, R>,
    /// Gradient of the forward output.
    pub grad_output: StridedView<'a, B, R>,
    /// Selected caller-owned gradient targets.
    pub gradients: ConvolutionGradientViews<'a, B, R>,
}
