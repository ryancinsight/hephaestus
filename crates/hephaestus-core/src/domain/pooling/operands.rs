use crate::domain::view::StridedView;

/// Borrowed operands for a pooling forward pass.
pub struct PoolingForwardOperands<'a, B, const R: usize> {
    /// Input tensor with batch and channel axes followed by spatial axes.
    pub input: StridedView<'a, B, R>,
    /// Caller-owned output tensor.
    pub output: StridedView<'a, B, R>,
}

/// Borrowed operands for an additive pooling backward pass.
pub struct PoolingBackwardOperands<'a, B, const R: usize> {
    /// Forward input tensor.
    pub input: StridedView<'a, B, R>,
    /// Gradient of the pooling output.
    pub grad_output: StridedView<'a, B, R>,
    /// Caller-owned input-gradient target. The operation adds into it.
    pub grad_input: StridedView<'a, B, R>,
}

impl<B, const R: usize> Clone for PoolingForwardOperands<'_, B, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, const R: usize> Copy for PoolingForwardOperands<'_, B, R> {}

impl<B, const R: usize> Clone for PoolingBackwardOperands<'_, B, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, const R: usize> Copy for PoolingBackwardOperands<'_, B, R> {}
