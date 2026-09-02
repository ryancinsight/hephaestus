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
    /// Optional forward input tensor.
    ///
    /// Maximum pooling needs the input values to identify the selected
    /// element. Average pooling only needs the input shape, which the plan
    /// derives from [`grad_input`](Self::grad_input) when this is `None`.
    pub input: Option<StridedView<'a, B, R>>,
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
