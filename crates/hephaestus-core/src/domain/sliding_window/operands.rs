use crate::domain::view::StridedView;

/// Borrowed operands for extracting spatial windows into columns.
pub struct SlidingWindowUnfoldOperands<'a, B, const R: usize> {
    /// Input tensor with batch and channel axes followed by spatial axes.
    pub input: StridedView<'a, B, R>,
    /// `[batch, channel * kernel_volume, output_locations]` destination.
    pub output: StridedView<'a, B, 3>,
}

/// Borrowed operands for accumulating columns into a spatial tensor.
pub struct SlidingWindowFoldOperands<'a, B, const R: usize> {
    /// `[batch, channel * kernel_volume, output_locations]` source.
    pub input: StridedView<'a, B, 3>,
    /// Caller-owned `[batch, channel, spatial...]` destination.
    pub output: StridedView<'a, B, R>,
}

impl<B, const R: usize> Clone for SlidingWindowUnfoldOperands<'_, B, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, const R: usize> Copy for SlidingWindowUnfoldOperands<'_, B, R> {}

impl<B, const R: usize> Clone for SlidingWindowFoldOperands<'_, B, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, const R: usize> Copy for SlidingWindowFoldOperands<'_, B, R> {}
