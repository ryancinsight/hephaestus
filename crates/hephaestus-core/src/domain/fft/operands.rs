use crate::domain::view::StridedView;

/// Borrowed split-complex device operands for an in-place dense FFT.
///
/// The real and imaginary views must describe the same dense C-order shape and
/// distinct full buffers. [`plan_fft`](super::plan_fft) validates those
/// requirements before a backend prepares kernels or mutates either component.
pub struct FftOperands<'a, B, const R: usize> {
    /// Real component, transformed in place.
    pub real: StridedView<'a, B, R>,
    /// Imaginary component, transformed in place.
    pub imaginary: StridedView<'a, B, R>,
}
