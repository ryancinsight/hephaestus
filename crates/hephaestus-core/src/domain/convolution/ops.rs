use eunomia::Pod;
use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;

use super::{ConvolutionBackwardOperands, ConvolutionForwardOperands};

/// Monomorphized accelerator convolution operations.
///
/// Implementors are backend-owned zero-sized operation markers. Preparation
/// validates every operand and compiles all requested kernels before dispatch,
/// so validation or compilation failures cannot partially mutate outputs.
pub trait ConvolutionOps<D: ComputeDevice, T: Pod> {
    /// Prepared regular-forward resources bound to fixed operands.
    type PreparedForward<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;
    /// Prepared regular-backward resources bound to fixed operands.
    type PreparedBackward<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;
    /// Prepared transposed-forward resources bound to fixed operands.
    type PreparedTransposedForward<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;
    /// Prepared transposed-backward resources bound to fixed operands.
    type PreparedTransposedBackward<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;

    /// Compute a regular convolution into caller-owned device storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn convolution_forward_into<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: ConvolutionForwardOperands<'_, D::Buffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<()> {
        let prepared = self.prepare_convolution_forward(device, operands, parameters)?;
        self.dispatch_convolution_forward(device, &prepared)
    }

    /// Validate and prepare a regular convolution forward pass.
    ///
    /// # Errors
    ///
    /// Returns a typed shape, storage, aliasing, capability, or compilation
    /// failure before output mutation.
    fn prepare_convolution_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: ConvolutionForwardOperands<'a, D::Buffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedForward<'a, R, S>>;

    /// Dispatch a prepared regular convolution forward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed dispatch or synchronization failure.
    fn dispatch_convolution_forward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()>;

    /// Add a regular convolution backward pass into selected gradients.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn convolution_backward_accumulate<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: ConvolutionBackwardOperands<'_, D::Buffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<()> {
        let prepared = self.prepare_convolution_backward(device, operands, parameters)?;
        self.dispatch_convolution_backward(device, &prepared)
    }

    /// Validate and prepare every selected regular backward kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed shape, storage, aliasing, capability, or compilation
    /// failure before any gradient mutation.
    fn prepare_convolution_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: ConvolutionBackwardOperands<'a, D::Buffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedBackward<'a, R, S>>;

    /// Dispatch a prepared regular convolution backward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed dispatch or synchronization failure.
    fn dispatch_convolution_backward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()>;

    /// Compute a transposed convolution into caller-owned device storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn convolution_transposed_forward_into<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: ConvolutionForwardOperands<'_, D::Buffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<()> {
        let prepared = self.prepare_convolution_transposed_forward(device, operands, parameters)?;
        self.dispatch_convolution_transposed_forward(device, &prepared)
    }

    /// Validate and prepare a transposed convolution forward pass.
    ///
    /// # Errors
    ///
    /// Returns a typed shape, storage, aliasing, capability, or compilation
    /// failure before output mutation.
    fn prepare_convolution_transposed_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: ConvolutionForwardOperands<'a, D::Buffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedForward<'a, R, S>>;

    /// Dispatch a prepared transposed convolution forward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed dispatch or synchronization failure.
    fn dispatch_convolution_transposed_forward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedTransposedForward<'_, R, S>,
    ) -> Result<()>;

    /// Add a transposed convolution backward pass into selected gradients.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn convolution_transposed_backward_accumulate<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: ConvolutionBackwardOperands<'_, D::Buffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<()> {
        let prepared =
            self.prepare_convolution_transposed_backward(device, operands, parameters)?;
        self.dispatch_convolution_transposed_backward(device, &prepared)
    }

    /// Validate and prepare every selected transposed backward kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed shape, storage, aliasing, capability, or compilation
    /// failure before any gradient mutation.
    fn prepare_convolution_transposed_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: ConvolutionBackwardOperands<'a, D::Buffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedBackward<'a, R, S>>;

    /// Dispatch a prepared transposed convolution backward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed dispatch or synchronization failure.
    fn dispatch_convolution_transposed_backward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedTransposedBackward<'_, R, S>,
    ) -> Result<()>;
}
