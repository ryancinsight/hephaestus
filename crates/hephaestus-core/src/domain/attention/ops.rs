use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;

use super::{AttentionBackwardOperands, AttentionForwardOperands, AttentionScalar};

/// Monomorphized accelerator scaled dot-product attention operations.
///
/// Implementors are backend-owned zero-sized markers. Preparation validates
/// every operand and compiles every requested kernel before mutation.
pub trait AttentionOps<D: ComputeDevice, T: AttentionScalar> {
    /// Prepared forward resources bound to fixed operands.
    type PreparedForward<'a>
    where
        D: 'a,
        T: 'a;
    /// Prepared backward resources bound to fixed operands.
    type PreparedBackward<'a>
    where
        D: 'a,
        T: 'a;

    /// Compute attention output and post-softmax weights into caller storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn attention_forward_into(
        &self,
        device: &D,
        operands: AttentionForwardOperands<'_, D::Buffer<T>, T>,
    ) -> Result<()> {
        let prepared = self.prepare_attention_forward(device, operands)?;
        self.dispatch_attention_forward(device, &prepared)
    }

    /// Validate and prepare an attention forward pass.
    ///
    /// # Errors
    ///
    /// Returns before output mutation on shape, storage, alias, scalar,
    /// capability, or compilation failure.
    fn prepare_attention_forward<'a>(
        &self,
        device: &'a D,
        operands: AttentionForwardOperands<'a, D::Buffer<T>, T>,
    ) -> Result<Self::PreparedForward<'a>>;

    /// Dispatch a prepared attention forward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed dispatch or synchronization failure.
    fn dispatch_attention_forward(
        &self,
        device: &D,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()>;

    /// Add an attention backward pass into selected gradient destinations.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or device-dispatch failure.
    fn attention_backward_accumulate(
        &self,
        device: &D,
        operands: AttentionBackwardOperands<'_, D::Buffer<T>, T>,
    ) -> Result<()> {
        let prepared = self.prepare_attention_backward(device, operands)?;
        self.dispatch_attention_backward(device, &prepared)
    }

    /// Validate and prepare every selected attention backward kernel.
    ///
    /// # Errors
    ///
    /// Returns before gradient mutation on validation or compilation failure.
    fn prepare_attention_backward<'a>(
        &self,
        device: &'a D,
        operands: AttentionBackwardOperands<'a, D::Buffer<T>, T>,
    ) -> Result<Self::PreparedBackward<'a>>;

    /// Dispatch a prepared additive attention backward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed dispatch or synchronization failure.
    fn dispatch_attention_backward(
        &self,
        device: &D,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()>;
}
