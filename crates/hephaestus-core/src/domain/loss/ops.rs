use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;

use super::{CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyScalar};

/// Monomorphized accelerator mean cross-entropy operations.
///
/// Implementors are backend-owned zero-sized markers. Preparation validates
/// host-visible structure and compiles device preflight plus compute kernels
/// before any output mutation.
pub trait CrossEntropyOps<D: ComputeDevice, T: CrossEntropyScalar> {
    /// Prepared forward resources bound to fixed provider-resident operands.
    type PreparedForward<'a>
    where
        D: 'a,
        T: 'a;
    /// Prepared backward resources bound to fixed provider-resident operands.
    type PreparedBackward<'a>
    where
        D: 'a,
        T: 'a;

    /// Compute normalized probabilities and scalar mean loss.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, preflight, or dispatch failure.
    fn cross_entropy_forward_into(
        &self,
        device: &D,
        operands: CrossEntropyForwardOperands<'_, D::Buffer<T>, D::Buffer<u32>>,
    ) -> Result<()> {
        let prepared = self.prepare_cross_entropy_forward(device, operands)?;
        self.dispatch_cross_entropy_forward(device, &prepared)
    }

    /// Validate and prepare cross-entropy forward resources.
    ///
    /// # Errors
    ///
    /// Returns before output mutation on structural, capability, allocation,
    /// or compilation failure.
    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a D,
        operands: CrossEntropyForwardOperands<'a, D::Buffer<T>, D::Buffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>>;

    /// Run device preflight and dispatch a prepared forward pass.
    ///
    /// # Errors
    ///
    /// Returns a typed target, finite-value, arithmetic, or device failure.
    fn dispatch_cross_entropy_forward(
        &self,
        device: &D,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()>;

    /// Add the mean cross-entropy gradient into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, preflight, or dispatch failure.
    fn cross_entropy_backward_accumulate(
        &self,
        device: &D,
        operands: CrossEntropyBackwardOperands<'_, D::Buffer<T>, D::Buffer<u32>>,
    ) -> Result<()> {
        let prepared = self.prepare_cross_entropy_backward(device, operands)?;
        self.dispatch_cross_entropy_backward(device, &prepared)
    }

    /// Validate and prepare additive backward resources.
    ///
    /// # Errors
    ///
    /// Returns before destination mutation on structural, capability,
    /// allocation, or compilation failure.
    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a D,
        operands: CrossEntropyBackwardOperands<'a, D::Buffer<T>, D::Buffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>>;

    /// Run device preflight and dispatch a prepared additive backward pass.
    ///
    /// # Errors
    ///
    /// Returns a typed target, finite-value, arithmetic, or device failure.
    fn dispatch_cross_entropy_backward(
        &self,
        device: &D,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()>;
}
