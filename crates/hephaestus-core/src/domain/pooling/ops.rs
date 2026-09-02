use super::{PoolingBackwardOperands, PoolingForwardOperands};
use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
use bytemuck::Pod;

/// Pooling reduction selected at the operation boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingMode {
    /// Select the first maximum in each window.
    Maximum,
    /// Average the valid input points in each window.
    Average,
}

/// Monomorphized accelerator pooling operations.
pub trait PoolingOps<D: ComputeDevice, T: Pod> {
    /// Prepared forward resources bound to fixed operands and geometry.
    type PreparedForward<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;
    /// Prepared backward resources bound to fixed operands and geometry.
    type PreparedBackward<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;

    /// Compute pooling into caller-owned device storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or dispatch failure.
    fn pooling_forward_into<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: PoolingForwardOperands<'_, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<()> {
        let prepared = self.prepare_pooling_forward(device, operands, parameters, mode)?;
        self.dispatch_pooling_forward(device, &prepared)
    }

    /// Validate and prepare a pooling forward pass.
    ///
    /// # Errors
    ///
    /// Returns before output mutation on shape, layout, aliasing, capability,
    /// allocation, or compilation failure.
    fn prepare_pooling_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: PoolingForwardOperands<'a, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedForward<'a, R, S>>;

    /// Dispatch a prepared pooling forward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch or synchronization failure.
    fn dispatch_pooling_forward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()>;

    /// Add pooling's input gradient into caller-owned device storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or dispatch failure.
    fn pooling_backward_accumulate<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: PoolingBackwardOperands<'_, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<()> {
        let prepared = self.prepare_pooling_backward(device, operands, parameters, mode)?;
        self.dispatch_pooling_backward(device, &prepared)
    }

    /// Validate and prepare an additive pooling backward pass.
    ///
    /// # Errors
    ///
    /// Returns before gradient mutation on shape, layout, aliasing,
    /// capability, allocation, or compilation failure.
    fn prepare_pooling_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: PoolingBackwardOperands<'a, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedBackward<'a, R, S>>;

    /// Dispatch a prepared additive pooling backward pass.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch or synchronization failure.
    fn dispatch_pooling_backward<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()>;
}
