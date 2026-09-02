use super::{SlidingWindowFoldOperands, SlidingWindowUnfoldOperands};
use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
use bytemuck::Pod;

/// Monomorphized accelerator unfold/fold operations.
pub trait SlidingWindowOps<D: ComputeDevice, T: Pod> {
    /// Prepared resources bound to one unfold operation.
    type PreparedUnfold<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;
    /// Prepared resources bound to one fold operation.
    type PreparedFold<'a, const R: usize, const S: usize>
    where
        D: 'a,
        T: 'a;

    /// Extract spatial windows into caller-owned column storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or dispatch failure.
    fn unfold_into<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: SlidingWindowUnfoldOperands<'_, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
    ) -> Result<()> {
        let prepared = self.prepare_unfold(device, operands, parameters)?;
        self.dispatch_unfold(device, &prepared)
    }

    /// Validate and prepare an unfold operation.
    ///
    /// # Errors
    ///
    /// Returns before output mutation on shape, layout, aliasing, capability,
    /// allocation, or compilation failure.
    fn prepare_unfold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: SlidingWindowUnfoldOperands<'a, D::Buffer<T>, R>,
        parameters: leto::WindowParameters<S>,
    ) -> Result<Self::PreparedUnfold<'a, R, S>>;

    /// Dispatch a prepared unfold operation.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch or synchronization failure.
    fn dispatch_unfold<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedUnfold<'_, R, S>,
    ) -> Result<()>;

    /// Accumulate column storage into caller-owned spatial output.
    ///
    /// The output is zeroed before accumulation, so this operation is the
    /// adjoint of [`Self::unfold_into`].
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or dispatch failure.
    fn fold_into<const R: usize, const S: usize>(
        &self,
        device: &D,
        operands: SlidingWindowFoldOperands<'_, D::Buffer<T>, R>,
        output_spatial_shape: [usize; S],
        parameters: leto::WindowParameters<S>,
    ) -> Result<()> {
        let prepared = self.prepare_fold(device, operands, output_spatial_shape, parameters)?;
        self.dispatch_fold(device, &prepared)
    }

    /// Validate and prepare a fold operation.
    ///
    /// # Errors
    ///
    /// Returns before output mutation on shape, layout, aliasing, capability,
    /// allocation, or compilation failure.
    fn prepare_fold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a D,
        operands: SlidingWindowFoldOperands<'a, D::Buffer<T>, R>,
        output_spatial_shape: [usize; S],
        parameters: leto::WindowParameters<S>,
    ) -> Result<Self::PreparedFold<'a, R, S>>;

    /// Dispatch a prepared fold operation.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch or synchronization failure.
    fn dispatch_fold<const R: usize, const S: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedFold<'_, R, S>,
    ) -> Result<()>;
}
