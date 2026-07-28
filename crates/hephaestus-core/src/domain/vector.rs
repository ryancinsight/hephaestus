//! Dense rank-one vector operations over device-resident buffers.

use bytemuck::Pod;

use super::device::ComputeDevice;
use super::error::Result;

/// Backend-neutral dense vector operations for iterative solvers.
///
/// # Why this seam exists
///
/// Each backend crate already provides `dot`, `norm_l2`, and the elementwise
/// families, but as free functions taking that backend's concrete device type,
/// and the elementwise family rejects an aliased output so it cannot express an
/// in-place update at all. [`ComputeDevice`] covers allocation and transfer
/// only. A consumer generic over the device could therefore reach no vector
/// operation, and had to bind to one backend or hand-write kernels of its own.
/// This trait closes that gap: a Krylov solver, a preconditioner, or any other
/// vector-recurrence consumer programs against `V: DenseVectorOps<D, T>` and
/// runs on every backend implementing it.
///
/// # Shape
///
/// `Self` is a backend-provided bundle of prepared kernels, constructed once
/// against a device, and the device is passed per call. This mirrors
/// [`super::kernel::MultiStorageKernel`] and keeps kernel compilation out of the
/// operation path — preparing a shader per call would dominate every operation
/// it performs.
///
/// # Contract
///
/// All operands are dense, contiguous, rank-one buffers of equal length; an
/// implementation returns [`crate::HephaestusError::LengthMismatch`] otherwise.
/// Operations naming a `target` write it in place, which is what keeps a solver
/// iteration free of scratch allocation and of round trips through host memory.
/// A zero-length operand is a no-op, not an error.
///
/// # Prepared reductions
///
/// `dot` and `norm_l2` reduce a whole vector to one scalar the host must read to
/// test convergence. Preparing a reduction binds its intermediate storage and
/// dispatch resources to a fixed operand allocation once, so a solver reusing
/// the same buffers across iterations allocates nothing after setup. A prepared
/// handle is valid only for the allocations it was prepared against;
/// implementations reject a mismatched operand rather than reading the wrong
/// memory.
pub trait DenseVectorOps<D: ComputeDevice, T: Pod> {
    /// Prepared resources for a dot product over a fixed operand pair.
    type PreparedDot<'a>
    where
        Self: 'a;
    /// Prepared resources for a Euclidean norm over a fixed operand.
    type PreparedNorm<'a>
    where
        Self: 'a;

    /// Copy `source` into `target` device-to-device.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend transfer failure.
    fn copy_vector(&self, device: &D, source: &D::Buffer<T>, target: &D::Buffer<T>) -> Result<()>;

    /// Scale `target` in place by `factor`.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn scale_vector(&self, device: &D, target: &D::Buffer<T>, factor: T) -> Result<()>;

    /// Apply `target += factor * source` in place.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn axpy(
        &self,
        device: &D,
        target: &D::Buffer<T>,
        source: &D::Buffer<T>,
        factor: T,
    ) -> Result<()>;

    /// Apply `target = source + factor * target` in place.
    ///
    /// The companion of [`Self::axpy`] with the scaling on the accumulator
    /// rather than the increment. Krylov direction recurrences take this shape,
    /// and one kernel keeps it to a single traversal where a scale followed by
    /// an axpy would take two.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn xpay(
        &self,
        device: &D,
        target: &D::Buffer<T>,
        source: &D::Buffer<T>,
        factor: T,
    ) -> Result<()>;

    /// Compute `output = left - right` into distinct caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, an aliased output, or the backend dispatch
    /// failure.
    fn subtract_into(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
        output: &D::Buffer<T>,
    ) -> Result<()>;

    /// Prepare a dot product bound to `left` and `right`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend preparation failure.
    fn prepare_dot<'a>(
        &self,
        device: &D,
        left: &'a D::Buffer<T>,
        right: &'a D::Buffer<T>,
    ) -> Result<Self::PreparedDot<'a>>;

    /// Execute a prepared dot product and read the scalar back to the host.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dot_prepared<'a>(
        &self,
        device: &D,
        prepared: &Self::PreparedDot<'a>,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
    ) -> Result<T>;

    /// Prepare a Euclidean norm bound to `vector`.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation failure.
    fn prepare_norm_l2<'a>(
        &self,
        device: &D,
        vector: &'a D::Buffer<T>,
    ) -> Result<Self::PreparedNorm<'a>>;

    /// Execute a prepared Euclidean norm and read the scalar back to the host.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn norm_l2_prepared<'a>(
        &self,
        device: &D,
        prepared: &Self::PreparedNorm<'a>,
        vector: &D::Buffer<T>,
    ) -> Result<T>;

    /// Compute a one-shot dot product.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation or dispatch failure.
    fn dot(&self, device: &D, left: &D::Buffer<T>, right: &D::Buffer<T>) -> Result<T> {
        let prepared = self.prepare_dot(device, left, right)?;
        self.dot_prepared(device, &prepared, left, right)
    }

    /// Compute a one-shot Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation or dispatch failure.
    fn norm_l2(&self, device: &D, vector: &D::Buffer<T>) -> Result<T> {
        let prepared = self.prepare_norm_l2(device, vector)?;
        self.norm_l2_prepared(device, &prepared, vector)
    }
}
