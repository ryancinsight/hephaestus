//! Device-neutral dense product operations (ADR 0044).
//!
//! Covers the kernel-product tier of the linalg family: dense matrix
//! multiplication, batched matrix multiplication, and the Kronecker
//! product, each a single device kernel over strided operands. The
//! host-orchestrated compositions (`matexp`, `matpow`, `det`, `pinv`,
//! `matrix_rank`) are staged behind this trio per the ADR.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::error::Result;
use super::view::StridedView;

/// Device-neutral dense products over strided views.
///
/// Implementors are zero-sized per-backend markers, so a bound of
/// `P: DenseProductOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch. Scalar bounds are
/// per-implementation: each backend constrains `T` to its dialect's
/// requirements.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`](crate::KernelDialect::IEEE_SPECIAL_VALUES)
/// (ADR 0043) for what is and is not promised per dialect.
pub trait DenseProductOps<D: ComputeDevice, T: Pod> {
    /// Compute `output = lhs · rhs` for rank-2 operands.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch (including the shared dimension), an
    /// aliased output, a layout validation failure, or the backend
    /// dispatch failure.
    fn matmul_into(
        &self,
        device: &D,
        lhs: StridedView<'_, D::Buffer<T>, 2>,
        rhs: StridedView<'_, D::Buffer<T>, 2>,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>;

    /// Compute `output[b] = lhs[b] · rhs[b]` for rank-3 batched operands.
    ///
    /// # Errors
    ///
    /// Returns a batch or shape mismatch, an aliased output, a layout
    /// validation failure, or the backend dispatch failure.
    fn batched_matmul_into(
        &self,
        device: &D,
        lhs: StridedView<'_, D::Buffer<T>, 3>,
        rhs: StridedView<'_, D::Buffer<T>, 3>,
        output: StridedView<'_, D::Buffer<T>, 3>,
    ) -> Result<()>;

    /// Compute the Kronecker product `output = lhs ⊗ rhs`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch against the product shape, an aliased
    /// output, a layout validation failure, or the backend dispatch
    /// failure.
    fn kron_into(
        &self,
        device: &D,
        lhs: StridedView<'_, D::Buffer<T>, 2>,
        rhs: StridedView<'_, D::Buffer<T>, 2>,
        output: StridedView<'_, D::Buffer<T>, 2>,
    ) -> Result<()>;
}
