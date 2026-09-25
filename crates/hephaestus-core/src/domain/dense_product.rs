//! Device-neutral dense product operations (ADR 0044).
//!
//! Covers three roles in the linalg family. [`crate::DenseProductOps`] owns the
//! single-kernel products: dense matrix multiplication, batched matrix
//! multiplication, and the Kronecker product. [`crate::DenseCompositionOps`]
//! owns matrix power, determinant, and numerical rank.
//! [`crate::DenseMatrixFunctionOps`] owns the feature-gated host-delegated
//! matrix exponential and pseudoinverse.

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

/// Device-neutral matrix compositions built on provider linalg machinery.
///
/// This role is deliberately separate from [`DenseProductOps`]: kernel
/// products are direct strided dispatches, while matrix power, determinant,
/// and rank own provider scheduling and scalar policy. Keeping the roles
/// separate prevents one backend's composition implementation from becoming a
/// breaking requirement for providers that only implement the kernel tier.
///
/// Implementors are zero-sized backend markers. Calls monomorphize to the
/// provider entry point and introduce no virtual dispatch or adapter state.
pub trait DenseCompositionOps<D: ComputeDevice> {
    /// Compute `matrix^exponent`, with `matrix^0` equal to the identity.
    ///
    /// # Errors
    ///
    /// Returns a shape, layout, allocation, or backend dispatch error.
    fn matpow(
        &self,
        device: &D,
        matrix: StridedView<'_, D::Buffer<f32>, 2>,
        exponent: u32,
    ) -> Result<D::Buffer<f32>>;

    /// Compute the determinant of a square matrix.
    ///
    /// # Errors
    ///
    /// Returns a shape, layout, or backend dispatch error.
    fn det(&self, device: &D, matrix: StridedView<'_, D::Buffer<f32>, 2>)
    -> Result<D::Buffer<f32>>;

    /// Estimate numerical rank using a relative pivot threshold.
    ///
    /// # Errors
    ///
    /// Returns a layout or backend dispatch error.
    fn matrix_rank_with_tolerance(
        &self,
        device: &D,
        matrix: StridedView<'_, D::Buffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize>;

    /// Estimate numerical rank using Leto's default relative tolerance.
    ///
    /// # Errors
    ///
    /// Returns a layout or backend dispatch error.
    fn matrix_rank(&self, device: &D, matrix: StridedView<'_, D::Buffer<f32>, 2>) -> Result<usize> {
        self.matrix_rank_with_tolerance(device, matrix, 1.0e-9)
    }
}

/// Device-neutral matrix functions delegated to Leto by the provider.
///
/// This role is separate from [`DenseCompositionOps`] because WGPU exposes
/// these functions only when `decomposition` or `sparse` enables `leto-ops`.
/// Keeping the feature-gated pair separate preserves the unconditional
/// power/determinant/rank role without adding a default unsupported method or
/// widening WGPU's minimal feature surface. Implementors remain zero-sized
/// backend markers with static dispatch.
pub trait DenseMatrixFunctionOps<D: ComputeDevice> {
    /// Compute the matrix exponential `exp(matrix)`.
    ///
    /// # Errors
    ///
    /// Returns a shape, layout, host-compute, allocation, or backend dispatch
    /// error.
    fn matexp(
        &self,
        device: &D,
        matrix: StridedView<'_, D::Buffer<f32>, 2>,
    ) -> Result<D::Buffer<f32>>;

    /// Compute the Moore-Penrose pseudoinverse of a rectangular or square
    /// matrix.
    ///
    /// # Errors
    ///
    /// Returns a layout, host-compute, allocation, or backend dispatch error.
    fn pinv(
        &self,
        device: &D,
        matrix: StridedView<'_, D::Buffer<f32>, 2>,
    ) -> Result<D::Buffer<f32>>;
}
