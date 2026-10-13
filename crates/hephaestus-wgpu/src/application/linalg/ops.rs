//! Operand extension traits for GPU matrices.

use super::{
    det, kron, matmul, matpow, matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max,
    trace,
};
#[cfg(any(feature = "decomposition", feature = "sparse"))]
use super::{matexp, pinv};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
#[cfg(not(feature = "decomposition"))]
use hephaestus_core::HephaestusError;
use hephaestus_core::Result;

/// Borrow any rank-2 receiver as a read-only `StridedOperand<'_, T, 2>`.
pub trait AsGpuMatrixOperand<'a, T> {
    /// Return the strided operand.
    fn as_operand(&self) -> StridedOperand<'a, T, 2>;
}

impl<'a, T> AsGpuMatrixOperand<'a, T> for StridedOperand<'a, T, 2> {
    #[inline]
    fn as_operand(&self) -> StridedOperand<'a, T, 2> {
        StridedOperand {
            buffer: self.buffer,
            layout: self.layout,
        }
    }
}

/// Matrix product surface on the GPU.
pub trait MatrixProduct<T> {
    /// Matrix multiply `self · rhs`, allocating a new buffer.
    fn matmul<'a, R: AsGpuMatrixOperand<'a, T>>(
        &self,
        device: &WgpuDevice,
        rhs: &R,
    ) -> Result<WgpuBuffer<T>>;
    /// Kronecker (tensor) product `self ⊗ rhs`, shape `[m·p, n·q]`.
    fn kron<'a, R: AsGpuMatrixOperand<'a, T>>(
        &self,
        device: &WgpuDevice,
        rhs: &R,
    ) -> Result<WgpuBuffer<T>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixProduct<f32> for M {
    #[inline]
    fn matmul<'b, R: AsGpuMatrixOperand<'b, f32>>(
        &self,
        device: &WgpuDevice,
        rhs: &R,
    ) -> Result<WgpuBuffer<f32>> {
        matmul(device, self.as_operand(), rhs.as_operand())
    }
    #[inline]
    fn kron<'b, R: AsGpuMatrixOperand<'b, f32>>(
        &self,
        device: &WgpuDevice,
        rhs: &R,
    ) -> Result<WgpuBuffer<f32>> {
        kron(device, self.as_operand(), rhs.as_operand())
    }
}

/// Matrix norms on the GPU.
pub trait MatrixNorm<T> {
    /// Entrywise L1 norm `Σ |aᵢⱼ|`.
    fn norm_l1(&self, device: &WgpuDevice) -> Result<WgpuBuffer<T>>;
    /// Frobenius (entrywise L2) norm `sqrt(Σ aᵢⱼ²)`.
    fn norm_l2(&self, device: &WgpuDevice) -> Result<WgpuBuffer<T>>;
    /// Max-magnitude norm `max |aᵢⱼ|`.
    fn norm_max(&self, device: &WgpuDevice) -> Result<WgpuBuffer<T>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixNorm<f32> for M {
    #[inline]
    fn norm_l1(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        norm_l1(device, self.as_operand())
    }
    #[inline]
    fn norm_l2(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        norm_l2(device, self.as_operand())
    }
    #[inline]
    fn norm_max(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        norm_max(device, self.as_operand())
    }
}

/// Matrix factorizations on the GPU.
#[cfg(feature = "decomposition")]
pub trait MatrixDecompose {
    /// LU decomposition with partial pivoting (`P·A = L·U`).
    fn lu(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuLuDecomposition>;
    /// LU with complete (full) pivoting (`P A Q = L U`); rank-revealing.
    fn full_piv_lu(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuFullPivLuDecomposition>;
    /// Householder QR decomposition (`A = Q·R`).
    fn qr(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuQrDecomposition>;
    /// Column-pivoted (rank-revealing) QR (`A P = Q R`).
    fn col_piv_qr(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuColPivQrDecomposition>;
    /// Cholesky factorization of a symmetric positive-definite matrix.
    fn cholesky(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuCholesky>;
    /// Symmetric indefinite unpivoted `U D Uᵀ` factorization.
    fn udu(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuUduDecomposition>;
    /// Stable Bunch–Kaufman factorization.
    fn bunch_kaufman(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuBunchKaufmanDecomposition>;
    /// Upper Hessenberg reduction.
    fn hessenberg(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuHessenbergDecomposition>;
    /// Golub–Kahan bidiagonalization.
    fn bidiagonalize(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuBidiagonalDecomposition>;
    /// Thin SVD.
    fn svd(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition>;
    /// Rank-revealing SVD.
    fn svd_rank_revealing(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition>;
    /// Singular values.
    fn singular_values(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
    /// Symmetric eigendecomposition.
    fn symmetric_eigen(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuSymmetricEigenDecomposition>;
    /// Symmetric eigenvalues only.
    fn symmetric_eigenvalues(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
    /// All eigenvalues of a general (non-symmetric) matrix.
    fn eigenvalues(&self, device: &WgpuDevice) -> Result<WgpuBuffer<eunomia::Complex<f32>>>;
    /// Real Schur decomposition.
    fn schur(&self, device: &WgpuDevice)
    -> Result<crate::application::decomposition::GpuRealSchur>;
}

#[cfg(feature = "decomposition")]
impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixDecompose for M {
    #[inline]
    fn lu(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuLuDecomposition> {
        crate::application::decomposition::lu_decompose(device, self.as_operand())
    }
    #[inline]
    fn full_piv_lu(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuFullPivLuDecomposition> {
        crate::application::decomposition::full_piv_lu(device, self.as_operand())
    }
    #[inline]
    fn qr(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuQrDecomposition> {
        crate::application::decomposition::qr_decompose(device, self.as_operand())
    }
    #[inline]
    fn col_piv_qr(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuColPivQrDecomposition> {
        crate::application::decomposition::col_piv_qr(device, self.as_operand())
    }
    #[inline]
    fn cholesky(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuCholesky> {
        crate::application::decomposition::cholesky_decompose(device, self.as_operand())
    }
    #[inline]
    fn udu(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuUduDecomposition> {
        crate::application::decomposition::udu_decompose(device, self.as_operand())
    }
    #[inline]
    fn bunch_kaufman(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuBunchKaufmanDecomposition> {
        crate::application::decomposition::bunch_kaufman(device, self.as_operand())
    }
    #[inline]
    fn hessenberg(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuHessenbergDecomposition> {
        crate::application::decomposition::hessenberg(device, self.as_operand())
    }
    #[inline]
    fn bidiagonalize(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuBidiagonalDecomposition> {
        crate::application::decomposition::bidiagonalize(device, self.as_operand())
    }
    #[inline]
    fn svd(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition> {
        crate::application::decomposition::svd_decompose(device, self.as_operand())
    }
    #[inline]
    fn svd_rank_revealing(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition> {
        crate::application::decomposition::svd_rank_revealing(device, self.as_operand())
    }
    #[inline]
    fn singular_values(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        crate::application::decomposition::singular_values(device, self.as_operand())
    }
    #[inline]
    fn symmetric_eigen(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuSymmetricEigenDecomposition> {
        crate::application::decomposition::symmetric_eigen_jacobi(device, self.as_operand())
    }
    #[inline]
    fn symmetric_eigenvalues(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        crate::application::decomposition::symmetric_eigenvalues_jacobi(device, self.as_operand())
    }
    #[inline]
    fn eigenvalues(&self, device: &WgpuDevice) -> Result<WgpuBuffer<eunomia::Complex<f32>>> {
        crate::application::decomposition::eigenvalues(device, self.as_operand())
    }
    #[inline]
    fn schur(
        &self,
        device: &WgpuDevice,
    ) -> Result<crate::application::decomposition::GpuRealSchur> {
        crate::application::decomposition::schur(device, self.as_operand())
    }
}

/// Direct linear-algebra answers (solve / inverse / determinant / pseudoinverse) on the GPU.
pub trait MatrixSolve {
    /// Solve `self · x = rhs` for a square system via LU.
    fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>>;
    /// Least-squares solution of an overdetermined system via QR.
    fn solve_least_squares(
        &self,
        device: &WgpuDevice,
        rhs: &WgpuBuffer<f32>,
    ) -> Result<WgpuBuffer<f32>>;
    /// Matrix inverse.
    fn inv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
    /// Determinant.
    fn det(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
    /// Moore-Penrose pseudoinverse.
    fn pinv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixSolve for M {
    #[inline]
    fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        #[cfg(feature = "decomposition")]
        {
            let lu = crate::application::decomposition::lu_decompose(device, self.as_operand())?;
            lu.solve(device, rhs)
        }
        #[cfg(not(feature = "decomposition"))]
        {
            let _ = (device, rhs);
            Err(HephaestusError::DispatchFailed {
                message: "solve requires the decomposition feature".to_string(),
            })
        }
    }
    #[inline]
    fn solve_least_squares(
        &self,
        device: &WgpuDevice,
        rhs: &WgpuBuffer<f32>,
    ) -> Result<WgpuBuffer<f32>> {
        #[cfg(feature = "decomposition")]
        {
            let qr = crate::application::decomposition::qr_decompose(device, self.as_operand())?;
            qr.solve_least_squares(device, rhs)
        }
        #[cfg(not(feature = "decomposition"))]
        {
            let _ = (device, rhs);
            Err(HephaestusError::DispatchFailed {
                message: "solve_least_squares requires the decomposition feature".to_string(),
            })
        }
    }
    #[inline]
    fn inv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        #[cfg(feature = "decomposition")]
        {
            let lu = crate::application::decomposition::lu_decompose(device, self.as_operand())?;
            lu.inv(device)
        }
        #[cfg(not(feature = "decomposition"))]
        {
            let _ = device;
            Err(HephaestusError::DispatchFailed {
                message: "inv requires the decomposition feature".to_string(),
            })
        }
    }
    #[inline]
    fn det(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        det(device, self.as_operand())
    }
    #[inline]
    fn pinv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        #[cfg(any(feature = "decomposition", feature = "sparse"))]
        {
            pinv(device, self.as_operand())
        }
        #[cfg(not(any(feature = "decomposition", feature = "sparse")))]
        {
            let _ = device;
            Err(HephaestusError::DispatchFailed {
                message: "pinv requires the decomposition or sparse feature".to_string(),
            })
        }
    }
}

/// Matrix properties on the GPU.
pub trait MatrixProperties {
    /// Trace.
    fn trace(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
    /// Numerical rank.
    fn rank(&self, device: &WgpuDevice) -> Result<usize>;
    /// Numerical rank with an explicit tolerance.
    fn rank_with_tolerance(&self, device: &WgpuDevice, relative_tolerance: f32) -> Result<usize>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixProperties for M {
    #[inline]
    fn trace(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        trace(device, self.as_operand())
    }
    #[inline]
    fn rank(&self, device: &WgpuDevice) -> Result<usize> {
        matrix_rank(device, self.as_operand())
    }
    #[inline]
    fn rank_with_tolerance(&self, device: &WgpuDevice, relative_tolerance: f32) -> Result<usize> {
        matrix_rank_with_tolerance(device, self.as_operand(), relative_tolerance)
    }
}

/// Matrix functions on the GPU.
pub trait MatrixFunction {
    /// Integer matrix power.
    fn matpow(&self, device: &WgpuDevice, exponent: u32) -> Result<WgpuBuffer<f32>>;
    /// Matrix exponential.
    fn matexp(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixFunction for M {
    #[inline]
    fn matpow(&self, device: &WgpuDevice, exponent: u32) -> Result<WgpuBuffer<f32>> {
        matpow(device, self.as_operand(), exponent)
    }
    #[inline]
    fn matexp(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        #[cfg(any(feature = "decomposition", feature = "sparse"))]
        {
            matexp(device, self.as_operand())
        }
        #[cfg(not(any(feature = "decomposition", feature = "sparse")))]
        {
            let _ = device;
            Err(HephaestusError::DispatchFailed {
                message: "matexp requires the decomposition or sparse feature".to_string(),
            })
        }
    }
}
