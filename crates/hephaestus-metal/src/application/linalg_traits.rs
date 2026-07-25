//! Fluent dense-matrix traits for the Metal backend.

use crate::application::linalg::{
    det, kron, matexp, matmul, matpow, matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2,
    norm_max, pinv, trace,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;
use hephaestus_core::{HephaestusError, Result};

/// Borrow a rank-2 receiver as a read-only Metal strided matrix operand.
pub trait AsGpuMatrixOperand<'a, T> {
    /// Return the strided operand represented by `self`.
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

/// Matrix products on the Metal device.
pub trait MatrixProduct<T> {
    /// Compute `self · rhs` into a newly allocated Metal buffer.
    fn matmul<'a, R: AsGpuMatrixOperand<'a, T>>(
        &self,
        device: &MetalDevice,
        rhs: &R,
    ) -> Result<MetalBuffer<T>>;
    /// Compute the Kronecker product `self ⊗ rhs`.
    fn kron<'a, R: AsGpuMatrixOperand<'a, T>>(
        &self,
        device: &MetalDevice,
        rhs: &R,
    ) -> Result<MetalBuffer<T>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixProduct<f32> for M {
    #[inline]
    fn matmul<'b, R: AsGpuMatrixOperand<'b, f32>>(
        &self,
        device: &MetalDevice,
        rhs: &R,
    ) -> Result<MetalBuffer<f32>> {
        matmul(device, self.as_operand(), rhs.as_operand())
    }

    #[inline]
    fn kron<'b, R: AsGpuMatrixOperand<'b, f32>>(
        &self,
        device: &MetalDevice,
        rhs: &R,
    ) -> Result<MetalBuffer<f32>> {
        kron(device, self.as_operand(), rhs.as_operand())
    }
}

/// Entrywise matrix norms on the Metal device.
pub trait MatrixNorm<T> {
    /// Compute the entrywise L1 norm.
    fn norm_l1(&self, device: &MetalDevice) -> Result<MetalBuffer<T>>;
    /// Compute the Frobenius norm.
    fn norm_l2(&self, device: &MetalDevice) -> Result<MetalBuffer<T>>;
    /// Compute the maximum entry magnitude.
    fn norm_max(&self, device: &MetalDevice) -> Result<MetalBuffer<T>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixNorm<f32> for M {
    #[inline]
    fn norm_l1(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        norm_l1(device, self.as_operand())
    }

    #[inline]
    fn norm_l2(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        norm_l2(device, self.as_operand())
    }

    #[inline]
    fn norm_max(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        norm_max(device, self.as_operand())
    }
}

/// Matrix factorizations on the Metal device.
#[cfg(feature = "decomposition")]
pub trait MatrixDecompose {
    /// Compute an LU factorization with partial pivoting.
    fn lu(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuLuDecomposition>;
    /// Compute a rank-revealing LU factorization with complete pivoting.
    fn full_piv_lu(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuFullPivLuDecomposition>;
    /// Compute a Householder QR factorization.
    fn qr(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuQrDecomposition>;
    /// Compute a rank-revealing column-pivoted QR factorization.
    fn col_piv_qr(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuColPivQrDecomposition>;
    /// Compute a Cholesky factorization of a symmetric positive-definite matrix.
    fn cholesky(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuCholesky>;
    /// Compute an unpivoted symmetric-indefinite `U D Uᵀ` factorization.
    fn udu(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuUduDecomposition>;
    /// Compute a pivoted Bunch–Kaufman factorization.
    fn bunch_kaufman(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuBunchKaufmanDecomposition>;
    /// Reduce a square matrix to upper Hessenberg form.
    fn hessenberg(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuHessenbergDecomposition>;
    /// Compute a Golub–Kahan bidiagonalization.
    fn bidiagonalize(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuBidiagonalDecomposition>;
    /// Compute a thin singular-value decomposition.
    fn svd(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition>;
    /// Compute a rank-revealing singular-value decomposition.
    fn svd_rank_revealing(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition>;
    /// Compute the singular values of a matrix.
    fn singular_values(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
    /// Compute a symmetric eigendecomposition.
    fn symmetric_eigen(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuSymmetricEigenDecomposition>;
    /// Compute only the eigenvalues of a symmetric matrix.
    fn symmetric_eigenvalues(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
    /// Compute all eigenvalues of a general square matrix.
    fn eigenvalues(&self, device: &MetalDevice) -> Result<MetalBuffer<eunomia::Complex<f32>>>;
    /// Compute a real Schur decomposition.
    fn schur(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuRealSchur>;
}

#[cfg(feature = "decomposition")]
impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixDecompose for M {
    #[inline]
    fn lu(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuLuDecomposition> {
        crate::application::decomposition::lu_decompose(device, self.as_operand())
    }

    #[inline]
    fn full_piv_lu(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuFullPivLuDecomposition> {
        crate::application::decomposition::full_piv_lu(device, self.as_operand())
    }

    #[inline]
    fn qr(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuQrDecomposition> {
        crate::application::decomposition::qr_decompose(device, self.as_operand())
    }

    #[inline]
    fn col_piv_qr(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuColPivQrDecomposition> {
        crate::application::decomposition::col_piv_qr(device, self.as_operand())
    }

    #[inline]
    fn cholesky(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuCholesky> {
        crate::application::decomposition::cholesky_decompose(device, self.as_operand())
    }

    #[inline]
    fn udu(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuUduDecomposition> {
        crate::application::decomposition::udu_decompose(device, self.as_operand())
    }

    #[inline]
    fn bunch_kaufman(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuBunchKaufmanDecomposition> {
        crate::application::decomposition::bunch_kaufman(device, self.as_operand())
    }

    #[inline]
    fn hessenberg(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuHessenbergDecomposition> {
        crate::application::decomposition::hessenberg(device, self.as_operand())
    }

    #[inline]
    fn bidiagonalize(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuBidiagonalDecomposition> {
        crate::application::decomposition::bidiagonalize(device, self.as_operand())
    }

    #[inline]
    fn svd(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition> {
        crate::application::decomposition::svd_decompose(device, self.as_operand())
    }

    #[inline]
    fn svd_rank_revealing(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuSvdDecomposition> {
        crate::application::decomposition::svd_rank_revealing(device, self.as_operand())
    }

    #[inline]
    fn singular_values(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        crate::application::decomposition::singular_values(device, self.as_operand())
    }

    #[inline]
    fn symmetric_eigen(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuSymmetricEigenDecomposition> {
        crate::application::decomposition::symmetric_eigen_jacobi(device, self.as_operand())
    }

    #[inline]
    fn symmetric_eigenvalues(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        crate::application::decomposition::symmetric_eigenvalues_jacobi(device, self.as_operand())
    }

    #[inline]
    fn eigenvalues(&self, device: &MetalDevice) -> Result<MetalBuffer<eunomia::Complex<f32>>> {
        crate::application::decomposition::eigenvalues(device, self.as_operand())
    }

    #[inline]
    fn schur(
        &self,
        device: &MetalDevice,
    ) -> Result<crate::application::decomposition::GpuRealSchur> {
        crate::application::decomposition::schur(device, self.as_operand())
    }
}

/// Direct linear-algebra answers on the Metal device.
pub trait MatrixSolve {
    /// Solve `self · x = rhs` for a square system via LU.
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>>;
    /// Solve an overdetermined system in the least-squares sense via QR.
    fn solve_least_squares(
        &self,
        device: &MetalDevice,
        rhs: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>>;
    /// Compute the matrix inverse.
    fn inv(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
    /// Compute the determinant.
    fn det(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
    /// Compute the Moore–Penrose pseudoinverse.
    fn pinv(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixSolve for M {
    #[inline]
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        #[cfg(feature = "decomposition")]
        {
            let lu = crate::application::decomposition::lu_decompose(device, self.as_operand())?;
            let inner = lu.solve(device.wgpu_device(), rhs.wgpu_buffer())?;
            Ok(MetalBuffer { inner })
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
        device: &MetalDevice,
        rhs: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>> {
        #[cfg(feature = "decomposition")]
        {
            let qr = crate::application::decomposition::qr_decompose(device, self.as_operand())?;
            let inner = qr.solve_least_squares(device.wgpu_device(), rhs.wgpu_buffer())?;
            Ok(MetalBuffer { inner })
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
    fn inv(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        #[cfg(feature = "decomposition")]
        {
            let lu = crate::application::decomposition::lu_decompose(device, self.as_operand())?;
            let inner = lu.inv(device.wgpu_device())?;
            Ok(MetalBuffer { inner })
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
    fn det(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        det(device, self.as_operand())
    }

    #[inline]
    fn pinv(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        pinv(device, self.as_operand())
    }
}

/// Matrix properties on the Metal device.
pub trait MatrixProperties {
    /// Compute the trace.
    fn trace(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
    /// Compute the numerical rank.
    fn rank(&self, device: &MetalDevice) -> Result<usize>;
    /// Compute the numerical rank with an explicit relative tolerance.
    fn rank_with_tolerance(&self, device: &MetalDevice, relative_tolerance: f32) -> Result<usize>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixProperties for M {
    #[inline]
    fn trace(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        trace(device, self.as_operand())
    }

    #[inline]
    fn rank(&self, device: &MetalDevice) -> Result<usize> {
        matrix_rank(device, self.as_operand())
    }

    #[inline]
    fn rank_with_tolerance(&self, device: &MetalDevice, relative_tolerance: f32) -> Result<usize> {
        matrix_rank_with_tolerance(device, self.as_operand(), relative_tolerance)
    }
}

/// Matrix functions on the Metal device.
pub trait MatrixFunction {
    /// Raise a square matrix to a non-negative integer power.
    fn matpow(&self, device: &MetalDevice, exponent: u32) -> Result<MetalBuffer<f32>>;
    /// Compute the matrix exponential.
    fn matexp(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>>;
}

impl<'a, M: AsGpuMatrixOperand<'a, f32>> MatrixFunction for M {
    #[inline]
    fn matpow(&self, device: &MetalDevice, exponent: u32) -> Result<MetalBuffer<f32>> {
        matpow(device, self.as_operand(), exponent)
    }

    #[inline]
    fn matexp(&self, device: &MetalDevice) -> Result<MetalBuffer<f32>> {
        matexp(device, self.as_operand())
    }
}
