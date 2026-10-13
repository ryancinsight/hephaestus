use hephaestus_core::{DecompositionOps, Result, StridedView};

use crate::application::decomposition::{
    bidiagonalize, bunch_kaufman, cholesky_decompose, col_piv_qr, eigenvalues, full_piv_lu,
    hessenberg, lu_decompose, qr_decompose, schur, singular_values, svd_decompose,
    symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi, udu_decompose,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

use super::basic::{MetalCholeskyDecomposition, MetalLuDecomposition, MetalQrDecomposition};
use super::orthogonal::{
    MetalBidiagonalDecomposition, MetalHessenbergDecomposition, MetalRealSchur,
};
use super::pivoted::{MetalColPivQrDecomposition, MetalFullPivLuDecomposition};
use super::spectral::{MetalSvdDecomposition, MetalSymmetricEigenDecomposition};
use super::symmetric::{MetalBunchKaufmanDecomposition, MetalUduDecomposition};

/// Dense decompositions for one Metal device.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalDecompositionOps;

impl DecompositionOps<MetalDevice> for MetalDecompositionOps {
    type Lu<'op> = MetalLuDecomposition;
    type Qr<'op> = MetalQrDecomposition;
    type Cholesky<'op> = MetalCholeskyDecomposition;

    fn lu<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>> {
        let inner = lu_decompose(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let factors = MetalBuffer {
            inner: inner.factors().clone(),
        };
        Ok(MetalLuDecomposition { inner, factors })
    }

    fn qr<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>> {
        let inner = qr_decompose(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let r = MetalBuffer {
            inner: inner.r_buffer().clone(),
        };
        Ok(MetalQrDecomposition { inner, r })
    }

    type ColPivQr<'op> = MetalColPivQrDecomposition;
    type FullPivLu<'op> = MetalFullPivLuDecomposition;

    fn col_piv_qr<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>> {
        Ok(MetalColPivQrDecomposition {
            inner: col_piv_qr(
                device,
                StridedOperand {
                    buffer: input.buffer,
                    layout: input.layout,
                },
            )?,
        })
    }

    fn full_piv_lu<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>> {
        let inner = full_piv_lu(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let factors = MetalBuffer {
            inner: inner.lu_buffer().clone(),
        };
        Ok(MetalFullPivLuDecomposition { inner, factors })
    }

    type SymmetricEigen<'op> = MetalSymmetricEigenDecomposition;

    fn symmetric_eigen<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::SymmetricEigen<'op>> {
        let inner = symmetric_eigen_jacobi(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let eigenvalues = MetalBuffer {
            inner: inner.eigenvalues().clone(),
        };
        let eigenvectors = MetalBuffer {
            inner: inner.eigenvectors().clone(),
        };
        Ok(MetalSymmetricEigenDecomposition {
            inner,
            eigenvalues,
            eigenvectors,
        })
    }

    fn symmetric_eigenvalues(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<f32>, 2>,
    ) -> Result<MetalBuffer<f32>> {
        symmetric_eigenvalues_jacobi(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )
    }

    type Svd<'op> = MetalSvdDecomposition;

    fn svd<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Svd<'op>> {
        let inner = svd_decompose(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let u = MetalBuffer {
            inner: inner.u().clone(),
        };
        let v = MetalBuffer {
            inner: inner.v().clone(),
        };
        let singular_values = MetalBuffer {
            inner: inner.singular_values().clone(),
        };
        Ok(MetalSvdDecomposition {
            inner,
            u,
            v,
            singular_values,
        })
    }

    fn singular_values(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<f32>, 2>,
    ) -> Result<MetalBuffer<f32>> {
        singular_values(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )
    }

    type BunchKaufman<'op> = MetalBunchKaufmanDecomposition;
    type Udu<'op> = MetalUduDecomposition;

    fn bunch_kaufman<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::BunchKaufman<'op>> {
        let inner = bunch_kaufman(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let l = MetalBuffer {
            inner: inner.l_buffer().clone(),
        };
        let d = MetalBuffer {
            inner: inner.d_buffer().clone(),
        };
        Ok(MetalBunchKaufmanDecomposition { inner, l, d })
    }

    fn udu<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Udu<'op>> {
        let inner = udu_decompose(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let u = MetalBuffer {
            inner: inner.u_buffer().clone(),
        };
        let d = MetalBuffer {
            inner: inner.d_buffer().clone(),
        };
        Ok(MetalUduDecomposition { inner, u, d })
    }

    type Schur<'op> = MetalRealSchur;
    type Hessenberg<'op> = MetalHessenbergDecomposition;
    type Bidiagonal<'op> = MetalBidiagonalDecomposition;

    fn eigenvalues(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<f32>, 2>,
    ) -> Result<MetalBuffer<eunomia::Complex<f32>>> {
        eigenvalues(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )
    }

    fn schur<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Schur<'op>> {
        let inner = schur(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let q = MetalBuffer {
            inner: inner.q_buffer().clone(),
        };
        let t = MetalBuffer {
            inner: inner.t_buffer().clone(),
        };
        Ok(MetalRealSchur { inner, q, t })
    }

    fn hessenberg<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Hessenberg<'op>> {
        let inner = hessenberg(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let q = MetalBuffer {
            inner: inner.q_buffer().clone(),
        };
        let h = MetalBuffer {
            inner: inner.h_buffer().clone(),
        };
        Ok(MetalHessenbergDecomposition { inner, q, h })
    }

    fn bidiagonalize<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Bidiagonal<'op>> {
        let inner = bidiagonalize(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let u = MetalBuffer {
            inner: inner.u_buffer().clone(),
        };
        let b = MetalBuffer {
            inner: inner.b_buffer().clone(),
        };
        let v = MetalBuffer {
            inner: inner.v_buffer().clone(),
        };
        Ok(MetalBidiagonalDecomposition { inner, u, b, v })
    }

    fn cholesky<'op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        let inner = cholesky_decompose(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
        )?;
        let lower = MetalBuffer {
            inner: inner.lower().clone(),
        };
        Ok(MetalCholeskyDecomposition { inner, lower })
    }
}
