//! Metal implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! Metal's decomposition machinery re-exports WGPU's handle types, which
//! implement the core handle traits only for `WgpuDevice`; the orphan rule
//! forecloses re-implementing them here for `MetalDevice`. Each result is
//! therefore wrapped in a thin newtype holding the WGPU handle plus a
//! rewrapped factor buffer (WGPU buffers clone as handles, so the rewrap is
//! a reference-count bump, not a copy).

use hephaestus_core::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, DecompositionOps,
    FullPivLuHandle, HessenbergHandle, LuHandle, QrHandle, Result, SchurHandle, StridedView,
    SvdHandle, SymmetricEigenHandle, UduHandle,
};

use crate::application::decomposition::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition, bidiagonalize, bunch_kaufman,
    cholesky_decompose, col_piv_qr, eigenvalues, full_piv_lu, hessenberg, lu_decompose,
    qr_decompose, schur, singular_values, svd_decompose, symmetric_eigen_jacobi,
    symmetric_eigenvalues_jacobi, udu_decompose,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Column-pivoted QR result: the WGPU handle wrapped for this device.
pub struct MetalColPivQrDecomposition {
    inner: GpuColPivQrDecomposition,
}

impl ColPivQrHandle<MetalDevice> for MetalColPivQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn permutation(&self) -> &[usize] {
        self.inner.permutation()
    }
    fn solve_least_squares(
        &self,
        device: &MetalDevice,
        rhs: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve_least_squares(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

/// Fully pivoted LU result: the WGPU handle plus its factors rewrapped
/// for this device.
pub struct MetalFullPivLuDecomposition {
    inner: GpuFullPivLuDecomposition,
    factors: MetalBuffer<f32>,
}

impl FullPivLuHandle<MetalDevice> for MetalFullPivLuDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn factors(&self) -> &MetalBuffer<f32> {
        &self.factors
    }
    fn row_permutation(&self) -> &[usize] {
        self.inner.row_permutation()
    }
    fn col_permutation(&self) -> &[usize] {
        self.inner.col_permutation()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

/// Symmetric eigendecomposition result: the WGPU handle plus its factors
/// rewrapped for this device.
pub struct MetalSymmetricEigenDecomposition {
    inner: GpuSymmetricEigenDecomposition,
    eigenvalues: MetalBuffer<f32>,
    eigenvectors: MetalBuffer<f32>,
}

impl SymmetricEigenHandle<MetalDevice> for MetalSymmetricEigenDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn eigenvalues(&self) -> &MetalBuffer<f32> {
        &self.eigenvalues
    }
    fn eigenvectors(&self) -> &MetalBuffer<f32> {
        &self.eigenvectors
    }
}

/// SVD result: the WGPU handle plus its factors rewrapped for this
/// device.
pub struct MetalSvdDecomposition {
    inner: GpuSvdDecomposition,
    u: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
    singular_values: MetalBuffer<f32>,
}

impl SvdHandle<MetalDevice> for MetalSvdDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn u(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn v(&self) -> &MetalBuffer<f32> {
        &self.v
    }
    fn singular_values(&self) -> &MetalBuffer<f32> {
        &self.singular_values
    }
}

/// Bunch–Kaufman result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalBunchKaufmanDecomposition {
    inner: GpuBunchKaufmanDecomposition,
    l: MetalBuffer<f32>,
    d: MetalBuffer<f32>,
}

impl BunchKaufmanHandle<MetalDevice> for MetalBunchKaufmanDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn l_buffer(&self) -> &MetalBuffer<f32> {
        &self.l
    }
    fn d_buffer(&self) -> &MetalBuffer<f32> {
        &self.d
    }
    fn permutation(&self) -> &[usize] {
        self.inner.permutation()
    }
}

/// `U·D·Uᵀ` result: the WGPU handle plus its factors rewrapped for this
/// device.
pub struct MetalUduDecomposition {
    inner: GpuUduDecomposition,
    u: MetalBuffer<f32>,
    d: MetalBuffer<f32>,
}

impl UduHandle<MetalDevice> for MetalUduDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn u_buffer(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn d_buffer(&self) -> &MetalBuffer<f32> {
        &self.d
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

/// Real Schur result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalRealSchur {
    inner: GpuRealSchur,
    q: MetalBuffer<f32>,
    t: MetalBuffer<f32>,
}

impl SchurHandle<MetalDevice> for MetalRealSchur {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn q_buffer(&self) -> &MetalBuffer<f32> {
        &self.q
    }
    fn t_buffer(&self) -> &MetalBuffer<f32> {
        &self.t
    }
}

/// Hessenberg result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalHessenbergDecomposition {
    inner: GpuHessenbergDecomposition,
    q: MetalBuffer<f32>,
    h: MetalBuffer<f32>,
}

impl HessenbergHandle<MetalDevice> for MetalHessenbergDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn q_buffer(&self) -> &MetalBuffer<f32> {
        &self.q
    }
    fn h_buffer(&self) -> &MetalBuffer<f32> {
        &self.h
    }
}

/// Bidiagonal result: the WGPU handle plus its factors rewrapped for
/// this device.
pub struct MetalBidiagonalDecomposition {
    inner: GpuBidiagonalDecomposition,
    u: MetalBuffer<f32>,
    b: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
}

impl BidiagonalHandle<MetalDevice> for MetalBidiagonalDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn u_buffer(&self) -> &MetalBuffer<f32> {
        &self.u
    }
    fn b_buffer(&self) -> &MetalBuffer<f32> {
        &self.b
    }
    fn v_buffer(&self) -> &MetalBuffer<f32> {
        &self.v
    }
}

/// Dense decompositions for one Metal device.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalDecompositionOps;

/// LU result: the WGPU handle plus its factors rewrapped for this device.
pub struct MetalLuDecomposition {
    inner: GpuLuDecomposition,
    factors: MetalBuffer<f32>,
}

/// QR result: the WGPU handle plus its `R` rewrapped for this device.
pub struct MetalQrDecomposition {
    inner: GpuQrDecomposition,
    r: MetalBuffer<f32>,
}

/// Cholesky result: the WGPU handle plus its `L` rewrapped for this device.
pub struct MetalCholeskyDecomposition {
    inner: GpuCholesky,
    lower: MetalBuffer<f32>,
}

impl LuHandle<MetalDevice> for MetalLuDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn factors(&self) -> &MetalBuffer<f32> {
        &self.factors
    }
    fn pivots(&self) -> &[usize] {
        self.inner.pivots()
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

impl QrHandle<MetalDevice> for MetalQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    fn r_buffer(&self) -> &MetalBuffer<f32> {
        &self.r
    }
    fn solve_least_squares(
        &self,
        device: &MetalDevice,
        rhs: &MetalBuffer<f32>,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve_least_squares(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

impl CholeskyHandle<MetalDevice> for MetalCholeskyDecomposition {
    fn order(&self) -> usize {
        self.inner.n()
    }
    fn lower(&self) -> &MetalBuffer<f32> {
        &self.lower
    }
    fn det(&self) -> f32 {
        self.inner.det()
    }
    fn solve(&self, device: &MetalDevice, rhs: &MetalBuffer<f32>) -> Result<MetalBuffer<f32>> {
        self.inner
            .solve(device.wgpu_device(), &rhs.inner)
            .map(|inner| MetalBuffer { inner })
    }
}

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
