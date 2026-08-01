//! Metal implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! Metal's decomposition machinery re-exports WGPU's handle types, which
//! implement the core handle traits only for `WgpuDevice`; the orphan rule
//! forecloses re-implementing them here for `MetalDevice`. Each result is
//! therefore wrapped in a thin newtype holding the WGPU handle plus a
//! rewrapped factor buffer (WGPU buffers clone as handles, so the rewrap is
//! a reference-count bump, not a copy).

use hephaestus_core::{
    CholeskyHandle, ColPivQrHandle, DecompositionOps, FullPivLuHandle, LuHandle, QrHandle, Result,
    StridedView, SymmetricEigenHandle,
};

use crate::application::decomposition::{
    GpuCholesky, GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuLuDecomposition,
    GpuQrDecomposition, GpuSymmetricEigenDecomposition, cholesky_decompose, col_piv_qr,
    full_piv_lu, lu_decompose, qr_decompose, symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Column-pivoted QR result: the WGPU handle wrapped for this device.
pub struct MetalColPivQrDecomposition {
    inner: GpuColPivQrDecomposition,
}

impl ColPivQrHandle<MetalDevice> for MetalColPivQrDecomposition {
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
