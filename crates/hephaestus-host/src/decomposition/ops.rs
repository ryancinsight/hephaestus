use hephaestus_core::{ComputeDevice, DecompositionOps, Result, StridedView};

use crate::operands::{upload_array, with_matrix_view};
use crate::{HostBuffer, HostDevice};

use super::basic::HostCholesky;
use super::basic::HostLu;
use super::basic::HostQr;
use super::orthogonal::HostBidiagonal;
use super::orthogonal::HostHessenberg;
use super::orthogonal::HostSchur;
use super::pivoted::HostColPivQr;
use super::pivoted::HostFullPivLu;
use super::spectral::HostSvd;
use super::spectral::HostSymmetricEigen;
use super::symmetric::HostBunchKaufman;
use super::symmetric::HostUdu;
use super::types::HostDecompositionOps;

impl DecompositionOps<HostDevice> for HostDecompositionOps {
    type Lu<'op> = HostLu;
    type Qr<'op> = HostQr;
    type Cholesky<'op> = HostCholesky;
    type ColPivQr<'op> = HostColPivQr;
    type FullPivLu<'op> = HostFullPivLu;
    type SymmetricEigen<'op> = HostSymmetricEigen;
    type Svd<'op> = HostSvd;
    type Schur<'op> = HostSchur;
    type Hessenberg<'op> = HostHessenberg;
    type Bidiagonal<'op> = HostBidiagonal;
    type BunchKaufman<'op> = HostBunchKaufman;
    type Udu<'op> = HostUdu;

    fn lu<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>> {
        let inner = with_matrix_view(&input, |view| leto_ops::lu_decompose(&view))?;
        let factors = upload_array(device, inner.factors())?;
        Ok(HostLu { inner, factors })
    }

    fn qr<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>> {
        let inner = with_matrix_view(&input, |view| leto_ops::qr_decompose(&view))?;
        let r = upload_array(device, &inner.r())?;
        Ok(HostQr { inner, r })
    }

    fn cholesky<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        let inner = with_matrix_view(&input, |view| leto_ops::cholesky_decompose(&view))?;
        let lower = upload_array(device, inner.lower())?;
        Ok(HostCholesky { inner, lower })
    }

    fn col_piv_qr<'op>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>> {
        let shape = (input.layout.shape()[0], input.layout.shape()[1]);
        let inner = with_matrix_view(&input, |view| leto_ops::col_piv_qr(&view))?;
        Ok(HostColPivQr { inner, shape })
    }

    fn full_piv_lu<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_matrix_view(&input, |view| leto_ops::full_piv_lu(&view))?;
        let factors = device.upload(inner.lu_factors())?;
        Ok(HostFullPivLu {
            inner,
            order,
            factors,
        })
    }

    fn symmetric_eigen<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::SymmetricEigen<'op>> {
        let inner = with_matrix_view(&input, |view| leto_ops::symmetric_eigen_jacobi(&view))?;
        Ok(HostSymmetricEigen {
            order: inner.eigenvalues.len(),
            eigenvalues: device.upload(&inner.eigenvalues)?,
            eigenvectors: upload_array(device, &inner.eigenvectors)?,
        })
    }

    fn symmetric_eigenvalues(
        &self,
        device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let values =
            with_matrix_view(&input, |view| leto_ops::symmetric_eigenvalues_jacobi(&view))?;
        device.upload(&values)
    }

    fn svd<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Svd<'op>> {
        let shape = (input.layout.shape()[0], input.layout.shape()[1]);
        let inner = with_matrix_view(&input, |view| leto_ops::svd_decompose(&view))?;
        Ok(HostSvd {
            shape,
            u: upload_array(device, &inner.left_singular_vectors)?,
            v: upload_array(device, &inner.right_singular_vectors)?,
            singular_values: device.upload(&inner.singular_values)?,
        })
    }

    fn singular_values(
        &self,
        device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let values = with_matrix_view(&input, |view| leto_ops::singular_values(&view))?;
        device.upload(&values)
    }

    fn eigenvalues(
        &self,
        device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<eunomia::Complex<f32>>> {
        let values = with_matrix_view(&input, |view| leto_ops::eigenvalues(&view))?;
        device.upload(&values)
    }

    fn schur<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Schur<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_matrix_view(&input, |view| leto_ops::schur(&view))?;
        Ok(HostSchur {
            order,
            q: upload_array(device, &inner.q())?,
            t: upload_array(device, &inner.t())?,
        })
    }

    fn hessenberg<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Hessenberg<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_matrix_view(&input, |view| leto_ops::hessenberg(&view))?;
        Ok(HostHessenberg {
            order,
            q: upload_array(device, inner.q())?,
            h: upload_array(device, inner.h())?,
        })
    }

    fn bidiagonalize<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Bidiagonal<'op>> {
        let shape = (input.layout.shape()[0], input.layout.shape()[1]);
        let inner = with_matrix_view(&input, |view| leto_ops::bidiagonalize(&view))?;
        Ok(HostBidiagonal {
            shape,
            u: upload_array(device, inner.u())?,
            b: upload_array(device, inner.b())?,
            v: upload_array(device, inner.v())?,
        })
    }

    fn bunch_kaufman<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::BunchKaufman<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_matrix_view(&input, |view| leto_ops::bunch_kaufman(&view))?;
        Ok(HostBunchKaufman {
            order,
            l: upload_array(device, &inner.l())?,
            d: upload_array(device, &inner.d())?,
            inner,
        })
    }

    fn udu<'op>(
        &self,
        device: &HostDevice,
        input: StridedView<'op, HostBuffer<f32>, 2>,
    ) -> Result<Self::Udu<'op>> {
        let order = input.layout.shape()[0];
        let inner = with_matrix_view(&input, |view| leto_ops::udu_decompose(&view))?;
        Ok(HostUdu {
            order,
            u: upload_array(device, &inner.u())?,
            d: device.upload(inner.diagonal())?,
            inner,
        })
    }
}
