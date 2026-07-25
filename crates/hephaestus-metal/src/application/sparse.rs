//! CSR sparse matrix operations delegated through native Metal-selected WGPU.

use bytemuck::Pod;
use hephaestus_core::{DialectScalar, Result, Wgsl};
use hephaestus_wgpu as wgpu_backend;

use crate::application::strided::{StridedOperand, to_wgpu_strided};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// A CSR matrix whose values and index metadata reside on the Metal device.
#[derive(Clone, Debug)]
pub struct GpuCsrMatrix<T> {
    inner: wgpu_backend::GpuCsrMatrix<T>,
}

impl<T> GpuCsrMatrix<T>
where
    T: DialectScalar<Wgsl> + Pod + leto_ops::Scalar,
{
    /// Upload a CPU-side Leto CSR matrix to the Metal device.
    ///
    /// # Errors
    ///
    /// Returns a typed upload or CSR-index conversion error.
    pub fn from_cpu(device: &MetalDevice, matrix: &leto_ops::CsrMatrix<T>) -> Result<Self> {
        Ok(Self {
            inner: wgpu_backend::GpuCsrMatrix::from_cpu(device.wgpu_device(), matrix)?,
        })
    }

    /// Download the Metal-resident CSR matrix to a CPU Leto matrix.
    ///
    /// # Errors
    ///
    /// Returns a typed download or CSR reconstruction error.
    pub fn to_cpu(&self, device: &MetalDevice) -> Result<leto_ops::CsrMatrix<T>> {
        self.inner.to_cpu(device.wgpu_device())
    }
}

impl<T> GpuCsrMatrix<T> {
    /// Return `(rows, columns)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Return the number of stored non-zero values.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }
}

/// Prepared Metal CSR matrix-vector product for repeated dispatch.
pub struct PreparedSpmv<T> {
    inner: wgpu_backend::PreparedSpmv<T>,
}

impl<T> PreparedSpmv<T> {
    /// Dispatch the prepared sparse matrix-vector product.
    pub fn dispatch(&self) {
        self.inner.dispatch();
    }
}

/// Prepared Metal CSR matrix-matrix product for repeated dispatch.
pub struct PreparedSpmm<T> {
    inner: wgpu_backend::PreparedSpmm<T>,
}

impl<T> PreparedSpmm<T> {
    /// Dispatch the prepared sparse matrix-matrix product.
    pub fn dispatch(&self) {
        self.inner.dispatch();
    }
}

/// A prepared Metal sparse operation in the closed batchable operation set.
pub enum PreparedSparseDispatch<'a, T> {
    /// Prepared CSR matrix-vector product.
    Spmv(&'a PreparedSpmv<T>),
    /// Prepared CSR matrix-matrix product.
    Spmm(&'a PreparedSpmm<T>),
}

/// Submit prepared Metal sparse operations in one native WGPU command batch.
///
/// # Errors
///
/// Returns a typed error when the operations belong to different underlying
/// WGPU devices.
pub fn submit_prepared_sparse_batch<T: DialectScalar<Wgsl> + Pod>(
    operations: &[PreparedSparseDispatch<'_, T>],
) -> Result<()> {
    let operations = operations
        .iter()
        .map(|operation| match operation {
            PreparedSparseDispatch::Spmv(operation) => {
                wgpu_backend::PreparedSparseDispatch::Spmv(&operation.inner)
            }
            PreparedSparseDispatch::Spmm(operation) => {
                wgpu_backend::PreparedSparseDispatch::Spmm(&operation.inner)
            }
        })
        .collect::<Vec<_>>();
    wgpu_backend::submit_prepared_sparse_batch(&operations)
}

/// Prepare `y = A · x` for repeated dispatch into fixed Metal storage.
pub fn prepare_spmv<T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    x: &MetalBuffer<T>,
    output: &mut MetalBuffer<T>,
) -> Result<PreparedSpmv<T>>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    Ok(PreparedSpmv {
        inner: wgpu_backend::prepare_spmv(
            device.wgpu_device(),
            &matrix.inner,
            x.wgpu_buffer(),
            &mut output.inner,
        )?,
    })
}

/// Prepare `C = A · B` for repeated dispatch into fixed Metal storage.
pub fn prepare_spmm<'a, T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    rhs: &StridedOperand<'a, T, 2>,
    output: &mut MetalBuffer<T>,
) -> Result<PreparedSpmm<T>>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    let rhs = to_wgpu_strided(*rhs);
    Ok(PreparedSpmm {
        inner: wgpu_backend::prepare_spmm(
            device.wgpu_device(),
            &matrix.inner,
            &rhs,
            &mut output.inner,
        )?,
    })
}

/// Prepare multiple RHS SpMV using the shared sparse-dense kernel.
#[inline]
pub fn prepare_spmv_many<'a, T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    rhs: &StridedOperand<'a, T, 2>,
    output: &mut MetalBuffer<T>,
) -> Result<PreparedSpmm<T>>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    prepare_spmm(device, matrix, rhs, output)
}

/// Compute `y = A · x`, allocating the Metal output.
pub fn spmv<T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    x: &MetalBuffer<T>,
) -> Result<MetalBuffer<T>>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    Ok(MetalBuffer {
        inner: wgpu_backend::spmv(device.wgpu_device(), &matrix.inner, x.wgpu_buffer())?,
    })
}

/// Compute `y = A · x` into a caller-owned Metal buffer.
pub fn spmv_into<T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    x: &MetalBuffer<T>,
    output: &mut MetalBuffer<T>,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    wgpu_backend::spmv_into(
        device.wgpu_device(),
        &matrix.inner,
        x.wgpu_buffer(),
        &mut output.inner,
    )
}

/// Compute `C = A · B`, allocating the Metal output.
pub fn spmm<'a, T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    rhs: &StridedOperand<'a, T, 2>,
) -> Result<MetalBuffer<T>>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    let rhs = to_wgpu_strided(*rhs);
    Ok(MetalBuffer {
        inner: wgpu_backend::spmm(device.wgpu_device(), &matrix.inner, &rhs)?,
    })
}

/// Compute `C = A · B` into a caller-owned Metal buffer.
pub fn spmm_into<'a, T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    rhs: &StridedOperand<'a, T, 2>,
    output: &mut MetalBuffer<T>,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    let rhs = to_wgpu_strided(*rhs);
    wgpu_backend::spmm_into(device.wgpu_device(), &matrix.inner, &rhs, &mut output.inner)
}

/// Compute multiple RHS SpMV, allocating the Metal output batch.
#[inline]
pub fn spmv_many<'a, T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    rhs: &StridedOperand<'a, T, 2>,
) -> Result<MetalBuffer<T>>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    spmm(device, matrix, rhs)
}

/// Compute multiple RHS SpMV into a caller-owned Metal output batch.
#[inline]
pub fn spmv_many_into<'a, T>(
    device: &MetalDevice,
    matrix: &GpuCsrMatrix<T>,
    rhs: &StridedOperand<'a, T, 2>,
    output: &mut MetalBuffer<T>,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + wgpu_backend::MatmulZero + Pod,
{
    spmm_into(device, matrix, rhs, output)
}
