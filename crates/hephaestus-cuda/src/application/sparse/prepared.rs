//! Reusable CUDA CSR sparse products.

use std::sync::Arc;

use eunomia::Pod;
use hephaestus_core::{BlockWidth, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};

use super::GpuCsrMatrix;
use crate::CudaDevice;
use crate::application::linalg::AsGpuMatrixOperand;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided::map_layout_err;
use crate::infrastructure::buffer::CudaBuffer;

/// Prepared CUDA CSR matrix-vector product for repeated dispatch.
pub struct PreparedSpmv<'a, T> {
    device: &'a CudaDevice,
    matrix: &'a GpuCsrMatrix<T>,
    x: &'a CudaBuffer<T>,
    output: &'a CudaBuffer<T>,
    kernel: Arc<SafeCachedKernel>,
    width: BlockWidth,
    grid: u32,
    nrows: u32,
}

impl<T: DialectScalar<CudaC> + leto_ops::Scalar + Pod> PreparedSpmv<'_, T> {
    /// Dispatch the prepared CSR matrix-vector product.
    ///
    /// # Errors
    ///
    /// Returns a typed native CUDA launch error.
    pub fn dispatch(&self) -> Result<()> {
        let mut values_ptr = self.matrix.values().raw();
        let mut col_indices_ptr = self.matrix.col_indices().raw();
        let mut row_ptr_ptr = self.matrix.row_ptr().raw();
        let mut x_ptr = self.x.raw();
        let mut output_ptr = self.output.raw();
        let mut nrows = self.nrows;
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut values_ptr as *mut u64).cast(),
            (&mut col_indices_ptr as *mut u64).cast(),
            (&mut row_ptr_ptr as *mut u64).cast(),
            (&mut x_ptr as *mut u64).cast(),
            (&mut output_ptr as *mut u64).cast(),
            (&mut nrows as *mut u32).cast(),
        ];
        launch_kernel(
            self.device,
            &self.kernel,
            LaunchConfig::linear(self.grid, self.width),
            &mut args,
        )
    }

    pub(crate) fn device(&self) -> &CudaDevice {
        self.device
    }
}

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct SpmmMeta {
    rows: u32,
    cols: u32,
    b_stride_row: i32,
    b_stride_col: i32,
    b_offset: u32,
}

/// Prepared CUDA CSR matrix-matrix product for repeated dispatch.
pub struct PreparedSpmm<'a, T> {
    device: &'a CudaDevice,
    matrix: &'a GpuCsrMatrix<T>,
    rhs: StridedOperand<'a, T, 2>,
    output: &'a CudaBuffer<T>,
    kernel: Arc<SafeCachedKernel>,
    width: BlockWidth,
    grid: u32,
    meta: SpmmMeta,
}

impl<T: DialectScalar<CudaC> + leto_ops::Scalar + Pod> PreparedSpmm<'_, T> {
    /// Dispatch the prepared CSR matrix-matrix product.
    ///
    /// # Errors
    ///
    /// Returns a typed native CUDA launch error.
    pub fn dispatch(&self) -> Result<()> {
        let mut meta = self.meta;
        let mut values_ptr = self.matrix.values().raw();
        let mut col_indices_ptr = self.matrix.col_indices().raw();
        let mut row_ptr_ptr = self.matrix.row_ptr().raw();
        let mut rhs_ptr = self.rhs.buffer.raw();
        let mut output_ptr = self.output.raw();
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut meta as *mut SpmmMeta).cast(),
            (&mut values_ptr as *mut u64).cast(),
            (&mut col_indices_ptr as *mut u64).cast(),
            (&mut row_ptr_ptr as *mut u64).cast(),
            (&mut rhs_ptr as *mut u64).cast(),
            (&mut output_ptr as *mut u64).cast(),
        ];
        launch_kernel(
            self.device,
            &self.kernel,
            LaunchConfig::linear(self.grid, self.width),
            &mut args,
        )
    }

    pub(crate) fn device(&self) -> &CudaDevice {
        self.device
    }
}

/// A prepared CUDA sparse operation in the closed batchable operation set.
pub enum PreparedSparseDispatch<'plan, 'device, T> {
    /// Prepared CSR matrix-vector product.
    Spmv(&'plan PreparedSpmv<'device, T>),
    /// Prepared CSR matrix-matrix product.
    Spmm(&'plan PreparedSpmm<'device, T>),
}

impl<T: DialectScalar<CudaC> + leto_ops::Scalar + Pod> PreparedSparseDispatch<'_, '_, T> {
    fn device(&self) -> &CudaDevice {
        match self {
            Self::Spmv(operation) => operation.device(),
            Self::Spmm(operation) => operation.device(),
        }
    }

    fn dispatch(&self) -> Result<()> {
        match self {
            Self::Spmv(operation) => operation.dispatch(),
            Self::Spmm(operation) => operation.dispatch(),
        }
    }
}

/// Submit prepared CUDA sparse operations in order on their native stream.
///
/// # Errors
///
/// Returns an error when operations belong to different CUDA contexts or when
/// a native launch fails.
pub fn submit_prepared_sparse_batch<T: DialectScalar<CudaC> + leto_ops::Scalar + Pod>(
    operations: &[PreparedSparseDispatch<'_, '_, T>],
) -> Result<()> {
    let Some((first, rest)) = operations.split_first() else {
        return Ok(());
    };
    if rest
        .iter()
        .any(|operation| !first.device().same_context(operation.device()))
    {
        return Err(HephaestusError::DispatchFailed {
            message: "prepared sparse batch contains operations from different CUDA contexts"
                .to_string(),
        });
    }
    for operation in operations {
        operation.dispatch()?;
    }
    Ok(())
}

/// Prepare `y = A · x` for repeated dispatch into a fixed output buffer.
pub fn prepare_spmv<'a, T: DialectScalar<CudaC> + leto_ops::Scalar + Pod>(
    device: &'a CudaDevice,
    matrix: &'a GpuCsrMatrix<T>,
    x: &'a CudaBuffer<T>,
    output: &'a CudaBuffer<T>,
) -> Result<PreparedSpmv<'a, T>> {
    let (nrows, ncols) = matrix.shape();
    if x.len() != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: x.len(),
        });
    }
    if output.len() != nrows {
        return Err(HephaestusError::LengthMismatch {
            host_len: nrows,
            device_len: output.len(),
        });
    }
    let width = BlockWidth::DEFAULT;
    let grid = grid_size(nrows, width)?;
    let nrows = super::spmm::to_u32(nrows, "CSR row count")?;
    let kernel = cached_kernel(
        device,
        PipelineKey::Spmv {
            marker: std::any::TypeId::of::<super::spmv::SparseSpmvKernel<T>>(),
            scalar: std::any::TypeId::of::<T>(),
        },
        "spmv_kernel",
        super::spmv::spmv_shader_source::<T>,
    )?;
    Ok(PreparedSpmv {
        device,
        matrix,
        x,
        output,
        kernel,
        width,
        grid,
        nrows,
    })
}

/// Prepare `C = A · B` for repeated dispatch into a fixed output buffer.
pub fn prepare_spmm<'a, T, B>(
    device: &'a CudaDevice,
    matrix: &'a GpuCsrMatrix<T>,
    rhs: &'a B,
    output: &'a mut CudaBuffer<T>,
) -> Result<PreparedSpmm<'a, T>>
where
    T: DialectScalar<CudaC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
{
    let rhs = rhs.as_operand();
    let (nrows, ncols) = matrix.shape();
    let [rhs_rows, rhs_cols] = rhs.layout.shape();
    if rhs_rows != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: rhs_rows,
        });
    }
    let output_len =
        nrows
            .checked_mul(rhs_cols)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("spmm output size {nrows}×{rhs_cols} overflows usize"),
            })?;
    if output.len() != output_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: output_len,
            device_len: output.len(),
        });
    }
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    let meta = SpmmMeta {
        rows: super::spmm::to_u32(nrows, "CSR row count")?,
        cols: super::spmm::to_u32(rhs_cols, "dense RHS column count")?,
        b_stride_row: super::spmm::to_i32(rhs.layout.strides()[0], "dense RHS row stride")?,
        b_stride_col: super::spmm::to_i32(rhs.layout.strides()[1], "dense RHS column stride")?,
        b_offset: super::spmm::to_u32(rhs.layout.offset(), "dense RHS offset")?,
    };
    let width = BlockWidth::DEFAULT;
    let grid = grid_size(output_len, width)?;
    let kernel = cached_kernel(
        device,
        PipelineKey::Spmm {
            marker: std::any::TypeId::of::<super::spmm::SparseSpmmKernel<T>>(),
            scalar: std::any::TypeId::of::<T>(),
        },
        "spmm_kernel",
        super::spmm::spmm_shader_source::<T>,
    )?;
    Ok(PreparedSpmm {
        device,
        matrix,
        rhs,
        output,
        kernel,
        width,
        grid,
        meta,
    })
}

/// Prepare multiple RHS SpMV using the shared sparse-dense kernel.
#[inline]
pub fn prepare_spmv_many<'a, T, B>(
    device: &'a CudaDevice,
    matrix: &'a GpuCsrMatrix<T>,
    rhs: &'a B,
    output: &'a mut CudaBuffer<T>,
) -> Result<PreparedSpmm<'a, T>>
where
    T: DialectScalar<CudaC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
{
    prepare_spmm(device, matrix, rhs, output)
}
