//! Reusable ROCm CSR sparse products.

use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result};

use super::GpuCsrMatrix;
use crate::RocmDevice;
use crate::application::linalg::AsGpuMatrixOperand;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// Prepared ROCm CSR matrix-vector product for repeated dispatch.
pub struct PreparedSpmv<'a, T> {
    device: &'a RocmDevice,
    matrix: &'a GpuCsrMatrix<T>,
    x: &'a RocmBuffer<T>,
    output: &'a RocmBuffer<T>,
    kernel: Arc<RocmKernel>,
    width: BlockWidth,
    grid: u32,
    nrows: u32,
}

impl<T: DialectScalar<HipC> + leto_ops::Scalar + Pod> PreparedSpmv<'_, T> {
    /// Dispatch the prepared CSR matrix-vector product.
    ///
    /// # Errors
    ///
    /// Returns a typed native HIP launch error.
    pub fn dispatch(&self) -> Result<()> {
        let mut values_ptr: DevicePtr = self.matrix.values().raw();
        let mut col_indices_ptr: DevicePtr = self.matrix.col_indices().raw();
        let mut row_ptr_ptr: DevicePtr = self.matrix.row_ptr().raw();
        let mut x_ptr: DevicePtr = self.x.raw();
        let mut output_ptr: DevicePtr = self.output.raw();
        let mut nrows = self.nrows;
        let mut args: [*mut core::ffi::c_void; 6] = [
            (&mut values_ptr as *mut DevicePtr).cast(),
            (&mut col_indices_ptr as *mut DevicePtr).cast(),
            (&mut row_ptr_ptr as *mut DevicePtr).cast(),
            (&mut x_ptr as *mut DevicePtr).cast(),
            (&mut output_ptr as *mut DevicePtr).cast(),
            (&mut nrows as *mut u32).cast(),
        ];
        launch_kernel(
            self.device,
            &self.kernel,
            LaunchConfig::linear(self.grid, self.width),
            &mut args,
        )
    }

    pub(crate) fn device(&self) -> &RocmDevice {
        self.device
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpmmMeta {
    rows: u32,
    cols: u32,
    b_stride_row: i32,
    b_stride_col: i32,
    b_offset: u32,
}

/// Prepared ROCm CSR matrix-matrix product for repeated dispatch.
pub struct PreparedSpmm<'a, T> {
    device: &'a RocmDevice,
    matrix: &'a GpuCsrMatrix<T>,
    rhs: StridedOperand<'a, T, 2>,
    output: &'a RocmBuffer<T>,
    kernel: Arc<RocmKernel>,
    width: BlockWidth,
    grid: u32,
    meta: SpmmMeta,
}

impl<T: DialectScalar<HipC> + leto_ops::Scalar + Pod> PreparedSpmm<'_, T> {
    /// Dispatch the prepared CSR matrix-matrix product.
    ///
    /// # Errors
    ///
    /// Returns a typed native HIP launch error.
    pub fn dispatch(&self) -> Result<()> {
        let mut meta = self.meta;
        let mut values_ptr: DevicePtr = self.matrix.values().raw();
        let mut col_indices_ptr: DevicePtr = self.matrix.col_indices().raw();
        let mut row_ptr_ptr: DevicePtr = self.matrix.row_ptr().raw();
        let mut rhs_ptr: DevicePtr = self.rhs.buffer.raw();
        let mut output_ptr: DevicePtr = self.output.raw();
        let mut args: [*mut core::ffi::c_void; 6] = [
            (&mut meta as *mut SpmmMeta).cast(),
            (&mut values_ptr as *mut DevicePtr).cast(),
            (&mut col_indices_ptr as *mut DevicePtr).cast(),
            (&mut row_ptr_ptr as *mut DevicePtr).cast(),
            (&mut rhs_ptr as *mut DevicePtr).cast(),
            (&mut output_ptr as *mut DevicePtr).cast(),
        ];
        launch_kernel(
            self.device,
            &self.kernel,
            LaunchConfig::linear(self.grid, self.width),
            &mut args,
        )
    }

    pub(crate) fn device(&self) -> &RocmDevice {
        self.device
    }
}

/// A prepared ROCm sparse operation in the closed batchable operation set.
pub enum PreparedSparseDispatch<'plan, 'device, T> {
    /// Prepared CSR matrix-vector product.
    Spmv(&'plan PreparedSpmv<'device, T>),
    /// Prepared CSR matrix-matrix product.
    Spmm(&'plan PreparedSpmm<'device, T>),
}

impl<T: DialectScalar<HipC> + leto_ops::Scalar + Pod> PreparedSparseDispatch<'_, '_, T> {
    fn device(&self) -> &RocmDevice {
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

/// Submit prepared ROCm sparse operations in order on their native stream.
///
/// # Errors
///
/// Returns an error when operations belong to different HIP contexts or when
/// a native launch fails.
pub fn submit_prepared_sparse_batch<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
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
            message: "prepared sparse batch contains operations from different HIP contexts"
                .to_string(),
        });
    }
    for operation in operations {
        operation.dispatch()?;
    }
    Ok(())
}

/// Prepare `y = A · x` for repeated dispatch into a fixed output buffer.
pub fn prepare_spmv<'a, T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &'a RocmDevice,
    matrix: &'a GpuCsrMatrix<T>,
    x: &'a RocmBuffer<T>,
    output: &'a mut RocmBuffer<T>,
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
            marker: core::any::TypeId::of::<super::spmv::SparseSpmvKernel<T>>(),
            scalar: core::any::TypeId::of::<T>(),
        },
        "spmv_kernel",
        super::spmv::shader_source::<T>,
    )?;
    Ok(PreparedSpmv {
        device,
        matrix,
        x,
        output: &*output,
        kernel,
        width,
        grid,
        nrows,
    })
}

/// Prepare `C = A · B` for repeated dispatch into a fixed output buffer.
pub fn prepare_spmm<'a, T, B>(
    device: &'a RocmDevice,
    matrix: &'a GpuCsrMatrix<T>,
    rhs: &'a B,
    output: &'a mut RocmBuffer<T>,
) -> Result<PreparedSpmm<'a, T>>
where
    T: DialectScalar<HipC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
{
    let rhs = rhs.as_operand();
    let (nrows, ncols) = matrix.shape();
    let [rhs_rows, rhs_cols] = rhs.layout.shape;
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
        .map_err(super::spmm::map_layout_error)?;
    let meta = SpmmMeta {
        rows: super::spmm::to_u32(nrows, "CSR row count")?,
        cols: super::spmm::to_u32(rhs_cols, "dense RHS column count")?,
        b_stride_row: super::spmm::to_i32(rhs.layout.strides[0], "dense RHS row stride")?,
        b_stride_col: super::spmm::to_i32(rhs.layout.strides[1], "dense RHS column stride")?,
        b_offset: super::spmm::to_u32(rhs.layout.offset, "dense RHS offset")?,
    };
    let width = BlockWidth::DEFAULT;
    let grid = grid_size(output_len, width)?;
    let kernel = cached_kernel(
        device,
        PipelineKey::Spmm {
            marker: core::any::TypeId::of::<super::spmm::SparseSpmmKernel<T>>(),
            scalar: core::any::TypeId::of::<T>(),
        },
        "spmm_kernel",
        super::spmm::shader_source::<T>,
    )?;
    Ok(PreparedSpmm {
        device,
        matrix,
        rhs,
        output: &*output,
        kernel,
        width,
        grid,
        meta,
    })
}

/// Prepare multiple RHS SpMV using the shared sparse-dense kernel.
#[inline]
pub fn prepare_spmv_many<'a, T, B>(
    device: &'a RocmDevice,
    matrix: &'a GpuCsrMatrix<T>,
    rhs: &'a B,
    output: &'a mut RocmBuffer<T>,
) -> Result<PreparedSpmm<'a, T>>
where
    T: DialectScalar<HipC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
{
    prepare_spmm(device, matrix, rhs, output)
}
