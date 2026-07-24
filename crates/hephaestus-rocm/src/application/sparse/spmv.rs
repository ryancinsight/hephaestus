//! Sparse matrix-vector product `y = A · x` over ROCm CSR buffers.

use super::GpuCsrMatrix;
use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};
use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result,
};

struct SparseSpmvKernel<T>(PhantomData<T>);

fn shader_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"
extern "C" __global__ void spmv_kernel(
    const {ty}* values,
    const unsigned int* col_indices,
    const unsigned int* row_ptr,
    const {ty}* x,
    {ty}* y,
    unsigned int nrows
) {{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {{
        return;
    }}

    unsigned int begin = row_ptr[row];
    unsigned int end = row_ptr[row + 1u];
    {ty} acc = 0;
    for (unsigned int index = begin; index < end; index++) {{
        acc += values[index] * x[col_indices[index]];
    }}
    y[row] = acc;
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

/// Compute `y = A · x` into a caller-owned ROCm buffer.
///
/// # Errors
///
/// Returns a typed length, dispatch, module-compilation, or HIP launch error
/// when the vector or output does not match the CSR matrix contract.
pub fn spmv_into<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &RocmDevice,
    matrix: &GpuCsrMatrix<T>,
    x: &RocmBuffer<T>,
    y: &mut RocmBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = matrix.shape();
    if x.len() != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: x.len(),
        });
    }
    if y.len() != nrows {
        return Err(HephaestusError::LengthMismatch {
            host_len: nrows,
            device_len: y.len(),
        });
    }
    if nrows == 0 {
        return Ok(());
    }

    let nrows_u32 = u32::try_from(nrows).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("CSR row count {nrows} exceeds u32 range"),
    })?;
    let width = BlockWidth::DEFAULT;
    let grid = grid_size(nrows, width)?;
    let key = PipelineKey::Spmv {
        marker: core::any::TypeId::of::<SparseSpmvKernel<T>>(),
        scalar: core::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "spmv_kernel", shader_source::<T>)?;

    let mut values_ptr: DevicePtr = matrix.values().raw();
    let mut col_indices_ptr: DevicePtr = matrix.col_indices().raw();
    let mut row_ptr_ptr: DevicePtr = matrix.row_ptr().raw();
    let mut x_ptr: DevicePtr = x.raw();
    let mut y_ptr: DevicePtr = y.raw();
    let mut nrows_arg = nrows_u32;
    let mut args: [*mut core::ffi::c_void; 6] = [
        (&mut values_ptr as *mut DevicePtr).cast(),
        (&mut col_indices_ptr as *mut DevicePtr).cast(),
        (&mut row_ptr_ptr as *mut DevicePtr).cast(),
        (&mut x_ptr as *mut DevicePtr).cast(),
        (&mut y_ptr as *mut DevicePtr).cast(),
        (&mut nrows_arg as *mut u32).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, width),
        &mut args,
    )
}

/// Compute `y = A · x`, allocating the length-`A.rows()` ROCm output.
///
/// # Errors
///
/// Returns a typed allocation, length, dispatch, module-compilation, or HIP
/// launch error when the vector does not match the CSR matrix contract.
pub fn spmv<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &RocmDevice,
    matrix: &GpuCsrMatrix<T>,
    x: &RocmBuffer<T>,
) -> Result<RocmBuffer<T>> {
    let (nrows, _) = matrix.shape();
    let mut y = device.alloc_zeroed::<T>(nrows)?;
    spmv_into(device, matrix, x, &mut y)?;
    Ok(y)
}
