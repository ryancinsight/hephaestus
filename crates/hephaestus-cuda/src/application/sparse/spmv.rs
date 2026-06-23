//! GPU-resident sparse matrix–vector product `y = A · x` on CUDA CSR buffers.

use super::GpuCsrMatrix;
use crate::application::cuda_type::CudaScalar;
use crate::application::pipeline::{cached_kernel, grid_size};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

struct SparseSpmvKernel<T>(PhantomData<T>);

fn spmv_shader_source<T: CudaScalar>() -> String {
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
    for (unsigned int idx = begin; idx < end; idx++) {{
        acc += values[idx] * x[col_indices[idx]];
    }}
    y[row] = acc;
}}
"#,
        ty = T::CUDA_TYPE
    )
}

/// Compute `y = A · x` into a pre-allocated output buffer `y` (length `nrows`).
pub fn spmv_into<T: CudaScalar + leto_ops::Scalar + Pod>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    x: &CudaBuffer<T>,
    y: &mut CudaBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = a.shape();
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

    let width = BlockWidth::DEFAULT;
    let grid = grid_size(nrows, width)?;

    let key = format!(
        "spmv_{}_{}",
        std::any::type_name::<SparseSpmvKernel<T>>(),
        std::any::type_name::<T>()
    );

    let kernel = cached_kernel(device, key, "spmv_kernel", || spmv_shader_source::<T>())?;

    #[cfg(feature = "cuda")]
    {
        device.bind()?;
        let mut values_ptr = a.values().raw();
        let mut col_indices_ptr = a.col_indices().raw();
        let mut row_ptr_ptr = a.row_ptr().raw();
        let mut x_ptr = x.raw();
        let mut y_ptr = y.raw();
        let mut nrows_val = nrows as u32;

        let mut args: [*mut std::ffi::c_void; 6] = [
            &mut values_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut col_indices_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut row_ptr_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut x_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut y_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut nrows_val as *mut u32 as *mut std::ffi::c_void,
        ];

        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                grid,
                1,
                1,
                width.get(),
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuLaunchKernel failed with code: {res}"),
                });
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, grid, width);
    }

    Ok(())
}

/// Compute `y = A · x`, allocating the result buffer.
pub fn spmv<T: CudaScalar + leto_ops::Scalar + Pod>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    x: &CudaBuffer<T>,
) -> Result<CudaBuffer<T>> {
    let (nrows, _) = a.shape();
    let mut y = device.alloc_zeroed::<T>(nrows)?;
    spmv_into(device, a, x, &mut y)?;
    Ok(y)
}
