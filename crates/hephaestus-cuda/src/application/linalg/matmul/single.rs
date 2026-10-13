use eunomia::Pod;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};
use leto::Layout;

use crate::application::linalg::{map_layout, map_layout_err};
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::{CudaBuffer, CudaDevice};

use super::shaders::{MatmulKernel, matmul_shader_source};

/// Perform matrix multiplication `out = lhs * rhs` on the CUDA device.
pub fn matmul_into<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [rows, lhs_shared] = lhs.layout.shape();
    let [rhs_shared, cols] = rhs.layout.shape();
    let [out_rows, out_cols] = out.layout.shape();

    if lhs_shared != rhs_shared || rows != out_rows || cols != out_cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape(),
                rhs.layout.shape(),
                out.layout.shape()
            ),
        });
    }

    if lhs.buffer.aliases(out.buffer) || rhs.buffer.aliases(out.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }

    lhs.layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    out.layout
        .validate_storage_len(out.buffer.len())
        .map_err(map_layout_err)?;

    if out.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "matmul output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if rows == 0 || cols == 0 || lhs_shared == 0 {
        return Ok(());
    }

    let a_meta = map_layout(lhs.layout)?;
    let b_meta = map_layout(rhs.layout)?;
    let c_meta = map_layout(out.layout)?;

    let key = PipelineKey::Matmul {
        marker: std::any::TypeId::of::<MatmulKernel<T>>(),
        scalar: std::any::TypeId::of::<T>(),
    };

    let kernel = cached_kernel(device, key, "matmul_kernel", || matmul_shader_source::<T>())?;

    let workgroups_x = cols.div_ceil(16);
    let workgroups_y = rows.div_ceil(16);

    let mut a_ptr = lhs.buffer.raw();
    let mut b_ptr = rhs.buffer.raw();
    let mut c_ptr = out.buffer.raw();
    let mut a_meta_val = a_meta;
    let mut b_meta_val = b_meta;
    let mut c_meta_val = c_meta;

    // Argument list mirrors `matmul_kernel(const T*, const T*, T*, MatrixLayout,
    // MatrixLayout, MatrixLayout)`.
    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut c_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut a_meta_val as *mut crate::application::linalg::GpuMatrixLayout
            as *mut std::ffi::c_void,
        &mut b_meta_val as *mut crate::application::linalg::GpuMatrixLayout
            as *mut std::ffi::c_void,
        &mut c_meta_val as *mut crate::application::linalg::GpuMatrixLayout
            as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(workgroups_x as u32, workgroups_y as u32, 16, 16),
        &mut args,
    )
}

/// Allocate and compute matrix multiplication `lhs * rhs` on the CUDA device.
///
/// The returned buffer has C-contiguous shape `[lhs.rows, rhs.cols]`.
pub fn matmul<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [rows, lhs_shared] = lhs.layout.shape();
    let [rhs_shared, cols] = rhs.layout.shape();
    if lhs_shared != rhs_shared {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape(),
                rhs.layout.shape()
            ),
        });
    }
    let out_layout = Layout::c_contiguous([rows, cols]).map_err(map_layout_err)?;
    let out =
        device.alloc_uninitialized::<T>(out_layout.checked_size().map_err(map_layout_err)?)?;
    matmul_into(
        device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
    )?;
    Ok(out)
}
