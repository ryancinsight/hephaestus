//! GPU-resident sparse–dense matrix product `C = A · B` on CUDA CSR buffers.

use super::GpuCsrMatrix;
use crate::application::linalg::AsGpuMatrixOperand;
use crate::application::pipeline::{
    cached_kernel, grid_size, launch_kernel, LaunchConfig, PipelineKey,
};
use crate::application::strided::map_layout_err;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpmmMeta {
    rows: u32,
    cols: u32,
    b_stride_row: i32,
    b_stride_col: i32,
    b_offset: u32,
}

struct SparseSpmmKernel<T>(PhantomData<T>);

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

fn spmm_shader_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"
struct SpmmMeta {{
    unsigned int rows;
    unsigned int cols;
    int b_stride_row;
    int b_stride_col;
    unsigned int b_offset;
}};

extern "C" __global__ void spmm_kernel(
    SpmmMeta meta,
    const {ty}* values,
    const unsigned int* col_indices,
    const unsigned int* row_ptr,
    const {ty}* b,
    {ty}* c
) {{
    unsigned int flat = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int len = meta.rows * meta.cols;
    if (flat >= len) {{
        return;
    }}

    unsigned int row = flat / meta.cols;
    unsigned int col = flat - row * meta.cols;
    
    unsigned int begin = row_ptr[row];
    unsigned int end = row_ptr[row + 1u];
    {ty} acc = 0;
    for (unsigned int idx = begin; idx < end; idx++) {{
        unsigned int b_row = col_indices[idx];
        int b_offset = (int)meta.b_offset + (int)b_row * meta.b_stride_row + (int)col * meta.b_stride_col;
        acc += values[idx] * b[b_offset];
    }}
    c[flat] = acc;
}}
"#,
        ty = T::TYPE_TOKEN
    )
}

/// Compute `C = A · B` into a pre-allocated output buffer `c`.
pub fn spmm_into<
    'a,
    T: DialectScalar<CudaC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    b: &B,
    c: &mut CudaBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = a.shape();
    let b_op = b.as_operand();
    let [b_rows, bcols] = b_op.layout.shape;

    if b_rows != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: b_rows,
        });
    }

    let expected_c_len = nrows * bcols;
    if c.len() != expected_c_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: expected_c_len,
            device_len: c.len(),
        });
    }
    if expected_c_len == 0 {
        return Ok(());
    }

    b_op.layout
        .validate_storage_len(b_op.buffer.len())
        .map_err(map_layout_err)?;

    let meta = SpmmMeta {
        rows: to_u32(nrows, "CSR row count")?,
        cols: to_u32(bcols, "dense rhs column count")?,
        b_stride_row: to_i32(b_op.layout.strides[0], "dense rhs row stride")?,
        b_stride_col: to_i32(b_op.layout.strides[1], "dense rhs column stride")?,
        b_offset: to_u32(b_op.layout.offset, "dense rhs offset")?,
    };

    let width = BlockWidth::DEFAULT;
    let grid = grid_size(expected_c_len, width)?;

    let key = PipelineKey::Spmm {
        marker: std::any::TypeId::of::<SparseSpmmKernel<T>>(),
        scalar: std::any::TypeId::of::<T>(),
    };

    let kernel = cached_kernel(device, key, "spmm_kernel", || spmm_shader_source::<T>())?;

    let mut meta_val = meta;
    let mut values_ptr = a.values().raw();
    let mut col_indices_ptr = a.col_indices().raw();
    let mut row_ptr_ptr = a.row_ptr().raw();
    let mut b_ptr = b_op.buffer.raw();
    let mut c_ptr = c.raw();

    // Argument list mirrors `spmm_kernel(SpmmMeta, const T*, const unsigned int*,
    // const unsigned int*, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut meta_val as *mut SpmmMeta as *mut std::ffi::c_void,
        &mut values_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut col_indices_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut row_ptr_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut c_ptr as *mut u64 as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, width),
        &mut args,
    )
}

/// Compute multiple sparse matrix-vector products into a pre-allocated output
/// batch.
///
/// Columns of `x_batch` are independent RHS vectors, and columns of `y_batch`
/// are the corresponding `A · x_j` outputs. CUDA uses the same sparse-dense
/// kernel as [`spmm_into`] so multi-RHS SpMV amortizes launch overhead without a
/// duplicate kernel.
pub fn spmv_many_into<
    'a,
    T: DialectScalar<CudaC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    x_batch: &B,
    y_batch: &mut CudaBuffer<T>,
) -> Result<()> {
    spmm_into(device, a, x_batch, y_batch)
}

/// Compute `C = A · B`, allocating the result buffer.
pub fn spmm<'a, T: DialectScalar<CudaC> + leto_ops::Scalar + Pod, B: AsGpuMatrixOperand<'a, T>>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    b: &B,
) -> Result<CudaBuffer<T>> {
    let (nrows, _) = a.shape();
    let b_op = b.as_operand();
    let [_, bcols] = b_op.layout.shape;

    let mut c = device.alloc_zeroed::<T>(nrows * bcols)?;
    spmm_into(device, a, b, &mut c)?;
    Ok(c)
}

/// Compute multiple sparse matrix-vector products, allocating the output batch.
pub fn spmv_many<
    'a,
    T: DialectScalar<CudaC> + leto_ops::Scalar + Pod,
    B: AsGpuMatrixOperand<'a, T>,
>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    x_batch: &B,
) -> Result<CudaBuffer<T>> {
    spmm(device, a, x_batch)
}
