//! Sparse-dense matrix product `C = A · B` over ROCm CSR buffers.

use super::GpuCsrMatrix;
use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result,
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

const _: () = assert!(core::mem::size_of::<SpmmMeta>() == 20);

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

fn map_layout_error(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {error}"),
    }
}

fn shader_source<T: DialectScalar<HipC>>() -> String {
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
    unsigned int length = meta.rows * meta.cols;
    if (flat >= length) {{
        return;
    }}

    unsigned int row = flat / meta.cols;
    unsigned int col = flat - row * meta.cols;
    unsigned int begin = row_ptr[row];
    unsigned int end = row_ptr[row + 1u];
    {ty} acc = 0;
    for (unsigned int index = begin; index < end; index++) {{
        unsigned int b_row = col_indices[index];
        int b_offset = (int)meta.b_offset
            + (int)b_row * meta.b_stride_row
            + (int)col * meta.b_stride_col;
        acc += values[index] * b[b_offset];
    }}
    c[flat] = acc;
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

/// Compute `C = A · B` into a caller-owned row-major ROCm buffer.
///
/// `B` may use any validated rank-2 Leto layout. Its rows must match the CSR
/// column count, and `c` must contain `A.rows() * B.columns()` elements.
///
/// # Errors
///
/// Returns a typed shape, layout, aliasing, dispatch, module-compilation, or
/// HIP launch error when the operands violate the sparse-dense contract.
pub fn spmm_into<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &RocmDevice,
    matrix: &GpuCsrMatrix<T>,
    b: StridedOperand<'_, T, 2>,
    c: &mut RocmBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = matrix.shape();
    let [b_rows, bcols] = b.layout.shape;
    if b_rows != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: b_rows,
        });
    }
    let expected_c_len =
        nrows
            .checked_mul(bcols)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("SpMM output length overflows usize: {nrows} * {bcols}"),
            })?;
    if c.len() != expected_c_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: expected_c_len,
            device_len: c.len(),
        });
    }
    b.layout
        .validate_storage_len(b.buffer.len())
        .map_err(map_layout_error)?;
    if b.buffer.aliases(c) || matrix.values().aliases(c) {
        return Err(HephaestusError::DispatchFailed {
            message: "SpMM output buffer must not alias an input buffer".to_string(),
        });
    }
    if expected_c_len == 0 {
        return Ok(());
    }

    let meta = SpmmMeta {
        rows: to_u32(nrows, "CSR row count")?,
        cols: to_u32(bcols, "dense RHS column count")?,
        b_stride_row: to_i32(b.layout.strides[0], "dense RHS row stride")?,
        b_stride_col: to_i32(b.layout.strides[1], "dense RHS column stride")?,
        b_offset: to_u32(b.layout.offset, "dense RHS offset")?,
    };
    let width = BlockWidth::DEFAULT;
    let grid = grid_size(expected_c_len, width)?;
    let key = PipelineKey::Spmm {
        marker: core::any::TypeId::of::<SparseSpmmKernel<T>>(),
        scalar: core::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "spmm_kernel", shader_source::<T>)?;

    let mut meta_arg = meta;
    let mut values_ptr: DevicePtr = matrix.values().raw();
    let mut col_indices_ptr: DevicePtr = matrix.col_indices().raw();
    let mut row_ptr_ptr: DevicePtr = matrix.row_ptr().raw();
    let mut b_ptr: DevicePtr = b.buffer.raw();
    let mut c_ptr: DevicePtr = c.raw();
    let mut args: [*mut core::ffi::c_void; 6] = [
        (&mut meta_arg as *mut SpmmMeta).cast(),
        (&mut values_ptr as *mut DevicePtr).cast(),
        (&mut col_indices_ptr as *mut DevicePtr).cast(),
        (&mut row_ptr_ptr as *mut DevicePtr).cast(),
        (&mut b_ptr as *mut DevicePtr).cast(),
        (&mut c_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, width),
        &mut args,
    )
}

/// Compute `C = A · B`, allocating a row-major ROCm output.
///
/// # Errors
///
/// Returns the typed errors described by [`spmm_into`].
pub fn spmm<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &RocmDevice,
    matrix: &GpuCsrMatrix<T>,
    b: StridedOperand<'_, T, 2>,
) -> Result<RocmBuffer<T>> {
    let (nrows, _) = matrix.shape();
    let bcols = b.layout.shape[1];
    let output_len = nrows
        .checked_mul(bcols)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("SpMM output length overflows usize: {nrows} * {bcols}"),
        })?;
    let mut c = device.alloc_zeroed::<T>(output_len)?;
    spmm_into(device, matrix, b, &mut c)?;
    Ok(c)
}

/// Compute multiple sparse matrix-vector products using the sparse-dense
/// kernel. Each column of `x_batch` is an independent right-hand side.
pub fn spmv_many_into<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &RocmDevice,
    matrix: &GpuCsrMatrix<T>,
    x_batch: StridedOperand<'_, T, 2>,
    y_batch: &mut RocmBuffer<T>,
) -> Result<()> {
    spmm_into(device, matrix, x_batch, y_batch)
}

/// Compute multiple sparse matrix-vector products, allocating the output
/// batch.
pub fn spmv_many<T: DialectScalar<HipC> + leto_ops::Scalar + Pod>(
    device: &RocmDevice,
    matrix: &GpuCsrMatrix<T>,
    x_batch: StridedOperand<'_, T, 2>,
) -> Result<RocmBuffer<T>> {
    spmm(device, matrix, x_batch)
}
