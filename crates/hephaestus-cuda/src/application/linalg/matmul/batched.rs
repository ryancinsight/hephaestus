use eunomia::Pod;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result};
use leto::Layout;

use crate::application::linalg::{map_layout, map_layout_err, to_i64, to_u32};
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::{CudaBuffer, CudaDevice};

use super::shaders::{BatchedMatmulKernel, batched_matmul_shader_source};

/// CUDA's hardware grid.z dimension limit (fixed at 65535 across every
/// current compute capability, unlike grid.x/y which scale with it).
const MAX_GRID_Z: u32 = 65_535;

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` on the CUDA device.
///
/// Dispatches the batch dimension via `blockIdx.z` in one kernel launch (or
/// the minimum number of launches CUDA's grid.z limit requires — batches
/// beyond `MAX_GRID_Z` chunk into further launches, each still covering up
/// to 65535 batch elements) rather than one `matmul_into` launch per batch
/// element. `lhs`/`rhs` broadcast when their batch dimension is 1.
pub fn batched_matmul_into<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
    out: StridedOperand<'_, T, 3>,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [lhs_batch, m, lhs_k] = lhs.layout.shape();
    let [rhs_batch, rhs_k, n] = rhs.layout.shape();
    let [out_batch, out_m, out_n] = out.layout.shape();

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k || m != out_m || n != out_n {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape(),
                rhs.layout.shape(),
                out.layout.shape()
            ),
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

    // Empty output is a no-op. A zero shared extent still dispatches: the
    // kernel runs zero tiles and writes the additive identity to every
    // logical output cell without reading either operand.
    if batch == 0 || m == 0 || n == 0 {
        return Ok(());
    }

    // The kernel uses an unsigned 32-bit batch index, including chunk offsets.
    // Validate the complete extent before compiling or dispatching any chunk.
    let batch_words = to_u32(batch, "batched matmul batch count")?;

    let lhs_batch_stride = if lhs_batch == 1 {
        0
    } else {
        lhs.layout.strides()[0]
    };
    let rhs_batch_stride = if rhs_batch == 1 {
        0
    } else {
        rhs.layout.strides()[0]
    };
    let out_batch_stride = out.layout.strides()[0];

    // Per-matrix layout at batch index 0; the kernel adds `batch_idx *
    // {a,b,c}_batch_stride` on the device side, so the host only maps the
    // base (rank-2) slice once.
    let lhs_mat_layout = Layout::try_new(
        [m, lhs_k],
        [lhs.layout.strides()[1], lhs.layout.strides()[2]],
        lhs.layout.offset(),
    )
    .expect("invariant: submatrix layout derives from a validated parent");
    let rhs_mat_layout = Layout::try_new(
        [rhs_k, n],
        [rhs.layout.strides()[1], rhs.layout.strides()[2]],
        rhs.layout.offset(),
    )
    .expect("invariant: submatrix layout derives from a validated parent");
    let out_mat_layout = Layout::try_new(
        [out_m, out_n],
        [out.layout.strides()[1], out.layout.strides()[2]],
        out.layout.offset(),
    )
    .expect("invariant: submatrix layout derives from a validated parent");

    let a_meta = map_layout(&lhs_mat_layout)?;
    let b_meta = map_layout(&rhs_mat_layout)?;
    let c_meta = map_layout(&out_mat_layout)?;

    let a_batch_stride = to_i64(lhs_batch_stride, "lhs batch stride")?;
    let b_batch_stride = to_i64(rhs_batch_stride, "rhs batch stride")?;
    let c_batch_stride = to_i64(out_batch_stride, "out batch stride")?;

    let key = PipelineKey::BatchedMatmul {
        marker: std::any::TypeId::of::<BatchedMatmulKernel<T>>(),
        scalar: std::any::TypeId::of::<T>(),
    };
    let kernel = cached_kernel(device, key, "batched_matmul_kernel", || {
        batched_matmul_shader_source::<T>()
    })?;

    let workgroups_x = b_meta.shape[1].div_ceil(16);
    let workgroups_y = a_meta.shape[0].div_ceil(16);

    let mut a_ptr = lhs.buffer.raw();
    let mut b_ptr = rhs.buffer.raw();
    let mut c_ptr = out.buffer.raw();
    let mut a_meta_val = a_meta;
    let mut b_meta_val = b_meta;
    let mut c_meta_val = c_meta;
    let mut a_batch_stride_val = a_batch_stride;
    let mut b_batch_stride_val = b_batch_stride;
    let mut c_batch_stride_val = c_batch_stride;

    let mut batch_offset = 0_u32;
    while batch_offset < batch_words {
        let chunk = (batch_words - batch_offset).min(MAX_GRID_Z);
        let mut batch_offset_val = batch_offset;

        // Argument list mirrors `batched_matmul_kernel(const T*, const T*, T*,
        // MatrixLayout, MatrixLayout, MatrixLayout, long long, long long,
        // long long, unsigned int)`.
        let mut args: [*mut std::ffi::c_void; 10] = [
            std::ptr::from_mut(&mut a_ptr).cast(),
            std::ptr::from_mut(&mut b_ptr).cast(),
            std::ptr::from_mut(&mut c_ptr).cast(),
            std::ptr::from_mut(&mut a_meta_val).cast(),
            std::ptr::from_mut(&mut b_meta_val).cast(),
            std::ptr::from_mut(&mut c_meta_val).cast(),
            std::ptr::from_mut(&mut a_batch_stride_val).cast(),
            std::ptr::from_mut(&mut b_batch_stride_val).cast(),
            std::ptr::from_mut(&mut c_batch_stride_val).cast(),
            std::ptr::from_mut(&mut batch_offset_val).cast(),
        ];

        launch_kernel(
            device,
            &kernel,
            LaunchConfig::batched_planar(workgroups_x, workgroups_y, chunk, 16, 16),
            &mut args,
        )?;

        batch_offset += chunk;
    }

    Ok(())
}

/// Allocate and compute batched matrix multiplication on the CUDA device.
///
/// Singleton batches broadcast to the other operand's batch count. The returned
/// buffer has C-contiguous shape `[batch, lhs.rows, rhs.cols]`.
pub fn batched_matmul<T>(
    device: &CudaDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let [lhs_batch, m, lhs_k] = lhs.layout.shape();
    let [rhs_batch, rhs_k, n] = rhs.layout.shape();
    let batch = lhs_batch.max(rhs_batch);
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape(),
                rhs.layout.shape()
            ),
        });
    }
    let out_layout = Layout::c_contiguous([batch, m, n]).map_err(map_layout_err)?;
    let out =
        device.alloc_uninitialized::<T>(out_layout.checked_size().map_err(map_layout_err)?)?;
    batched_matmul_into(
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
