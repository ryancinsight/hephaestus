//! Device-resident triangular unpacking of a packed LU factorisation.
//!
//! [`hephaestus_core::split_packed_lu`] performs the same split on the host.
//! Callers holding packed factors on CUDA would otherwise download one full
//! matrix and upload two full matrices merely to apply a triangular mask.
//! This module writes **L** and **U** directly from the packed device buffer.
//!
//! The split copies packed entries and writes the structural constants `0`
//! and `1`; it performs no arithmetic, so its result is bitwise identical to
//! the host oracle.

use core::ffi::c_void;
use std::sync::Arc;

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::{CudaBuffer, CudaDevice};

const SPLIT_PACKED_LU_ENTRY: &str = "split_packed_lu_kernel";

fn split_packed_lu_source() -> String {
    r#"extern "C" __global__ void split_packed_lu_kernel(
    const float* packed,
    float* lower,
    float* upper,
    unsigned long long n,
    unsigned long long total
) {
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const unsigned long long row = idx / n;
    const unsigned long long column = idx % n;
    const float value = packed[idx];
    if (row > column) {
        lower[idx] = value;
        upper[idx] = 0.0f;
    } else if (row == column) {
        lower[idx] = 1.0f;
        upper[idx] = value;
    } else {
        lower[idx] = 0.0f;
        upper[idx] = value;
    }
}
"#
    .to_string()
}

/// Split a device-resident packed LU factorisation into explicit dense **L**
/// and **U** buffers without staging either factor through the host.
///
/// `packed` holds the in-place result of a packed LU factorisation of an
/// *n* × *n* row-major matrix. The strictly-lower triangle carries the
/// unit-lower **L** entries (with an implicit diagonal), while the upper
/// triangle including the diagonal carries **U**. The returned buffers are
/// dense row-major matrices with an explicit unit diagonal in **L**.
///
/// This is the CUDA counterpart of [`hephaestus_core::split_packed_lu`]. Every
/// output entry is either copied from `packed` or set to a structural `0`/`1`,
/// so the two operations agree bitwise.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when `n * n` overflows
/// or `packed` belongs to another CUDA device, and
/// [`HephaestusError::LengthMismatch`] when its length differs from `n * n`.
/// Allocation, kernel compilation, grid planning, and launch failures retain
/// their provider error variants.
pub fn split_packed_lu(
    device: &CudaDevice,
    packed: &CudaBuffer<f32>,
    n: usize,
) -> Result<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    let total = n
        .checked_mul(n)
        .ok_or_else(|| HephaestusError::InvalidConfiguration {
            message: format!("packed LU dimension {n} overflows an element count"),
        })?;
    if packed.len() != total {
        return Err(HephaestusError::LengthMismatch {
            host_len: total,
            device_len: packed.len(),
        });
    }
    if !packed
        .context
        .as_ref()
        .is_some_and(|context| Arc::ptr_eq(context, device.cuda_context()))
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: "packed LU buffer must belong to the dispatch device".to_string(),
        });
    }
    if total == 0 {
        return Ok((
            device.alloc_uninitialized::<f32>(0)?,
            device.alloc_uninitialized::<f32>(0)?,
        ));
    }

    let mut n_arg = u64::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("packed LU dimension {n} exceeds CUDA's 64-bit index range"),
    })?;
    let mut total_arg = u64::try_from(total).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("packed LU element count {total} exceeds CUDA's 64-bit index range"),
    })?;
    let width = BlockWidth::DEFAULT;
    let launch = LaunchConfig::linear(grid_size(total, width)?, width);
    let kernel = cached_kernel(
        device,
        PipelineKey::SplitPackedLu,
        SPLIT_PACKED_LU_ENTRY,
        split_packed_lu_source,
    )?;

    // Every invocation below writes one element of each output, so neither
    // allocation needs a zero-initialization pass.
    let lower = device.alloc_uninitialized::<f32>(total)?;
    let upper = device.alloc_uninitialized::<f32>(total)?;

    let mut packed_ptr = packed.raw();
    let mut lower_ptr = lower.raw();
    let mut upper_ptr = upper.raw();
    let mut args: [*mut c_void; 5] = [
        (&raw mut packed_ptr).cast(),
        (&raw mut lower_ptr).cast(),
        (&raw mut upper_ptr).cast(),
        (&raw mut n_arg).cast(),
        (&raw mut total_arg).cast(),
    ];
    launch_kernel(device, &kernel, launch, &mut args)?;

    Ok((lower, upper))
}
