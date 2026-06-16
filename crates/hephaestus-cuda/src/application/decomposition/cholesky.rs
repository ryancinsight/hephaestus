//! GPU-resident Cholesky decomposition.
//!
//! Computes **A** = **L** **L**ᵀ for symmetric positive-definite matrices.
//!
//! Two entry points are provided:
//!
//! - [`cholesky_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`cholesky_decompose_blocked`] — blocked algorithm where panel
//!   factorization runs on the CPU but the O(n³) trailing SYRK update
//!   (`A₂₂ -= L₂₁ L₂₁ᵀ`) runs on the GPU via a dedicated compute kernel.

#[cfg(feature = "cuda")]
use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use super::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Lower-triangular Cholesky factor on the device, with host-side
/// decomposition for solve/inv/det without re-factorization.
pub struct GpuCholesky {
    /// Host-side leto-ops decomposition (owns the factor data).
    inner: leto_ops::CholeskyDecomposition<f32>,
    /// Device-resident lower-triangular factor **L** (*n* × *n*, row-major).
    lower: CudaBuffer<f32>,
    n: usize,
}

impl GpuCholesky {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lower(&self) -> &CudaBuffer<f32> {
        &self.lower
    }

    /// Consume and return the lower-triangular factor buffer.
    #[must_use]
    #[inline]
    pub fn into_lower(self) -> CudaBuffer<f32> {
        self.lower
    }

    /// Determinant det(**A**) = Πᵢ Lᵢᵢ² via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side forward/back substitution.
    ///
    /// Downloads the RHS from the device, solves on the host using the
    /// stored decomposition, and uploads the solution vector.
    pub fn solve(&self, device: &CudaDevice, rhs: &CudaBuffer<f32>) -> Result<CudaBuffer<f32>> {
        if rhs.len() != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len(),
            });
        }
        if self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0f32; self.n];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([self.n]).unwrap(),
            &rhs_host,
        );
        let x = self
            .inner
            .solve(&rhs_view)
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky solve failed: {e}"),
            })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }

    /// Compute the inverse **A**⁻¹ via the host-side decomposition.
    pub fn inv(&self, device: &CudaDevice) -> Result<CudaBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inv = self
            .inner
            .inv()
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

/// Compute the Cholesky factorization **A** = **L** **L**ᵀ on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`]. The result is stored on the device for downstream GPU
/// consumers.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let lower = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::cholesky_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([0, 0]).unwrap(),
            &[],
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Cholesky decomposition failed: {e}"),
        })?;
        return Ok(GpuCholesky { inner, lower, n: 0 });
    }

    // Download input to host.
    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    // Create a leto ArrayView over the downloaded data.
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    // Factor on CPU using leto-ops.
    let chol =
        leto_ops::cholesky_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Cholesky decomposition failed: {e}"),
        })?;

    // Upload the lower-triangular factor to the device.
    let lower = device.upload(leto::Storage::as_slice(chol.lower().storage()))?;

    Ok(GpuCholesky {
        inner: chol,
        lower,
        n,
    })
}

// ---------------------------------------------------------------------------
// Blocked Cholesky - GPU Trailing update
// ---------------------------------------------------------------------------

/// Blocked Cholesky factorization **A** = **L** **L**ᵀ with GPU-accelerated
/// trailing-matrix SYRK updates.
pub fn cholesky_decompose_blocked(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    #[cfg(feature = "cuda")]
    {
        let n = validate_square(&matrix)?;
        if n == 0 {
            let lower = device.alloc_zeroed::<f32>(0)?;
            let inner = leto_ops::cholesky_decompose(&leto::ArrayView::<f32, 2>::new(
                leto::Layout::c_contiguous([0, 0]).unwrap(),
                &[],
            ))
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky decomposition failed: {e}"),
            })?;
            return Ok(GpuCholesky { inner, lower, n: 0 });
        }

        // Download the full matrix to host (panels are factored on CPU).
        let mut host = vec![0.0f32; n * n];
        device.download(matrix.buffer, &mut host)?;
        let original_host = host.clone();

        // Device-resident buffer for the packed lower factor, updated in-place
        // by the SYRK trailing kernel.
        let lower_buf = device.upload(&host)?;

        const BLOCK_SIZE: usize = 64;
        let block_size = BLOCK_SIZE.min(n);

        for k in (0..n).step_by(block_size) {
            let b = block_size.min(n - k);

            // ── Step 1: factor the diagonal block A[k..k+b, k..k+b] on CPU ──
            let mut diag_host = vec![0.0f32; b * b];
            for i in 0..b {
                for j in 0..b {
                    diag_host[i * b + j] = host[(k + i) * n + (k + j)];
                }
            }
            let diag_view = leto::ArrayView::<f32, 2>::new(
                leto::Layout::c_contiguous([b, b]).unwrap(),
                &diag_host,
            );
            let diag_chol = leto_ops::cholesky_decompose(&diag_view).map_err(|e| {
                HephaestusError::DispatchFailed {
                    message: format!("Cholesky panel factorisation failed: {e}"),
                }
            })?;
            let diag_lower = diag_chol.lower();
            let diag_slice = leto::Storage::as_slice(diag_lower.storage());

            // Write the factored diagonal block back to the host array and
            // upload to the device buffer.
            for i in 0..b {
                for j in 0..b {
                    host[(k + i) * n + (k + j)] = diag_slice[i * b + j];
                }
            }
            write_device_buffer(device, &host, &lower_buf)?;

            let trail_rows = n - k - b;
            if trail_rows == 0 {
                continue;
            }

            // ── Step 2: panel solve  L₂₁ = A₂₁ · L₁₁⁻ᵀ  on CPU ──
            let mut rhs = vec![0.0f32; trail_rows * b];
            for i in 0..trail_rows {
                for j in 0..b {
                    rhs[i * b + j] = host[(k + b + i) * n + (k + j)];
                }
            }

            for col in 0..b {
                for i in 0..trail_rows {
                    let mut s = rhs[i * b + col];
                    for p in 0..col {
                        s -= rhs[i * b + p] * diag_slice[col * b + p];
                    }
                    rhs[i * b + col] = s / diag_slice[col * b + col];
                }
            }

            // Write L₂₁ back to host and upload to the device.
            for i in 0..trail_rows {
                for j in 0..b {
                    host[(k + b + i) * n + (k + j)] = rhs[i * b + j];
                }
            }
            write_device_buffer(device, &host, &lower_buf)?;
            let panel_buf = device.upload(&rhs)?;

            // ── Step 3: trailing SYRK update on GPU ──
            let trail_layout = leto::Layout::new(
                [trail_rows, trail_rows],
                [n as isize, 1],
                (k + b) * n + (k + b),
            );
            syrk_trailing_update(device, &lower_buf, &trail_layout, &panel_buf, b)?;

            // Download the updated trailing matrix back to host.
            device.download(&lower_buf, &mut host)?;
        }

        for row in 0..n {
            for col in (row + 1)..n {
                host[row * n + col] = 0.0;
            }
        }
        write_device_buffer(device, &host, &lower_buf)?;

        let original_view = leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([n, n]).unwrap(),
            &original_host,
        );
        let inner = leto_ops::cholesky_decompose(&original_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("Cholesky blocked finalisation failed: {e}"),
            }
        })?;

        Ok(GpuCholesky {
            inner,
            lower: lower_buf,
            n,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, matrix);
        Err(HephaestusError::AdapterUnavailable {
            message: "hephaestus-cuda built without the `cuda` feature".to_string(),
        })
    }
}

// Helper structures and functions for SYRK update (only compiled when CUDA is enabled)
#[cfg(feature = "cuda")]
mod syrk_impl {
    use super::*;
    use crate::application::linalg::{to_i32, to_u32};
    use crate::application::pipeline::cached_kernel;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable)]
    pub struct SyrkMeta {
        shape: [u32; 2],
        strides: [i32; 2],
        offset: u32,
        panel_cols: u32,
        _pad: [u32; 2],
    }

    // SAFETY: SyrkMeta is `#[repr(C)]` and every field is Pod.
    unsafe impl Pod for SyrkMeta {}

    pub fn write_device_buffer<T: Pod>(
        device: &CudaDevice,
        host: &[T],
        buffer: &CudaBuffer<T>,
    ) -> Result<()> {
        if host.len() != buffer.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: buffer.len(),
            });
        }
        if host.is_empty() {
            return Ok(());
        }
        device.bind()?;
        let bytes = std::mem::size_of_val(host);
        // SAFETY: `buffer.raw()` is a valid device pointer. `host` is valid host memory of `bytes` size.
        unsafe {
            let res = cuda_core::sys::cuMemcpyHtoD_v2(
                buffer.raw(),
                host.as_ptr() as *const std::ffi::c_void,
                bytes,
            );
            if res != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!("write_device_buffer cuMemcpyHtoD_v2 failed with code: {res}"),
                });
            }
        }
        Ok(())
    }

    fn syrk_shader_source() -> String {
        r#"
    struct SyrkMeta {
        unsigned int shape[2];
        int strides[2];
        unsigned int offset;
        unsigned int panel_cols;
    };

    extern "C" __global__ void syrk_kernel(
        const float* panel,
        float* trail,
        SyrkMeta meta
    ) {
        __shared__ float panel_row_shared[16][16];

        unsigned int col = blockIdx.x * 16u + threadIdx.x;
        unsigned int row = blockIdx.y * 16u + threadIdx.y;
        unsigned int local_col = threadIdx.x;
        unsigned int local_row = threadIdx.y;

        unsigned int rows = meta.shape[0];
        unsigned int cols = meta.shape[1];
        unsigned int k    = meta.panel_cols;
        int stride_row = meta.strides[0];
        int stride_col = meta.strides[1];
        unsigned int off = meta.offset;

        if (row >= rows || col >= cols) {
            return;
        }

        // Only the lower triangle is touched (col <= row).
        if (col > row) {
            return;
        }

        float sum = 0.0f;
        unsigned int num_tiles = (k + 15u) / 16u;

        for (unsigned int tile = 0u; tile < num_tiles; tile++) {
            unsigned int panel_col = tile * 16u + local_col;
            if (panel_col < k) {
                panel_row_shared[local_row][local_col] = panel[row * k + panel_col];
            } else {
                panel_row_shared[local_row][local_col] = 0.0f;
            }

            __syncthreads();

            for (unsigned int i = 0u; i < 16u; i++) {
                unsigned int ki = tile * 16u + i;
                if (ki < k) {
                    float a_val = panel_row_shared[local_row][i];
                    float b_val = panel[col * k + ki];
                    sum += a_val * b_val;
                }
            }

            __syncthreads();
        }

        int c_off = (int)off + (int)row * stride_row + (int)col * stride_col;
        trail[c_off] -= sum;
    }
        "#
        .to_string()
    }

    pub fn syrk_trailing_update(
        device: &CudaDevice,
        trail: &CudaBuffer<f32>,
        trail_layout: &leto::Layout<2>,
        panel: &CudaBuffer<f32>,
        panel_cols: usize,
    ) -> Result<()> {
        let [rows, cols] = trail_layout.shape;
        if rows == 0 || cols == 0 || panel_cols == 0 {
            return Ok(());
        }

        let meta = SyrkMeta {
            shape: [to_u32(rows, "SYRK rows")?, to_u32(cols, "SYRK cols")?],
            strides: [
                to_i32(trail_layout.strides[0], "SYRK row stride")?,
                to_i32(trail_layout.strides[1], "SYRK col stride")?,
            ],
            offset: to_u32(trail_layout.offset, "SYRK offset")?,
            panel_cols: to_u32(panel_cols, "SYRK panel cols")?,
            _pad: [0; 2],
        };

        let key = "cholesky_syrk".to_string();
        let kernel = cached_kernel(device, key, "syrk_kernel", syrk_shader_source)?;

        let workgroups_x = cols.div_ceil(16);
        let workgroups_y = rows.div_ceil(16);

        let mut panel_ptr = panel.raw();
        let mut trail_ptr = trail.raw();
        let mut meta_val = meta;

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut panel_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut trail_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut meta_val as *mut SyrkMeta as *mut std::ffi::c_void,
        ];

        // SAFETY: Buffers are valid, dimensions match.
        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                workgroups_x as u32,
                workgroups_y as u32,
                1,
                16,
                16,
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

        Ok(())
    }
}

#[cfg(feature = "cuda")]
use syrk_impl::{syrk_trailing_update, write_device_buffer};
