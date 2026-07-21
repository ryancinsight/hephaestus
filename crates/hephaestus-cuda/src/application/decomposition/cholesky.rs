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
use hephaestus_core::{
    ComputeDevice, DeviceBuffer, HephaestusError, Result, factor_cholesky_panel,
};

#[cfg(feature = "cuda")]
use super::region::{MatrixRegion, download_matrix_region_compact, write_matrix_region_compact};
use super::validate::{validate_dense_operand, validate_square};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::{CudaDevice, cuda_byte_count};

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
            leto::Layout::c_contiguous([self.n]).expect("infallible: valid contiguous layout"),
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
            leto::Layout::c_contiguous([0, 0]).expect("infallible: empty matrix layout"),
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
///
/// The operand must be dense C-contiguous at offset 0 (the blocked path
/// bulk-copies the matrix storage on the device); transposed, offset, or
/// broadcast views are rejected with a typed error — materialize them
/// first.
///
/// # Errors
///
/// - Non-square or non-dense (non-C-contiguous / offset / broadcast) operand.
/// - Non-finite values in the input; matrix not positive-definite.
pub fn cholesky_decompose_blocked(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    #[cfg(feature = "cuda")]
    {
        let n = validate_square(&matrix)?;
        validate_dense_operand("cholesky", &matrix)?;
        if n == 0 {
            let lower = device.alloc_zeroed::<f32>(0)?;
            let inner = leto_ops::cholesky_decompose(&leto::ArrayView::<f32, 2>::new(
                leto::Layout::c_contiguous([0, 0]).expect("infallible: empty matrix layout"),
                &[],
            ))
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky decomposition failed: {e}"),
            })?;
            return Ok(GpuCholesky { inner, lower, n: 0 });
        }

        // Allocate device-resident buffer and copy matrix.buffer into it on the GPU
        let lower_buf = device.alloc_zeroed::<f32>(n * n)?;
        device.bind()?;
        let bytes = n * n * std::mem::size_of::<f32>();
        let byte_count = cuda_byte_count(bytes, "blocked Cholesky startup copy byte count")?;
        // SAFETY: this device's context is current (`bind` above). `lower_buf`
        // is a live, freshly allocated `n * n`-element device allocation, and
        // `matrix.buffer` holds at least `n * n` elements: the operand is
        // enforced dense C-contiguous at offset 0 (`validate_dense_operand`
        // above), so the layout's validated storage extent
        // (`validate_square`) equals the `bytes` read here. The copy is
        // asynchronous on the null stream; both allocations outlive it
        // because frees route through synchronizing `cuMemFree`-family
        // calls.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyDtoD_v2(lower_buf.raw(), matrix.buffer.raw(), byte_count)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("cholesky startup cuMemcpyDtoD_v2 failed: {res}"),
            });
        }

        const BLOCK_SIZE: usize = 64;
        let block_size = BLOCK_SIZE.min(n);

        for k in (0..n).step_by(block_size) {
            let b = block_size.min(n - k);
            let panel_rows = n - k;

            // Download only the active panel column (diagonal block + off-diagonal panel)
            let panel_region = MatrixRegion {
                stride: n,
                row_start: k,
                col_start: k,
                rows: panel_rows,
                cols: b,
            };
            let mut panel = download_matrix_region_compact(device, &lower_buf, panel_region)?;

            let trail_rows = n - k - b;
            // Factor this panel on the host (diagonal-block Cholesky +
            // off-diagonal triangular solve) — the shared computation.
            factor_cholesky_panel(&mut panel, b, trail_rows)?;

            if trail_rows == 0 {
                // Write the final diagonal block back to the device buffer
                write_matrix_region_compact(device, &lower_buf, &panel, panel_region)?;
                continue;
            }

            // Write the entire updated active panel back to device buffer
            write_matrix_region_compact(device, &lower_buf, &panel, panel_region)?;

            // ── Step 3: trailing SYRK update on GPU ──
            let trail_layout = leto::Layout::new(
                [trail_rows, trail_rows],
                [n as isize, 1],
                (k + b) * n + (k + b),
            );
            syrk_trailing_update(
                device,
                &lower_buf,
                &trail_layout,
                &lower_buf,
                b,
                (k + b) * n + k,
                n,
            )?;
        }

        // Download the final factored matrix back to host.
        let mut host = vec![0.0f32; n * n];
        device.download(&lower_buf, &mut host)?;

        for row in 0..n {
            for col in (row + 1)..n {
                host[row * n + col] = 0.0;
            }
        }
        device.write_buffer(&lower_buf, &host)?;

        let inner = leto_ops::CholeskyDecomposition::from_raw_parts(
            leto::Array2::from_shape_vec([n, n], host).expect("valid square factor"),
        );

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
    use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable)]
    pub struct SyrkMeta {
        shape: [u32; 2],
        strides: [i32; 2],
        offset: u32,
        panel_cols: u32,
        panel_offset: u32,
        panel_stride: u32,
    }

    // SAFETY: SyrkMeta is `#[repr(C)]` and every field is Pod.
    unsafe impl Pod for SyrkMeta {}

    fn syrk_shader_source() -> String {
        r#"
    struct SyrkMeta {
        unsigned int shape[2];
        int strides[2];
        unsigned int offset;
        unsigned int panel_cols;
        unsigned int panel_offset;
        unsigned int panel_stride;
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

        float sum = 0.0f;
        unsigned int num_tiles = (k + 15u) / 16u;

        for (unsigned int tile = 0u; tile < num_tiles; tile++) {
            unsigned int panel_col = tile * 16u + local_col;
            if (row < rows && panel_col < k) {
                panel_row_shared[local_row][local_col] = panel[meta.panel_offset + row * meta.panel_stride + panel_col];
            } else {
                panel_row_shared[local_row][local_col] = 0.0f;
            }

            __syncthreads();

            if (row < rows && col < cols && col <= row) {
                for (unsigned int i = 0u; i < 16u; i++) {
                    unsigned int ki = tile * 16u + i;
                    if (ki < k) {
                        float a_val = panel_row_shared[local_row][i];
                        float b_val = panel[meta.panel_offset + col * meta.panel_stride + ki];
                        sum += a_val * b_val;
                    }
                }
            }

            __syncthreads();
        }

        if (row < rows && col < cols && col <= row) {
            int c_off = (int)off + (int)row * stride_row + (int)col * stride_col;
            trail[c_off] -= sum;
        }
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
        panel_offset: usize,
        panel_stride: usize,
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
            panel_offset: to_u32(panel_offset, "SYRK panel offset")?,
            panel_stride: to_u32(panel_stride, "SYRK panel stride")?,
        };

        let kernel = cached_kernel(
            device,
            PipelineKey::CholeskySyrk,
            "syrk_kernel",
            syrk_shader_source,
        )?;

        let workgroups_x = cols.div_ceil(16);
        let workgroups_y = rows.div_ceil(16);

        let mut panel_ptr = panel.raw();
        let mut trail_ptr = trail.raw();
        let mut meta_val = meta;

        // Argument list mirrors `syrk_kernel(const float*, float*, SyrkMeta)`.
        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut panel_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut trail_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut meta_val as *mut SyrkMeta as *mut std::ffi::c_void,
        ];

        launch_kernel(
            device,
            &kernel,
            LaunchConfig::planar(workgroups_x as u32, workgroups_y as u32, 16, 16),
            &mut args,
        )
    }
}

#[cfg(feature = "cuda")]
use syrk_impl::syrk_trailing_update;
