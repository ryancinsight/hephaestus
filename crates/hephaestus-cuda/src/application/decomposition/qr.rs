//! GPU-resident QR decomposition via Householder reflectors.
//!
//! Computes **A** = **Q R** where **Q** is orthogonal and **R** is
//! upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`qr_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`qr_decompose_blocked`] — blocked algorithm where panel
//!   factorization runs on the CPU but the trailing Householder application
//!   runs on the GPU via a dedicated CUDA kernel.
//!
//! # Mathematical Foundations
//!
//! ## Blocked QR with GPU Trailing Application
//!
//! For large *m*, the dominant cost is applying the *b* Householder
//! reflectors from each panel to the trailing submatrix.  Each application
//! costs O(m(n−k)) flops and is embarrassingly parallel across columns.
//!
//! **Theorem (Blocked QR complexity).** For *m × n* with block size *b*,
//! the total flop count is 2n²(m − n/3), identical to unblocked QR. ∎

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

#[cfg(feature = "cuda")]
use hephaestus_core::panel_qr_packed;

#[cfg(feature = "cuda")]
use super::region::{download_matrix_region_compact, write_matrix_region_compact, MatrixRegion};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// QR decomposition result: device-resident R factor with host-side
/// decomposition for solve_least_squares.
pub struct GpuQrDecomposition {
    /// Host-side leto-ops decomposition (owns packed/heads/betas).
    inner: leto_ops::QrDecomposition<f32>,
    /// Device-resident upper-triangular factor **R** (*m* × *n*, row-major).
    r: CudaBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuQrDecomposition {
    /// (rows, cols) of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r_buffer(&self) -> &CudaBuffer<f32> {
        &self.r
    }

    /// Borrow the host-side Leto decomposition.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::QrDecomposition<f32> {
        &self.inner
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ (least squares).
    pub fn solve_least_squares(
        &self,
        device: &CudaDevice,
        rhs: &CudaBuffer<f32>,
    ) -> Result<CudaBuffer<f32>> {
        let (m, n) = (self.rows, self.cols);
        if rhs.len() != m {
            return Err(HephaestusError::LengthMismatch {
                host_len: m,
                device_len: rhs.len(),
            });
        }
        if m == 0 || n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0f32; m];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view =
            leto::ArrayView::<f32, 1>::new(leto::Layout::c_contiguous([m]).unwrap(), &rhs_host);
        let x = self.inner.solve_least_squares(&rhs_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("QR least-squares solve failed: {e}"),
            }
        })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }
}

/// Compute the Householder QR factorization on the GPU.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-finite values in the input.
/// - Exactly-zero pivot column norm (rank-deficient input).
pub fn qr_decompose(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 || cols == 0 {
        let r_buf = device.alloc_zeroed::<f32>(0)?;
        let placeholder = vec![0.0f32];
        let inner = leto_ops::qr_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([1, 1]).unwrap(),
            &placeholder,
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("QR decomposition failed: {e}"),
        })?;
        return Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
            rows,
            cols,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let qr = leto_ops::qr_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("QR decomposition failed: {e}"),
    })?;

    let r_host = qr.r();
    let r_buf = device.upload(leto::Storage::as_slice(r_host.storage()))?;

    Ok(GpuQrDecomposition {
        inner: qr,
        r: r_buf,
        rows,
        cols,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing Householder application
// ---------------------------------------------------------------------------

/// Panel block size for the blocked QR algorithm.
#[cfg(feature = "cuda")]
const QR_BLOCK_SIZE: usize = 32;

/// Blocked QR factorization **A = Q R** with GPU-accelerated trailing
/// Householder application.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-finite values in the input.
/// - Rank-deficient input (zero column norm).
pub fn qr_decompose_blocked(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    #[cfg(feature = "cuda")]
    {
        let [m, n] = matrix.layout.shape;
        if m < n {
            return Err(HephaestusError::DispatchFailed {
                message: format!("QR requires m ≥ n, got shape [{m}, {n}]"),
            });
        }
        matrix
            .layout
            .validate_storage_len(matrix.buffer.len())
            .map_err(map_layout_err)?;

        if m == 0 || n == 0 {
            let r_buf = device.alloc_zeroed::<f32>(0)?;
            let placeholder = vec![0.0f32];
            let inner = leto_ops::qr_decompose(&leto::ArrayView::<f32, 2>::new(
                leto::Layout::c_contiguous([1, 1]).unwrap(),
                &placeholder,
            ))
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("QR decomposition failed: {e}"),
            })?;
            return Ok(GpuQrDecomposition {
                inner,
                r: r_buf,
                rows: m,
                cols: n,
            });
        }

        let work_buf = device.alloc_zeroed::<f32>(m * n)?;
        device.bind()?;
        let bytes = m * n * std::mem::size_of::<f32>();
        // SAFETY: this device's context is current (`bind` above). `work_buf`
        // is a live, freshly allocated `m * n`-element device allocation, and
        // `matrix.buffer` holds at least the layout's validated storage extent
        // (`validate_storage_len` above), which covers the `bytes` read for
        // the dense zero-offset `[m, n]` operands this blocked entry point
        // operates on. The copy is asynchronous on the null stream; both
        // allocations outlive it because frees route through synchronizing
        // `cuMemFree`-family calls.
        let res =
            unsafe { cuda_core::sys::cuMemcpyDtoD_v2(work_buf.raw(), matrix.buffer.raw(), bytes) };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("QR startup cuMemcpyDtoD_v2 failed: {res}"),
            });
        }

        let block_size = QR_BLOCK_SIZE.min(n);

        let mut packed = vec![0.0f32; m * n];
        let mut cumulative_heads = Vec::with_capacity(n.min(m));
        let mut cumulative_betas = Vec::with_capacity(n.min(m));

        // Pre-allocate vectors buffer.
        let vectors_dev = device.alloc_zeroed::<f32>(m * block_size)?;

        // Pre-allocate reflector buffer.
        let reflector_dev = device.alloc_zeroed::<hh_impl::HhReflectorMeta>(block_size)?;

        for k in (0..n).step_by(block_size) {
            let b = block_size.min(n - k);
            let panel_rows = m - k;
            let trail_cols = n - k - b;

            // ── Step 1 & 2: Download active panel from work_buf directly to host panel ──
            let panel_region = MatrixRegion {
                stride: n,
                row_start: k,
                col_start: k,
                rows: panel_rows,
                cols: b,
            };
            let mut panel = download_matrix_region_compact(device, &work_buf, panel_region)?;

            // ── Step 3: Factor active panel region on CPU ──
            let (heads, betas) = panel_qr_packed(&mut panel, panel_rows, b)?;

            cumulative_heads.extend_from_slice(&heads);
            cumulative_betas.extend_from_slice(&betas);

            for j in 0..b {
                let col = k + j;
                for r in (col + 1)..m {
                    let panel_row = r - k;
                    packed[r * n + col] = panel[panel_row * b + j];
                }
            }

            // Zero out the strictly lower-triangular part of panel before writing back
            let mut packed_vectors = Vec::with_capacity(panel_rows * b);
            let mut vector_offsets = Vec::with_capacity(b);
            for j in 0..b {
                let vec_len = panel_rows - j;
                vector_offsets.push(packed_vectors.len());
                packed_vectors.push(heads[j]);
                for i in 1..vec_len {
                    packed_vectors.push(panel[(j + i) * b + j]);
                }
            }

            for r in 0..panel_rows {
                for c in 0..b {
                    if c < r {
                        panel[r * b + c] = 0.0;
                    }
                }
            }

            // ── Step 4 & 5: Write the factored panel with sub-diagonal zeroes back to the device ──
            write_matrix_region_compact(device, &work_buf, &panel, panel_region)?;

            if trail_cols == 0 {
                continue;
            }

            // ── Step 6: Apply b Householder reflectors on GPU in-place ──
            device.write_sub_buffer(&vectors_dev, 0, &packed_vectors)?;

            hh_impl::hh_trailing_update(
                device,
                &vectors_dev,
                &work_buf,
                &reflector_dev,
                panel_rows,
                trail_cols,
                n,
                k,
                b,
                &vector_offsets,
                &betas,
            )?;
        }

        // Download final matrix to extract R.
        let mut host = vec![0.0f32; m * n];
        device.download(&work_buf, &mut host)?;

        // Merge R (upper triangle of host) with the accumulated reflector tails.
        for i in 0..m {
            for j in i..n {
                packed[i * n + j] = host[i * n + j];
            }
        }

        let inner = leto_ops::QrDecomposition::from_raw_parts(
            packed,
            cumulative_heads,
            cumulative_betas,
            m,
            n,
        );

        Ok(GpuQrDecomposition {
            inner,
            r: work_buf,
            rows: m,
            cols: n,
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

// Custom gather/scatter compute kernels removed in favor of generic MatrixRegion transfers.

#[cfg(feature = "cuda")]
mod hh_impl {
    use super::*;
    use crate::application::linalg::to_u32;
    use crate::application::pipeline::{cached_kernel, launch_kernel, LaunchConfig};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable)]
    pub(super) struct HhReflectorMeta {
        pub(super) vector_offset: u32,
        pub(super) beta: f32,
    }

    // SAFETY: `HhReflectorMeta` is `#[repr(C)]` and contains one `u32` and
    // one `f32` field of identical size and alignment, so it has no padding
    // bytes, and every bit pattern is valid for both types.
    unsafe impl bytemuck::Pod for HhReflectorMeta {}

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable)]
    pub struct HhMeta {
        panel_rows: u32,
        reflector_count: u32,
        trail_cols: u32,
        matrix_cols: u32,
        k: u32,
        _pad: [u32; 3],
    }

    // SAFETY: `HhMeta` is `#[repr(C)]` and contains only `u32` fields of
    // identical size and alignment (the trailing `[u32; 3]` pads the struct
    // to 32 bytes explicitly), so it has no implicit padding bytes, and
    // every bit pattern is a valid value.
    unsafe impl bytemuck::Pod for HhMeta {}

    fn hh_shader_source() -> String {
        r#"
    struct ReflectorMeta {
        unsigned int vector_offset;
        float beta;
    };

    struct HhMeta {
        unsigned int panel_rows;
        unsigned int reflector_count;
        unsigned int trail_cols;
        unsigned int matrix_cols;
        unsigned int k;
        unsigned int _pad0;
        unsigned int _pad1;
        unsigned int _pad2;
    };

    extern "C" __global__ void householder_kernel(
        const float* v_buf,
        float* a_buf,
        const ReflectorMeta* reflector_buf,
        HhMeta meta
    ) {
        __shared__ float sdata[256];

        unsigned int wid_x = blockIdx.x;
        unsigned int col = meta.k + meta.reflector_count + wid_x;
        unsigned int tid = threadIdx.x;

        if (col >= meta.matrix_cols) {
            return;
        }

        unsigned int n_cols = meta.matrix_cols;
        unsigned int k_offset = meta.k;

        for (unsigned int reflector = 0u; reflector < meta.reflector_count; reflector++) {
            unsigned int n_rows = meta.panel_rows - reflector;
            unsigned int start_row = k_offset + reflector;
            unsigned int v_off = reflector_buf[reflector].vector_offset;
            float beta = reflector_buf[reflector].beta;

            // Phase 1: partial dot = v^T · A[start_row:m, col]
            float partial = 0.0f;
            unsigned int row = tid;
            while (row < n_rows) {
                unsigned int a_idx = (start_row + row) * n_cols + col;
                partial += v_buf[v_off + row] * a_buf[a_idx];
                row += 256u;
            }
            sdata[tid] = partial;
            __syncthreads();

            // Parallel tree reduction
            for (unsigned int s = 128u; s > 0u; s >>= 1u) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            float dot = sdata[0];
            __syncthreads();

            // Phase 2: A[start_row:m, col] -= beta * v * dot
            row = tid;
            while (row < n_rows) {
                unsigned int a_idx = (start_row + row) * n_cols + col;
                a_buf[a_idx] -= beta * v_buf[v_off + row] * dot;
                row += 256u;
            }
            __syncthreads();
        }
    }
        "#
        .to_string()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hh_trailing_update(
        device: &CudaDevice,
        v_buf: &CudaBuffer<f32>,
        a_buf: &CudaBuffer<f32>,
        reflector_buf: &CudaBuffer<HhReflectorMeta>,
        panel_rows: usize,
        trail_cols: usize,
        matrix_cols: usize,
        k: usize,
        reflector_count: usize,
        vector_offsets: &[usize],
        betas: &[f32],
    ) -> Result<()> {
        if panel_rows == 0 || trail_cols == 0 || reflector_count == 0 {
            return Ok(());
        }

        let meta = HhMeta {
            panel_rows: to_u32(panel_rows, "HH panel_rows")?,
            reflector_count: to_u32(reflector_count, "HH reflector_count")?,
            trail_cols: to_u32(trail_cols, "HH trail_cols")?,
            matrix_cols: to_u32(matrix_cols, "HH matrix_cols")?,
            k: to_u32(k, "HH k")?,
            _pad: [0; 3],
        };

        let reflector_host: Vec<HhReflectorMeta> = vector_offsets
            .iter()
            .copied()
            .zip(betas.iter().copied())
            .map(|(offset, beta)| {
                let vector_offset = to_u32(offset, "HH vector_offset")?;
                Ok(HhReflectorMeta {
                    vector_offset,
                    beta,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        device.write_sub_buffer(reflector_buf, 0, &reflector_host)?;

        let key = "qr_householder".to_string();
        let kernel = cached_kernel(device, key, "householder_kernel", hh_shader_source)?;

        let mut v_ptr = v_buf.raw();
        let mut a_ptr = a_buf.raw();
        let mut ref_ptr = reflector_buf.raw();
        let mut meta_val = meta;

        // Argument list mirrors `householder_kernel(const float*, float*,
        // const ReflectorMeta*, HhMeta)`.
        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut v_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut ref_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut meta_val as *mut HhMeta as *mut std::ffi::c_void,
        ];

        // 256-thread blocks match the kernel's fixed `sdata[256]` reduction tile.
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::planar(trail_cols as u32, 1, 256, 1),
            &mut args,
        )
    }
}
