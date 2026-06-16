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

/// In-place Householder QR factorization of an (*m* × *n*) packed row-major
/// matrix, returning the Householder vector heads and β coefficients.
#[cfg(feature = "cuda")]
fn panel_qr_packed(a: &mut [f32], m: usize, n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    if a.len() != m * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: m * n,
            device_len: a.len(),
        });
    }
    if m < n {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR panel requires m ≥ n, got [{m}, {n}]"),
        });
    }

    if let Some((idx, value)) = a
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR panel factorisation failed: entry {idx} is non-finite ({value})"),
        });
    }

    let mut heads = vec![0.0f32; n];
    let mut betas = vec![0.0f32; n];

    for k in 0..n {
        let mut norm_sq = 0.0f32;
        for r in k..m {
            let x = a[r * n + k];
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("QR pivot column {k} has zero norm: matrix is rank-deficient"),
            });
        }

        let pivot = a[k * n + k];
        let alpha = if pivot > 0.0 { -norm } else { norm };
        let head = pivot - alpha;

        let mut v_norm_sq = head * head;
        for r in (k + 1)..m {
            let x = a[r * n + k];
            v_norm_sq += x * x;
        }
        let beta = 2.0 / v_norm_sq;

        for c in (k + 1)..n {
            let mut s = head * a[k * n + c];
            for r in (k + 1)..m {
                s += a[r * n + k] * a[r * n + c];
            }
            let bs = beta * s;
            a[k * n + c] -= bs * head;
            for r in (k + 1)..m {
                a[r * n + c] -= bs * a[r * n + k];
            }
        }

        a[k * n + k] = alpha;
        heads[k] = head;
        betas[k] = beta;
    }

    Ok((heads, betas))
}

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

        let mut host = vec![0.0f32; m * n];
        device.download(matrix.buffer, &mut host)?;
        let original_host = host.clone();

        let work_buf = device.alloc_zeroed::<f32>(m * n)?;

        let block_size = QR_BLOCK_SIZE.min(n);

        for k in (0..n).step_by(block_size) {
            let b = block_size.min(n - k);
            let panel_rows = m - k;
            let trail_cols = n - k - b;

            // ── Step 1: Factor the panel A[k:m, k:k+b] on CPU ──
            let mut panel = vec![0.0f32; panel_rows * b];
            for i in 0..panel_rows {
                for j in 0..b {
                    panel[i * b + j] = host[(k + i) * n + (k + j)];
                }
            }
            let (heads, betas) = panel_qr_packed(&mut panel, panel_rows, b)?;

            // Write R (upper triangle) back to host, zero below diagonal.
            for i in 0..panel_rows {
                for j in 0..b {
                    if j <= i {
                        host[(k + i) * n + (k + j)] = panel[i * b + j];
                    } else {
                        host[(k + i) * n + (k + j)] = 0.0;
                    }
                }
            }

            if trail_cols == 0 {
                hh_impl::write_device_buffer(device, &host, &work_buf)?;
                continue;
            }

            // ── Step 2: Upload working matrix to device ──
            hh_impl::write_device_buffer(device, &host, &work_buf)?;

            // ── Step 3: Apply b Householder reflectors on GPU ──
            for j in 0..b {
                let vec_len = panel_rows - j;

                let mut v = vec![0.0f32; vec_len];
                v[0] = heads[j];
                for i in 1..vec_len {
                    v[i] = panel[(j + i) * b + j];
                }

                let v_dev = device.upload(&v)?;
                let c_offset = (k + j) * n + (k + b);

                hh_impl::hh_trailing_update(
                    device, &v_dev, &work_buf, vec_len, trail_cols, n, c_offset, betas[j],
                )?;
            }

            // ── Step 4: Download updated working matrix ──
            device.download(&work_buf, &mut host)?;
        }

        let original_view = leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([m, n]).unwrap(),
            &original_host,
        );
        let inner = leto_ops::qr_decompose(&original_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("QR blocked finalisation failed: {e}"),
            }
        })?;

        // Materialize R from blocked result.
        let mut r_host = vec![0.0f32; m * n];
        for i in 0..m.min(n) {
            for j in i..n {
                r_host[i * n + j] = host[i * n + j];
            }
        }
        let r_buf = device.upload(&r_host)?;

        Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
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

// ---------------------------------------------------------------------------
// Householder apply kernel (CUDA PTX)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod hh_impl {
    use super::*;
    use crate::application::linalg::{to_i32, to_u32};
    use crate::application::pipeline::cached_kernel;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable)]
    pub struct HhMeta {
        vec_len: u32,
        trail_cols: u32,
        trail_stride: u32,
        c_offset: u32,
        beta: f32,
        _pad: [u32; 3],
    }

    unsafe impl bytemuck::Pod for HhMeta {}

    pub fn write_device_buffer<T: bytemuck::Pod>(
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

    fn hh_shader_source() -> String {
        r#"
    struct HhMeta {
        unsigned int vec_len;
        unsigned int trail_cols;
        unsigned int trail_stride;
        unsigned int c_offset;
        float beta;
        unsigned int _pad0;
        unsigned int _pad1;
        unsigned int _pad2;
    };

    extern "C" __global__ void householder_kernel(
        const float* v_buf,
        float* a_buf,
        HhMeta meta
    ) {
        __shared__ float sdata[256];

        unsigned int col = blockIdx.x;
        unsigned int tid = threadIdx.x;

        if (col >= meta.trail_cols) {
            return;
        }

        unsigned int n = meta.vec_len;
        unsigned int stride = meta.trail_stride;
        unsigned int off = meta.c_offset;
        float beta = meta.beta;

        // Phase 1: partial dot = v^T · A[:, col]
        float partial = 0.0f;
        unsigned int row = tid;
        while (row < n) {
            partial += v_buf[row] * a_buf[off + row * stride + col];
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

        // Phase 2: A[:, col] -= beta * v * dot
        row = tid;
        while (row < n) {
            unsigned int idx = off + row * stride + col;
            a_buf[idx] -= beta * v_buf[row] * dot;
            row += 256u;
        }
    }
        "#
        .to_string()
    }

    pub fn hh_trailing_update(
        device: &CudaDevice,
        v_buf: &CudaBuffer<f32>,
        a_buf: &CudaBuffer<f32>,
        vec_len: usize,
        trail_cols: usize,
        trail_stride: usize,
        c_offset: usize,
        beta: f32,
    ) -> Result<()> {
        if vec_len == 0 || trail_cols == 0 {
            return Ok(());
        }

        let meta = HhMeta {
            vec_len: to_u32(vec_len, "HH vec_len")?,
            trail_cols: to_u32(trail_cols, "HH trail_cols")?,
            trail_stride: to_u32(trail_stride, "HH trail_stride")?,
            c_offset: to_u32(c_offset, "HH c_offset")?,
            beta,
            _pad: [0; 3],
        };

        let key = "qr_householder".to_string();
        let kernel = cached_kernel(device, key, "householder_kernel", hh_shader_source)?;

        let mut v_ptr = v_buf.raw();
        let mut a_ptr = a_buf.raw();
        let mut meta_val = meta;

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut v_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut meta_val as *mut HhMeta as *mut std::ffi::c_void,
        ];

        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                trail_cols as u32,
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuLaunchKernel HH failed with code: {res}"),
                });
            }
        }

        Ok(())
    }
}
