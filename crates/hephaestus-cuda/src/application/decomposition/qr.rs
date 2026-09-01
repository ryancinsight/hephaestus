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
use super::region::{MatrixRegion, download_matrix_region_compact, write_matrix_region_compact};
#[cfg(feature = "cuda")]
use super::validate::validate_dense_operand;

use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
#[cfg(feature = "cuda")]
use crate::infrastructure::device::cuda_byte_count;

#[cfg(feature = "cuda")]
mod householder;
#[cfg(feature = "cuda")]
mod q;

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

    /// Take ownership of the device-resident **R** buffer.
    ///
    /// This avoids a device-to-device copy when a caller needs **R** as an
    /// independent result after consuming the decomposition.
    #[must_use]
    #[inline]
    pub fn into_r_buffer(self) -> CudaBuffer<f32> {
        self.r
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

        let rhs_host = device.download_owned(rhs)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([m]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
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
    let [rows, cols] = matrix.layout.shape();
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    let host_data = device.download_owned(matrix.buffer)?;

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
/// The operand must be dense C-contiguous at offset 0 (the blocked path
/// bulk-copies the matrix storage on the device); transposed, offset, or
/// broadcast views are rejected with a typed error — materialize them
/// first.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-dense (non-C-contiguous / offset / broadcast) operand.
/// - Non-finite values in the input.
/// - Rank-deficient input (zero column norm).
pub fn qr_decompose_blocked(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    #[cfg(feature = "cuda")]
    {
        let [m, n] = matrix.layout.shape();
        if m < n {
            return Err(HephaestusError::DispatchFailed {
                message: format!("QR requires m ≥ n, got shape [{m}, {n}]"),
            });
        }
        matrix
            .layout
            .validate_storage_len(matrix.buffer.len())
            .map_err(map_layout_err)?;
        validate_dense_operand("QR", &matrix)?;

        if m == 0 || n == 0 {
            let r_buf = device.alloc_zeroed::<f32>(0)?;
            let inner =
                leto_ops::QrDecomposition::from_raw_parts(Vec::new(), Vec::new(), Vec::new(), m, n);
            return Ok(GpuQrDecomposition {
                inner,
                r: r_buf,
                rows: m,
                cols: n,
            });
        }

        let work_buf = device.alloc_uninitialized::<f32>(m * n)?;
        device.bind()?;
        let bytes = m * n * std::mem::size_of::<f32>();
        let byte_count = cuda_byte_count(bytes, "blocked QR startup copy byte count")?;
        // SAFETY: this device's context is current (`bind` above). `work_buf`
        // is a live, freshly allocated `m * n`-element device allocation, and
        // `matrix.buffer` holds at least `m * n` elements: the operand is
        // enforced dense C-contiguous at offset 0 (`validate_dense_operand`
        // above), so the layout's validated storage extent
        // (`validate_storage_len`) equals the `bytes` read here. The copy is
        // asynchronous on the null stream; both allocations outlive it
        // because frees route through synchronizing `cuMemFree`-family
        // calls.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyDtoD_v2(work_buf.raw(), matrix.buffer.raw(), byte_count)
        };
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
        let vectors_dev = device.alloc_uninitialized::<f32>(m * block_size)?;

        // Pre-allocate reflector buffer.
        let reflector_dev =
            device.alloc_uninitialized::<householder::HhReflectorMeta>(block_size)?;

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

            householder::hh_trailing_update(
                device,
                householder::HhTrailingUpdate {
                    vectors: &vectors_dev,
                    matrix: &work_buf,
                    reflectors: &reflector_dev,
                    panel_rows,
                    trail_cols,
                    matrix_cols: n,
                    panel_start: k,
                    vector_offsets: &vector_offsets,
                    betas: &betas,
                },
            )?;
        }

        // Download final matrix to extract R.
        let host = device.download_owned(&work_buf)?;

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
