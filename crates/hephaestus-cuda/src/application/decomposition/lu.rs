//! GPU-resident LU decomposition with partial pivoting.
//!
//! Computes **P A** = **L U** where **P** is a permutation matrix, **L** is
//! unit lower-triangular, and **U** is upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`lu_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`lu_decompose_blocked`] — blocked algorithm where panel
//!   factorization uses the same partial-pivoting rule as Leto's LU and the
//!   O(2n³/3) trailing GEMM update (`A₂₂ -= L₂₁ · U₁₂`) runs on the GPU
//!   via a dedicated CUDA kernel.
//!
//! # Mathematical Foundations
//!
//! ## Blocked LU with GPU Trailing GEMM
//!
//! Partition **P A = L U** into *b × b* blocks:
//!
//! ```text
//! ┌           ┐   ┌       ┐ ┌       ┐
//! │ A₁₁  A₁₂ │   │ L₁₁ 0 │ │ U₁₁ U₁₂│
//! │ A₂₁  A₂₂ │ = │ L₂₁ I │ │  0  S₂₂│
//! └           ┘   └       ┘ └       ┘
//! ```
//!
//! The Schur complement is **S₂₂ = A₂₂ − L₂₁ U₁₂** and the dominant
//! cost is the rank-b GEMM update which runs on the GPU.
//!
//! **Theorem (Blocked LU complexity).** For *n × n* with block size *b*,
//! the total flop count is 2n³/3, identical to unblocked LU.  The blocked
//! variant improves performance by:
//! (a) moving the O(b(n−k)²) trailing GEMM to the GPU, and
//! (b) improving CPU cache locality for the O(b²(n−k)) panel operations.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

#[cfg(feature = "cuda")]
use super::region::{download_matrix_region, write_matrix_region, MatrixRegion};
#[cfg(feature = "cuda")]
use hephaestus_core::panel_lu_packed;

use super::validate::validate_square;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// LU decomposition result: device-resident packed factors with host-side
/// decomposition for solve/inv/det.
pub struct GpuLuDecomposition {
    /// Host-side leto-ops decomposition (owns pivots, sign, factors).
    inner: leto_ops::LuDecomposition<f32>,
    /// Device-resident packed L/U factors (*n* × *n*, row-major).
    factors: CudaBuffer<f32>,
    n: usize,
}

impl GpuLuDecomposition {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the packed factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn factors(&self) -> &CudaBuffer<f32> {
        &self.factors
    }

    /// Return the permutation pivots.
    #[must_use]
    #[inline]
    pub fn pivots(&self) -> &[usize] {
        self.inner.pivots()
    }

    /// Determinant: sign × Πᵢ Uᵢᵢ via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side forward/back substitution.
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
                message: format!("LU solve failed: {e}"),
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
                message: format!("LU inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

/// Compute the LU decomposition with partial pivoting on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For large matrices where the O(2n³/3) trailing update should
/// run on the GPU, prefer [`lu_decompose_blocked`].
///
/// # Errors
///
/// - Non-square input.
/// - Non-finite values in the input.
/// - Singular matrix (exact zero pivot).
pub fn lu_decompose(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let factors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::lu_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([0, 0]).unwrap(),
            &[],
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("LU decomposition failed: {e}"),
        })?;
        return Ok(GpuLuDecomposition {
            inner,
            factors,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let lu = leto_ops::lu_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("LU decomposition failed: {e}"),
    })?;

    let factors = device.upload(leto::Storage::as_slice(lu.factors().storage()))?;

    Ok(GpuLuDecomposition {
        inner: lu,
        factors,
        n,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing GEMM
// ---------------------------------------------------------------------------

/// Panel block size for the blocked LU algorithm.
#[cfg(feature = "cuda")]
const LU_BLOCK_SIZE: usize = 64;

/// Blocked LU factorization **P A = L U** with GPU-accelerated trailing-matrix
/// GEMM updates.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Singular matrix (exact zero pivot).
pub fn lu_decompose_blocked(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    #[cfg(feature = "cuda")]
    {
        let n = validate_square(&matrix)?;
        if n == 0 {
            let factors = device.alloc_zeroed::<f32>(0)?;
            let inner = leto_ops::lu_decompose(&leto::ArrayView::<f32, 2>::new(
                leto::Layout::c_contiguous([0, 0]).unwrap(),
                &[],
            ))
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("LU decomposition failed: {e}"),
            })?;
            return Ok(GpuLuDecomposition {
                inner,
                factors,
                n: 0,
            });
        }

        let mut host = vec![0.0f32; n * n];
        device.download(matrix.buffer, &mut host)?;

        let factors_buf = device.upload(&host)?;

        let mut perm: Vec<usize> = (0..n).collect();
        let mut sign = 1i8;

        let block_size = LU_BLOCK_SIZE.min(n);

        for k in (0..n).step_by(block_size) {
            let b = block_size.min(n - k);
            let trail = n - k - b;

            // ── Step 1: Factor the diagonal block on CPU ──
            let mut diag = vec![0.0f32; b * b];
            for i in 0..b {
                for j in 0..b {
                    diag[i * b + j] = host[(k + i) * n + (k + j)];
                }
            }
            let pivots = panel_lu_packed(&mut diag, b)?;

            // Apply the panel's row swaps to the trailing columns and to
            // the cumulative permutation vector.
            for (i, &pivot) in pivots.iter().enumerate().take(b) {
                if pivot != i {
                    let row_a = k + i;
                    let row_b = k + pivot;
                    // Swap entire rows in host.
                    for j in 0..n {
                        host.swap(row_a * n + j, row_b * n + j);
                    }
                    // Update cumulative permutation.
                    perm.swap(row_a, row_b);
                    sign = -sign;
                }
            }

            // Write factored diagonal back to host.
            for i in 0..b {
                for j in 0..b {
                    host[(k + i) * n + (k + j)] = diag[i * b + j];
                }
            }

            if trail == 0 {
                write_matrix_region(
                    device,
                    &factors_buf,
                    &host,
                    MatrixRegion {
                        stride: n,
                        row_start: k,
                        col_start: 0,
                        rows: b,
                        cols: n,
                    },
                )?;
                continue;
            }

            // ── Step 2: Solve L₂₁ = A₂₁ · U₁₁⁻¹ on CPU ──
            for i in 0..trail {
                for j in 0..b {
                    let mut s = host[(k + b + i) * n + (k + j)];
                    for p in 0..j {
                        s -= host[(k + b + i) * n + (k + p)] * diag[p * b + j];
                    }
                    host[(k + b + i) * n + (k + j)] = s / diag[j * b + j];
                }
            }

            // ── Step 3: Solve U₁₂ = L₁₁⁻¹ · A₁₂ on CPU ──
            for j in 0..trail {
                for i in 0..b {
                    let mut s = host[(k + i) * n + (k + b + j)];
                    for p in 0..i {
                        s -= diag[i * b + p] * host[(k + p) * n + (k + b + j)];
                    }
                    host[(k + i) * n + (k + b + j)] = s;
                }
            }

            // Upload only the updated active column panel (L₂₁) and U₁₂ row panel (covering columns 0..n).
            let col_region = MatrixRegion {
                stride: n,
                row_start: k + b,
                col_start: k,
                rows: trail,
                cols: b,
            };
            let row_region = MatrixRegion {
                stride: n,
                row_start: k,
                col_start: 0,
                rows: b,
                cols: n,
            };
            write_matrix_region(device, &factors_buf, &host, col_region)?;
            write_matrix_region(device, &factors_buf, &host, row_region)?;

            gemm_impl::gemm_trailing_update(
                device,
                &factors_buf,
                (k + b) * n + k,
                n,
                trail,
                b,
                &factors_buf,
                k * n + (k + b),
                n,
                trail,
                &factors_buf,
                (k + b) * n + (k + b),
                n,
            )?;

            // Download only the next column and row panels instead of the full trailing matrix.
            let k_next = k + b;
            if k_next < n {
                let b_next = block_size.min(n - k_next);
                // Download the next column panel
                download_matrix_region(
                    device,
                    &factors_buf,
                    &mut host,
                    MatrixRegion {
                        stride: n,
                        row_start: k_next,
                        col_start: k_next,
                        rows: n - k_next,
                        cols: b_next,
                    },
                )?;
                // Download the next row panel
                if k_next + b_next < n {
                    download_matrix_region(
                        device,
                        &factors_buf,
                        &mut host,
                        MatrixRegion {
                            stride: n,
                            row_start: k_next,
                            col_start: k_next + b_next,
                            rows: b_next,
                            cols: n - k_next - b_next,
                        },
                    )?;
                }
            }
        }

        // Download the final factored matrix back to host.
        device.download(&factors_buf, &mut host)?;

        let inner = leto_ops::LuDecomposition::from_raw_parts(
            leto::Array2::from_shape_vec([n, n], host).expect("valid square factor"),
            perm,
            sign,
        );

        Ok(GpuLuDecomposition {
            inner,
            factors: factors_buf,
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

// ---------------------------------------------------------------------------
// GEMM kernel implementation (CUDA PTX)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod gemm_impl {
    use super::*;
    use crate::application::linalg::to_u32;
    use crate::application::pipeline::cached_kernel;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable)]
    pub struct GemmMeta {
        /// Shape: [m, n, k].
        shape: [u32; 3],
        /// Row strides: [C row-stride, A row-stride, B row-stride].
        strides: [u32; 3],
        /// Element offsets: [C offset, A offset, B offset].
        offsets: [u32; 3],
    }

    unsafe impl bytemuck::Pod for GemmMeta {}

    fn gemm_shader_source() -> String {
        r#"
    struct GemmMeta {
        unsigned int shape[3];
        unsigned int strides[3];
        unsigned int offsets[3];
    };

    extern "C" __global__ void gemm_kernel(
        const float* a_buf,
        const float* b_buf,
        float* c_buf,
        GemmMeta meta
    ) {
        __shared__ float tile_a[16][16];
        __shared__ float tile_b[16][16];

        unsigned int row = blockIdx.y * 16u + threadIdx.y;
        unsigned int col = blockIdx.x * 16u + threadIdx.x;
        unsigned int m = meta.shape[0];
        unsigned int n = meta.shape[1];
        unsigned int k = meta.shape[2];
        unsigned int c_stride = meta.strides[0];
        unsigned int a_stride = meta.strides[1];
        unsigned int b_stride = meta.strides[2];
        unsigned int c_off = meta.offsets[0];
        unsigned int a_off = meta.offsets[1];
        unsigned int b_off = meta.offsets[2];

        if (row >= m || col >= n) {
            return;
        }

        float sum = 0.0f;
        unsigned int num_tiles = (k + 15u) / 16u;

        for (unsigned int tile = 0u; tile < num_tiles; tile++) {
            // Load tile of A: A[row, tile*16 + threadIdx.x]
            unsigned int a_col = tile * 16u + threadIdx.x;
            if (a_col < k) {
                tile_a[threadIdx.y][threadIdx.x] = a_buf[a_off + row * a_stride + a_col];
            } else {
                tile_a[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Load tile of B: B[tile*16 + threadIdx.y, col]
            unsigned int b_row = tile * 16u + threadIdx.y;
            if (b_row < k) {
                tile_b[threadIdx.y][threadIdx.x] = b_buf[b_off + b_row * b_stride + col];
            } else {
                tile_b[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for (unsigned int i = 0u; i < 16u; i++) {
                sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
            }

            __syncthreads();
        }

        // C -= A * B
        unsigned int c_idx = c_off + row * c_stride + col;
        c_buf[c_idx] -= sum;
    }
        "#
        .to_string()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_trailing_update(
        device: &CudaDevice,
        a_buf: &CudaBuffer<f32>,
        a_offset: usize,
        a_stride: usize,
        a_rows: usize,
        a_cols: usize,
        b_buf: &CudaBuffer<f32>,
        b_offset: usize,
        b_stride: usize,
        b_cols: usize,
        c_buf: &CudaBuffer<f32>,
        c_offset: usize,
        c_stride: usize,
    ) -> Result<()> {
        let m = a_rows;
        let k = a_cols;
        let n = b_cols;
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let meta = GemmMeta {
            shape: [
                to_u32(m, "GEMM m")?,
                to_u32(n, "GEMM n")?,
                to_u32(k, "GEMM k")?,
            ],
            strides: [
                to_u32(c_stride, "GEMM C stride")?,
                to_u32(a_stride, "GEMM A stride")?,
                to_u32(b_stride, "GEMM B stride")?,
            ],
            offsets: [
                to_u32(c_offset, "GEMM C offset")?,
                to_u32(a_offset, "GEMM A offset")?,
                to_u32(b_offset, "GEMM B offset")?,
            ],
        };

        let key = "lu_gemm".to_string();
        let kernel = cached_kernel(device, key, "gemm_kernel", gemm_shader_source)?;

        let workgroups_x = n.div_ceil(16);
        let workgroups_y = m.div_ceil(16);

        let mut a_ptr = a_buf.raw();
        let mut b_ptr = b_buf.raw();
        let mut c_ptr = c_buf.raw();
        let mut meta_val = meta;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut c_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut meta_val as *mut GemmMeta as *mut std::ffi::c_void,
        ];

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
                    message: format!("cuLaunchKernel GEMM failed with code: {res}"),
                });
            }
        }

        Ok(())
    }
}
