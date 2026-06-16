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
//!   O(2n³/3) trailing GEMM update (`A₂₂ -= L₂₁ · U₁₂`) runs on the GPU via a
//!   dedicated compute kernel.
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
//!
//! **Proof.** Each block iteration costs:
//! - Panel factor: 2b³/3
//! - Panel solve (L₂₁): b²(n−k−b)/2
//! - Panel solve (U₁₂): b²(n−k−b)/2
//! - Trailing GEMM: 2b(n−k−b)²
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n³/3 total flops (the same
//! as unblocked LU).  The key performance gain is that the trailing GEMM,
//! which dominates for large n, executes on the GPU's massively parallel
//! compute units rather than on the CPU's sequential cores.

use bytemuck::Pod;
use std::any::TypeId;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Layout;

use super::validate::validate_square;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// LU decomposition result: device-resident packed factors with host-side
/// decomposition for solve/inv/det.
pub struct GpuLuDecomposition {
    /// Host-side leto-ops decomposition (owns pivots, sign, factors).
    inner: leto_ops::LuDecomposition<f32>,
    /// Device-resident packed L/U factors (*n* × *n*, row-major).
    factors: WgpuBuffer<f32>,
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
    pub fn factors(&self) -> &WgpuBuffer<f32> {
        &self.factors
    }

    /// Determinant: sign × Πᵢ Uᵢᵢ via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side forward/back substitution.
    pub fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        if rhs.len != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len,
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
    pub fn inv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
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

// ---------------------------------------------------------------------------
// GEMM uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the trailing GEMM compute kernel.
/// Computes **C -= A · B** where A is (m×k), B is (k×n), C is (m×n).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct GemmMeta {
    /// Shape: [m, n, k].
    shape: [u32; 3],
    /// Row strides: [C row-stride, A row-stride, B row-stride].
    strides: [u32; 3],
    /// Element offset into the C (trailing) buffer.
    c_offset: u32,
}

// SAFETY: GemmMeta is `#[repr(C)]` and every field is Pod.
unsafe impl Pod for GemmMeta {}

fn write_device_buffer<T: Pod>(
    device: &WgpuDevice,
    host: &[T],
    buffer: &WgpuBuffer<T>,
) -> Result<()> {
    if host.len() != buffer.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: host.len(),
            device_len: buffer.len,
        });
    }
    device
        .queue()
        .write_buffer(buffer.raw(), 0, bytemuck::cast_slice(host));
    Ok(())
}

// ---------------------------------------------------------------------------
// GEMM kernel  C -= A · B
// ---------------------------------------------------------------------------

/// WGSL source for the trailing GEMM kernel
///
/// ```text
/// C[i,j] -= Σₖ A[i,k] · B[k,j]
/// ```
///
/// where A is L₂₁ (m×k), B is U₁₂ (k×n), and C is the trailing
/// submatrix A₂₂ (m×n).  Uses 16×16 workgroup tiles with shared-memory
/// cooperative loading for both A and B tiles.
fn gemm_shader_source() -> String {
    const TY: &str = "f32";
    const ZERO: &str = "0.0";

    format!(
        r#"struct GemmMeta {{
    shape: vec3<u32>,
    strides: vec3<u32>,
    c_offset: u32,
}}

@group(0) @binding(0) var<storage, read>      a_buf: array<{ty}>;
@group(0) @binding(1) var<storage, read>      b_buf: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> c_buf: array<{ty}>;
@group(0) @binding(3) var<uniform>             meta: GemmMeta;

var<workgroup> tile_a: array<{ty}, 256>;
var<workgroup> tile_b: array<{ty}, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
) {{
    let row = gid.y;
    let col = gid.x;
    let m = meta.shape.x;
    let n = meta.shape.y;
    let k = meta.shape.z;
    let c_stride = meta.strides.x;
    let a_stride = meta.strides.y;
    let b_stride = meta.strides.z;
    let c_off = meta.c_offset;

    if (row >= m || col >= n) {{
        return;
    }}

    var sum = {ty}({zero});
    let num_tiles = (k + 15u) / 16u;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {{
        // Load tile of A: A[row, tile*16 + lid.x]
        let a_col = tile * 16u + lid.x;
        if (a_col < k) {{
            tile_a[lid.y * 16u + lid.x] = a_buf[row * a_stride + a_col];
        }} else {{
            tile_a[lid.y * 16u + lid.x] = {ty}({zero});
        }}

        // Load tile of B: B[tile*16 + lid.y, col]
        let b_row = tile * 16u + lid.y;
        if (b_row < k) {{
            tile_b[lid.y * 16u + lid.x] = b_buf[b_row * b_stride + col];
        }} else {{
            tile_b[lid.y * 16u + lid.x] = {ty}({zero});
        }}

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            sum = sum + tile_a[lid.y * 16u + i] * tile_b[i * 16u + lid.x];
        }}

        workgroupBarrier();
    }}

    // C -= A · B
    let c_idx = c_off + row * c_stride + col;
    c_buf[c_idx] = c_buf[c_idx] - sum;
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct GemmKernel;

struct GemmTrailingUpdate<'a> {
    a_buf: &'a WgpuBuffer<f32>,
    a_rows: usize,
    a_cols: usize,
    b_buf: &'a WgpuBuffer<f32>,
    b_cols: usize,
    c_buf: &'a WgpuBuffer<f32>,
    c_layout: &'a Layout<2>,
}

/// GPU dispatch for the trailing GEMM:  **C -= A · B**
///
/// A is (m×k) compact row-major, B is (k×n) compact row-major, C is (m×n)
/// starting at `c_layout.offset` with `c_layout.strides[0]` as row stride.
fn gemm_trailing_update(device: &WgpuDevice, update: GemmTrailingUpdate<'_>) -> Result<()> {
    let m = update.a_rows;
    let k = update.a_cols;
    let n = update.b_cols;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    // A is compact (m×k): row stride = k.
    // B is compact (k×n): row stride = n.
    let meta = GemmMeta {
        shape: [
            u32::try_from(m).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM m {m} exceeds u32"),
            })?,
            u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM n {n} exceeds u32"),
            })?,
            u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM k {k} exceeds u32"),
            })?,
        ],
        strides: [
            u32::try_from(update.c_layout.strides[0]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!(
                        "GEMM C row stride {} exceeds u32",
                        update.c_layout.strides[0]
                    ),
                }
            })?,
            u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM A row stride {k} exceeds u32"),
            })?,
            u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM B row stride {n} exceeds u32"),
            })?,
        ],
        c_offset: u32::try_from(update.c_layout.offset).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("GEMM C offset {} exceeds u32", update.c_layout.offset),
            }
        })?,
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<GemmKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-gemm",
        gemm_shader_source,
    );

    let meta_buf = device.get_uniform_buffer(std::mem::size_of::<GemmMeta>() as u64)?;
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-gemm"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: update.a_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: update.b_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: update.c_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-gemm"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x = u32::try_from(n.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("GEMM workgroup x {} exceeds u32", n.div_ceil(16)),
        })?;
        let wg_y = u32::try_from(m.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("GEMM workgroup y {} exceeds u32", m.div_ceil(16)),
        })?;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    device.recycle_uniform_buffer(meta_buf);

    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point 1 — host delegation
// ---------------------------------------------------------------------------

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
    device: &WgpuDevice,
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

    let mut host_data = vec![0.0f32; matrix.buffer.len];
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
///
/// A value of 64 balances CPU panel factorisation cost against GPU GEMM
/// launch overhead.
const LU_BLOCK_SIZE: usize = 64;

/// In-place partial pivoting LU factorization of a packed `n × n` row-major
/// matrix, returning the Leto-style cumulative row permutation vector.
///
/// The packed result stores L (unit lower) strictly below the diagonal and U
/// on and above it, identical to the leto-ops convention.
///
/// Returns the pivot vector where `pivots[k]` is the row swapped with row `k`
/// at step *k* (identity if no swap occurred).
fn panel_lu_packed(a: &mut [f32], n: usize) -> Result<Vec<usize>> {
    if a.len() != n * n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n * n,
            device_len: a.len(),
        });
    }
    if let Some((idx, value)) = a
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!("LU panel factorisation failed: entry {idx} is non-finite ({value})"),
        });
    }

    let mut pivots: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivot: find the row r ≥ k with the largest |a[r, k]|.
        let mut pivot_row = k;
        let mut pivot_mag = a[k * n + k].abs();
        for r in (k + 1)..n {
            let mag = a[r * n + k].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_row != k {
            // Swap entire rows in the working portion.
            for c in 0..n {
                a.swap(k * n + c, pivot_row * n + c);
            }
            pivots.swap(k, pivot_row);
        }

        if pivot_mag == 0.0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("LU panel factorisation failed: pivot column {k} is exactly zero"),
            });
        }

        let pivot = a[k * n + k];
        for r in (k + 1)..n {
            let factor = a[r * n + k] / pivot;
            a[r * n + k] = factor; // L entry
            for c in (k + 1)..n {
                a[r * n + c] -= factor * a[k * n + c];
            }
        }
    }

    Ok(pivots)
}

/// Blocked LU factorization **P A = L U** with GPU-accelerated trailing-matrix
/// GEMM updates.
///
/// The algorithm processes the matrix in `LU_BLOCK_SIZE × LU_BLOCK_SIZE`
/// panels.  For each panel *k*:
///
/// 1. The diagonal block is factored on the **CPU** via in-place partial
///    pivoting LU (O(2b³/3)).
/// 2. The row permutation is applied to the remaining rows of `A₂₁` and
///    `A₁₂`.
/// 3. The L₂₁ panel (below-diagonal) is solved on the **CPU** via
///    forward-substitution with U₁₁ (O(b²(n−k)/2)).
/// 4. The U₁₂ panel (right-of-diagonal) is solved on the **CPU** via
///    forward-substitution with L₁₁ (O(b²(n−k)/2)).
/// 5. The trailing submatrix is updated on the **GPU** via a dedicated GEMM
///    kernel: `A₂₂ -= L₂₁ · U₁₂` (O(2b(n−k)²)).
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Singular matrix (exact zero pivot).
pub fn lu_decompose_blocked(
    device: &WgpuDevice,
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

    // Download the full matrix to host.
    let mut host = vec![0.0f32; n * n];
    device.download(matrix.buffer, &mut host)?;

    // Keep a copy of the original matrix for the host-side solve/inv/det API.
    let original_host = host.clone();

    // Device-resident buffer for the packed L/U factors.
    // Allocate zeroed; the loop writes to it via write_device_buffer.
    let factors_buf = device.alloc_zeroed::<f32>(n * n)?;

    // Track cumulative row permutation applied to the full matrix.
    let mut perm: Vec<usize> = (0..n).collect();

    let block_size = LU_BLOCK_SIZE.min(n);

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let trail = n - k - b;

        // ── Step 1: Factor the diagonal block A[k..k+b, k..k+b] on CPU ──
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
                // Swap trailing columns in host.
                for j in (k + b)..n {
                    host.swap(row_a * n + j, row_b * n + j);
                }
                // Update cumulative permutation.
                perm.swap(row_a, row_b);
            }
        }

        // Write the factored diagonal block (packed L/U) back to host.
        for i in 0..b {
            for j in 0..b {
                host[(k + i) * n + (k + j)] = diag[i * b + j];
            }
        }

        if trail == 0 {
            write_device_buffer(device, &host, &factors_buf)?;
            continue;
        }

        // ── Step 2: Solve L₂₁ = A₂₁ · U₁₁⁻¹ on CPU ──
        // For each column j of U₁₁, solve forward:
        //   L₂₁[i,j] = (A₂₁[i,j] - Σₚ₌₀ʲ⁻¹ L₂₁[i,p] · U₁₁[p,j]) / U₁₁[j,j]
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
        // For each row i (unit diagonal L₁₁):
        //   U₁₂[i,j] = A₁₂[i,j] - Σₚ₌₀ⁱ⁻¹ L₁₁[i,p] · U₁₂[p,j]
        for j in 0..trail {
            for i in 0..b {
                let mut s = host[(k + i) * n + (k + b + j)];
                for p in 0..i {
                    s -= diag[i * b + p] * host[(k + p) * n + (k + b + j)];
                }
                host[(k + i) * n + (k + b + j)] = s;
            }
        }

        // ── Step 4: Trailing GEMM update on GPU: A₂₂ -= L₂₁ · U₁₂ ──
        // Upload the updated host so the device buffer has L₂₁, U₁₂.
        write_device_buffer(device, &host, &factors_buf)?;

        // Upload L₂₁ and U₁₂ as compact buffers for the GEMM kernel.
        let mut l21 = vec![0.0f32; trail * b];
        let mut u12 = vec![0.0f32; b * trail];
        for i in 0..trail {
            for j in 0..b {
                l21[i * b + j] = host[(k + b + i) * n + (k + j)];
            }
        }
        for i in 0..b {
            for j in 0..trail {
                u12[i * trail + j] = host[(k + i) * n + (k + b + j)];
            }
        }

        let l21_buf = device.upload(&l21)?;
        let u12_buf = device.upload(&u12)?;

        // A₂₂ lives at rows [k+b..n], cols [k+b..n] in the full buffer.
        let trail_layout =
            leto::Layout::new([trail, trail], [n as isize, 1], (k + b) * n + (k + b));

        gemm_trailing_update(
            device,
            GemmTrailingUpdate {
                a_buf: &l21_buf,
                a_rows: trail,
                a_cols: b,
                b_buf: &u12_buf,
                b_cols: trail,
                c_buf: &factors_buf,
                c_layout: &trail_layout,
            },
        )?;

        // Download the updated trailing matrix back to host.
        device.download(&factors_buf, &mut host)?;
    }

    // Build a leto-ops LU on the original (un-permuted) matrix for the
    // host-side solve/inv/det API.
    let original_view =
        leto::ArrayView::<f32, 2>::new(leto::Layout::c_contiguous([n, n]).unwrap(), &original_host);
    let inner =
        leto_ops::lu_decompose(&original_view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("LU blocked finalisation failed: {e}"),
        })?;

    Ok(GpuLuDecomposition {
        inner,
        factors: factors_buf,
        n,
    })
}
