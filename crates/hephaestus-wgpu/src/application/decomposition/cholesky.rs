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

use bytemuck::Pod;
use std::any::TypeId;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Layout;

use super::validate::validate_square;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

// ---------------------------------------------------------------------------
// SYRK uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the SYRK compute kernel, matching the WGSL `SyrkMeta`
/// struct.  The matrix layout fields describe the **trailing matrix** C;
/// `panel_cols` is the rank-k dimension of the panel L₂₁.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct SyrkMeta {
    /// Shape of the trailing matrix: `[rows, cols]`.
    shape: [u32; 2],
    /// Row-major strides of the trailing matrix.
    strides: [i32; 2],
    /// Element offset into the trailing-matrix buffer.
    offset: u32,
    /// Rank-k dimension (number of columns in the panel).
    panel_cols: u32,
    _pad: [u32; 2],
}

// SAFETY: SyrkMeta is `#[repr(C)]` and every field is Pod.
unsafe impl Pod for SyrkMeta {}

// ---------------------------------------------------------------------------
// SYRK kernel
// ---------------------------------------------------------------------------

/// WGSL source for the rank-k symmetric update
///
/// ```text
/// C[i,j] -= Σₖ B[i,k] · B[j,k]
/// ```
///
/// where `C` is the trailing matrix and `B` is the panel (`rows × k`).
/// Each workgroup processes a 16×16 tile of `C` using shared-memory
/// cooperative loading of panel rows, identical in spirit to the tiled matmul
/// kernel but specialised for the symmetric case.
fn syrk_shader_source() -> String {
    // WGSL f32 literal is always "f32(...)".
    const TY: &str = "f32";
    const ZERO: &str = "0.0";

    format!(
        r#"struct SyrkMeta {{
    shape: vec2<u32>,
    strides: vec2<i32>,
    offset: u32,
    panel_cols: u32,
    _pad: vec2<u32>,
}}

@group(0) @binding(0) var<storage, read>      panel: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> trail:  array<{ty}>;
@group(0) @binding(2) var<uniform>             syrk_meta: SyrkMeta;

var<workgroup> panel_row: array<array<{ty}, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wid:  vec3<u32>,
) {{
    let col = gid.x;
    let row = gid.y;
    let local_col = lid.x;
    let local_row = lid.y;

    let rows = syrk_meta.shape.x;
    let cols = syrk_meta.shape.y;
    let k    = syrk_meta.panel_cols;
    let stride_row = syrk_meta.strides.x;
    let stride_col = syrk_meta.strides.y;
    let off        = syrk_meta.offset;

    var sum = {ty}({zero});
    let num_tiles = (k + 15u) / 16u;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {{
        // Load `panel[row, tile*16 + local_col]` into shared memory.
        let panel_col = tile * 16u + local_col;
        if (row < rows && panel_col < k) {{
            panel_row[local_row][local_col] = panel[row * k + panel_col];
        }} else {{
            panel_row[local_row][local_col] = {ty}({zero});
        }}
        workgroupBarrier();

        // Each thread computes its own output element.
        // Re-load `panel[row, t*16 + i]` for each i in the tile from shared
        // memory, and load `panel[col, t*16 + i]` directly from global memory.
        if (row < rows && col < cols && col <= row) {{
            for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
                let ki = tile * 16u + i;
                if (ki < k) {{
                    let a_val = panel_row[local_row][i];
                    let b_val = panel[col * k + ki];
                    sum = sum + a_val * b_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write back: C[row, col] -= sum
    if (row < rows && col < cols && col <= row) {{
        let c_off = i32(off) + i32(row) * stride_row + i32(col) * stride_col;
        trail[u32(c_off)] = trail[u32(c_off)] - sum;
    }}
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct SyrkKernel;

/// GPU dispatch for the rank-k symmetric trailing-matrix update
///
/// ```text
/// trail[row, col] -= Σₖ panel[row, k] · panel[col, k]
/// ```
///
/// Only the **lower triangle** (`col <= row`) of the trailing matrix is
/// updated, which is sufficient for the blocked Cholesky loop.
fn syrk_trailing_update(
    device: &WgpuDevice,
    trail: &WgpuBuffer<f32>,
    trail_layout: &Layout<2>,
    panel: &WgpuBuffer<f32>,
    panel_cols: usize,
) -> Result<()> {
    let [rows, cols] = trail_layout.shape;
    if rows == 0 || cols == 0 || panel_cols == 0 {
        return Ok(());
    }

    let meta = SyrkMeta {
        shape: [
            u32::try_from(rows).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK row count {rows} exceeds u32"),
            })?,
            u32::try_from(cols).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK col count {cols} exceeds u32"),
            })?,
        ],
        strides: [
            i32::try_from(trail_layout.strides[0]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK row stride {} exceeds i32", trail_layout.strides[0]),
                }
            })?,
            i32::try_from(trail_layout.strides[1]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK col stride {} exceeds i32", trail_layout.strides[1]),
                }
            })?,
        ],
        offset: u32::try_from(trail_layout.offset).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("SYRK offset {} exceeds u32", trail_layout.offset),
            }
        })?,
        panel_cols: u32::try_from(panel_cols).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("SYRK panel cols {panel_cols} exceeds u32"),
        })?,
        _pad: [0; 2],
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<SyrkKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-syrk",
        syrk_shader_source,
    );

    let meta_buf = device.get_uniform_buffer(std::mem::size_of::<SyrkMeta>() as u64)?;
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-syrk"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: panel.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: trail.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-syrk"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-syrk"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x =
            u32::try_from(cols.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK workgroup x {} exceeds u32", cols.div_ceil(16)),
            })?;
        let wg_y =
            u32::try_from(rows.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK workgroup y {} exceeds u32", rows.div_ceil(16)),
            })?;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    device.recycle_uniform_buffer(meta_buf);

    Ok(())
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Lower-triangular Cholesky factor on the device, with host-side
/// decomposition for solve/inv/det without re-factorization.
pub struct GpuCholesky {
    /// Host-side leto-ops decomposition (owns the factor data).
    inner: leto_ops::CholeskyDecomposition<f32>,
    /// Device-resident lower-triangular factor **L** (*n* × *n*, row-major).
    lower: WgpuBuffer<f32>,
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
    pub fn lower(&self) -> &WgpuBuffer<f32> {
        &self.lower
    }

    /// Consume and return the lower-triangular factor buffer.
    #[must_use]
    #[inline]
    pub fn into_lower(self) -> WgpuBuffer<f32> {
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
                message: format!("Cholesky solve failed: {e}"),
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
                message: format!("Cholesky inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

// ---------------------------------------------------------------------------
// Entry point 1 — host delegation (unchanged)
// ---------------------------------------------------------------------------

/// Compute the Cholesky factorization **A** = **L** **L**ᵀ on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For large matrices where the O(n³) trailing update should run
/// on the GPU, prefer [`cholesky_decompose_blocked`].
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose(
    device: &WgpuDevice,
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
    let mut host_data = vec![0.0f32; matrix.buffer.len];
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
// Entry point 2 — blocked with GPU trailing SYRK
// ---------------------------------------------------------------------------

/// Panel block size for the blocked Cholesky algorithm.
///
/// A value of 64 balances CPU panel factorisation cost against GPU SYRK
/// launch overhead.  For matrices smaller than `BLOCK_SIZE` the algorithm
/// degrades gracefully to a single CPU panel pass.
const BLOCK_SIZE: usize = 64;

/// Blocked Cholesky factorization **A** = **L** **L**ᵀ with GPU-accelerated
/// trailing-matrix SYRK updates.
///
/// The algorithm processes the matrix in `BLOCK_SIZE × BLOCK_SIZE` panels.
/// For each panel *k*:
///
/// 1. The diagonal block is factored on the **CPU** via [`leto_ops`]
///    (O(b³/3)).
/// 2. The off-diagonal panel is solved on the **CPU** via triangular solve
///    (O(b²·(n−k)/2)).
/// 3. The trailing submatrix is updated on the **GPU** via a dedicated SYRK
///    kernel: `A₂₂ -= L₂₁ · L₂₁ᵀ` (O(b·(n−k)²/2)).
///
/// The SYRK trailing update is the dominant cost for large *n* and is the
/// reason this entry point exists: unlike [`cholesky_decompose`] which
/// delegates *all* O(n³) work to the CPU, this function offloads the
/// rank-k update to the GPU compute pipeline.
///
/// # Block-size tuning
///
/// `BLOCK_SIZE` is currently a compile-time constant.  A future refinement
/// could auto-tune based on the device's preferred workgroup size and
/// available shared memory.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose_blocked(
    device: &WgpuDevice,
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

    // Download the full matrix to host (panels are factored on CPU).
    let mut host = vec![0.0f32; n * n];
    device.download(matrix.buffer, &mut host)?;
    let original_host = host.clone();

    // Device-resident buffer for the packed lower factor, updated in-place
    // by the SYRK trailing kernel (the trailing submatrix *is* the Schur
    // complement, stored in the not-yet-factorised portion of L).
    let lower_buf = device.upload(&host)?;

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
        let diag_view =
            leto::ArrayView::<f32, 2>::new(leto::Layout::c_contiguous([b, b]).unwrap(), &diag_host);
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
        device.write_buffer(&lower_buf, &host)?;

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

        // Triangular solve: each column of L₂₁ via back-substitution with L₁₁ᵀ.
        for col in 0..b {
            // Back-substitute against L₁₁ᵀ (i.e. forward-substitute against L₁₁).
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
        device.write_buffer(&lower_buf, &host)?;
        let panel_buf = device.upload(&rhs)?;

        // ── Step 3: trailing SYRK update on GPU ──
        let trail_layout = leto::Layout::new(
            [trail_rows, trail_rows],
            [n as isize, 1],
            (k + b) * n + (k + b),
        );
        // WGPU storage bindings cannot bind the same buffer as read-only and
        // read-write in one dispatch, so the solved panel is uploaded as a
        // compact read-only buffer before the in-place trailing update.
        syrk_trailing_update(device, &lower_buf, &trail_layout, &panel_buf, b)?;

        // Download the updated trailing matrix back to host so the next
        // iteration's panel factorisation sees the Schur complement.
        device.download(&lower_buf, &mut host)?;
    }

    for row in 0..n {
        for col in (row + 1)..n {
            host[row * n + col] = 0.0;
        }
    }
    device.write_buffer(&lower_buf, &host)?;

    let original_view =
        leto::ArrayView::<f32, 2>::new(leto::Layout::c_contiguous([n, n]).unwrap(), &original_host);
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
