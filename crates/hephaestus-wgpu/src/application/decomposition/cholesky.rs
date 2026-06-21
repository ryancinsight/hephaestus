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
use crate::UniformBufferGuard;

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
    /// Row offset in the panel buffer where the active panel begins.
    panel_row_offset: u32,
    _pad: u32,
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
    panel_row_offset: u32,
    _pad: u32,
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
        // Load `panel[row + panel_row_offset, tile*16 + local_col]` into shared memory.
        let panel_col = tile * 16u + local_col;
        if (row < rows && panel_col < k) {{
            panel_row[local_row][local_col] = panel[(row + syrk_meta.panel_row_offset) * k + panel_col];
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
                    let b_val = panel[(col + syrk_meta.panel_row_offset) * k + ki];
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
    encoder: &mut wgpu::CommandEncoder,
    trail: &WgpuBuffer<f32>,
    trail_layout: &Layout<2>,
    panel: &WgpuBuffer<f32>,
    panel_cols: usize,
    panel_row_offset: usize,
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
        panel_row_offset: u32::try_from(panel_row_offset).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("SYRK panel row offset {panel_row_offset} exceeds u32"),
            }
        })?,
        _pad: 0,
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<SyrkKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-syrk",
        syrk_shader_source,
    );

    let raw_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<SyrkMeta>(1)?)?;
    let meta_buf = UniformBufferGuard::new(device.clone(), raw_meta_buf);
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Cholesky Panel Gather/Scatter compute kernels
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct CholCopyMeta {
    k: u32,
    b: u32,
    panel_rows: u32,
    n: u32,
}

// SAFETY: CholCopyMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for CholCopyMeta {}

struct CholCopyKernel;

fn chol_gather_shader_source() -> String {
    r#"struct CopyMeta {
    k: u32,
    b: u32,
    panel_rows: u32,
    n: u32,
}
@group(0) @binding(0) var<storage, read_write> main_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> panel_matrix: array<f32>;
@group(0) @binding(2) var<uniform> params: CopyMeta;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let total_elements = params.panel_rows * params.b;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }
    let r = idx / params.b;
    let c = idx % params.b;
    let src_idx = (params.k + r) * params.n + (params.k + c);
    panel_matrix[idx] = main_matrix[src_idx];
}
"#
    .to_string()
}

fn chol_scatter_shader_source() -> String {
    r#"struct CopyMeta {
    k: u32,
    b: u32,
    panel_rows: u32,
    n: u32,
}
@group(0) @binding(0) var<storage, read_write> main_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> panel_matrix: array<f32>;
@group(0) @binding(2) var<uniform> params: CopyMeta;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let total_elements = params.panel_rows * params.b;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }
    let r = idx / params.b;
    let c = idx % params.b;
    let dst_idx = (params.k + r) * params.n + (params.k + c);
    main_matrix[dst_idx] = panel_matrix[idx];
}
"#
    .to_string()
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

    // Device-resident buffer for the packed lower factor, updated in-place
    // by the SYRK trailing kernel.
    let lower_buf = device.upload(&host)?;

    let block_size = BLOCK_SIZE.min(n);

    // Pre-allocate device-resident panel buffer of maximum needed size: n * block_size.
    let panel_dev = device.alloc_zeroed::<f32>(n * block_size)?;

    let gather_pipeline = cached_pipeline(
        device,
        (TypeId::of::<CholCopyKernel>(), TypeId::of::<f32>(), 0),
        "hephaestus-chol-gather",
        chol_gather_shader_source,
    );
    let scatter_pipeline = cached_pipeline(
        device,
        (TypeId::of::<CholCopyKernel>(), TypeId::of::<f32>(), 1),
        "hephaestus-chol-scatter",
        chol_scatter_shader_source,
    );

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = n - k;

        // ── Step 1: Gather active panel column on the GPU, then download contiguously ──
        let copy_meta = CholCopyMeta {
            k: k as u32,
            b: b as u32,
            panel_rows: panel_rows as u32,
            n: n as u32,
        };
        let raw_copy_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<CholCopyMeta>(1)?)?;
        let copy_meta_buf = UniformBufferGuard::new(device.clone(), raw_copy_meta_buf);
        device
            .queue()
            .write_buffer(&copy_meta_buf, 0, bytemuck::bytes_of(&copy_meta));

        let gather_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-cholesky-gather-bind-group"),
                layout: &gather_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lower_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: panel_dev.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: copy_meta_buf.as_entire_binding(),
                    },
                ],
            });

        let scatter_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-cholesky-scatter-bind-group"),
                layout: &scatter_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lower_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: panel_dev.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: copy_meta_buf.as_entire_binding(),
                    },
                ],
            });

        let mut gather_encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cholesky-panel-gather"),
            });
        {
            let mut pass = gather_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-cholesky-panel-gather"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gather_pipeline);
            pass.set_bind_group(0, &gather_bind_group, &[]);
            let total_elements = panel_rows * b;
            let wg_x = total_elements.div_ceil(256);
            pass.dispatch_workgroups(wg_x as u32, 1, 1);
        }
        device.queue().submit(Some(gather_encoder.finish()));

        let mut panel_host = vec![0.0f32; panel_rows * b];
        device.download_sub_buffer(&panel_dev, 0, &mut panel_host)?;

        // ── Step 1.5: factor the diagonal block A[k..k+b, k..k+b] on CPU ──
        let mut diag_host = vec![0.0f32; b * b];
        for i in 0..b {
            for j in 0..b {
                diag_host[i * b + j] = panel_host[i * b + j];
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

        // Write the factored diagonal block back to panel_host
        for i in 0..b {
            for j in 0..b {
                panel_host[i * b + j] = diag_slice[i * b + j];
            }
        }

        let trail_rows = n - k - b;
        if trail_rows == 0 {
            // Upload the final diagonal block back to device buffer and scatter
            device.write_sub_buffer(&panel_dev, 0, &panel_host)?;

            let mut scatter_encoder = device
                .inner()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("hephaestus-cholesky-panel-scatter"),
                });
            {
                let mut pass = scatter_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hephaestus-cholesky-panel-scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&scatter_pipeline);
                pass.set_bind_group(0, &scatter_bind_group, &[]);
                let total_elements = panel_rows * b;
                let wg_x = total_elements.div_ceil(256);
                pass.dispatch_workgroups(wg_x as u32, 1, 1);
            }
            device.queue().submit(Some(scatter_encoder.finish()));
            continue;
        }

        // ── Step 2: panel solve  L₂₁ = A₂₁ · L₁₁⁻ᵀ  on CPU ──
        let mut rhs = vec![0.0f32; trail_rows * b];
        for i in 0..trail_rows {
            for j in 0..b {
                rhs[i * b + j] = panel_host[(b + i) * b + j];
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

        // Write L₂₁ back to panel_host
        for i in 0..trail_rows {
            for j in 0..b {
                panel_host[(b + i) * b + j] = rhs[i * b + j];
            }
        }

        // Upload the entire updated active panel (diagonal + off-diagonal) back to device buffer and scatter
        device.write_sub_buffer(&panel_dev, 0, &panel_host)?;

        // ── Step 3: trailing SYRK update on GPU ──
        let trail_layout = leto::Layout::new(
            [trail_rows, trail_rows],
            [n as isize, 1],
            (k + b) * n + (k + b),
        );

        let mut scatter_encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cholesky-panel-scatter-and-update"),
            });

        // 1. Scatter Compute Pass
        {
            let mut pass = scatter_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-cholesky-panel-scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&scatter_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            let total_elements = panel_rows * b;
            let wg_x = total_elements.div_ceil(256);
            pass.dispatch_workgroups(wg_x as u32, 1, 1);
        }

        // 2. Trailing SYRK Update
        syrk_trailing_update(
            device,
            &mut scatter_encoder,
            &lower_buf,
            &trail_layout,
            &panel_dev,
            b,
            b,
        )?;

        device.queue().submit(Some(scatter_encoder.finish()));
    }

    // Download the full factored matrix to host
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
