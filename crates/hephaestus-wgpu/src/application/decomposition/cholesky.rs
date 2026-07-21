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

use hephaestus_core::{ComputeDevice, HephaestusError, Result, factor_cholesky_panel};
use leto::Layout;

use super::region::{
    MatrixRegion, download_matrix_region_compact_into, write_matrix_region_compact_reusable,
};
use super::validate::{validate_dense_operand, validate_square};
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
    strides: [u32; 2],
    /// Element offset into the trailing-matrix buffer.
    offset: u32,
    /// Rank-k dimension (number of columns in the panel).
    panel_cols: u32,
    /// Element offset in the panel buffer where the active panel begins.
    panel_offset: u32,
    /// Row stride of the panel buffer.
    panel_stride: u32,
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
    strides: vec2<u32>,
    offset: u32,
    panel_cols: u32,
    panel_offset: u32,
    panel_stride: u32,
}}

@group(0) @binding(0) var<storage, read_write> trail:  array<{ty}>;
@group(0) @binding(1) var<uniform>             syrk_meta: SyrkMeta;

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
            panel_row[local_row][local_col] = trail[syrk_meta.panel_offset + row * syrk_meta.panel_stride + panel_col];
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
                    let b_val = trail[syrk_meta.panel_offset + col * syrk_meta.panel_stride + ki];
                    sum = sum + a_val * b_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write back: C[row, col] -= sum
    if (row < rows && col < cols && col <= row) {{
        let c_off = off + row * stride_row + col * stride_col;
        trail[c_off] = trail[c_off] - sum;
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
    panel_cols: usize,
    panel_offset: usize,
    panel_stride: usize,
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
            u32::try_from(trail_layout.strides[0]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK row stride {} exceeds u32", trail_layout.strides[0]),
                }
            })?,
            u32::try_from(trail_layout.strides[1]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK col stride {} exceeds u32", trail_layout.strides[1]),
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
        panel_offset: u32::try_from(panel_offset).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("SYRK panel offset {panel_offset} exceeds u32"),
        })?,
        panel_stride: u32::try_from(panel_stride).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("SYRK panel stride {panel_stride} exceeds u32"),
        })?,
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<SyrkKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-syrk",
        syrk_shader_source,
    );

    let raw_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<SyrkMeta>(1)?)?;
    let meta_buf = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buf);
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
                    resource: trail.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
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
            leto::Layout::c_contiguous([0, 0]).expect("infallible: empty matrix layout"),
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
/// - Non-dense (non-C-contiguous / offset / broadcast) operand: the
///   blocked path bulk-copies the matrix storage on the device.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
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
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cholesky-copy"),
        });
    // Raw whole-matrix copy: sound only for dense C-contiguous
    // zero-offset operands, enforced by `validate_dense_operand` at the
    // entry point (a strided/offset/broadcast view would copy the wrong
    // elements or exceed the operand's storage extent).
    encoder.copy_buffer_to_buffer(
        &matrix.buffer.buffer,
        0,
        &lower_buf.buffer,
        0,
        WgpuDevice::byte_size::<f32>(n * n)?,
    );
    device.queue().submit(Some(encoder.finish()));

    let block_size = BLOCK_SIZE.min(n);

    // Pre-allocate a reusable compact buffer for the maximum panel region:
    // panel_rows = n - k <= n, cols = b <= block_size  =>  max n * block_size elements.
    let panel_compact_buf = device.alloc_zeroed::<f32>(n * block_size)?;

    let mut host = vec![0.0f32; n * n];

    // Per-panel host scratch, allocated once and resized by the panel download
    // each iteration instead of allocating a fresh `Vec` per panel.
    let mut panel: Vec<f32> = Vec::with_capacity(n * block_size);

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = n - k;

        // ── Step 1: Download active panel region to host ──
        let panel_region = MatrixRegion {
            stride: n,
            row_start: k,
            col_start: k,
            rows: panel_rows,
            cols: b,
        };
        download_matrix_region_compact_into(
            device,
            &lower_buf,
            &panel_compact_buf,
            panel_region,
            &mut panel,
        )?;

        let trail_rows = n - k - b;
        // Factor this panel on the host (diagonal-block Cholesky + off-diagonal
        // triangular solve) — the backend-neutral shared computation.
        factor_cholesky_panel(&mut panel, b, trail_rows)?;

        if trail_rows == 0 {
            // Copy the finalized columns to the host-side packed matrix
            for j in 0..b {
                let col = k + j;
                for r in k..n {
                    host[r * n + col] = panel[(r - k) * b + j];
                }
            }

            // Upload the final diagonal block back to device buffer
            write_matrix_region_compact_reusable(
                device,
                &lower_buf,
                &panel_compact_buf,
                &panel,
                panel_region,
            )?;
            continue;
        }

        // Copy the finalized columns to the host-side packed matrix
        for j in 0..b {
            let col = k + j;
            for r in k..n {
                host[r * n + col] = panel[(r - k) * b + j];
            }
        }

        // Upload the entire updated active panel (diagonal + off-diagonal) back to device buffer
        write_matrix_region_compact_reusable(
            device,
            &lower_buf,
            &panel_compact_buf,
            &panel,
            panel_region,
        )?;

        // ── Step 3: trailing SYRK update on GPU ──
        let trail_layout = leto::Layout::new(
            [trail_rows, trail_rows],
            [n as isize, 1],
            (k + b) * n + (k + b),
        );

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cholesky-syrk-update"),
            });

        syrk_trailing_update(
            device,
            &mut encoder,
            &lower_buf,
            &trail_layout,
            b,
            (k + b) * n + k,
            n,
        )?;

        device.queue().submit(Some(encoder.finish()));
    }

    // Update the device buffer to zero out the strictly upper-triangular part asynchronously
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
