use std::any::TypeId;

use hephaestus_core::{BlockedCholeskyBackend, HephaestusError, Result, TrailingSyrk};
use leto::Layout;

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

// ---------------------------------------------------------------------------
// SYRK uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the SYRK compute kernel, matching the WGSL `SyrkMeta`
/// struct.  The matrix layout fields describe the **trailing matrix** C;
/// `panel_cols` is the rank-k dimension of the panel L₂₁.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
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
pub(super) fn syrk_trailing_update(
    device: &WgpuDevice,
    encoder: &mut wgpu::CommandEncoder,
    trail: &WgpuBuffer<f32>,
    trail_layout: &Layout<2>,
    panel_cols: usize,
    panel_offset: usize,
    panel_stride: usize,
) -> Result<()> {
    let [rows, cols] = trail_layout.shape();
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
            u32::try_from(trail_layout.strides()[0]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK row stride {} exceeds u32", trail_layout.strides()[0]),
                }
            })?,
            u32::try_from(trail_layout.strides()[1]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK col stride {} exceeds u32", trail_layout.strides()[1]),
                }
            })?,
        ],
        offset: u32::try_from(trail_layout.offset()).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("SYRK offset {} exceeds u32", trail_layout.offset()),
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

    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<SyrkMeta>(1)?)?;

    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

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
/// The WGPU blocked-Cholesky operation the shared `blocked_cholesky` loop
/// drives (ADR 0003).
///
/// Only the trailing SYRK kernel is WGPU-specific: the loop owns the panel
/// iteration, the CPU panel factorisation, the diagonal retention, and the
/// panel scatter. The SYRK reads both operands from the matrix buffer, and
/// this impl submits its own encoder because the trait hands it no batch.
impl BlockedCholeskyBackend for WgpuDevice {
    fn syrk_trailing(&self, matrix: &Self::Buffer, spec: TrailingSyrk) -> Result<()> {
        let trail_layout = Layout::try_new(
            [spec.trail_extent, spec.trail_extent],
            [spec.matrix_stride as isize, 1],
            spec.trail_offset,
        )
        .expect("invariant: submatrix layout derives from a validated parent");

        let mut encoder = self
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cholesky-syrk-update"),
            });
        syrk_trailing_update(
            self,
            &mut encoder,
            matrix,
            &trail_layout,
            spec.panel_cols,
            spec.panel_offset,
            spec.panel_stride,
        )?;
        self.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
