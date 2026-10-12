use std::any::TypeId;

use hephaestus_core::{HephaestusError, Result};

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

// ---------------------------------------------------------------------------
// GEMM uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the trailing GEMM compute kernel.
/// Computes **C -= A · B** where A is (m×k), B is (k×n), C is (m×n).
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct GemmMeta {
    /// Shape: [m, n, k, padding].
    shape: [u32; 4],
    /// Row strides: [C row-stride, A row-stride, B row-stride, padding].
    strides: [u32; 4],
    /// Element offsets: [C offset, A offset, B offset, padding].
    offsets: [u32; 4],
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
    shape: vec4<u32>,
    strides: vec4<u32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read_write> a_buf: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> b_buf: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> c_buf: array<{ty}>;
@group(0) @binding(3) var<uniform>             params: GemmMeta;

var<workgroup> tile_a: array<{ty}, 256>;
var<workgroup> tile_b: array<{ty}, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>
) {{
    let row = gid.y;
    let col = gid.x;
    let m = params.shape.x;
    let n = params.shape.y;
    let k = params.shape.z;
    let c_stride = params.strides.x;
    let a_stride = params.strides.y;
    let b_stride = params.strides.z;
    let c_off = params.offsets.x;
    let a_off = params.offsets.y;
    let b_off = params.offsets.z;

    var sum = {ty}({zero});
    let num_tiles = (k + 15u) / 16u;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {{
        // Load tile of A: A[row, tile*16 + lid.x]
        let a_col = tile * 16u + lid.x;
        if (row < m && a_col < k) {{
            tile_a[lid.y * 16u + lid.x] = a_buf[a_off + row * a_stride + a_col];
        }} else {{
            tile_a[lid.y * 16u + lid.x] = {ty}({zero});
        }}

        // Load tile of B: B[tile*16 + lid.y, col]
        let b_row = tile * 16u + lid.y;
        if (b_row < k && col < n) {{
            tile_b[lid.y * 16u + lid.x] = b_buf[b_off + b_row * b_stride + col];
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
    if (row < m && col < n) {{
        c_buf[c_idx] = c_buf[c_idx] - sum;
    }}
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct GemmKernel;

pub(crate) struct GemmTrailingUpdate<'a> {
    pub(crate) a_buf: &'a WgpuBuffer<f32>,
    pub(crate) a_offset: usize,
    pub(crate) a_stride: usize,
    pub(crate) a_rows: usize,
    pub(crate) a_cols: usize,
    pub(crate) b_buf: &'a WgpuBuffer<f32>,
    pub(crate) b_offset: usize,
    pub(crate) b_stride: usize,
    pub(crate) b_cols: usize,
    pub(crate) c_buf: &'a WgpuBuffer<f32>,
    pub(crate) c_offset: usize,
    pub(crate) c_stride: usize,
}

/// GPU dispatch for the trailing GEMM:  **C -= A · B**
///
/// A is (m×k) starting at `a_offset` with row stride `a_stride`,
/// B is (k×n) starting at `b_offset` with row stride `b_stride`,
/// C is (m×n) starting at `c_offset` with row stride `c_stride`.
pub(crate) fn gemm_trailing_update(
    device: &WgpuDevice,
    update: GemmTrailingUpdate<'_>,
) -> Result<()> {
    let m = update.a_rows;
    let k = update.a_cols;
    let n = update.b_cols;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

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
            0,
        ],
        strides: [
            u32::try_from(update.c_stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM C row stride {} exceeds u32", update.c_stride),
            })?,
            u32::try_from(update.a_stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM A row stride {} exceeds u32", update.a_stride),
            })?,
            u32::try_from(update.b_stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM B row stride {} exceeds u32", update.b_stride),
            })?,
            0,
        ],
        offsets: [
            u32::try_from(update.c_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM C offset {} exceeds u32", update.c_offset),
            })?,
            u32::try_from(update.a_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM A offset {} exceeds u32", update.a_offset),
            })?,
            u32::try_from(update.b_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM B offset {} exceeds u32", update.b_offset),
            })?,
            0,
        ],
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<GemmKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-gemm",
        gemm_shader_source,
    );

    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<GemmMeta>(1)?)?;

    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

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

    Ok(())
}
