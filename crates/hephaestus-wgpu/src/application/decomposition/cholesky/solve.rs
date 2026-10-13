//! Blocked Cholesky: substitution solves and the blocked backend.

use std::any::TypeId;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

// ---------------------------------------------------------------------------
// Device-side triangular solves against the resident factor
// ---------------------------------------------------------------------------

/// Rows per triangular-solve block: the workgroup width of the block kernel,
/// so one workgroup owns one diagonal block of **L** with a lane per row.
const SOLVE_BLOCK_WIDTH: u32 = 256;

/// [`SOLVE_BLOCK_WIDTH`] as an index count for host-side blocking arithmetic.
const SOLVE_BLOCK: usize = SOLVE_BLOCK_WIDTH as usize;

/// Packed metadata for one triangular-solve dispatch, matching the WGSL
/// `SolveMeta` struct.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct SolveMeta {
    /// Matrix dimension *n* of the *n* × *n* factor.
    n: u32,
    /// First row of the diagonal block this dispatch works on.
    block_start: u32,
    /// Rows in the block: `SOLVE_BLOCK`, or the ragged tail for the last one.
    block_len: u32,
    /// Rows the trailing update touches: everything after the block going
    /// forward, everything before it going backward. Unused by the block
    /// solve, which pads the struct to its 16-byte uniform stride.
    update_rows: u32,
}

/// Which triangular system a solve pass works on.
///
/// Forward substitution solves **L** · **y** = **b** from the first block
/// down; backward substitution solves **L**ᵀ · **x** = **y** from the last
/// block up, reading **L** transposed so no second factor is materialized.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Substitution {
    Forward,
    Backward,
}

impl Substitution {
    /// WGSL expression for the factor element coupling `row` to column
    /// `col` of the current system: `L[row, col]` forward, `L[col, row]`
    /// backward.
    fn coefficient(self) -> &'static str {
        match self {
            Self::Forward => "lower[row * params.n + col]",
            Self::Backward => "lower[col * params.n + row]",
        }
    }
}

struct ForwardBlockSolveKernel;

struct ForwardTrailingUpdateKernel;

struct BackwardBlockSolveKernel;

struct BackwardTrailingUpdateKernel;

/// WGSL source for the diagonal-block solve.
///
/// One workgroup solves the block's `block_len × block_len` triangle against
/// its slice of `rhs` in place: lane `i` owns local row `i`, and each step
/// `c` finalizes row `c` (one division by the diagonal), publishes it through
/// workgroup memory, then every later row subtracts its coupling to it.
/// Forward substitution walks `c` upward and couples rows below the pivot;
/// backward substitution walks it downward and couples rows above. The
/// barrier sits in uniform control flow (`block_len` is a uniform), and each
/// workgroup slot is written exactly once per solve, so one barrier per step
/// orders the publish before its readers.
fn block_solve_shader_source(substitution: Substitution) -> String {
    let (pivot, coupled) = match substitution {
        Substitution::Forward => ("step", "i > c"),
        Substitution::Backward => ("params.block_len - 1u - step", "i < c"),
    };
    let coefficient = substitution.coefficient();
    format!(
        r#"struct SolveMeta {{
    n: u32,
    block_start: u32,
    block_len: u32,
    update_rows: u32,
}}
@group(0) @binding(0) var<storage, read>       lower: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs:   array<f32>;
@group(0) @binding(2) var<uniform>             params: SolveMeta;

var<workgroup> solved: array<f32, {SOLVE_BLOCK}u>;

@compute @workgroup_size({SOLVE_BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let i = lid.x;
    let row = params.block_start + i;
    let owns_row = i < params.block_len;
    var acc = 0.0;
    if (owns_row) {{
        acc = rhs[row];
    }}
    for (var step = 0u; step < params.block_len; step = step + 1u) {{
        let c = {pivot};
        let col = params.block_start + c;
        if (i == c) {{
            let value = acc / {coefficient};
            solved[c] = value;
            rhs[row] = value;
        }}
        workgroupBarrier();
        if (owns_row && {coupled}) {{
            acc = acc - {coefficient} * solved[c];
        }}
    }}
}}
"#
    )
}

/// WGSL source for the trailing update after a block solve.
///
/// One invocation per row outside the solved block subtracts that row's
/// coupling to every just-solved entry: `rhs[row] -= Σ_c coefficient(row, c)
/// · rhs[c]` over the block's columns. Forward substitution updates the rows
/// after the block; backward substitution updates the rows before it. The
/// block's own entries are only read, and every written row lies outside it,
/// so the dispatch is race-free without atomics.
fn trailing_update_shader_source(substitution: Substitution) -> String {
    let first_row = match substitution {
        Substitution::Forward => "params.block_start + params.block_len",
        Substitution::Backward => "0u",
    };
    let coefficient = substitution.coefficient();
    format!(
        r#"struct SolveMeta {{
    n: u32,
    block_start: u32,
    block_len: u32,
    update_rows: u32,
}}
@group(0) @binding(0) var<storage, read>       lower: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs:   array<f32>;
@group(0) @binding(2) var<uniform>             params: SolveMeta;

@compute @workgroup_size({SOLVE_BLOCK})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {{
    if (gid.x >= params.update_rows) {{
        return;
    }}
    let row = {first_row} + gid.x;
    var acc = rhs[row];
    for (var c = 0u; c < params.block_len; c = c + 1u) {{
        let col = params.block_start + c;
        acc = acc - {coefficient} * rhs[col];
    }}
    rhs[row] = acc;
}}
"#
    )
}

/// Build and cache one blocked-solve kernel pipeline keyed by marker type.
///
/// The marker keeps each kernel in its own cache slot while this helper
/// centralizes the shared cache-key shape.
fn solve_pipeline<K: 'static>(
    device: &WgpuDevice,
    label: &'static str,
    source: impl FnOnce() -> String,
) -> wgpu::ComputePipeline {
    cached_pipeline(
        device,
        (TypeId::of::<K>(), TypeId::of::<f32>(), SOLVE_BLOCK_WIDTH),
        label,
        source,
    )
}

/// The four kernels of the blocked solve, in the order a solve issues them.
#[derive(Clone, Copy)]
enum SolveKernel {
    ForwardBlock,
    ForwardUpdate,
    BackwardBlock,
    BackwardUpdate,
}

/// One dispatch of the blocked solve: its kernel, metadata, and width.
struct SolveDispatch {
    kernel: SolveKernel,
    meta: SolveMeta,
    workgroups: u32,
}

/// Solve **A** · **x** = **rhs** on the device against the resident factor.
///
/// Forward substitution on **L** then backward substitution on **L**ᵀ, each
/// blocked by `SOLVE_BLOCK` rows: per block one workgroup solves the diagonal
/// triangle in place, then a row-parallel dispatch applies the rank-`block`
/// update to the rows still unsolved. That is `2 · ⌈n / SOLVE_BLOCK⌉` block
/// solves plus their trailing updates per solve, all recorded in one compute
/// pass so WebGPU's per-dispatch ordering serializes the chain. The right-hand
/// side is copied device-to-device into the solution buffer first, so the
/// caller's buffer is untouched and nothing crosses the host.
///
/// # Errors
///
/// - `LengthMismatch` when `rhs.len != n`.
/// - `DispatchFailed` when a dimension or workgroup count exceeds `u32`.
pub(crate) fn device_solve(
    device: &WgpuDevice,
    lower: &WgpuBuffer<f32>,
    n: usize,
    rhs: &WgpuBuffer<f32>,
) -> Result<WgpuBuffer<f32>> {
    if rhs.len != n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n,
            device_len: rhs.len,
        });
    }
    let solution = device.alloc_zeroed::<f32>(n)?;
    if n == 0 {
        return Ok(solution);
    }
    let to_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("cholesky solve {what} {value} exceeds u32"),
        })
    };
    let n_u32 = to_u32(n, "dimension")?;

    let forward_block = solve_pipeline::<ForwardBlockSolveKernel>(
        device,
        "hephaestus-cholesky-forward-block-solve",
        || block_solve_shader_source(Substitution::Forward),
    );
    let forward_update = solve_pipeline::<ForwardTrailingUpdateKernel>(
        device,
        "hephaestus-cholesky-forward-trailing-update",
        || trailing_update_shader_source(Substitution::Forward),
    );
    let backward_block = solve_pipeline::<BackwardBlockSolveKernel>(
        device,
        "hephaestus-cholesky-backward-block-solve",
        || block_solve_shader_source(Substitution::Backward),
    );
    let backward_update = solve_pipeline::<BackwardTrailingUpdateKernel>(
        device,
        "hephaestus-cholesky-backward-trailing-update",
        || trailing_update_shader_source(Substitution::Backward),
    );

    let block_count = n.div_ceil(SOLVE_BLOCK);
    let mut dispatches = Vec::with_capacity(4 * block_count);
    let dispatch =
        |kernel: SolveKernel, block_start: usize, block_len: usize, update_rows: usize| {
            Ok::<_, HephaestusError>(SolveDispatch {
                kernel,
                meta: SolveMeta {
                    n: n_u32,
                    block_start: to_u32(block_start, "block start")?,
                    block_len: to_u32(block_len, "block length")?,
                    update_rows: to_u32(update_rows, "update rows")?,
                },
                // A block solve is exactly one workgroup; `update_rows` is zero
                // there and the block kernel never reads it.
                workgroups: to_u32(update_rows.div_ceil(SOLVE_BLOCK).max(1), "workgroup count")?,
            })
        };
    for k in 0..block_count {
        let block_start = k * SOLVE_BLOCK;
        let block_len = SOLVE_BLOCK.min(n - block_start);
        let remaining = n - block_start - block_len;
        dispatches.push(dispatch(
            SolveKernel::ForwardBlock,
            block_start,
            block_len,
            0,
        )?);
        if remaining > 0 {
            dispatches.push(dispatch(
                SolveKernel::ForwardUpdate,
                block_start,
                block_len,
                remaining,
            )?);
        }
    }
    for k in (0..block_count).rev() {
        let block_start = k * SOLVE_BLOCK;
        let block_len = SOLVE_BLOCK.min(n - block_start);
        dispatches.push(dispatch(
            SolveKernel::BackwardBlock,
            block_start,
            block_len,
            0,
        )?);
        if block_start > 0 {
            dispatches.push(dispatch(
                SolveKernel::BackwardUpdate,
                block_start,
                block_len,
                block_start,
            )?);
        }
    }

    let pipeline_for = |kernel: SolveKernel| match kernel {
        SolveKernel::ForwardBlock => &forward_block,
        SolveKernel::ForwardUpdate => &forward_update,
        SolveKernel::BackwardBlock => &backward_block,
        SolveKernel::BackwardUpdate => &backward_update,
    };
    let meta_size = WgpuDevice::byte_size::<SolveMeta>(1)?;
    let mut metas = Vec::with_capacity(dispatches.len());
    let mut bind_groups = Vec::with_capacity(dispatches.len());
    for dispatch in &dispatches {
        let meta_buf = device.get_uniform_buffer(meta_size)?;

        device
            .queue()
            .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&dispatch.meta));
        bind_groups.push(
            device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hephaestus-cholesky-solve-bind-group"),
                    layout: &pipeline_for(dispatch.kernel).get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: lower.raw().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: solution.raw().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: meta_buf.as_entire_binding(),
                        },
                    ],
                }),
        );
        metas.push(meta_buf);
    }

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cholesky-solve"),
        });
    encoder.copy_buffer_to_buffer(
        rhs.raw(),
        0,
        solution.raw(),
        0,
        WgpuDevice::byte_size::<f32>(n)?,
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-cholesky-solve-pass"),
            timestamp_writes: None,
        });
        for (dispatch, bind_group) in dispatches.iter().zip(&bind_groups) {
            pass.set_pipeline(pipeline_for(dispatch.kernel));
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dispatch.workgroups, 1, 1);
        }
    }
    device.queue().submit(Some(encoder.finish()));
    // The uniform guards return their buffers to the pool on drop; the
    // submitted command buffer already holds its own references.
    drop(metas);

    Ok(solution)
}
