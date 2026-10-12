use std::any::TypeId;

use hephaestus_core::{HephaestusError, Result};

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

// ---------------------------------------------------------------------------
// Strict-upper triangular zero
// ---------------------------------------------------------------------------

/// Packed metadata for the triangular-zero kernel, matching the WGSL
/// `TriangleMeta` struct.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct TriangleMeta {
    /// Order of the square matrix.
    n: u32,
    /// Element count, carried so the shader needs no multiply that could
    /// overflow silently at `u32` width.
    total: u32,
}

struct StrictUpperZeroKernel;

/// WGSL source for the in-place strict-upper triangular zero.
///
/// Each invocation owns one flat index of the row-major matrix and writes only
/// when that index lies strictly above the diagonal, so the factor itself is
/// never read back or rewritten.
fn strict_upper_zero_shader_source() -> String {
    r#"struct TriangleMeta {
    n: u32,
    total: u32,
}
@group(0) @binding(0) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(1) var<uniform>             params: TriangleMeta;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= params.total) {
        return;
    }
    let r = idx / params.n;
    let c = idx % params.n;
    if (c > r) {
        matrix[idx] = 0.0;
    }
}
"#
    .to_string()
}

/// Zero the strictly-upper triangle of a row-major `n` x `n` device buffer in
/// place.
///
/// The blocked factorisation leaves the strict upper triangle holding the
/// input's values outside the diagonal blocks (the per-panel scatters cover
/// only `row >= blockstart(col)`), while the diagonal blocks are already
/// zeroed there by [`hephaestus_core::factor_cholesky_panel`]. This pass
/// finishes the factor on the device, replacing an `n^2` host upload whose
/// only remaining effect was that zeroing.
///
/// # Errors
///
/// - `LengthMismatch` when `matrix.len != n * n`.
/// - `DispatchFailed` when the element count exceeds the shader's `u32` index
///   width or the workgroup count exceeds `u32::MAX`.
pub(super) fn zero_strict_upper(
    device: &WgpuDevice,
    matrix: &WgpuBuffer<f32>,
    n: usize,
) -> Result<()> {
    let total = n
        .checked_mul(n)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("cholesky dimension {n} overflows an element count"),
        })?;
    if matrix.len != total {
        return Err(HephaestusError::LengthMismatch {
            host_len: total,
            device_len: matrix.len,
        });
    }
    if total == 0 {
        return Ok(());
    }

    let meta = TriangleMeta {
        n: u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("cholesky dimension {n} exceeds u32"),
        })?,
        total: u32::try_from(total).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("cholesky element count {total} exceeds u32"),
        })?,
    };

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<StrictUpperZeroKernel>(),
            TypeId::of::<f32>(),
            256,
        ),
        "hephaestus-cholesky-strict-upper-zero",
        strict_upper_zero_shader_source,
    );

    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<TriangleMeta>(1)?)?;

    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-cholesky-strict-upper-zero-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let workgroups =
        u32::try_from(total.div_ceil(256)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!(
                "cholesky triangular-zero workgroup count {} exceeds u32::MAX",
                total.div_ceil(256)
            ),
        })?;

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cholesky-strict-upper-zero"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-cholesky-strict-upper-zero-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}
