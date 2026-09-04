//! Device-resident triangular unpacking of a packed LU factorisation.
//!
//! [`hephaestus_core::split_packed_lu`] performs the same split on the host.
//! Callers holding the packed factor on the device would otherwise pay three
//! full-matrix transfers per factorisation — one download and two uploads — to
//! reach the same result. This module writes **L** and **U** directly from the
//! packed device buffer, so the factors never leave device memory.
//!
//! The split is a pure copy of packed entries plus the structural constants
//! `0` and `1`; no arithmetic is performed, so the device result is bitwise
//! identical to the host oracle rather than merely close to it.

use std::any::TypeId;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::uniform_guard;

/// Packed metadata for the split kernel, matching the WGSL `SplitMeta` struct.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct SplitMeta {
    /// Matrix dimension *n* of the *n* × *n* packed factor.
    n: u32,
    /// Element count `n * n`, precomputed to keep the bounds test in the
    /// shader free of a multiply that would overflow silently at `u32` width.
    total: u32,
}

struct SplitPackedLuKernel;

/// WGSL source for the triangular unpack.
///
/// Each invocation owns one flat index of the row-major matrix and writes
/// exactly one element of **L** and one of **U**, so both outputs are fully
/// defined by a single dispatch and neither needs pre-zeroing.
fn split_packed_lu_shader_source() -> String {
    r#"struct SplitMeta {
    n: u32,
    total: u32,
}
@group(0) @binding(0) var<storage, read>       packed: array<f32>;
@group(0) @binding(1) var<storage, read_write> lower:  array<f32>;
@group(0) @binding(2) var<storage, read_write> upper:  array<f32>;
@group(0) @binding(3) var<uniform>             params: SplitMeta;

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
    let value = packed[idx];
    if (r > c) {
        lower[idx] = value;
        upper[idx] = 0.0;
    } else if (r == c) {
        lower[idx] = 1.0;
        upper[idx] = value;
    } else {
        lower[idx] = 0.0;
        upper[idx] = value;
    }
}
"#
    .to_string()
}

/// Split a device-resident packed LU factorisation into explicit dense **L**
/// and **U** buffers without staging either factor through the host.
///
/// `packed` holds the in-place result of a packed LU factorisation of an
/// *n* × *n* row-major matrix: the strictly-lower triangle carries the
/// unit-lower **L** entries (its diagonal implicit) and the upper triangle
/// including the diagonal carries **U**. The returned buffers are dense
/// *n* × *n* row-major matrices with an explicit unit diagonal on **L**, so
/// **L** · **U** reproduces the packed factor product.
///
/// This is the device-side counterpart of
/// [`hephaestus_core::split_packed_lu`] and agrees with it bitwise: every
/// output entry is either a copied packed entry or a structural `0`/`1`.
///
/// # Errors
///
/// - `LengthMismatch` when `packed.len != n * n`.
/// - `DispatchFailed` when the element count exceeds the shader's `u32`
///   index width or the workgroup count exceeds `u32::MAX`.
pub fn split_packed_lu(
    device: &WgpuDevice,
    packed: &WgpuBuffer<f32>,
    n: usize,
) -> Result<(WgpuBuffer<f32>, WgpuBuffer<f32>)> {
    let total = n
        .checked_mul(n)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("packed LU dimension {n} overflows an element count"),
        })?;
    if packed.len != total {
        return Err(HephaestusError::LengthMismatch {
            host_len: total,
            device_len: packed.len,
        });
    }
    if total == 0 {
        return Ok((
            device.alloc_zeroed::<f32>(0)?,
            device.alloc_zeroed::<f32>(0)?,
        ));
    }

    let meta = SplitMeta {
        n: u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("packed LU dimension {n} exceeds u32"),
        })?,
        total: u32::try_from(total).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("packed LU element count {total} exceeds u32"),
        })?,
    };

    // Every element of both outputs is written by the dispatch below, so an
    // uninitialized allocation needs no zeroing pass.
    let lower = device.alloc_uninitialized::<f32>(total)?;
    let upper = device.alloc_uninitialized::<f32>(total)?;

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<SplitPackedLuKernel>(),
            TypeId::of::<f32>(),
            256,
        ),
        "hephaestus-split-packed-lu",
        split_packed_lu_shader_source,
    );

    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<SplitMeta>(1)?)?;
    let meta_buf = uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-split-packed-lu-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: packed.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lower.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: upper.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let workgroups =
        u32::try_from(total.div_ceil(256)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!(
                "packed LU split workgroup count {} exceeds u32::MAX",
                total.div_ceil(256)
            ),
        })?;

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-split-packed-lu"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-split-packed-lu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok((lower, upper))
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;

    fn device_or_skip() -> Option<WgpuDevice> {
        static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
        DEVICE
            .get_or_init(
                || match WgpuDevice::try_default("hephaestus-split-packed-lu-test") {
                    Ok(device) => Some(device),
                    Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
                        if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some() {
                            panic!("WGPU adapter required, but acquisition failed: {error}");
                        }
                        eprintln!("skipping wgpu packed-LU split test: adapter unavailable");
                        None
                    }
                    Err(error) => {
                        panic!("WGPU packed-LU split tests require a working provider: {error}")
                    }
                },
            )
            .clone()
    }

    /// Deterministic packed factor with a distinct value per position, so a
    /// transposed or off-by-one index cannot coincidentally match the oracle.
    fn packed_factor(n: usize) -> Vec<f32> {
        (0..n * n)
            .map(|idx| {
                let r = idx / n;
                let c = idx % n;
                (r * 16 + c) as f32 + 0.5
            })
            .collect()
    }

    fn assert_matches_host_oracle(n: usize) {
        let Some(device) = device_or_skip() else {
            return;
        };
        let host_packed = packed_factor(n);
        let (oracle_l, oracle_u) =
            hephaestus_core::split_packed_lu(&host_packed, n).expect("host oracle split");

        let packed = device.upload(&host_packed).expect("packed upload");
        let (lower, upper) = split_packed_lu(&device, &packed, n).expect("device split");

        let device_l = device.download_owned(&lower).expect("lower readback");
        let device_u = device.download_owned(&upper).expect("upper readback");

        // Copies and structural constants only: no arithmetic occurs on
        // either side, so bitwise equality is the correct assertion.
        assert_eq!(device_l, oracle_l, "L mismatch at n = {n}");
        assert_eq!(device_u, oracle_u, "U mismatch at n = {n}");
    }

    #[test]
    fn split_matches_host_oracle_at_workgroup_boundaries() {
        // 1 and 2 cover the degenerate shapes; 16 is a partial workgroup
        // (256 elements exactly); 17 crosses into a second workgroup with a
        // ragged tail.
        for n in [1, 2, 16, 17] {
            assert_matches_host_oracle(n);
        }
    }

    #[test]
    fn split_of_empty_matrix_yields_empty_factors() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let packed = device.alloc_zeroed::<f32>(0).expect("empty allocation");
        let (lower, upper) = split_packed_lu(&device, &packed, 0).expect("empty split");
        assert_eq!(lower.len, 0);
        assert_eq!(upper.len, 0);
    }

    #[test]
    fn split_rejects_a_buffer_whose_length_is_not_the_square_of_n() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let packed = device.alloc_zeroed::<f32>(5).expect("allocation");
        let error =
            split_packed_lu(&device, &packed, 3).expect_err("length mismatch must be rejected");
        assert!(matches!(
            error,
            HephaestusError::LengthMismatch {
                host_len: 9,
                device_len: 5,
            }
        ));
    }
}
