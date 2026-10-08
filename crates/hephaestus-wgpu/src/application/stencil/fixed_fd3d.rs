//! Provider-owned fixed-scheme three-dimensional first derivatives.
//!
//! The device counterpart of `leto_ops::FiniteDifference3D`'s per-axis
//! sweeps. One WGSL source serves all five schemes behind a uniform branch,
//! so the families cannot drift apart in the way five separately written
//! stencils could. Like the staggered pair beside it the kernel is f32-only:
//! WGSL does not guarantee f64 storage, and a generic scalar would be a
//! falsely generic boundary.
//!
//! # Bit-exactness against the CPU path
//!
//! Each scheme's lane reproduces the provider's arithmetic in the same
//! operation order — the sixth-order interior keeps the exact left-associated
//! `((−9·f[i+2] + 45·f[i+1]) + −45·f[i−1]) …` chain, and the fourth-order
//! fall-back keeps `((−8·m1 + 8·p1) + −p2) + m2` — with the reciprocal scales
//! arriving precomputed in the parameter block. WGSL arithmetic is IEEE
//! without implicit contraction, so the same operations round the same way.
//! The grid covers the output domain: every scheme but the forward sweep
//! writes the input grid, while forward drops one plane on its axis and each
//! lane reads `f[c]` and `f[c+1]` from the input grid.

use hephaestus_core::{DispatchGrid, HephaestusError, MultiStorageKernel, Result};
pub use hephaestus_core::{FixedFd3DParams, FixedFd3DScheme};

use crate::application::storage_kernel::{
    WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP: [usize; 3] = [4, 4, 4];

/// Compiled fixed-scheme sweep kernel.
///
/// Monomorphized at construction and reused across dispatches on the same
/// device; the scheme rides in each dispatch's parameter block.
#[derive(Debug)]
pub struct FixedFd3DKernel {
    sweep: WgslMultiStorageKernel,
}

impl FixedFd3DKernel {
    /// Compile the sweep for a device.
    ///
    /// # Errors
    ///
    /// Returns `HephaestusError::DispatchFailed` when the WGSL source or
    /// binding layout is rejected by the device.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        let bindings = [
            WgslStorageBindingLayout::read_only(0),
            WgslStorageBindingLayout::read_write(2),
        ];
        Ok(Self {
            sweep: WgslMultiStorageKernel::new(
                device,
                "hephaestus-fixed-fd-3d-sweep",
                FIXED_FD_3D_SHADER,
                "fixed_fd_sweep",
                &bindings,
                1,
            )?,
        })
    }

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the backend
    /// dispatch failure.
    pub fn sweep(
        &self,
        device: &WgpuDevice,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_storage(input.len, output.len)?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.output_dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        self.sweep.dispatch(
            device,
            [
                WgslStorageBinding::new(0, input),
                WgslStorageBinding::new(2, output),
            ],
            params,
            grid,
        )
    }
}

const FIXED_FD_3D_SHADER: &str = r"
struct Uniforms {
    dims_axis: vec4<u32>,
    scheme: vec4<u32>,
    scales: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

// Input-grid strides for a row-major [nx, ny, nz] field.
fn input_stride(axis: u32) -> u32 {
    if (axis == 0u) {
        return uniforms.dims_axis.y * uniforms.dims_axis.z;
    }
    if (axis == 1u) {
        return uniforms.dims_axis.z;
    }
    return 1u;
}

fn input_extent(axis: u32) -> u32 {
    if (axis == 0u) {
        return uniforms.dims_axis.x;
    }
    if (axis == 1u) {
        return uniforms.dims_axis.y;
    }
    return uniforms.dims_axis.z;
}

// Output extents: the forward sweep drops one plane on its axis, every other
// scheme keeps the input grid.
fn output_extent(axis: u32) -> u32 {
    let extent = input_extent(axis);
    if (uniforms.scheme.x == 3u && axis == uniforms.dims_axis.w) {
        return extent - 1u;
    }
    return extent;
}

fn coord_along(global_id: vec3<u32>, axis: u32) -> u32 {
    if (axis == 0u) {
        return global_id.x;
    }
    if (axis == 1u) {
        return global_id.y;
    }
    return global_id.z;
}

// First-order one-sided stencils.
fn first_forward(base: u32, stride: u32) -> f32 {
    return (field[base + stride] - field[base]) * uniforms.scales.x;
}

fn first_backward(base: u32, stride: u32, c: u32) -> f32 {
    let at = base + c * stride;
    return (field[at] - field[at - stride]) * uniforms.scales.x;
}

// Second-order central: (f[c+1] - f[c-1]) / 2h.
fn second(base: u32, stride: u32, c: u32) -> f32 {
    let at = base + c * stride;
    return (field[at + stride] - field[at - stride]) * uniforms.scales.y;
}

// Fourth-order central: ((-8*m1 + 8*p1) + -p2) + m2, over 12h — the exact
// association the provider evaluates.
fn fourth(base: u32, stride: u32, c: u32) -> f32 {
    let at = base + c * stride;
    let m1 = field[at - stride];
    let m2 = field[at - 2u * stride];
    let p1 = field[at + stride];
    let p2 = field[at + 2u * stride];
    return ((-8.0 * m1) + (8.0 * p1) + (-p2) + m2) * uniforms.scales.z;
}

// Sixth-order interior: ((-9*f[i+2] + 45*f[i+1]) + -45*f[i-1]) + 9*f[i-2]
// + -f[i-3] + f[i+3], over 60h — the exact chain the provider evaluates.
fn sixth_interior(base: u32, stride: u32, c: u32) -> f32 {
    let at = base + c * stride;
    return ((-9.0 * field[at + 2u * stride])
        + (45.0 * field[at + stride])
        + (-45.0 * field[at - stride])
        + (9.0 * field[at - 2u * stride])
        + (-field[at - 3u * stride])
        + field[at + 3u * stride]) * uniforms.scales.w;
}

fn central_second(base: u32, stride: u32, c: u32, n: u32) -> f32 {
    if (c == 0u) {
        return first_forward(base, stride);
    }
    if (c == n - 1u) {
        return first_backward(base, stride, c);
    }
    return second(base, stride, c);
}

// First matching row wins, exactly the provider's selection: flat on a
// singleton axis, one-sided at the ends, second order beside them, fourth
// order everywhere else.
fn central_fourth(base: u32, stride: u32, c: u32, n: u32) -> f32 {
    if (n == 1u) {
        return 0.0;
    }
    if (c == 0u) {
        return first_forward(base, stride);
    }
    if (c == n - 1u) {
        return first_backward(base, stride, c);
    }
    if (c < 2u || c >= n - 2u) {
        return second(base, stride, c);
    }
    return fourth(base, stride, c);
}

fn central_sixth(base: u32, stride: u32, c: u32, n: u32) -> f32 {
    if (c == 0u) {
        return first_forward(base, stride);
    }
    if (c == n - 1u) {
        return first_backward(base, stride, c);
    }
    if (c == 1u || c == n - 2u) {
        return second(base, stride, c);
    }
    if (c == 2u || c == n - 3u) {
        return fourth(base, stride, c);
    }
    return sixth_interior(base, stride, c);
}

fn staggered_backward(base: u32, stride: u32, c: u32) -> f32 {
    if (c == 0u) {
        return first_forward(base, stride);
    }
    return first_backward(base, stride, c);
}

@compute @workgroup_size(4, 4, 4)
fn fixed_fd_sweep(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let axis = uniforms.dims_axis.w;
    let nx = uniforms.dims_axis.x;
    let ny = uniforms.dims_axis.y;
    let nz = uniforms.dims_axis.z;
    let onx = output_extent(0u);
    let ony = output_extent(1u);
    let onz = output_extent(2u);
    if (global_id.x >= onx || global_id.y >= ony || global_id.z >= onz) {
        return;
    }

    let c = coord_along(global_id, axis);
    let n = input_extent(axis);
    let stride = input_stride(axis);
    // The output coordinate is a valid input coordinate on every axis: the
    // only shrunk axis is the swept one under the forward scheme, whose
    // output coordinate stays below n - 1.
    let input_flat = (global_id.x * ny + global_id.y) * nz + global_id.z;
    let base = input_flat - c * stride;
    let out_flat = (global_id.x * ony + global_id.y) * onz + global_id.z;

    let scheme = uniforms.scheme.x;
    var value = 0.0;
    if (scheme == 0u) {
        value = central_second(base, stride, c, n);
    } else if (scheme == 1u) {
        value = central_fourth(base, stride, c, n);
    } else if (scheme == 2u) {
        value = central_sixth(base, stride, c, n);
    } else if (scheme == 3u) {
        value = first_forward(base + c * stride, stride);
    } else {
        value = staggered_backward(base, stride, c);
    }
    result[out_flat] = value;
}
";

/// Provider-owned implementation of [`hephaestus_core::FixedFd3DOps`] for
/// WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFixedFd3DOps;

impl hephaestus_core::FixedFd3DOps<WgpuDevice> for WgpuFixedFd3DOps {
    type FixedFd3D = FixedFd3DKernel;

    fn prepare_fixed_fd_3d(&self, device: &WgpuDevice) -> Result<Self::FixedFd3D> {
        FixedFd3DKernel::new(device)
    }

    fn fixed_fd_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::FixedFd3D,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.sweep(device, input, output, params)
    }
}

/// Scheme discriminants for cross-checking the uniform branch.
#[cfg(test)]
pub(crate) fn scheme_discriminants() -> [(FixedFd3DScheme, u32); 5] {
    use FixedFd3DScheme::*;
    [
        (CentralSecondOrder, 0),
        (CentralFourthOrder, 1),
        (CentralSixthOrder, 2),
        (StaggeredForward, 3),
        (StaggeredBackward, 4),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_the_shader_branch() {
        for (scheme, discriminant) in scheme_discriminants() {
            assert_eq!(scheme as u32, discriminant, "{scheme:?}");
        }
    }
}
