//! Provider-owned three-dimensional staggered gradient/divergence pair.
//!
//! The device counterpart of `leto_ops::StaggeredLeapfrog3D`. Both kernels are
//! compiled from one WGSL source and share its indexing and reflection helpers,
//! so the pair cannot drift apart in the way two separately written stencils
//! could. Like the 2-D Laplacian beside it the kernel is f32-only: WGSL does
//! not guarantee f64 storage, and a generic scalar would be a falsely generic
//! boundary.
//!
//! # The divergence gathers rather than scatters
//!
//! Leto computes the divergence by scattering `-Gᵀ` directly, which makes the
//! adjoint identity true by construction. A GPU cannot scatter without atomics,
//! so this kernel gathers instead, which means writing the transpose out as its
//! own stencil — including its wall closure, which is the part the CPU comment
//! warns is easy to get wrong.
//!
//! It is derived rather than guessed. For a gradient
//! `G[i] = Σ_n c_n (f[R(i+n)] − f[R(i−n+1)])`, the transpose's column `j`
//! collects every `i` whose tap lands on `j`, so
//!
//! ```text
//!   D[j] = Σ_n c_n [ A_n(j) − B_n(j) ]
//!   A_n(j) = u[j+n−1]  when j+n−1 < extent      (the unreflected preimage)
//!          + u[n−2−j]  when j ≤ n−2             (reflected across the low wall)
//!   B_n(j) = u[j−n]              when j ≥ n
//!          + u[2·extent−1−j−n]   when j+n ≥ extent  (reflected across the high wall)
//! ```
//!
//! and the conformance suite checks it the only way worth checking it: against
//! the CPU pair, on every axis, plus the adjoint identity itself.
//!
//! # Why the grid must be at least as deep as the stencil
//!
//! Leto's reflection loops so that a stencil deeper than its axis still lands
//! in range. Under `extent >= 2N` a single reflection step is exact, which is
//! what lets these kernels stay branch-shallow and loop-free in their
//! reflection. [`Staggered3DParams::new`](hephaestus_core::Staggered3DParams::new) enforces that precondition, so the
//! difference is a rejected configuration rather than a silent divergence.

use hephaestus_core::{DispatchGrid, HephaestusError, MultiStorageKernel, Result};
pub use hephaestus_core::{Staggered3DParams, StaggeredAxis};

use crate::application::storage_kernel::{
    WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP: [usize; 3] = [4, 4, 4];

/// Compiled staggered gradient and divergence kernels.
///
/// Both are monomorphized at construction and reused across dispatches on the
/// same device.
#[derive(Debug)]
pub struct Staggered3DKernel {
    gradient: WgslMultiStorageKernel,
    divergence: WgslMultiStorageKernel,
}

impl Staggered3DKernel {
    /// Compile the staggered pair for a device.
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
            gradient: WgslMultiStorageKernel::new(
                device,
                "hephaestus-staggered-3d-gradient",
                STAGGERED_3D_SHADER,
                "staggered_gradient",
                &bindings,
                1,
            )?,
            divergence: WgslMultiStorageKernel::new(
                device,
                "hephaestus-staggered-3d-divergence",
                STAGGERED_3D_SHADER,
                "staggered_divergence",
                &bindings,
                1,
            )?,
        })
    }

    /// Dispatch the gradient over device buffers.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the grid, or the backend
    /// dispatch failure.
    pub fn gradient(
        &self,
        device: &WgpuDevice,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        self.dispatch(&self.gradient, device, input, output, params)
    }

    /// Dispatch the divergence over device buffers.
    ///
    /// # Errors
    ///
    /// See [`Self::gradient`].
    pub fn divergence(
        &self,
        device: &WgpuDevice,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        self.dispatch(&self.divergence, device, input, output, params)
    }

    fn dispatch(
        &self,
        kernel: &WgslMultiStorageKernel,
        device: &WgpuDevice,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        params.validate_storage(input.len, output.len)?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(&params.dims_axis[..3]) {
            *slot = usize::try_from(*extent).map_err(|error| {
                HephaestusError::InvalidConfiguration {
                    message: format!("staggered grid extent does not fit usize: {error}"),
                }
            })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        kernel.dispatch(
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

const STAGGERED_3D_SHADER: &str = r"
struct Uniforms {
    dims_axis: vec4<u32>,
    order: vec4<u32>,
    inv_spacing: vec4<f32>,
    taps_low: vec4<f32>,
    taps_high: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

fn tap(n: u32) -> f32 {
    // n is 0-based. Two vec4 lanes carry c_1..c_8.
    if (n < 4u) {
        return uniforms.taps_low[n];
    }
    return uniforms.taps_high[n - 4u];
}

// Stride between neighbours along the differentiated axis, for a row-major
// [nx, ny, nz] field.
fn axis_stride() -> u32 {
    let axis = uniforms.dims_axis.w;
    if (axis == 0u) {
        return uniforms.dims_axis.y * uniforms.dims_axis.z;
    }
    if (axis == 1u) {
        return uniforms.dims_axis.z;
    }
    return 1u;
}

fn axis_extent() -> u32 {
    let axis = uniforms.dims_axis.w;
    if (axis == 0u) {
        return uniforms.dims_axis.x;
    }
    if (axis == 1u) {
        return uniforms.dims_axis.y;
    }
    return uniforms.dims_axis.z;
}

fn axis_inv_spacing() -> f32 {
    let axis = uniforms.dims_axis.w;
    if (axis == 0u) {
        return uniforms.inv_spacing.x;
    }
    if (axis == 1u) {
        return uniforms.inv_spacing.y;
    }
    return uniforms.inv_spacing.z;
}

// One reflection step about the walls between cells: -1-m at the low end and
// 2*extent-1-m at the high end. Exact for |offset| <= extent, which
// Staggered3DParams guarantees.
fn reflect(m: i32, extent: i32) -> u32 {
    if (m < 0) {
        return u32(-1 - m);
    }
    if (m >= extent) {
        return u32(2 * extent - 1 - m);
    }
    return u32(m);
}

struct Site {
    inside: bool,
    base: u32,
    stride: u32,
    coord: i32,
    extent: i32,
    taps: u32,
    scale: f32,
}

fn site(global_id: vec3<u32>) -> Site {
    var s: Site;
    s.inside = !(global_id.x >= uniforms.dims_axis.x
        || global_id.y >= uniforms.dims_axis.y
        || global_id.z >= uniforms.dims_axis.z);
    s.stride = axis_stride();
    s.extent = i32(axis_extent());
    s.taps = uniforms.order.x;
    s.scale = axis_inv_spacing();

    let flat = (global_id.x * uniforms.dims_axis.y + global_id.y) * uniforms.dims_axis.z
        + global_id.z;
    let axis = uniforms.dims_axis.w;
    var coord = global_id.z;
    if (axis == 0u) {
        coord = global_id.x;
    } else if (axis == 1u) {
        coord = global_id.y;
    }
    s.coord = i32(coord);
    // Strip the axis coordinate so a neighbour is base + m * stride.
    s.base = flat - coord * s.stride;
    return s;
}

@compute @workgroup_size(4, 4, 4)
fn staggered_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = site(global_id);
    if (!s.inside) {
        return;
    }
    var sum = 0.0;
    for (var n: u32 = 1u; n <= s.taps; n = n + 1u) {
        let c = tap(n - 1u);
        let hi = reflect(s.coord + i32(n), s.extent);
        let lo = reflect(s.coord - i32(n) + 1, s.extent);
        sum = sum + c * (field[s.base + hi * s.stride] - field[s.base + lo * s.stride]);
    }
    result[s.base + u32(s.coord) * s.stride] = sum * s.scale;
}

@compute @workgroup_size(4, 4, 4)
fn staggered_divergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = site(global_id);
    if (!s.inside) {
        return;
    }
    let j = s.coord;
    var sum = 0.0;
    for (var n: u32 = 1u; n <= s.taps; n = n + 1u) {
        let c = tap(n - 1u);
        let step = i32(n);
        var gathered = 0.0;

        // The unreflected preimage under i -> R(i - n + 1).
        let direct_low = j + step - 1;
        if (direct_low < s.extent) {
            gathered = gathered + field[s.base + u32(direct_low) * s.stride];
        }
        // Its low-wall reflection.
        let mirrored_low = step - 2 - j;
        if (mirrored_low >= 0) {
            gathered = gathered + field[s.base + u32(mirrored_low) * s.stride];
        }
        // The unreflected preimage under i -> R(i + n).
        let direct_high = j - step;
        if (direct_high >= 0) {
            gathered = gathered - field[s.base + u32(direct_high) * s.stride];
        }
        // Its high-wall reflection.
        if (j + step >= s.extent) {
            let mirrored_high = 2 * s.extent - 1 - j - step;
            gathered = gathered - field[s.base + u32(mirrored_high) * s.stride];
        }

        sum = sum + c * gathered;
    }
    result[s.base + u32(j) * s.stride] = sum * s.scale;
}
";

/// Provider-owned implementation of [`hephaestus_core::Staggered3DOps`] for
/// WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuStaggered3DOps;

impl hephaestus_core::Staggered3DOps<WgpuDevice> for WgpuStaggered3DOps {
    type Staggered3D = Staggered3DKernel;

    fn prepare_staggered_3d(&self, device: &WgpuDevice) -> Result<Self::Staggered3D> {
        Staggered3DKernel::new(device)
    }

    fn staggered_gradient_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::Staggered3D,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        kernel.gradient(device, input, output, params)
    }

    fn staggered_divergence_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::Staggered3D,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        kernel.divergence(device, input, output, params)
    }
}
