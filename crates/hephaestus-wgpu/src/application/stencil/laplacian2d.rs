//! Provider-owned two-dimensional Laplacian stencil.
//!
//! This module moves the CFDrs 2D Laplacian WGSL kernel into the Hephaestus
//! provider so that `cfd-core`/`cfd-math` become thin typed consumers. The
//! kernel is intentionally f32-only: WGSL does not guarantee f64 storage
//! support, and exposing a generic scalar would be a falsely generic boundary.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{DispatchGrid, HephaestusError, MultiStorageKernel, Result};

use crate::application::storage_kernel::{WgslStorageBinding, WgslStorageBindingLayout, WgslMultiStorageKernel};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP: [usize; 3] = [8, 8, 1];

/// Boundary condition applied by the 2D Laplacian stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Homogeneous Dirichlet: `u = 0` on boundaries.
    Dirichlet,
    /// Homogeneous Neumann: `∂u/∂n = 0` on boundaries.
    Neumann,
    /// Periodic wrapping across domain boundaries.
    Periodic,
}

impl BoundaryCondition {
    const fn as_u32(self) -> u32 {
        match self {
            BoundaryCondition::Dirichlet => 0,
            BoundaryCondition::Neumann => 1,
            BoundaryCondition::Periodic => 2,
        }
    }
}

/// Uniform parameters for the 2D Laplacian dispatch.
///
/// The layout is kept 16-byte aligned to match WGSL uniform buffer layout
/// expectations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Laplacian2DParams {
    /// `(nx, ny, bc_type, pad)`
    pub dims_bc: [u32; 4],
    /// `(dx_inv2, dy_inv2, 0.0, 0.0)`
    pub inv2: [f32; 4],
}

impl Laplacian2DParams {
    /// Build parameters for a uniform Cartesian grid.
    ///
    /// # Errors
    /// Returns `HephaestusError::InvalidConfiguration` when `dx` or `dy` is not
    /// finite and positive, or when `nx` or `ny` is less than 2.
    pub fn new(nx: u32, ny: u32, dx: f32, dy: f32, bc: BoundaryCondition) -> Result<Self> {
        if nx < 2 || ny < 2 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("Laplacian grid axes must contain at least two points: nx={nx}, ny={ny}"),
            });
        }
        if !dx.is_finite() || dx <= 0.0 || !dy.is_finite() || dy <= 0.0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("Laplacian spacing must be finite and positive: dx={dx}, dy={dy}"),
            });
        }
        Ok(Self {
            dims_bc: [nx, ny, bc.as_u32(), 0],
            inv2: [dx.recip().powi(2), dy.recip().powi(2), 0.0, 0.0],
        })
    }
}

/// Compiled 2D Laplacian stencil kernel.
///
/// The kernel is monomorphized at construction time and can be reused for
/// multiple dispatches on the same device.
#[derive(Debug)]
pub struct Laplacian2DKernel {
    kernel: WgslMultiStorageKernel,
}

impl Laplacian2DKernel {
    /// Compile the 2D Laplacian kernel for a device.
    ///
    /// # Errors
    /// Returns `HephaestusError::DispatchFailed` when the WGSL source or
    /// binding layout is rejected by the device, or when the input/output
    /// buffers do not have the same length.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        let kernel = WgslMultiStorageKernel::new(
            device,
            "hephaestus-laplacian-2d",
            LAPLACIAN_2D_SHADER,
            "laplacian_2d",
            &[
                WgslStorageBindingLayout::read_only(0),
                WgslStorageBindingLayout::read_write(2),
            ],
            1,
        )?;
        Ok(Self { kernel })
    }

    /// Dispatch the Laplacian stencil over device buffers.
    ///
    /// # Errors
    /// Returns `HephaestusError::DispatchFailed` when the grid contract,
    /// buffer lengths, or dispatch is invalid.
    pub fn dispatch(
        &self,
        device: &WgpuDevice,
        input: &WgpuBuffer<f32>,
        output: &WgpuBuffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()> {
        if input.len != output.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: input.len,
                device_len: output.len,
            });
        }
        let nx = params.dims_bc[0] as usize;
        let ny = params.dims_bc[1] as usize;
        let grid = DispatchGrid::covering_domain([nx, ny, 1], WORKGROUP)?;
        self.kernel.dispatch(
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_valid_grid() {
        let p = Laplacian2DParams::new(4, 5, 0.1, 0.2, BoundaryCondition::Dirichlet).unwrap();
        assert_eq!(p.dims_bc, [4, 5, 0, 0]);
        assert!((p.inv2[0] - 100.0).abs() < f32::EPSILON);
        assert!((p.inv2[1] - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn params_rejects_too_small_axes() {
        assert!(matches!(
            Laplacian2DParams::new(1, 4, 1.0, 1.0, BoundaryCondition::Neumann),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            Laplacian2DParams::new(4, 1, 1.0, 1.0, BoundaryCondition::Neumann),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn params_rejects_bad_spacing() {
        for bad in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0, -1.0] {
            assert!(matches!(
                Laplacian2DParams::new(4, 4, bad, 1.0, BoundaryCondition::Periodic),
                Err(HephaestusError::InvalidConfiguration { .. })
            ));
            assert!(matches!(
                Laplacian2DParams::new(4, 4, 1.0, bad, BoundaryCondition::Periodic),
                Err(HephaestusError::InvalidConfiguration { .. })
            ));
        }
    }
}

const LAPLACIAN_2D_SHADER: &str = r"
struct Uniforms {
    dims_bc: vec4<u32>,
    inv2: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(8, 8)
fn laplacian_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uniforms.dims_bc.x || j >= uniforms.dims_bc.y) {
        return;
    }

    let idx = j * uniforms.dims_bc.x + i;
    var laplacian = 0.0;

    // X direction
    if (i > 0u && i < uniforms.dims_bc.x - 1u) {
        let left = field[j * uniforms.dims_bc.x + (i - 1u)];
        let center = field[idx];
        let right = field[j * uniforms.dims_bc.x + (i + 1u)];
        laplacian += (left - 2.0 * center + right) * uniforms.inv2.x;
    } else if (i == 0u) {
        if (uniforms.dims_bc.z == 0u) {
            let center = field[idx];
            laplacian += (-2.0 * center) * uniforms.inv2.x;
        } else if (uniforms.dims_bc.z == 1u) {
            let nx = uniforms.dims_bc.x;
            let center = field[idx];
            if (nx >= 4u) {
                let u1 = field[j * nx + 1u];
                let u2 = field[j * nx + 2u];
                let u3 = field[j * nx + 3u];
                laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) * uniforms.inv2.x;
            } else {
                let right = field[j * nx + (i + 1u)];
                laplacian += (right - 2.0 * center + right) * uniforms.inv2.x;
            }
        } else if (uniforms.dims_bc.z == 2u) {
            let left = field[j * uniforms.dims_bc.x + (uniforms.dims_bc.x - 2u)];
            let center = field[idx];
            let right = field[j * uniforms.dims_bc.x + (i + 1u)];
            laplacian += (left - 2.0 * center + right) * uniforms.inv2.x;
        }
    } else if (i == uniforms.dims_bc.x - 1u) {
        if (uniforms.dims_bc.z == 0u) {
            let center = field[idx];
            laplacian += (-2.0 * center) * uniforms.inv2.x;
        } else if (uniforms.dims_bc.z == 1u) {
            let nx = uniforms.dims_bc.x;
            let center = field[idx];
            if (nx >= 4u) {
                let u1 = field[j * nx + (nx - 2u)];
                let u2 = field[j * nx + (nx - 3u)];
                let u3 = field[j * nx + (nx - 4u)];
                laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) * uniforms.inv2.x;
            } else {
                let left = field[j * nx + (i - 1u)];
                laplacian += (left - 2.0 * center + left) * uniforms.inv2.x;
            }
        } else if (uniforms.dims_bc.z == 2u) {
            let left = field[j * uniforms.dims_bc.x + (i - 1u)];
            let center = field[idx];
            let right = field[j * uniforms.dims_bc.x + 1u];
            laplacian += (left - 2.0 * center + right) * uniforms.inv2.x;
        }
    }

    // Y direction
    if (j > 0u && j < uniforms.dims_bc.y - 1u) {
        let bottom = field[(j - 1u) * uniforms.dims_bc.x + i];
        let center = field[idx];
        let top = field[(j + 1u) * uniforms.dims_bc.x + i];
        laplacian += (bottom - 2.0 * center + top) * uniforms.inv2.y;
    } else if (j == 0u) {
        if (uniforms.dims_bc.z == 0u) {
            let center = field[idx];
            laplacian += (-2.0 * center) * uniforms.inv2.y;
        } else if (uniforms.dims_bc.z == 1u) {
            let ny = uniforms.dims_bc.y;
            let center = field[idx];
            if (ny >= 4u) {
                let u1 = field[(1u) * uniforms.dims_bc.x + i];
                let u2 = field[(2u) * uniforms.dims_bc.x + i];
                let u3 = field[(3u) * uniforms.dims_bc.x + i];
                laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) * uniforms.inv2.y;
            } else {
                let top = field[(j + 1u) * uniforms.dims_bc.x + i];
                laplacian += (top - 2.0 * center + top) * uniforms.inv2.y;
            }
        } else if (uniforms.dims_bc.z == 2u) {
            let bottom = field[(uniforms.dims_bc.y - 2u) * uniforms.dims_bc.x + i];
            let center = field[idx];
            let top = field[(j + 1u) * uniforms.dims_bc.x + i];
            laplacian += (bottom - 2.0 * center + top) * uniforms.inv2.y;
        }
    } else if (j == uniforms.dims_bc.y - 1u) {
        if (uniforms.dims_bc.z == 0u) {
            let center = field[idx];
            laplacian += (-2.0 * center) * uniforms.inv2.y;
        } else if (uniforms.dims_bc.z == 1u) {
            let ny = uniforms.dims_bc.y;
            let center = field[idx];
            if (ny >= 4u) {
                let u1 = field[(ny - 2u) * uniforms.dims_bc.x + i];
                let u2 = field[(ny - 3u) * uniforms.dims_bc.x + i];
                let u3 = field[(ny - 4u) * uniforms.dims_bc.x + i];
                laplacian += (2.0 * center - 5.0 * u1 + 4.0 * u2 - u3) * uniforms.inv2.y;
            } else {
                let bottom = field[(j - 1u) * uniforms.dims_bc.x + i];
                laplacian += (bottom - 2.0 * center + bottom) * uniforms.inv2.y;
            }
        } else if (uniforms.dims_bc.z == 2u) {
            let bottom = field[(j - 1u) * uniforms.dims_bc.x + i];
            let center = field[idx];
            let top = field[uniforms.dims_bc.x + i];
            laplacian += (bottom - 2.0 * center + top) * uniforms.inv2.y;
        }
    }

    result[idx] = laplacian;
}
";
