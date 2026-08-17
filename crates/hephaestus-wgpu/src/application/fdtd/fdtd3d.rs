//! WGPU implementation of the provider-neutral 3D acoustic FDTD contract.

use hephaestus_core::{
    DeviceBuffer, DispatchGrid, Fdtd3dOps, Fdtd3dParams, FdtdMedium, FdtdVelocity, HephaestusError,
    MultiStorageKernel, Result,
};

use crate::application::storage_kernel::{
    WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP: [usize; 3] = [8, 8, 4];

/// Provider-owned prepared FDTD pipeline bundle.
#[derive(Debug)]
pub struct Fdtd3dKernel {
    velocity: WgslMultiStorageKernel,
    pressure: WgslMultiStorageKernel,
}

impl Fdtd3dKernel {
    /// Compile the velocity and pressure update pipelines.
    ///
    /// # Errors
    ///
    /// Returns the provider's shader compilation or binding-layout error.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        let storage_layouts = [
            WgslStorageBindingLayout::read_write(0),
            WgslStorageBindingLayout::read_write(1),
            WgslStorageBindingLayout::read_only(2),
        ];
        Ok(Self {
            velocity: WgslMultiStorageKernel::new(
                device,
                "hephaestus-fdtd-3d-velocity",
                FDTD_3D_SHADER,
                "velocity_update",
                &storage_layouts,
                3,
            )?,
            pressure: WgslMultiStorageKernel::new(
                device,
                "hephaestus-fdtd-3d-pressure",
                FDTD_3D_SHADER,
                "pressure_update",
                &storage_layouts,
                3,
            )?,
        })
    }

    /// Dispatch one in-place velocity-then-pressure update.
    ///
    /// # Errors
    ///
    /// Returns a buffer-size or dispatch error.
    pub fn dispatch(
        &self,
        device: &WgpuDevice,
        pressure: &WgpuBuffer<f32>,
        velocity: &WgpuBuffer<FdtdVelocity>,
        medium: &WgpuBuffer<FdtdMedium>,
        params: &Fdtd3dParams,
    ) -> Result<()> {
        params.validate_storage(pressure.len(), velocity.len(), medium.len())?;
        let grid = DispatchGrid::covering_domain(
            [
                usize::try_from(params.nx()).map_err(|error| dimension_error("nx", error))?,
                usize::try_from(params.ny()).map_err(|error| dimension_error("ny", error))?,
                usize::try_from(params.nz()).map_err(|error| dimension_error("nz", error))?,
            ],
            WORKGROUP,
        )?;
        self.velocity.dispatch(
            device,
            [
                WgslStorageBinding::new(0, pressure),
                WgslStorageBinding::new(1, velocity),
                WgslStorageBinding::new(2, medium),
            ],
            params,
            grid,
        )?;
        self.pressure.dispatch(
            device,
            [
                WgslStorageBinding::new(0, pressure),
                WgslStorageBinding::new(1, velocity),
                WgslStorageBinding::new(2, medium),
            ],
            params,
            grid,
        )
    }
}

/// Zero-sized WGPU implementation of [`hephaestus_core::Fdtd3dOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFdtd3dOps;

impl Fdtd3dOps<WgpuDevice> for WgpuFdtd3dOps {
    type Kernel = Fdtd3dKernel;

    fn prepare_fdtd_3d(&self, device: &WgpuDevice) -> Result<Self::Kernel> {
        Fdtd3dKernel::new(device)
    }

    fn step_fdtd_3d(
        &self,
        device: &WgpuDevice,
        kernel: &Self::Kernel,
        pressure: &WgpuBuffer<f32>,
        velocity: &WgpuBuffer<FdtdVelocity>,
        medium: &WgpuBuffer<FdtdMedium>,
        params: &Fdtd3dParams,
    ) -> Result<()> {
        kernel.dispatch(device, pressure, velocity, medium, params)
    }
}

fn dimension_error(axis: &str, error: impl core::fmt::Display) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: format!("FDTD {axis} dimension does not fit usize: {error}"),
    }
}

const FDTD_3D_SHADER: &str = r#"
struct Params {
    dimensions: vec4<u32>,
    spacing_dt: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> pressure: array<f32>;

@group(0) @binding(1)
var<storage, read_write> velocity: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> medium: array<vec2<f32>>;

@group(0) @binding(3)
var<uniform> params: Params;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.dimensions.x + z * params.dimensions.x * params.dimensions.y;
}

@compute @workgroup_size(8, 8, 4)
fn velocity_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= params.dimensions.x || y >= params.dimensions.y || z >= params.dimensions.z) {
        return;
    }

    let idx = index_3d(x, y, z);
    if (x == 0u || x == params.dimensions.x - 1u ||
        y == 0u || y == params.dimensions.y - 1u ||
        z == 0u || z == params.dimensions.z - 1u) {
        velocity[idx] = vec4<f32>(0.0);
        return;
    }

    let density = medium[idx].x;
    let dx = params.spacing_dt.x;
    let dy = params.spacing_dt.y;
    let dz = params.spacing_dt.z;
    let dt = params.spacing_dt.w;
    let grad = vec3<f32>(
        (pressure[index_3d(x + 1u, y, z)] - pressure[index_3d(x - 1u, y, z)]) / (2.0 * dx),
        (pressure[index_3d(x, y + 1u, z)] - pressure[index_3d(x, y - 1u, z)]) / (2.0 * dy),
        (pressure[index_3d(x, y, z + 1u)] - pressure[index_3d(x, y, z - 1u)]) / (2.0 * dz)
    );
    velocity[idx] = velocity[idx] - vec4<f32>(dt / density * grad, 0.0);
}

@compute @workgroup_size(8, 8, 4)
fn pressure_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= params.dimensions.x || y >= params.dimensions.y || z >= params.dimensions.z) {
        return;
    }

    let idx = index_3d(x, y, z);
    if (x == 0u || x == params.dimensions.x - 1u ||
        y == 0u || y == params.dimensions.y - 1u ||
        z == 0u || z == params.dimensions.z - 1u) {
        pressure[idx] = 0.0;
        return;
    }

    let density = medium[idx].x;
    let sound_speed = medium[idx].y;
    let dx = params.spacing_dt.x;
    let dy = params.spacing_dt.y;
    let dz = params.spacing_dt.z;
    let dt = params.spacing_dt.w;
    let div_velocity =
        (velocity[index_3d(x + 1u, y, z)].x - velocity[index_3d(x - 1u, y, z)].x) / (2.0 * dx) +
        (velocity[index_3d(x, y + 1u, z)].y - velocity[index_3d(x, y - 1u, z)].y) / (2.0 * dy) +
        (velocity[index_3d(x, y, z + 1u)].z - velocity[index_3d(x, y, z - 1u)].z) / (2.0 * dz);
    pressure[idx] = pressure[idx] - dt * density * sound_speed * sound_speed * div_velocity;
}
"#;
