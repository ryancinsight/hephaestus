//! Allocation-free ordered WGPU dispatch for prepared dense complex FFT plans.

use hephaestus_core::{Binding, DispatchGrid, FftDirection, Result};

use crate::{
    application::stream::{WgpuBoundDispatch, WgpuCommandStream},
    infrastructure::{buffer::WgpuBuffer, device::WgpuDevice},
};

use super::{
    kernel::FusedParams,
    pipelines::FftPipelines,
    plan::{FftWorkspace, WgpuFftPlan},
    scalar::WgpuFftScalar,
    stages::RadixStages,
    strategy::{Axis, AxisStrategy, ChirpData},
};

#[derive(Clone, Copy)]
pub(super) struct FftComponents<'a, T> {
    pub(super) real: &'a WgpuBuffer<T>,
    pub(super) imaginary: &'a WgpuBuffer<T>,
}

fn grid(elements: usize) -> Result<DispatchGrid> {
    DispatchGrid::covering_domain([elements, 1, 1], [256, 1, 1])
}

fn product_grid(left: u32, right: u32) -> Result<DispatchGrid> {
    let elements = usize::try_from(u64::from(left) * u64::from(right)).map_err(|_| {
        hephaestus_core::HephaestusError::InvalidConfiguration {
            message: format!("FFT dispatch count {left} * {right} exceeds usize"),
        }
    })?;
    grid(elements)
}

impl<T: WgpuFftScalar> WgpuFftPlan<T> {
    fn workspace(&self) -> Result<&FftWorkspace<T>> {
        self.workspace
            .as_ref()
            .ok_or_else(|| hephaestus_core::HephaestusError::DispatchFailed {
                message: "staged FFT command requires prepared workspace".to_owned(),
            })
    }

    fn bind_pack(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        axis: Axis,
        components: FftComponents<'_, T>,
    ) -> Result<WgpuBoundDispatch> {
        let params = self.pack[axis.shader_index() as usize];
        let workspace = self.workspace()?;
        let bindings = [
            Binding::read_write(&workspace.real),
            Binding::read_write(&workspace.imaginary),
            Binding::read_write(components.real),
            Binding::read_write(components.imaginary),
        ];
        device.bind(
            &pipelines.pack,
            &bindings,
            &params,
            product_grid(params.n, params.batch_count)?,
        )
    }

    fn bind_unpack(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        axis: Axis,
        components: FftComponents<'_, T>,
    ) -> Result<WgpuBoundDispatch> {
        let params = self.pack[axis.shader_index() as usize];
        let workspace = self.workspace()?;
        let bindings = [
            Binding::read_write(&workspace.real),
            Binding::read_write(&workspace.imaginary),
            Binding::read_write(components.real),
            Binding::read_write(components.imaginary),
        ];
        device.bind(
            &pipelines.unpack,
            &bindings,
            &params,
            product_grid(params.n, params.batch_count)?,
        )
    }

    fn bind_radix(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        stages: &RadixStages,
        commands: &mut Vec<WgpuBoundDispatch>,
    ) -> Result<()> {
        if stages.fft_len == 0 {
            return Ok(());
        }
        let workspace = self.workspace()?;
        let roots = self.radix_twiddle.as_ref().ok_or_else(|| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: "prepared radix FFT is missing its twiddle table".to_owned(),
            }
        })?;
        let bindings = [
            Binding::read_write(&workspace.real),
            Binding::read_write(&workspace.imaginary),
            Binding::read(roots),
        ];
        let element_grid = product_grid(stages.batch_count, stages.fft_len)?;
        if stages.radix_four {
            commands.push(device.bind(
                &pipelines.radix_four_bit_reverse,
                &bindings,
                &stages.bit_reverse,
                element_grid,
            )?);
            let butterfly_grid = product_grid(stages.batch_count, stages.fft_len / 4)?;
            for params in &stages.butterflies {
                commands.push(device.bind(
                    &pipelines.radix_four_butterfly,
                    &bindings,
                    params,
                    butterfly_grid,
                )?);
            }
        } else {
            commands.push(device.bind(
                &pipelines.bit_reverse,
                &bindings,
                &stages.bit_reverse,
                element_grid,
            )?);
            let butterfly_grid = product_grid(stages.batch_count, stages.fft_len / 2)?;
            for params in &stages.butterflies {
                commands.push(device.bind(
                    &pipelines.butterfly,
                    &bindings,
                    params,
                    butterfly_grid,
                )?);
            }
        }
        if let Some(params) = stages.inverse_scale {
            commands.push(device.bind(&pipelines.scale, &bindings, &params, element_grid)?);
        }
        Ok(())
    }

    fn bind_chirp(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        chirp: &ChirpData<T>,
        inverse: bool,
        commands: &mut Vec<WgpuBoundDispatch>,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let kernel_bindings = [
            Binding::read_write(&workspace.real),
            Binding::read_write(&workspace.imaginary),
            Binding::read(&chirp.real_kernel),
            Binding::read(&chirp.imaginary_kernel),
        ];
        let direct_bindings = [
            Binding::read_write(&workspace.real),
            Binding::read_write(&workspace.imaginary),
            Binding::read(&chirp.direct_real),
            Binding::read(&chirp.direct_imaginary),
        ];
        let padded_grid = product_grid(chirp.params.m, chirp.params.batch_count)?;
        let output_grid = product_grid(chirp.params.n, chirp.params.batch_count)?;
        if inverse {
            commands.push(device.bind(
                &pipelines.chirp_negate_imaginary,
                &direct_bindings,
                &chirp.params,
                output_grid,
            )?);
        }
        commands.push(device.bind(
            &pipelines.chirp_premultiply,
            &direct_bindings,
            &chirp.params,
            padded_grid,
        )?);
        self.bind_radix(device, pipelines, &chirp.forward_radix, commands)?;
        commands.push(device.bind(
            &pipelines.chirp_point_multiply,
            &kernel_bindings,
            &chirp.params,
            padded_grid,
        )?);
        self.bind_radix(device, pipelines, &chirp.inverse_radix, commands)?;
        commands.push(device.bind(
            &pipelines.chirp_postmultiply,
            &direct_bindings,
            &chirp.params,
            output_grid,
        )?);
        if inverse {
            commands.push(device.bind(
                &pipelines.chirp_negate_imaginary,
                &direct_bindings,
                &chirp.params,
                output_grid,
            )?);
            commands.push(device.bind(
                &pipelines.chirp_scale,
                &direct_bindings,
                &chirp.params,
                output_grid,
            )?);
        }
        Ok(())
    }

    fn bind_fused(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        axis: Axis,
        inverse: bool,
        components: FftComponents<'_, T>,
    ) -> Result<WgpuBoundDispatch> {
        let pipeline = pipelines.fused.as_ref().ok_or_else(|| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: "fused FFT strategy requires the prepared fused pipeline".to_owned(),
            }
        })?;
        let twiddle = self.fused_twiddle.as_ref().ok_or_else(|| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: "fused FFT strategy requires prepared twiddle roots".to_owned(),
            }
        })?;
        let pack = self.pack[axis.shader_index() as usize];
        let max_grid_x = device.inner().limits().max_compute_workgroups_per_dimension;
        let grid_x = pack.batch_count.min(max_grid_x);
        let grid_y = pack.batch_count.div_ceil(grid_x);
        let params = FusedParams {
            n: pack.n,
            log2n: pack.n.trailing_zeros(),
            inverse: u32::from(inverse),
            batch_count: pack.batch_count,
            nx: pack.nx,
            ny: pack.ny,
            nz: pack.nz,
            axis: pack.axis,
            batch_grid_x: grid_x,
            padding: [0; 3],
        };
        let bindings = [
            Binding::read_write(components.real),
            Binding::read_write(components.imaginary),
            Binding::read(twiddle),
        ];
        device.bind(
            pipeline,
            &bindings,
            &params,
            DispatchGrid::new(grid_x, grid_y, 1),
        )
    }

    fn bind_axis(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        axis: Axis,
        inverse: bool,
        components: FftComponents<'_, T>,
        commands: &mut Vec<WgpuBoundDispatch>,
    ) -> Result<()> {
        let index = axis.shader_index() as usize;
        match self.strategy[index] {
            AxisStrategy::Identity => return Ok(()),
            AxisStrategy::FusedRadix2 => {
                commands.push(self.bind_fused(device, pipelines, axis, inverse, components)?);
                return Ok(());
            }
            AxisStrategy::StagedRadix2 => {
                commands.push(self.bind_pack(device, pipelines, axis, components)?);
                self.bind_radix(device, pipelines, &self.stages[index], commands)?;
            }
            AxisStrategy::ChirpZ { .. } => {
                commands.push(self.bind_pack(device, pipelines, axis, components)?);
                let chirp = self.chirp[index].as_ref().ok_or_else(|| {
                    hephaestus_core::HephaestusError::DispatchFailed {
                        message: format!(
                            "FFT {axis:?} axis selected Bluestein without prepared chirp data"
                        ),
                    }
                })?;
                self.bind_chirp(device, pipelines, chirp, inverse, commands)?;
            }
        }
        commands.push(self.bind_unpack(device, pipelines, axis, components)?);
        Ok(())
    }

    pub(crate) fn prepare_commands(
        &self,
        device: &WgpuDevice,
        pipelines: &FftPipelines<T>,
        direction: FftDirection,
        components: FftComponents<'_, T>,
    ) -> Result<Box<[WgpuBoundDispatch]>> {
        let mut commands = Vec::new();
        let axes = match direction {
            FftDirection::Forward => [Axis::Z, Axis::Y, Axis::X],
            FftDirection::Inverse => [Axis::X, Axis::Y, Axis::Z],
        };
        for axis in axes {
            if self.axis_is_active(axis) {
                self.bind_axis(
                    device,
                    pipelines,
                    axis,
                    matches!(direction, FftDirection::Inverse),
                    components,
                    &mut commands,
                )?;
            }
        }
        Ok(commands.into_boxed_slice())
    }

    pub(crate) fn encode(&self, stream: &mut WgpuCommandStream<'_>) -> Result<()> {
        stream.encode_bound_sequence("hephaestus-fft", &self.commands)
    }

    pub(crate) fn encode_in_pass(&self, pass: &mut wgpu::ComputePass<'_>) {
        for command in &self.commands {
            command.encode_in_pass(pass);
        }
    }
}
