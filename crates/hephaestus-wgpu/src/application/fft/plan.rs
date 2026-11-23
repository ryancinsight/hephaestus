//! WGPU planning and provider-owned storage for dense complex FFTs.

use eunomia::Zeroable;
use hephaestus_core::{ComputeDevice, FftDirection, FftPlan, Result};

use crate::infrastructure::{buffer::WgpuBuffer, device::WgpuDevice};

use super::{
    chirp::{chirp_angle, forward_radix_two},
    kernel::{ChirpParams, PackParams},
    pipelines::FftPipelines,
    scalar::WgpuFftScalar,
    sizing::{Sizing, dimension},
    stages::RadixStages,
    strategy::{Axis, AxisStrategy, ChirpData},
    twiddle::{build_fused_twiddle, build_radix_twiddle, radix_parameters, try_host_vector},
};

pub(crate) struct WgpuFftPlan<T> {
    pub(crate) real: WgpuBuffer<T>,
    pub(crate) imaginary: WgpuBuffer<T>,
    pub(crate) workspace: Option<FftWorkspace<T>>,
    pub(crate) fused_twiddle: Option<WgpuBuffer<T>>,
    pub(crate) radix_twiddle: Option<WgpuBuffer<T>>,
    pub(crate) strategy: [AxisStrategy; 3],
    pub(crate) pack: [PackParams; 3],
    pub(crate) chirp: [Option<ChirpData<T>>; 3],
    pub(crate) stages: [RadixStages; 3],
    pub(crate) commands: Box<[crate::application::stream::WgpuBoundDispatch]>,
}

pub(crate) struct FftWorkspace<T> {
    pub(crate) real: WgpuBuffer<T>,
    pub(crate) imaginary: WgpuBuffer<T>,
}

fn radix_stages(
    axis_len: usize,
    strategy: AxisStrategy,
    batch_count: u32,
    inverse: bool,
    root_length: usize,
) -> Result<RadixStages> {
    if !matches!(strategy, AxisStrategy::StagedRadix2) {
        return Ok(RadixStages::empty());
    }
    let fft_len = dimension(axis_len, "radix axis length")?;
    let (root_half, scale_index) = radix_parameters(root_length, fft_len)?;
    if fft_len.trailing_zeros() % 2 == 0 {
        Ok(RadixStages::radix_four(
            fft_len,
            batch_count,
            inverse,
            root_half,
            scale_index,
        ))
    } else {
        Ok(RadixStages::radix_two(
            fft_len,
            batch_count,
            inverse,
            root_half,
            scale_index,
        ))
    }
}

fn build_chirp_data<T: WgpuFftScalar>(
    device: &WgpuDevice,
    strategy: AxisStrategy,
    batch_count: u32,
    root_length: usize,
) -> Result<Option<ChirpData<T>>> {
    let AxisStrategy::ChirpZ { n, m } = strategy else {
        return Ok(None);
    };
    let n_param = dimension(n, "Bluestein axis length")?;
    let m_param = dimension(m, "Bluestein workspace length")?;
    let mut chirp = try_host_vector(m, "Bluestein kernel")?;
    chirp.resize(m, [0.0_f64, 0.0_f64]);
    let mut direct = try_host_vector(n, "Bluestein direct factor")?;
    for position in 0..n {
        let index = u32::try_from(position)
            .expect("invariant: the Bluestein axis length was validated as u32");
        let angle = chirp_angle(index, n_param);
        let value = [angle.cos(), angle.sin()];
        chirp[position] = value;
        if position > 0 {
            chirp[m - position] = value;
        }
        direct.push([
            T::from_fft_coefficient(value[0]),
            T::from_fft_coefficient(-value[1]),
        ]);
    }
    forward_radix_two(&mut chirp);

    let mut component = try_host_vector(m, "Bluestein upload component")?;
    component.extend(chirp.iter().map(|value| T::from_fft_coefficient(value[0])));
    let real_kernel = device.upload(&component)?;
    component.clear();
    component.extend(chirp.iter().map(|value| T::from_fft_coefficient(value[1])));
    let imaginary_kernel = device.upload(&component)?;
    drop(chirp);

    component.clear();
    component.extend(direct.iter().map(|value| value[0]));
    component.push(T::from_fft_coefficient(1.0 / f64::from(n_param)));
    let direct_real = device.upload(&component)?;
    component.clear();
    component.extend(direct.iter().map(|value| value[1]));
    let direct_imaginary = device.upload(&component)?;

    let (root_half, scale_index) = radix_parameters(root_length, m_param)?;
    Ok(Some(ChirpData {
        real_kernel,
        imaginary_kernel,
        direct_real,
        direct_imaginary,
        params: ChirpParams {
            n: n_param,
            m: m_param,
            batch_count,
            padding: 0,
        },
        forward_radix: RadixStages::radix_two(m_param, batch_count, false, root_half, scale_index),
        inverse_radix: RadixStages::radix_two(m_param, batch_count, true, root_half, scale_index),
    }))
}

impl<T: WgpuFftScalar> WgpuFftPlan<T> {
    pub(crate) fn new<const R: usize>(
        device: &WgpuDevice,
        plan: FftPlan<R>,
        real: WgpuBuffer<T>,
        imaginary: WgpuBuffer<T>,
    ) -> Result<Self> {
        let Sizing {
            dimensions,
            batch,
            strategy,
            radix_root_length,
            workspace_elements,
            ..
        } = Sizing::for_device::<T, R>(device, &plan)?;
        let axes = [Axis::X, Axis::Y, Axis::Z];
        let pipelines = FftPipelines::new(device)?;

        let mut pack = [PackParams::zeroed(); 3];
        for (index, axis) in axes.into_iter().enumerate() {
            let fft_len = match strategy[index] {
                AxisStrategy::Identity => 1,
                AxisStrategy::FusedRadix2 | AxisStrategy::StagedRadix2 => {
                    dimension(axis.len(dimensions), "axis length")?
                }
                AxisStrategy::ChirpZ { m, .. } => dimension(m, "Bluestein workspace length")?,
            };
            pack[index] = PackParams {
                n: dimension(axis.len(dimensions), "axis length")?,
                stage: 0,
                inverse: 0,
                batch_count: batch[index],
                nx: dimension(dimensions[0], "x extent")?,
                ny: dimension(dimensions[1], "y extent")?,
                nz: dimension(dimensions[2], "z extent")?,
                axis: axis.shader_index(),
                fft_len,
                padding: [0; 3],
            };
        }

        let inverse = matches!(plan.direction, FftDirection::Inverse);
        let workspace = workspace_elements
            .map(|elements| {
                Ok(FftWorkspace {
                    real: device.alloc_zeroed(elements)?,
                    imaginary: device.alloc_zeroed(elements)?,
                })
            })
            .transpose()?;
        let fused_twiddle = if strategy.contains(&AxisStrategy::FusedRadix2) {
            Some(build_fused_twiddle(device)?)
        } else {
            None
        };
        let radix_twiddle = build_radix_twiddle(device, radix_root_length)?;
        let mut prepared = Self {
            real,
            imaginary,
            workspace,
            fused_twiddle,
            radix_twiddle,
            chirp: [
                build_chirp_data(device, strategy[0], batch[0], radix_root_length)?,
                build_chirp_data(device, strategy[1], batch[1], radix_root_length)?,
                build_chirp_data(device, strategy[2], batch[2], radix_root_length)?,
            ],
            stages: [
                radix_stages(
                    dimensions[0],
                    strategy[0],
                    batch[0],
                    inverse,
                    radix_root_length,
                )?,
                radix_stages(
                    dimensions[1],
                    strategy[1],
                    batch[1],
                    inverse,
                    radix_root_length,
                )?,
                radix_stages(
                    dimensions[2],
                    strategy[2],
                    batch[2],
                    inverse,
                    radix_root_length,
                )?,
            ],
            strategy,
            pack,
            commands: Box::default(),
        };
        prepared.commands = prepared.prepare_commands(
            device,
            &pipelines,
            plan.direction,
            super::dispatch::FftComponents {
                real: &prepared.real,
                imaginary: &prepared.imaginary,
            },
        )?;
        Ok(prepared)
    }

    pub(crate) fn axis_is_active(&self, axis: Axis) -> bool {
        let index = axis.shader_index() as usize;
        !matches!(self.strategy[index], AxisStrategy::Identity)
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
