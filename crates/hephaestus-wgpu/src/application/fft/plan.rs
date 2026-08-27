//! WGPU planning and provider-owned storage for dense complex FFTs.

use bytemuck::Zeroable;
use hephaestus_core::{
    ComputeDevice, DeviceLimits, FftDirection, FftPlan, HephaestusError, Result,
};

use crate::infrastructure::{buffer::WgpuBuffer, device::WgpuDevice};

use super::{
    kernel::{
        ChirpParams, FUSED_MAX_LENGTH, FUSED_WORKGROUP_SIZE, FUSED_WORKGROUP_STORAGE_BYTES,
        PackParams, WORKGROUP_SIZE,
    },
    pipelines::FftPipelines,
    stages::RadixStages,
    strategy::{Axis, AxisStrategy, ChirpData},
};

pub(crate) struct WgpuFftPlan {
    pub(crate) rank: usize,
    pub(crate) real: WgpuBuffer<f32>,
    pub(crate) imaginary: WgpuBuffer<f32>,
    pub(crate) workspace: Option<FftWorkspace>,
    pub(crate) fused_twiddle: Option<WgpuBuffer<f32>>,
    pub(crate) strategy: [AxisStrategy; 3],
    pub(crate) pack: [PackParams; 3],
    pub(crate) chirp: [Option<ChirpData>; 3],
    pub(crate) stages: [RadixStages; 3],
    pub(crate) commands: Box<[crate::application::stream::WgpuBoundDispatch]>,
}

pub(crate) struct FftWorkspace {
    pub(crate) real: WgpuBuffer<f32>,
    pub(crate) imaginary: WgpuBuffer<f32>,
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

fn try_host_vector<T>(capacity: usize, role: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| HephaestusError::AllocationFailed {
            message: format!("FFT {role} host allocation for {capacity} values failed: {error}"),
        })?;
    Ok(values)
}

fn next_power_of_two(n: usize) -> Result<usize> {
    let mut power = 1usize;
    while power < n {
        power = power.checked_mul(2).ok_or_else(|| {
            invalid(format!(
                "FFT length {n} cannot be rounded to a power of two"
            ))
        })?;
    }
    Ok(power)
}

fn axis_strategy_for(n: usize) -> Result<AxisStrategy> {
    if n.is_power_of_two() {
        return Ok(AxisStrategy::StagedRadix2);
    }
    let convolution_len = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            invalid(format!(
                "Bluestein convolution length overflows for axis {n}"
            ))
        })?;
    Ok(AxisStrategy::ChirpZ {
        n,
        m: next_power_of_two(convolution_len)?,
    })
}

fn select_fused_strategy(
    strategy: AxisStrategy,
    axis_len: usize,
    batch_count: u32,
    limits: DeviceLimits,
    max_workgroups_per_dimension: u32,
) -> AxisStrategy {
    let grid_capacity = u64::from(max_workgroups_per_dimension)
        .saturating_mul(u64::from(max_workgroups_per_dimension));
    if matches!(strategy, AxisStrategy::StagedRadix2)
        && axis_len > 1
        && axis_len <= FUSED_MAX_LENGTH
        && limits.max_compute_workgroup_size_x >= FUSED_WORKGROUP_SIZE
        && limits.max_compute_invocations_per_workgroup >= FUSED_WORKGROUP_SIZE
        && limits.max_compute_workgroup_storage_size >= FUSED_WORKGROUP_STORAGE_BYTES
        && u64::from(batch_count) <= grid_capacity
    {
        AxisStrategy::FusedRadix2
    } else {
        strategy
    }
}

fn dimension(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid(format!("FFT {name}={value} exceeds the shader u32 domain")))
}

fn axis_workspace_elements(
    dimensions: [usize; 3],
    axis: Axis,
    strategy: AxisStrategy,
) -> Result<Option<usize>> {
    if axis.len(dimensions) == 1 {
        return Ok(None);
    }
    let transform_len = match strategy {
        AxisStrategy::FusedRadix2 => return Ok(None),
        AxisStrategy::StagedRadix2 => axis.len(dimensions),
        AxisStrategy::ChirpZ { m, .. } => m,
    };
    transform_len
        .checked_mul(
            axis.batch_count(dimensions)
                .ok_or_else(|| invalid("FFT axis batch count overflows"))?,
        )
        .ok_or_else(|| {
            invalid(format!(
                "FFT workspace element count overflows for {axis:?}"
            ))
        })
        .map(Some)
}

fn validate_storage_limit(
    max_buffer_size: u64,
    max_workgroups_per_dimension: u32,
    volume_elements: usize,
    dimensions: [usize; 3],
    strategies: [AxisStrategy; 3],
) -> Result<Option<usize>> {
    for (name, value) in [
        ("x extent", dimensions[0]),
        ("y extent", dimensions[1]),
        ("z extent", dimensions[2]),
    ] {
        dimension(value, name)?;
    }
    let workspace_elements = [
        axis_workspace_elements(dimensions, Axis::X, strategies[0])?,
        axis_workspace_elements(dimensions, Axis::Y, strategies[1])?,
        axis_workspace_elements(dimensions, Axis::Z, strategies[2])?,
    ]
    .into_iter()
    .flatten()
    .max();
    if let Some(workspace_elements) = workspace_elements
        && workspace_elements > u32::MAX as usize
    {
        return Err(invalid(format!(
            "FFT workspace element count {workspace_elements} exceeds the shader u32 address domain"
        )));
    }
    for (name, elements) in core::iter::once(("volume", volume_elements))
        .chain(workspace_elements.map(|workspace| ("workspace", workspace)))
    {
        let bytes = u64::try_from(elements)
            .map_err(|_| invalid(format!("FFT {name} element count exceeds u64")))?
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("FFT {name} byte count overflows")))?;
        if bytes > max_buffer_size {
            return Err(invalid(format!(
                "FFT {name} requires {bytes} bytes, exceeding device max_buffer_size={max_buffer_size}"
            )));
        }
    }
    if let Some(workspace_elements) = workspace_elements {
        let workgroups = workspace_elements.div_ceil(WORKGROUP_SIZE as usize);
        if workgroups > max_workgroups_per_dimension as usize {
            return Err(invalid(format!(
                "FFT dispatch requires {workgroups} x workgroups, exceeding device max_compute_workgroups_per_dimension={max_workgroups_per_dimension}"
            )));
        }
    }
    Ok(workspace_elements)
}

fn radix_stages(
    axis_len: usize,
    strategy: AxisStrategy,
    batch_count: u32,
    inverse: bool,
) -> Result<RadixStages> {
    if !matches!(strategy, AxisStrategy::StagedRadix2) {
        return Ok(RadixStages::empty());
    }
    let fft_len = dimension(axis_len, "radix axis length")?;
    if fft_len.trailing_zeros() % 2 == 0 {
        Ok(RadixStages::radix_four(fft_len, batch_count, inverse))
    } else {
        Ok(RadixStages::radix_two(fft_len, batch_count, inverse))
    }
}

fn build_fused_twiddle(device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
    let mut roots = [0.0_f32; FUSED_MAX_LENGTH];
    for index in 0..(FUSED_MAX_LENGTH / 2) {
        let angle = -core::f64::consts::TAU * index as f64 / FUSED_MAX_LENGTH as f64;
        roots[index] = angle.cos() as f32;
        roots[FUSED_MAX_LENGTH / 2 + index] = angle.sin() as f32;
    }
    device.upload(&roots)
}

fn forward_radix_two(values: &mut [[f64; 2]]) {
    let n = values.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    let mut reversed = 0usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut span = 2usize;
    loop {
        let angle = -core::f64::consts::TAU / span as f64;
        let step = [angle.cos(), angle.sin()];
        for start in (0..n).step_by(span) {
            let mut twiddle = [1.0, 0.0];
            for offset in 0..span / 2 {
                let even = values[start + offset];
                let odd = values[start + offset + span / 2];
                let rotated = [
                    odd[0].mul_add(twiddle[0], -(odd[1] * twiddle[1])),
                    odd[0].mul_add(twiddle[1], odd[1] * twiddle[0]),
                ];
                values[start + offset] = [even[0] + rotated[0], even[1] + rotated[1]];
                values[start + offset + span / 2] = [even[0] - rotated[0], even[1] - rotated[1]];
                twiddle = [
                    twiddle[0].mul_add(step[0], -(twiddle[1] * step[1])),
                    twiddle[0].mul_add(step[1], twiddle[1] * step[0]),
                ];
            }
        }
        if span == n {
            break;
        }
        span *= 2;
    }
}

fn chirp_angle(index: u32, n: u32) -> f64 {
    debug_assert!(n != 0);
    let index = u64::from(index);
    let n_wide = u64::from(n);
    let phase_index = (index * index) % (2 * n_wide);
    let whole_pi = u32::try_from(phase_index / n_wide)
        .expect("invariant: a phase reduced modulo 2N contains at most one whole pi");
    let remainder = u32::try_from(phase_index % n_wide)
        .expect("invariant: the phase remainder is less than the u32 transform length");
    f64::from(whole_pi) * core::f64::consts::PI
        + core::f64::consts::PI * f64::from(remainder) / f64::from(n)
}

fn build_chirp_data(
    device: &WgpuDevice,
    strategy: AxisStrategy,
    batch_count: u32,
) -> Result<Option<ChirpData>> {
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
        direct.push([value[0] as f32, -value[1] as f32]);
    }
    forward_radix_two(&mut chirp);

    let mut component = try_host_vector(m, "Bluestein upload component")?;
    component.extend(chirp.iter().map(|value| value[0] as f32));
    let real_kernel = device.upload(&component)?;
    component.clear();
    component.extend(chirp.iter().map(|value| value[1] as f32));
    let imaginary_kernel = device.upload(&component)?;
    drop(chirp);

    component.clear();
    component.extend(direct.iter().map(|value| value[0]));
    let direct_real = device.upload(&component)?;
    component.clear();
    component.extend(direct.iter().map(|value| value[1]));
    let direct_imaginary = device.upload(&component)?;

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
        forward_radix: RadixStages::radix_two(m_param, batch_count, false),
        inverse_radix: RadixStages::radix_two(m_param, batch_count, true),
    }))
}

impl WgpuFftPlan {
    pub(crate) fn new<const R: usize>(
        device: &WgpuDevice,
        plan: FftPlan<R>,
        real: WgpuBuffer<f32>,
        imaginary: WgpuBuffer<f32>,
    ) -> Result<Self> {
        let mut dimensions = [1usize; 3];
        dimensions[..R].copy_from_slice(&plan.shape);
        let axes = [Axis::X, Axis::Y, Axis::Z];
        let limits = device.device_limits();
        let max_workgroups_per_dimension =
            device.inner().limits().max_compute_workgroups_per_dimension;
        if limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
            || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        {
            return Err(invalid(format!(
                "FFT kernels require {WORKGROUP_SIZE} x invocations; device reports max x={} and max invocations={}",
                limits.max_compute_workgroup_size_x, limits.max_compute_invocations_per_workgroup
            )));
        }
        let mut batch = [0u32; 3];
        for (index, axis) in axes.into_iter().enumerate() {
            batch[index] = dimension(
                axis.batch_count(dimensions)
                    .ok_or_else(|| invalid("FFT axis batch count overflows"))?,
                "axis batch count",
            )?;
        }
        let strategy = [
            select_fused_strategy(
                axis_strategy_for(dimensions[0])?,
                dimensions[0],
                batch[0],
                limits,
                max_workgroups_per_dimension,
            ),
            select_fused_strategy(
                axis_strategy_for(dimensions[1])?,
                dimensions[1],
                batch[1],
                limits,
                max_workgroups_per_dimension,
            ),
            select_fused_strategy(
                axis_strategy_for(dimensions[2])?,
                dimensions[2],
                batch[2],
                limits,
                max_workgroups_per_dimension,
            ),
        ];
        let workspace_elements = validate_storage_limit(
            limits.max_buffer_size,
            max_workgroups_per_dimension,
            plan.elements,
            dimensions,
            strategy,
        )?;
        let pipelines = FftPipelines::new(device)?;

        let mut pack = [PackParams::zeroed(); 3];
        for (index, axis) in axes.into_iter().enumerate() {
            let fft_len = match strategy[index] {
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
        let mut prepared = Self {
            rank: R,
            real,
            imaginary,
            workspace,
            fused_twiddle,
            chirp: [
                build_chirp_data(device, strategy[0], batch[0])?,
                build_chirp_data(device, strategy[1], batch[1])?,
                build_chirp_data(device, strategy[2], batch[2])?,
            ],
            stages: [
                radix_stages(dimensions[0], strategy[0], batch[0], inverse)?,
                radix_stages(dimensions[1], strategy[1], batch[1], inverse)?,
                radix_stages(dimensions[2], strategy[2], batch[2], inverse)?,
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
        index < self.rank && self.pack[index].n > 1
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
