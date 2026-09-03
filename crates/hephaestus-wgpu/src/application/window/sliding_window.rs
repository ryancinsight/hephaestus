use core::any::TypeId;
use core::marker::PhantomData;

use hephaestus_core::{
    HephaestusError, Result, SlidingWindowFoldOperands, SlidingWindowOps,
    SlidingWindowUnfoldOperands, plan_sliding_window_fold, plan_sliding_window_unfold,
};
use leto::WindowParameters;

use super::WgpuWindowScalar;
use super::metadata::{WindowLayoutMeta, WindowMeta};
use super::prepared::PreparedWindowKernel;
use super::shader;
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::{UniformBufferGuard, uniform_guard};

const WORKGROUP_WIDTH: hephaestus_core::BlockWidth = hephaestus_core::BlockWidth::DEFAULT;

/// Prepared sliding-window unfold resources.
pub struct PreparedSlidingWindowUnfold<T, const R: usize, const S: usize> {
    kernel: PreparedWindowKernel,
    marker: PhantomData<T>,
}

/// Prepared sliding-window fold resources.
pub struct PreparedSlidingWindowFold<T, const R: usize, const S: usize> {
    kernel: PreparedWindowKernel,
    marker: PhantomData<T>,
}

/// WGPU implementation of generic sliding-window operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuSlidingWindowOps;

impl<T> SlidingWindowOps<WgpuDevice, T> for WgpuSlidingWindowOps
where
    T: WgpuWindowScalar,
{
    type PreparedUnfold<'a, const R: usize, const S: usize>
        = PreparedSlidingWindowUnfold<T, R, S>
    where
        WgpuDevice: 'a,
        T: 'a;
    type PreparedFold<'a, const R: usize, const S: usize>
        = PreparedSlidingWindowFold<T, R, S>
    where
        WgpuDevice: 'a,
        T: 'a;

    fn prepare_unfold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: SlidingWindowUnfoldOperands<'a, WgpuBuffer<T>, R>,
        parameters: WindowParameters<S>,
    ) -> Result<Self::PreparedUnfold<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = operands.input.buffer.aliases(operands.output.buffer);
        let plan =
            plan_sliding_window_unfold::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let metadata = WindowMeta::new(
            WindowLayoutMeta::new(operands.input.layout)?,
            WindowLayoutMeta::new(operands.output.layout)?,
            WindowLayoutMeta::empty(),
            &plan.geometry,
        )?;
        let elements = operands.output.layout.checked_size().map_err(|error| {
            HephaestusError::InvalidConfiguration {
                message: format!("unfold output layout rejected: {error}"),
            }
        })?;
        let kernel = prepare_kernel::<T, 2, true>(
            device,
            elements,
            metadata,
            [
                binding(0, operands.input.buffer),
                binding(1, operands.output.buffer),
            ],
            "hephaestus-sliding-window-unfold",
        )?;
        Ok(PreparedSlidingWindowUnfold {
            kernel,
            marker: PhantomData,
        })
    }

    fn dispatch_unfold<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedUnfold<'_, R, S>,
    ) -> Result<()> {
        prepared.kernel.dispatch(device)
    }

    fn prepare_fold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: SlidingWindowFoldOperands<'a, WgpuBuffer<T>, R>,
        output_spatial_shape: [usize; S],
        parameters: WindowParameters<S>,
    ) -> Result<Self::PreparedFold<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = operands.input.buffer.aliases(operands.output.buffer);
        let plan = plan_sliding_window_fold::<T, _, R, S>(
            &operands,
            output_spatial_shape,
            parameters,
            illegal_aliasing,
        )?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let metadata = WindowMeta::new(
            WindowLayoutMeta::new(operands.input.layout)?,
            WindowLayoutMeta::new(operands.output.layout)?,
            WindowLayoutMeta::empty(),
            &plan.geometry,
        )?;
        let elements = operands.output.layout.checked_size().map_err(|error| {
            HephaestusError::InvalidConfiguration {
                message: format!("fold output layout rejected: {error}"),
            }
        })?;
        let kernel = prepare_kernel::<T, 2, false>(
            device,
            elements,
            metadata,
            [
                binding(0, operands.input.buffer),
                binding(1, operands.output.buffer),
            ],
            "hephaestus-sliding-window-fold",
        )?;
        Ok(PreparedSlidingWindowFold {
            kernel,
            marker: PhantomData,
        })
    }

    fn dispatch_fold<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedFold<'_, R, S>,
    ) -> Result<()> {
        prepared.kernel.dispatch(device)
    }
}

fn validate_spatial_rank<const S: usize>() -> Result<()> {
    if !(1..=3).contains(&S) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("WGPU window operations support spatial ranks 1 through 3, got {S}"),
        });
    }
    Ok(())
}

fn binding<T>(binding: u32, buffer: &WgpuBuffer<T>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.raw().as_entire_binding(),
    }
}

fn metadata_buffer(device: &WgpuDevice, metadata: &WindowMeta) -> Result<UniformBufferGuard> {
    let raw = device.get_uniform_buffer(WgpuDevice::byte_size::<WindowMeta>(1)?)?;
    let buffer = uniform_guard(device.clone(), raw);
    device
        .queue()
        .write_buffer(&buffer, 0, eunomia::layout::bytes_of(metadata));
    Ok(buffer)
}

fn prepare_kernel<T, const N: usize, const UNFOLD: bool>(
    device: &WgpuDevice,
    elements: usize,
    metadata: WindowMeta,
    bindings: [wgpu::BindGroupEntry<'_>; N],
    label: &'static str,
) -> Result<PreparedWindowKernel>
where
    T: WgpuWindowScalar,
{
    if elements == 0 {
        return Ok(PreparedWindowKernel::empty(device, label));
    }
    let key = (
        TypeId::of::<SlidingWindowKernel<UNFOLD>>(),
        TypeId::of::<T>(),
        WORKGROUP_WIDTH.get(),
    );
    let pipeline = try_cached_pipeline(device, key, label, || {
        shader::sliding_window::<T>(UNFOLD, WORKGROUP_WIDTH.get())
    })?;
    let metadata_buffer = metadata_buffer(device, &metadata)?;
    let mut entries = bindings.to_vec();
    entries.push(wgpu::BindGroupEntry {
        binding: N as u32,
        resource: metadata_buffer.as_entire_binding(),
    });
    let bind_group = checked_bind_group(device, &pipeline, label, &entries)?;
    Ok(PreparedWindowKernel::ready(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(elements, WORKGROUP_WIDTH)?,
        label,
    ))
}

struct SlidingWindowKernel<const UNFOLD: bool>;
