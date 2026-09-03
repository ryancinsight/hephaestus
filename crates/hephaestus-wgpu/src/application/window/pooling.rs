use core::any::TypeId;
use core::marker::PhantomData;

use hephaestus_core::{
    HephaestusError, PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps,
    Result, plan_pooling_backward, plan_pooling_forward,
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

/// Prepared pooling forward resources.
pub struct PreparedPoolingForward<T, const R: usize, const S: usize> {
    kernel: PreparedWindowKernel,
    marker: PhantomData<T>,
}

/// Prepared pooling backward resources.
pub struct PreparedPoolingBackward<T, const R: usize, const S: usize> {
    kernel: PreparedWindowKernel,
    marker: PhantomData<T>,
}

/// WGPU implementation of generic pooling operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuPoolingOps;

impl<T> PoolingOps<WgpuDevice, T> for WgpuPoolingOps
where
    T: WgpuWindowScalar,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = PreparedPoolingForward<T, R, S>
    where
        WgpuDevice: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = PreparedPoolingBackward<T, R, S>
    where
        WgpuDevice: 'a,
        T: 'a;

    fn prepare_pooling_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: PoolingForwardOperands<'a, WgpuBuffer<T>, R>,
        parameters: WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = operands.input.buffer.aliases(operands.output.buffer);
        let plan = plan_pooling_forward::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let metadata = WindowMeta::new(
            WindowLayoutMeta::new(operands.input.layout)?,
            WindowLayoutMeta::new(operands.output.layout)?,
            WindowLayoutMeta::empty(),
            &plan.geometry,
        )?;
        let elements = operands.output.layout.checked_size().map_err(|error| {
            HephaestusError::InvalidConfiguration {
                message: format!("pooling output layout rejected: {error}"),
            }
        })?;
        let kernel = prepare_kernel::<T, 2>(
            device,
            elements,
            mode,
            metadata,
            [
                binding(0, operands.input.buffer),
                binding(1, operands.output.buffer),
            ],
            "hephaestus-pooling-forward",
        )?;
        Ok(PreparedPoolingForward {
            kernel,
            marker: PhantomData,
        })
    }

    fn dispatch_pooling_forward<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        prepared.kernel.dispatch(device)
    }

    fn prepare_pooling_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: PoolingBackwardOperands<'a, WgpuBuffer<T>, R>,
        parameters: WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let input_aliases = operands
            .input
            .is_some_and(|input| operands.grad_input.buffer.aliases(input.buffer));
        let illegal_aliasing = input_aliases
            || operands
                .grad_input
                .buffer
                .aliases(operands.grad_output.buffer);
        let plan =
            plan_pooling_backward::<T, _, R, S>(&operands, parameters, mode, illegal_aliasing)?;
        plan.validate_address_limit(i32::MAX as usize)?;
        let input_layout = operands
            .input
            .map_or(operands.grad_input.layout, |input| input.layout);
        let metadata = WindowMeta::new(
            WindowLayoutMeta::new(input_layout)?,
            WindowLayoutMeta::new(operands.grad_output.layout)?,
            WindowLayoutMeta::new(operands.grad_input.layout)?,
            &plan.geometry,
        )?;
        let elements = operands.grad_input.layout.checked_size().map_err(|error| {
            HephaestusError::InvalidConfiguration {
                message: format!("pooling gradient-input layout rejected: {error}"),
            }
        })?;
        let bindings = match mode {
            PoolingMode::Maximum => [
                binding(
                    0,
                    operands
                        .input
                        .expect("invariant: maximum input was planned")
                        .buffer,
                ),
                binding(1, operands.grad_output.buffer),
                binding(2, operands.grad_input.buffer),
            ],
            PoolingMode::Average => [
                binding(0, operands.grad_output.buffer),
                binding(1, operands.grad_input.buffer),
                binding(2, operands.grad_input.buffer),
            ],
        };
        let kernel = match mode {
            PoolingMode::Maximum => prepare_kernel::<T, 3>(
                device,
                elements,
                mode,
                metadata,
                [
                    bindings[0].clone(),
                    bindings[1].clone(),
                    bindings[2].clone(),
                ],
                "hephaestus-pooling-backward",
            )?,
            PoolingMode::Average => prepare_average_backward_kernel::<T>(
                device,
                elements,
                metadata,
                [bindings[0].clone(), bindings[1].clone()],
                "hephaestus-pooling-backward",
            )?,
        };
        Ok(PreparedPoolingBackward {
            kernel,
            marker: PhantomData,
        })
    }

    fn dispatch_pooling_backward<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
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

fn prepare_kernel<T, const N: usize>(
    device: &WgpuDevice,
    elements: usize,
    mode: PoolingMode,
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
        pooling_pipeline_key::<N>(mode),
        TypeId::of::<T>(),
        WORKGROUP_WIDTH.get(),
    );
    let pipeline = try_cached_pipeline(device, key, label, || {
        if N == 2 {
            shader::pooling_forward::<T>(mode, WORKGROUP_WIDTH.get())
        } else {
            shader::pooling_backward::<T>(mode, WORKGROUP_WIDTH.get())
        }
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

fn prepare_average_backward_kernel<T>(
    device: &WgpuDevice,
    elements: usize,
    metadata: WindowMeta,
    bindings: [wgpu::BindGroupEntry<'_>; 2],
    label: &'static str,
) -> Result<PreparedWindowKernel>
where
    T: WgpuWindowScalar,
{
    if elements == 0 {
        return Ok(PreparedWindowKernel::empty(device, label));
    }
    let key = (
        pooling_pipeline_key::<3>(PoolingMode::Average),
        TypeId::of::<T>(),
        WORKGROUP_WIDTH.get(),
    );
    let pipeline = try_cached_pipeline(device, key, label, || {
        shader::pooling_backward::<T>(PoolingMode::Average, WORKGROUP_WIDTH.get())
    })?;
    let metadata_buffer = metadata_buffer(device, &metadata)?;
    let entries = [
        bindings[0].clone(),
        bindings[1].clone(),
        wgpu::BindGroupEntry {
            binding: 2,
            resource: metadata_buffer.as_entire_binding(),
        },
    ];
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

struct PoolForwardKernel<const S: usize>;
struct PoolBackwardKernel<const S: usize, const N: usize>;

fn pooling_pipeline_key<const N: usize>(mode: PoolingMode) -> TypeId {
    match (mode, N) {
        (PoolingMode::Maximum, 2) => TypeId::of::<PoolForwardKernel<2>>(),
        (PoolingMode::Average, 2) => TypeId::of::<PoolForwardKernel<3>>(),
        (PoolingMode::Maximum, _) => TypeId::of::<PoolBackwardKernel<1, N>>(),
        (PoolingMode::Average, _) => TypeId::of::<PoolBackwardKernel<2, N>>(),
    }
}
