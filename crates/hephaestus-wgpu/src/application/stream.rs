//! WGPU implementation of the backend-neutral kernel command stream.

use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{
    validate_bindings, validate_grouped_bindings, Binding, CommandStream, DispatchGrid,
    GroupedBinding, GroupedCommandStream, GroupedKernelDevice, GroupedKernelSequence,
    GroupedKernelSource, HephaestusError, KernelDevice, KernelSource, Result, Wgsl,
};

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Prepared WGPU pipeline for a kernel source `K`.
pub struct WgpuPrepared<K> {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    parameter_binding: u32,
    label: &'static str,
    marker: PhantomData<K>,
}

/// Prepared WGPU pipeline for a grouped kernel source `K`.
pub struct WgpuGroupedPrepared<K> {
    pipeline: wgpu::ComputePipeline,
    bind_group_layouts: Vec<(u32, wgpu::BindGroupLayout)>,
    parameter_group: u32,
    parameter_binding: u32,
    label: &'static str,
    marker: PhantomData<K>,
}

impl<K> core::fmt::Debug for WgpuGroupedPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuGroupedPrepared")
            .field("pipeline", &self.pipeline)
            .field("bind_group_layouts", &self.bind_group_layouts)
            .field("parameter_group", &self.parameter_group)
            .field("parameter_binding", &self.parameter_binding)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for WgpuGroupedPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            bind_group_layouts: self.bind_group_layouts.clone(),
            parameter_group: self.parameter_group,
            parameter_binding: self.parameter_binding,
            label: self.label,
            marker: PhantomData,
        }
    }
}

impl<K> core::fmt::Debug for WgpuPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuPrepared")
            .field("pipeline", &self.pipeline)
            .field("bind_group_layout", &self.bind_group_layout)
            .field("parameter_binding", &self.parameter_binding)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for WgpuPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            bind_group_layout: self.bind_group_layout.clone(),
            parameter_binding: self.parameter_binding,
            label: self.label,
            marker: PhantomData,
        }
    }
}

/// WGPU command stream for ordered kernel dispatch, copies, and fills.
pub struct WgpuCommandStream<'d> {
    device: &'d WgpuDevice,
    encoder: wgpu::CommandEncoder,
    uniform_buffers: Vec<wgpu::Buffer>,
}

/// Active WGPU grouped-kernel sequence encoded into one compute pass.
pub struct WgpuGroupedSequence<'s> {
    device: &'s WgpuDevice,
    pass: wgpu::ComputePass<'s>,
    uniform_buffers: &'s mut Vec<wgpu::Buffer>,
}

impl KernelDevice for WgpuDevice {
    type Dialect = Wgsl;
    type BindingHandle<'a> = &'a wgpu::Buffer;
    type Prepared<K: KernelSource<Wgsl>> = WgpuPrepared<K>;
    type Stream<'d> = WgpuCommandStream<'d>;

    #[inline]
    fn binding_handle<T: Pod>(buffer: &Self::Buffer<T>) -> Self::BindingHandle<'_> {
        buffer.raw()
    }

    fn prepare<K: KernelSource<Wgsl>>(&self, kernel: &K) -> Result<Self::Prepared<K>> {
        let parameter_binding =
            u32::try_from(K::BINDINGS.len()).map_err(|_| HephaestusError::DispatchFailed {
                message: format!(
                    "{}: binding count {} exceeds u32::MAX",
                    K::LABEL,
                    K::BINDINGS.len()
                ),
            })?;

        let source = kernel.source();
        let shader = self
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(K::LABEL),
                source: wgpu::ShaderSource::Wgsl(source),
            });

        let mut entries = Vec::with_capacity(K::BINDINGS.len() + 1);
        for (binding, decl) in K::BINDINGS.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: u32::try_from(binding).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("{label}: binding index exceeds u32::MAX", label = K::LABEL),
                })?,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: matches!(decl.access, hephaestus_core::Access::ReadOnly),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: parameter_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout =
            self.inner()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(K::LABEL),
                    entries: &entries,
                });
        let pipeline_layout =
            self.inner()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(K::LABEL),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = self
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(K::LABEL),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(K::ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(WgpuPrepared {
            pipeline,
            bind_group_layout,
            parameter_binding,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn stream(&self) -> Result<Self::Stream<'_>> {
        let encoder = self
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-wgpu-command-stream"),
            });
        Ok(WgpuCommandStream {
            device: self,
            encoder,
            uniform_buffers: Vec::new(),
        })
    }
}

impl GroupedKernelDevice for WgpuDevice {
    type GroupedPrepared<K: GroupedKernelSource<Wgsl>> = WgpuGroupedPrepared<K>;
    type GroupedStream<'d> = WgpuCommandStream<'d>;

    fn prepare_grouped<K: GroupedKernelSource<Wgsl>>(
        &self,
        kernel: &K,
    ) -> Result<Self::GroupedPrepared<K>> {
        let source = kernel.source();
        let shader = self
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(K::LABEL),
                source: wgpu::ShaderSource::Wgsl(source),
            });

        let bind_group_layouts =
            grouped_layouts::<K>(self.inner(), K::PARAM_GROUP, K::PARAM_BINDING)?;
        let layout_refs: Vec<&wgpu::BindGroupLayout> = bind_group_layouts
            .iter()
            .map(|(_, layout)| layout)
            .collect();
        let pipeline_layout =
            self.inner()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(K::LABEL),
                    bind_group_layouts: &layout_refs,
                    push_constant_ranges: &[],
                });
        let pipeline = self
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(K::LABEL),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(K::ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(WgpuGroupedPrepared {
            pipeline,
            bind_group_layouts,
            parameter_group: K::PARAM_GROUP,
            parameter_binding: K::PARAM_BINDING,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn grouped_stream(&self) -> Result<Self::GroupedStream<'_>> {
        self.stream()
    }
}

impl<'d> CommandStream<'d, WgpuDevice> for WgpuCommandStream<'d> {
    fn encode<K: KernelSource<Wgsl>>(
        &mut self,
        prepared: &WgpuPrepared<K>,
        bindings: &[Binding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let raw_params = self
            .device
            .get_uniform_buffer(WgpuDevice::byte_size::<K::Params>(1)?)?;
        self.device
            .queue()
            .write_buffer(&raw_params, 0, bytemuck::bytes_of(params));

        let mut entries = Vec::with_capacity(bindings.len() + 1);
        for (binding, bound) in bindings.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: u32::try_from(binding).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("{}: binding index exceeds u32::MAX", K::LABEL),
                })?,
                resource: bound.handle.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: prepared.parameter_binding,
            resource: raw_params.as_entire_binding(),
        });

        let bind_group = self
            .device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(prepared.label),
                layout: &prepared.bind_group_layout,
                entries: &entries,
            });
        self.uniform_buffers.push(raw_params);

        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(prepared.label),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&prepared.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }

        Ok(())
    }

    fn copy<T: Pod>(&mut self, src: &WgpuBuffer<T>, dst: &WgpuBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if src.len() != dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: src.len(),
                device_len: dst.len(),
            });
        }
        let byte_len = WgpuDevice::byte_size::<T>(src.len())?;
        if byte_len != 0 {
            self.encoder
                .copy_buffer_to_buffer(src.raw(), 0, dst.raw(), 0, byte_len);
        }
        Ok(())
    }

    fn fill_zero<T: Pod>(&mut self, dst: &WgpuBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        let byte_len = WgpuDevice::byte_size::<T>(dst.len())?;
        if byte_len != 0 {
            self.encoder.clear_buffer(dst.raw(), 0, Some(byte_len));
        }
        Ok(())
    }

    fn submit(self) -> Result<()> {
        self.device.queue().submit(Some(self.encoder.finish()));
        for buffer in self.uniform_buffers {
            self.device.recycle_uniform_buffer(buffer);
        }
        Ok(())
    }
}

impl<'d> GroupedCommandStream<'d, WgpuDevice> for WgpuCommandStream<'d> {
    type Sequence<'s> = WgpuGroupedSequence<'s>;

    fn encode_grouped<K: GroupedKernelSource<Wgsl>>(
        &mut self,
        prepared: &WgpuGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(prepared.label),
                    timestamp_writes: None,
                });
            encode_grouped_on_pass(
                self.device,
                &mut self.uniform_buffers,
                &mut pass,
                prepared,
                bindings,
                params,
                grid,
            )?;
        }

        Ok(())
    }

    fn encode_grouped_sequence<F>(&mut self, label: &str, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>,
    {
        let pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        let mut sequence = WgpuGroupedSequence {
            device: self.device,
            pass,
            uniform_buffers: &mut self.uniform_buffers,
        };
        encode(&mut sequence)
    }

    fn submit_grouped(self) -> Result<()> {
        CommandStream::submit(self)
    }
}

impl<'s> GroupedKernelSequence<'s, WgpuDevice> for WgpuGroupedSequence<'s> {
    fn encode_grouped<K: GroupedKernelSource<Wgsl>>(
        &mut self,
        prepared: &WgpuGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }
        encode_grouped_on_pass(
            self.device,
            self.uniform_buffers,
            &mut self.pass,
            prepared,
            bindings,
            params,
            grid,
        )
    }
}

fn encode_grouped_on_pass<K: GroupedKernelSource<Wgsl>>(
    device: &WgpuDevice,
    uniform_buffers: &mut Vec<wgpu::Buffer>,
    pass: &mut wgpu::ComputePass<'_>,
    prepared: &WgpuGroupedPrepared<K>,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    params: &K::Params,
    grid: DispatchGrid,
) -> Result<()> {
    let raw_params = device.get_uniform_buffer(WgpuDevice::byte_size::<K::Params>(1)?)?;
    device
        .queue()
        .write_buffer(&raw_params, 0, bytemuck::bytes_of(params));

    let bind_groups = build_grouped_bind_groups(device.inner(), prepared, bindings, &raw_params)?;
    uniform_buffers.push(raw_params);

    pass.set_pipeline(&prepared.pipeline);
    for (group, bind_group) in &bind_groups {
        pass.set_bind_group(*group, bind_group, &[]);
    }
    pass.dispatch_workgroups(grid.x, grid.y, grid.z);
    Ok(())
}

fn grouped_layouts<K: GroupedKernelSource<Wgsl>>(
    device: &wgpu::Device,
    parameter_group: u32,
    parameter_binding: u32,
) -> Result<Vec<(u32, wgpu::BindGroupLayout)>> {
    let mut groups = std::collections::BTreeMap::<u32, Vec<wgpu::BindGroupLayoutEntry>>::new();
    let mut seen = std::collections::BTreeSet::<(u32, u32)>::new();
    for decl in K::BINDINGS {
        if !seen.insert((decl.group, decl.binding)) {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{}: duplicate grouped binding ({}, {})",
                    K::LABEL,
                    decl.group,
                    decl.binding
                ),
            });
        }
        if decl.group == parameter_group && decl.binding == parameter_binding {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{}: storage binding ({parameter_group}, {parameter_binding}) collides with parameter binding",
                    K::LABEL
                ),
            });
        }
        groups
            .entry(decl.group)
            .or_default()
            .push(wgpu::BindGroupLayoutEntry {
                binding: decl.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: matches!(decl.access, hephaestus_core::Access::ReadOnly),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
    }
    groups
        .entry(parameter_group)
        .or_default()
        .push(wgpu::BindGroupLayoutEntry {
            binding: parameter_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

    if groups.keys().copied().ne(0..groups.len() as u32) {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "{}: WGPU grouped kernels require dense groups starting at 0",
                K::LABEL
            ),
        });
    }

    let mut layouts = Vec::with_capacity(groups.len());
    for (group, entries) in groups {
        layouts.push((
            group,
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(K::LABEL),
                entries: &entries,
            }),
        ));
    }
    Ok(layouts)
}

fn build_grouped_bind_groups<K>(
    device: &wgpu::Device,
    prepared: &WgpuGroupedPrepared<K>,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    params: &wgpu::Buffer,
) -> Result<Vec<(u32, wgpu::BindGroup)>> {
    let mut entries = std::collections::BTreeMap::<u32, Vec<wgpu::BindGroupEntry<'_>>>::new();
    for bound in bindings {
        entries
            .entry(bound.group)
            .or_default()
            .push(wgpu::BindGroupEntry {
                binding: bound.binding,
                resource: bound.handle.as_entire_binding(),
            });
    }
    entries
        .entry(prepared.parameter_group)
        .or_default()
        .push(wgpu::BindGroupEntry {
            binding: prepared.parameter_binding,
            resource: params.as_entire_binding(),
        });

    let mut bind_groups = Vec::with_capacity(prepared.bind_group_layouts.len());
    for (group, layout) in &prepared.bind_group_layouts {
        let Some(group_entries) = entries.get(group) else {
            return Err(HephaestusError::DispatchFailed {
                message: format!("{}: missing bindings for group {group}", prepared.label),
            });
        };
        bind_groups.push((
            *group,
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(prepared.label),
                layout,
                entries: group_entries,
            }),
        ));
    }
    Ok(bind_groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use hephaestus_core::{
        Access, BindingDecl, ComputeDevice, GroupedBindingDecl, GroupedKernelInterface,
        KernelInterface,
    };

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ScaleParams {
        len: u32,
        factor: f32,
    }

    struct ScaleKernel;

    impl KernelInterface for ScaleKernel {
        type Params = ScaleParams;
        const LABEL: &'static str = "hephaestus-wgpu-stream-scale";
        const BINDINGS: &'static [BindingDecl] = &[
            BindingDecl::read_only::<f32>(),
            BindingDecl::read_write::<f32>(),
        ];
        const WORKGROUP: [u32; 3] = [64, 1, 1];
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GroupedParams {
        len: u32,
        addend: f32,
    }

    struct GroupedAddKernel;

    impl GroupedKernelInterface for GroupedAddKernel {
        type Params = GroupedParams;
        const LABEL: &'static str = "hephaestus-wgpu-grouped-add";
        const BINDINGS: &'static [GroupedBindingDecl] = &[
            GroupedBindingDecl::read_only::<f32>(0, 0),
            GroupedBindingDecl::read_only::<f32>(1, 0),
            GroupedBindingDecl::read_write::<f32>(1, 1),
        ];
        const PARAM_GROUP: u32 = 0;
        const PARAM_BINDING: u32 = 1;
        const WORKGROUP: [u32; 3] = [64, 1, 1];
    }

    impl GroupedKernelSource<Wgsl> for GroupedAddKernel {
        const ENTRY: &'static str = "main";

        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    addend: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> right: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = left[i] + right[i] + params.addend;
    }
}
"#,
            )
        }
    }

    impl KernelSource<Wgsl> for ScaleKernel {
        const ENTRY: &'static str = "main";

        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    factor: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = input[i] * params.factor;
    }
}
"#,
            )
        }
    }

    fn try_device() -> Option<WgpuDevice> {
        WgpuDevice::try_with_power_preference_and_adapter_config(
            "hephaestus-wgpu-stream-test",
            wgpu::PowerPreference::HighPerformance,
            |_| wgpu::Features::empty(),
            |_| wgpu::Limits::default(),
        )
        .ok()
    }

    #[test]
    fn command_stream_dispatches_prepared_wgsl_kernel() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping command stream test");
            return;
        };
        let input = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();

        device
            .dispatch(
                &prepared,
                &[Binding::read(&input), Binding::read_write(&output)],
                &ScaleParams {
                    len: 4,
                    factor: 2.5,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();

        let mut host = [0.0f32; 4];
        device.download(&output, &mut host).unwrap();
        assert_eq!(host, [2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn command_stream_preserves_order_for_fill_copy_and_dispatch() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping command stream order test");
            return;
        };
        let input = device.upload(&[3.0f32, 4.0, 5.0, 6.0]).unwrap();
        let scratch = device.alloc_zeroed::<f32>(4).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();

        let mut stream = device.stream().unwrap();
        stream.fill_zero(&scratch).unwrap();
        stream.copy(&input, &scratch).unwrap();
        stream
            .encode(
                &prepared,
                &[Binding::read(&scratch), Binding::read_write(&output)],
                &ScaleParams {
                    len: 4,
                    factor: 3.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit().unwrap();

        let mut host = [0.0f32; 4];
        device.download(&output, &mut host).unwrap();
        assert_eq!(host, [9.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn command_stream_rejects_binding_contract_mismatch() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping binding contract test");
            return;
        };
        let input = device.upload(&[1.0f32, 2.0]).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();

        let err = device
            .dispatch(
                &prepared,
                &[Binding::read(&input)],
                &ScaleParams {
                    len: 2,
                    factor: 1.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap_err();

        match err {
            HephaestusError::DispatchFailed { message } => {
                assert!(message.contains("declares 2 storage bindings, got 1"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn command_stream_copy_rejects_length_mismatch() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping copy contract test");
            return;
        };
        let src = device.upload(&[1u32, 2, 3]).unwrap();
        let dst = device.alloc_zeroed::<u32>(2).unwrap();
        let mut stream = device.stream().unwrap();

        let err = stream.copy(&src, &dst).unwrap_err();

        assert!(matches!(
            err,
            HephaestusError::LengthMismatch {
                host_len: 3,
                device_len: 2
            }
        ));
    }

    #[test]
    fn grouped_command_stream_dispatches_multi_group_wgsl_kernel() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping grouped command stream test");
            return;
        };
        let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

        device
            .dispatch_grouped(
                &prepared,
                &[
                    GroupedBinding::read(0, 0, &left),
                    GroupedBinding::read(1, 0, &right),
                    GroupedBinding::read_write(1, 1, &output),
                ],
                &GroupedParams {
                    len: 4,
                    addend: 0.5,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();

        let mut host = [0.0f32; 4];
        device.download(&output, &mut host).unwrap();
        assert_eq!(host, [11.5, 22.5, 33.5, 44.5]);
    }

    #[test]
    fn grouped_sequence_preserves_order_in_one_wgpu_pass() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping grouped sequence test");
            return;
        };
        let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let scratch = device.alloc_zeroed::<f32>(4).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-wgpu-grouped-sequence", |sequence| {
                sequence.encode_grouped(
                    &prepared,
                    &[
                        GroupedBinding::read(0, 0, &left),
                        GroupedBinding::read(1, 0, &right),
                        GroupedBinding::read_write(1, 1, &scratch),
                    ],
                    &GroupedParams {
                        len: 4,
                        addend: 0.5,
                    },
                    DispatchGrid::new(1, 1, 1),
                )?;
                sequence.encode_grouped(
                    &prepared,
                    &[
                        GroupedBinding::read(0, 0, &scratch),
                        GroupedBinding::read(1, 0, &right),
                        GroupedBinding::read_write(1, 1, &output),
                    ],
                    &GroupedParams {
                        len: 4,
                        addend: 1.0,
                    },
                    DispatchGrid::new(1, 1, 1),
                )
            })
            .unwrap();
        stream.submit_grouped().unwrap();

        let mut host = [0.0f32; 4];
        device.download(&output, &mut host).unwrap();
        assert_eq!(host, [22.5, 43.5, 64.5, 85.5]);
    }

    #[test]
    fn grouped_command_stream_rejects_group_mismatch() {
        let Some(device) = try_device() else {
            eprintln!("No WGPU adapter available; skipping grouped mismatch test");
            return;
        };
        let left = device.upload(&[1.0f32]).unwrap();
        let right = device.upload(&[2.0f32]).unwrap();
        let output = device.alloc_zeroed::<f32>(1).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

        let err = device
            .dispatch_grouped(
                &prepared,
                &[
                    GroupedBinding::read(0, 0, &left),
                    GroupedBinding::read(0, 0, &right),
                    GroupedBinding::read_write(1, 1, &output),
                ],
                &GroupedParams {
                    len: 1,
                    addend: 0.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap_err();

        match err {
            HephaestusError::DispatchFailed { message } => {
                assert!(message.contains("declared group 1 binding 0"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn kernel_interface_declares_wgpu_parameter_binding_after_storage() {
        assert_eq!(ScaleKernel::BINDINGS.len(), 2);
        assert_eq!(ScaleKernel::BINDINGS[0].access, Access::ReadOnly);
        assert_eq!(ScaleKernel::BINDINGS[1].access, Access::ReadWrite);
    }
}
