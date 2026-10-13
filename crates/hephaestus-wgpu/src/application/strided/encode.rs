use hephaestus_core::{BlockWidth, Result, StridedMeta};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::workgroups;
use crate::infrastructure::device::WgpuDevice;

/// Upload the meta uniform, bind `buffers` after it at consecutive slots, and
/// dispatch `len` invocations: the single encode path shared by every strided
/// kernel.
pub(crate) fn encode_strided(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    meta: &StridedMeta,
    buffers: &[&wgpu::Buffer],
    len: usize,
    width: BlockWidth,
    label: &str,
) -> Result<()> {
    let groups = workgroups(len, width)?;

    // Pooled meta uniform: queue.write_buffer is ordered on the queue
    // timeline, so recycling after submit cannot race in-flight dispatches.
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<StridedMeta>(1)?)?;

    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(meta));

    let mut entries = BindGroupEntries::with_capacity(buffers.len() + 1);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: meta_buffer.as_entire_binding(),
    });
    for (slot, buffer) in buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: u32::try_from(slot + 1).expect("invariant: strided bind slot count fits u32"),
            resource: buffer.as_entire_binding(),
        });
    }
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
