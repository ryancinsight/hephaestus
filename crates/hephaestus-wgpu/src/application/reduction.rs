use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Zero-sized reduction operation marker selecting the WGSL combine expression.
pub trait ReductionWgslOp: Copy + Send + Sync + 'static {
    /// WGSL expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"` or `"min(lhs, rhs)"`).
    const WGSL_EXPR: &'static str;
}

/// Associates a scalar type and reduction operation with the identity value.
pub trait ReductionIdentity<Op>: WgslScalar {
    /// The identity value on the host side.
    const IDENTITY: Self;
    /// The WGSL literal for the identity value.
    const WGSL_IDENTITY: &'static str;
}

/// Sum reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumOp;

/// Minimum reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinOp;

/// Maximum reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxOp;

impl ReductionWgslOp for SumOp {
    const WGSL_EXPR: &'static str = "lhs + rhs";
}

impl ReductionWgslOp for MinOp {
    const WGSL_EXPR: &'static str = "min(lhs, rhs)";
}

impl ReductionWgslOp for MaxOp {
    const WGSL_EXPR: &'static str = "max(lhs, rhs)";
}

// ── SumOp Identity implementations ──
impl ReductionIdentity<SumOp> for f32 {
    const IDENTITY: Self = 0.0;
    const WGSL_IDENTITY: &'static str = "0.0";
}
impl ReductionIdentity<SumOp> for u32 {
    const IDENTITY: Self = 0;
    const WGSL_IDENTITY: &'static str = "0u";
}
impl ReductionIdentity<SumOp> for i32 {
    const IDENTITY: Self = 0;
    const WGSL_IDENTITY: &'static str = "0";
}

// ── MinOp Identity implementations ──
impl ReductionIdentity<MinOp> for f32 {
    const IDENTITY: Self = f32::MAX;
    const WGSL_IDENTITY: &'static str = "3.402823466e+38";
}
impl ReductionIdentity<MinOp> for u32 {
    const IDENTITY: Self = u32::MAX;
    const WGSL_IDENTITY: &'static str = "4294967295u";
}
impl ReductionIdentity<MinOp> for i32 {
    const IDENTITY: Self = i32::MAX;
    const WGSL_IDENTITY: &'static str = "2147483647";
}

// ── MaxOp Identity implementations ──
impl ReductionIdentity<MaxOp> for f32 {
    const IDENTITY: Self = f32::MIN;
    const WGSL_IDENTITY: &'static str = "-3.402823466e+38";
}
impl ReductionIdentity<MaxOp> for u32 {
    const IDENTITY: Self = u32::MIN;
    const WGSL_IDENTITY: &'static str = "0u";
}
impl ReductionIdentity<MaxOp> for i32 {
    const IDENTITY: Self = i32::MIN;
    const WGSL_IDENTITY: &'static str = "-2147483648";
}

/// ZST wrapper to generate a unique TypeId in the pipeline cache for reduction operations.
struct ReductionOpWrapper<Op>(PhantomData<Op>);

fn shader_source<Op: ReductionWgslOp, T: ReductionIdentity<Op>>(width: BlockWidth) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> input: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let i = global_id.x;
    let num_elements = arrayLength(&input);
    
    if (i < num_elements) {{
        shared_data[local_id.x] = input[i];
    }} else {{
        shared_data[local_id.x] = {identity};
    }}
    
    workgroupBarrier();
    
    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (local_id.x < stride) {{
            let lhs = shared_data[local_id.x];
            let rhs = shared_data[local_id.x + stride];
            shared_data[local_id.x] = {expr};
        }}
        workgroupBarrier();
    }}
    
    if (local_id.x == 0u) {{
        output[workgroup_id.x] = shared_data[0];
    }}
}}
"#,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        identity = T::WGSL_IDENTITY,
        expr = Op::WGSL_EXPR,
    )
}

fn validate_reduction_width(width: BlockWidth) -> Result<()> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "reduction block width {} must be a power of two",
                width.get()
            ),
        });
    }
    Ok(())
}

/// Run reduction on the device, returning a 1-element buffer holding the result.
///
/// If the input buffer is empty, it returns a 1-element buffer containing the operation's identity value.
pub fn reduction<Op, T>(device: &WgpuDevice, input: &WgpuBuffer<T>) -> Result<WgpuBuffer<T>>
where
    Op: ReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op>,
{
    reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}

/// Run reduction on the device with a caller-selected power-of-two block width.
///
/// If the input buffer is empty, it returns a 1-element buffer containing the
/// operation's identity value. `width` is part of the monomorphized pipeline
/// cache key and WGSL workgroup size; non-power-of-two widths are rejected
/// because the workgroup tree halves its active lane count every step.
pub fn reduction_with_width<Op, T>(
    device: &WgpuDevice,
    input: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: ReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op>,
{
    validate_reduction_width(width)?;

    if input.len == 0 {
        return device.upload(&[T::IDENTITY]);
    }
    if input.len == 1 {
        // Create a copy of the buffer
        let out = device.alloc_zeroed::<T>(1)?;
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-reduction-copy-1"),
            });
        encoder.copy_buffer_to_buffer(
            &input.buffer,
            0,
            &out.buffer,
            0,
            std::mem::size_of::<T>() as u64,
        );
        device.queue().submit(Some(encoder.finish()));
        return Ok(out);
    }
    let mut current_len = input.len;
    let mut temp_buffers: Vec<WgpuBuffer<T>> = Vec::new();

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-reduction-multi-pass"),
        });

    while current_len > 1 {
        let out_len = current_len.div_ceil(width.get() as usize);
        let out_buffer = device.alloc_zeroed::<T>(out_len)?;

        let key = (
            TypeId::of::<ReductionOpWrapper<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        );

        let pipeline = cached_pipeline(device, key, "hephaestus-reduction", || {
            shader_source::<Op, T>(width)
        });

        let source_resource = if temp_buffers.is_empty() {
            input.buffer.as_entire_binding()
        } else {
            temp_buffers
                .last()
                .expect("invariant: non-initial reduction pass has a previous buffer")
                .buffer
                .as_entire_binding()
        };

        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-reduction"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.buffer.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-reduction-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups(current_len, width)?, 1, 1);
        }

        temp_buffers.push(out_buffer);
        current_len = out_len;
    }

    device.queue().submit(Some(encoder.finish()));

    // The final result is in the last allocated buffer.
    Ok(temp_buffers
        .pop()
        .expect("invariant: multi-element reduction allocates a final buffer"))
}
