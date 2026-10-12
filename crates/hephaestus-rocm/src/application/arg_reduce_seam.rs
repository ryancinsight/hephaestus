//! ROCm implementation of [`hephaestus_core::ArgReduceOps`].
//!
//! One thread per output lane scans the full reduced axis in a plain loop —
//! parallelism is across lanes, not within one lane's scan, mirroring the
//! host reference's first-strict-improvement tie-break exactly. The
//! comparison direction (argmax vs argmin) bakes into the generated source
//! at pipeline-cache-key granularity; the axis itself needs no baking
//! because the host selects the axis/lane strides before dispatch.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    ArgReduceOps, BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result,
    StridedView, map_layout_err, validate_arg_reduce_shape,
};

use crate::RocmDevice;
use crate::application::linalg::{to_i32, to_u32};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

fn arg_reduce_source<T: DialectScalar<HipC>>(min: bool) -> String {
    let better = if min {
        "candidate < best_val"
    } else {
        "candidate > best_val"
    };
    format!(
        r#"{prelude}extern "C" __global__ void arg_reduce_kernel(
    const {ty}* input,
    unsigned int* out,
    int in_stride_axis,
    int in_stride_lane,
    int out_stride_lane,
    unsigned int in_offset,
    unsigned int out_offset,
    unsigned int axis_len,
    unsigned int lanes
) {{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {{
        return;
    }}
    int lane_base = (int)in_offset + (int)lane * in_stride_lane;
    {ty} best_val = input[lane_base];
    unsigned int best_idx = 0u;
    for (unsigned int a = 1u; a < axis_len; ++a) {{
        {ty} candidate = input[lane_base + (int)a * in_stride_axis];
        if ({better}) {{
            best_val = candidate;
            best_idx = a;
        }}
    }}
    int out_off = (int)out_offset + (int)lane * out_stride_lane;
    out[out_off] = best_idx;
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        better = better,
    )
}

/// ROCm device marker implementing [`ArgReduceOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmArgReduceOps;

impl<T> ArgReduceOps<RocmDevice, T> for RocmArgReduceOps
where
    T: DialectScalar<HipC> + Pod + PartialOrd,
{
    fn argmax_axis_into(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, RocmBuffer<u32>, 2>,
    ) -> Result<()> {
        arg_reduce_into::<T>(device, input, axis, output, false)
    }

    fn argmin_axis_into(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, RocmBuffer<u32>, 2>,
    ) -> Result<()> {
        arg_reduce_into::<T>(device, input, axis, output, true)
    }
}

fn arg_reduce_into<T>(
    device: &RocmDevice,
    input: StridedView<'_, RocmBuffer<T>, 2>,
    axis: usize,
    output: StridedView<'_, RocmBuffer<u32>, 2>,
    min: bool,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod,
{
    let in_layout = input.layout;
    let out_layout = output.layout;
    let (lanes, axis_len) = validate_arg_reduce_shape(axis, in_layout.shape(), out_layout.shape())?;

    in_layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len())
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "arg-reduce output layout must be non-overlapping".to_string(),
        });
    }
    if lanes == 0 {
        return Ok(());
    }
    let other = 1 - axis;
    let in_strides = in_layout.strides();
    let out_strides = out_layout.strides();

    let width = BlockWidth::DEFAULT;
    let key = PipelineKey::ArgReduce {
        scalar: TypeId::of::<T>(),
        width: width.get(),
        min,
    };
    let kernel = cached_kernel(device, key, "arg_reduce_kernel", || {
        arg_reduce_source::<T>(min)
    })?;

    let mut input_ptr: DevicePtr = input.buffer.raw();
    let mut out_ptr: DevicePtr = output.buffer.raw();
    let mut in_stride_axis = to_i32(in_strides[axis], "input stride")?;
    let mut in_stride_lane = to_i32(in_strides[other], "input stride")?;
    let mut out_stride_lane = to_i32(out_strides[other], "output stride")?;
    let mut in_offset = to_u32(in_layout.offset(), "input offset")?;
    let mut out_offset = to_u32(out_layout.offset(), "output offset")?;
    let mut axis_len_val = to_u32(axis_len, "reduced axis length")?;
    let mut lanes_val = to_u32(lanes, "dispatch size")?;

    // Argument list mirrors `arg_reduce_kernel(const T*, unsigned int*,
    // int x3, unsigned int x4)`.
    let mut args: [*mut core::ffi::c_void; 9] = [
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut out_ptr as *mut DevicePtr).cast(),
        (&mut in_stride_axis as *mut i32).cast(),
        (&mut in_stride_lane as *mut i32).cast(),
        (&mut out_stride_lane as *mut i32).cast(),
        (&mut in_offset as *mut u32).cast(),
        (&mut out_offset as *mut u32).cast(),
        (&mut axis_len_val as *mut u32).cast(),
        (&mut lanes_val as *mut u32).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size(lanes, width)?, width),
        &mut args,
    )
}
