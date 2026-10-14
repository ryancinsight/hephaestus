//! ROCm implementation of [`hephaestus_core::AdaptivePoolingOps`].
//!
//! One thread per `(lane, out_idx)` pair (flat-dispatched over
//! `lanes * out_len`, matching `interpolation_seam`'s dispatch shape)
//! computes its own `[start, end)` window and reduces it directly — no
//! shared state between threads. `mode` is baked into the generated source
//! at pipeline-cache-key granularity, so each mode monomorphizes to its own
//! kernel with no per-invocation branch on it.
//!
//! Window bounds are exact integer arithmetic, mirroring the WGSL kernel:
//! floor division for the start, ceiling division for the end. Averages
//! divide by the window size in `T`'s native precision; maxima compare with
//! strict `>`, so the first maximal element wins ties.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    AdaptivePoolingMode, AdaptivePoolingOps, BlockWidth, DeviceBuffer, DialectScalar,
    HephaestusError, HipC, Result, StridedView, map_layout_err, validate_adaptive_pooling_shape,
};

use crate::RocmDevice;
use crate::application::linalg::{to_i32, to_u32};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

fn adaptive_pooling_source<T: DialectScalar<HipC>>(mode: AdaptivePoolingMode) -> String {
    let reduce_body = match mode {
        AdaptivePoolingMode::Average => format!(
            r"    {ty} acc = ({ty})0.0;
    for (unsigned int i = start; i < end; ++i) {{
        acc = acc + input[in_base + (int)i * in_stride_axis];
    }}
    {ty} value = acc / ({ty})(end - start);",
            ty = T::TYPE_TOKEN,
        ),
        AdaptivePoolingMode::Maximum => format!(
            r"    {ty} acc = input[in_base + (int)start * in_stride_axis];
    for (unsigned int i = start + 1u; i < end; ++i) {{
        {ty} candidate = input[in_base + (int)i * in_stride_axis];
        if (candidate > acc) {{
            acc = candidate;
        }}
    }}
    {ty} value = acc;",
            ty = T::TYPE_TOKEN,
        ),
    };
    format!(
        r#"{prelude}extern "C" __global__ void adaptive_pooling_kernel(
    const {ty}* input,
    {ty}* out,
    int in_stride_axis,
    int in_stride_lane,
    int out_stride_axis,
    int out_stride_lane,
    unsigned int in_offset,
    unsigned int out_offset,
    unsigned int in_len,
    unsigned int out_len,
    unsigned int lanes
) {{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = lanes * out_len;
    if (gid >= total) {{
        return;
    }}
    unsigned int lane = gid / out_len;
    unsigned int out_idx = gid % out_len;

    unsigned int start = out_idx * in_len / out_len;
    unsigned int end = ((out_idx + 1u) * in_len + out_len - 1u) / out_len;
    int in_base = (int)in_offset + (int)lane * in_stride_lane;

{reduce_body}

    int out_off = (int)out_offset + (int)lane * out_stride_lane
        + (int)out_idx * out_stride_axis;
    out[out_off] = value;
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        reduce_body = reduce_body,
    )
}

/// ROCm device marker implementing [`AdaptivePoolingOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmAdaptivePoolingOps;

impl<T> AdaptivePoolingOps<RocmDevice, T> for RocmAdaptivePoolingOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn adaptive_pool_axis_into(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        mode: AdaptivePoolingMode,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let out_layout = output.layout;
        let (lanes, in_len, out_len) =
            validate_adaptive_pooling_shape(axis, in_layout.shape(), out_layout.shape())?;

        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "adaptive-pooling output layout must be non-overlapping".to_string(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        let other = 1 - axis;
        let in_strides = in_layout.strides();
        let out_strides = out_layout.strides();

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::AdaptivePooling {
            scalar: TypeId::of::<T>(),
            width: width.get(),
            maximum: matches!(mode, AdaptivePoolingMode::Maximum),
        };
        let kernel = cached_kernel(device, key, "adaptive_pooling_kernel", || {
            adaptive_pooling_source::<T>(mode)
        })?;

        let total =
            lanes
                .checked_mul(out_len)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: "adaptive-pooling dispatch size overflows".to_string(),
                })?;

        let mut input_ptr: DevicePtr = input.buffer.raw();
        let mut out_ptr: DevicePtr = output.buffer.raw();
        let mut in_stride_axis = to_i32(in_strides[axis], "input stride")?;
        let mut in_stride_lane = to_i32(in_strides[other], "input stride")?;
        let mut out_stride_axis = to_i32(out_strides[axis], "output stride")?;
        let mut out_stride_lane = to_i32(out_strides[other], "output stride")?;
        let mut in_offset = to_u32(in_layout.offset(), "input offset")?;
        let mut out_offset = to_u32(out_layout.offset(), "output offset")?;
        let mut in_len_val = to_u32(in_len, "input axis length")?;
        let mut out_len_val = to_u32(out_len, "output axis length")?;
        let mut lanes_val = to_u32(lanes, "dispatch size")?;

        // Argument list mirrors `adaptive_pooling_kernel(const T*, T*,
        // int x4, unsigned int x5)`.
        let mut args: [*mut core::ffi::c_void; 11] = [
            (&mut input_ptr as *mut DevicePtr).cast(),
            (&mut out_ptr as *mut DevicePtr).cast(),
            (&mut in_stride_axis as *mut i32).cast(),
            (&mut in_stride_lane as *mut i32).cast(),
            (&mut out_stride_axis as *mut i32).cast(),
            (&mut out_stride_lane as *mut i32).cast(),
            (&mut in_offset as *mut u32).cast(),
            (&mut out_offset as *mut u32).cast(),
            (&mut in_len_val as *mut u32).cast(),
            (&mut out_len_val as *mut u32).cast(),
            (&mut lanes_val as *mut u32).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(total, width)?, width),
            &mut args,
        )
    }
}
