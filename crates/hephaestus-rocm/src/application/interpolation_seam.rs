//! ROCm implementation of [`hephaestus_core::InterpolationOps`].
//!
//! One thread per `(lane, out_idx)` pair (flat-dispatched over
//! `lanes * out_len`) computes its own mapped source coordinate and samples
//! or blends directly — no shared state between threads, so the kernel needs
//! no synchronization. `mode` is baked into the generated source at
//! pipeline-cache-key granularity, so each mode monomorphizes to its own
//! kernel with no per-invocation branch on it.
//!
//! The source-coordinate arithmetic mirrors the WGSL kernel exactly:
//! align-corners mapping in `T`'s native precision, truncation toward zero
//! for the lower index (a floor for the always-non-negative coordinate),
//! and round-half-up nearest selection (`frac < 0.5` keeps lower).

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, InterpolationMode,
    InterpolationOps, Result, StridedView, map_layout_err, validate_interpolation_shape,
};

use crate::RocmDevice;
use crate::application::linalg::{to_i32, to_u32};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

fn interpolation_source<T: DialectScalar<HipC>>(mode: InterpolationMode) -> String {
    let sample_body = match mode {
        InterpolationMode::Nearest => format!(
            r"    {ty} value;
    if (frac < ({ty})0.5) {{
        value = input[lower_off];
    }} else {{
        unsigned int upper = min(lower + 1u, in_len - 1u);
        int upper_off = in_base + (int)upper * in_stride_axis;
        value = input[upper_off];
    }}",
            ty = T::TYPE_TOKEN,
        ),
        InterpolationMode::Linear => format!(
            r"    unsigned int upper = min(lower + 1u, in_len - 1u);
    int upper_off = in_base + (int)upper * in_stride_axis;
    {ty} lower_val = input[lower_off];
    {ty} upper_val = input[upper_off];
    {ty} value = lower_val + (upper_val - lower_val) * frac;",
            ty = T::TYPE_TOKEN,
        ),
    };
    format!(
        r#"{prelude}extern "C" __global__ void interpolation_kernel(
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

    int in_base = (int)in_offset + (int)lane * in_stride_lane;
    unsigned int lower = 0u;
    {ty} frac = ({ty})0.0;
    if (in_len > 1u && out_len > 1u) {{
        {ty} src = ({ty})out_idx * ({ty})(in_len - 1u) / ({ty})(out_len - 1u);
        lower = min((unsigned int)src, in_len - 1u);
        frac = src - ({ty})lower;
    }}
    int lower_off = in_base + (int)lower * in_stride_axis;

{sample_body}

    int out_off = (int)out_offset + (int)lane * out_stride_lane
        + (int)out_idx * out_stride_axis;
    out[out_off] = value;
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        sample_body = sample_body,
    )
}

/// ROCm device marker implementing [`InterpolationOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmInterpolationOps;

impl<T> InterpolationOps<RocmDevice, T> for RocmInterpolationOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn interpolate_axis_into(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        mode: InterpolationMode,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let out_layout = output.layout;
        let (lanes, in_len, out_len) =
            validate_interpolation_shape(axis, in_layout.shape(), out_layout.shape())?;

        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "interpolation output layout must be non-overlapping".to_string(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        let other = 1 - axis;
        let in_strides = in_layout.strides();
        let out_strides = out_layout.strides();

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Interpolation {
            scalar: TypeId::of::<T>(),
            width: width.get(),
            nearest: matches!(mode, InterpolationMode::Nearest),
        };
        let kernel = cached_kernel(device, key, "interpolation_kernel", || {
            interpolation_source::<T>(mode)
        })?;

        let total =
            lanes
                .checked_mul(out_len)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: "interpolation dispatch size overflows".to_string(),
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

        // Argument list mirrors `interpolation_kernel(const T*, T*,
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
