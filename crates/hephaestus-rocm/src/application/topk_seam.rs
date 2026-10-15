//! ROCm implementation of [`hephaestus_core::TopKOps`].
//!
//! One thread per output lane maintains its `k`-slot sorted-descending
//! result directly in the output buffers — the same incremental insertion
//! the host reference uses (a tie never displaces, so the earlier index
//! always wins), with each output addressed through its own strides and
//! offset so either may be a non-dense view. `k` is an ordinary runtime
//! loop bound; the axis needs no baking because the host selects the
//! axis/lane strides before dispatch.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result, StridedView, TopKOps,
    map_layout_err, validate_topk_shape,
};

use crate::RocmDevice;
use crate::application::linalg::{to_i32, to_u32};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

fn topk_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}extern "C" __global__ void topk_kernel(
    const {ty}* input,
    {ty}* out_vals,
    unsigned int* out_idx,
    int in_stride_axis,
    int in_stride_lane,
    int val_stride_axis,
    int val_stride_lane,
    int idx_stride_axis,
    int idx_stride_lane,
    unsigned int in_offset,
    unsigned int val_offset,
    unsigned int idx_offset,
    unsigned int axis_len,
    unsigned int lanes,
    unsigned int k
) {{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) {{
        return;
    }}
    int lane_base = (int)in_offset + (int)lane * in_stride_lane;
    int val_base = (int)val_offset + (int)lane * val_stride_lane;
    int idx_base = (int)idx_offset + (int)lane * idx_stride_lane;

    for (unsigned int a = 0u; a < axis_len; ++a) {{
        {ty} val = input[lane_base + (int)a * in_stride_axis];
        if (a < k) {{
            unsigned int pos = a;
            out_vals[val_base + (int)pos * val_stride_axis] = val;
            out_idx[idx_base + (int)pos * idx_stride_axis] = a;
            while (pos > 0u) {{
                int vp = val_base + (int)pos * val_stride_axis;
                int vq = val_base + (int)(pos - 1u) * val_stride_axis;
                if (!(out_vals[vq] < out_vals[vp])) {{
                    break;
                }}
                int ip = idx_base + (int)pos * idx_stride_axis;
                int iq = idx_base + (int)(pos - 1u) * idx_stride_axis;
                {ty} tv = out_vals[vq];
                unsigned int ti = out_idx[iq];
                out_vals[vq] = out_vals[vp];
                out_idx[iq] = out_idx[ip];
                out_vals[vp] = tv;
                out_idx[ip] = ti;
                pos = pos - 1u;
            }}
        }} else if (val > out_vals[val_base + (int)(k - 1u) * val_stride_axis]) {{
            unsigned int pos = k - 1u;
            out_vals[val_base + (int)pos * val_stride_axis] = val;
            out_idx[idx_base + (int)pos * idx_stride_axis] = a;
            while (pos > 0u) {{
                int vp = val_base + (int)pos * val_stride_axis;
                int vq = val_base + (int)(pos - 1u) * val_stride_axis;
                if (!(out_vals[vq] < out_vals[vp])) {{
                    break;
                }}
                int ip = idx_base + (int)pos * idx_stride_axis;
                int iq = idx_base + (int)(pos - 1u) * idx_stride_axis;
                {ty} tv = out_vals[vq];
                unsigned int ti = out_idx[iq];
                out_vals[vq] = out_vals[vp];
                out_idx[iq] = out_idx[ip];
                out_vals[vp] = tv;
                out_idx[ip] = ti;
                pos = pos - 1u;
            }}
        }}
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`TopKOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmTopKOps;

impl<T> TopKOps<RocmDevice, T> for RocmTopKOps
where
    T: DialectScalar<HipC> + Pod + PartialOrd,
{
    fn topk_axis_into(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        k: usize,
        values: StridedView<'_, RocmBuffer<T>, 2>,
        indices: StridedView<'_, RocmBuffer<u32>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let val_layout = values.layout;
        let idx_layout = indices.layout;
        let (lanes, axis_len) = validate_topk_shape(
            axis,
            k,
            in_layout.shape(),
            val_layout.shape(),
            idx_layout.shape(),
        )?;

        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        val_layout
            .validate_storage_len(values.buffer.len())
            .map_err(map_layout_err)?;
        idx_layout
            .validate_storage_len(indices.buffer.len())
            .map_err(map_layout_err)?;
        if !val_layout.is_injective().map_err(map_layout_err)?
            || !idx_layout.is_injective().map_err(map_layout_err)?
        {
            return Err(HephaestusError::DispatchFailed {
                message: "top-k output layouts must be non-overlapping".to_string(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        let other = 1 - axis;
        let in_strides = in_layout.strides();
        let val_strides = val_layout.strides();
        let idx_strides = idx_layout.strides();

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::TopK {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "topk_kernel", topk_source::<T>)?;

        let mut input_ptr: DevicePtr = input.buffer.raw();
        let mut vals_ptr: DevicePtr = values.buffer.raw();
        let mut idx_ptr: DevicePtr = indices.buffer.raw();
        let mut in_stride_axis = to_i32(in_strides[axis], "input stride")?;
        let mut in_stride_lane = to_i32(in_strides[other], "input stride")?;
        let mut val_stride_axis = to_i32(val_strides[axis], "output stride")?;
        let mut val_stride_lane = to_i32(val_strides[other], "output stride")?;
        let mut idx_stride_axis = to_i32(idx_strides[axis], "output stride")?;
        let mut idx_stride_lane = to_i32(idx_strides[other], "output stride")?;
        let mut in_offset = to_u32(in_layout.offset(), "input offset")?;
        let mut val_offset = to_u32(val_layout.offset(), "output offset")?;
        let mut idx_offset = to_u32(idx_layout.offset(), "output offset")?;
        let mut axis_len_val = to_u32(axis_len, "reduced axis length")?;
        let mut lanes_val = to_u32(lanes, "dispatch size")?;
        let mut k_val = to_u32(k, "k")?;

        // Argument list mirrors `topk_kernel(const T*, T*, unsigned int*,
        // int x6, unsigned int x6)`.
        let mut args: [*mut core::ffi::c_void; 15] = [
            core::ptr::addr_of_mut!(input_ptr)
                .cast::<DevicePtr>()
                .cast(),
            core::ptr::addr_of_mut!(vals_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(idx_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(in_stride_axis).cast::<i32>().cast(),
            core::ptr::addr_of_mut!(in_stride_lane).cast::<i32>().cast(),
            core::ptr::addr_of_mut!(val_stride_axis)
                .cast::<i32>()
                .cast(),
            core::ptr::addr_of_mut!(val_stride_lane)
                .cast::<i32>()
                .cast(),
            core::ptr::addr_of_mut!(idx_stride_axis)
                .cast::<i32>()
                .cast(),
            core::ptr::addr_of_mut!(idx_stride_lane)
                .cast::<i32>()
                .cast(),
            core::ptr::addr_of_mut!(in_offset).cast::<u32>().cast(),
            core::ptr::addr_of_mut!(val_offset).cast::<u32>().cast(),
            core::ptr::addr_of_mut!(idx_offset).cast::<u32>().cast(),
            core::ptr::addr_of_mut!(axis_len_val).cast::<u32>().cast(),
            core::ptr::addr_of_mut!(lanes_val).cast::<u32>().cast(),
            core::ptr::addr_of_mut!(k_val).cast::<u32>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(lanes, width)?, width),
            &mut args,
        )
    }
}
