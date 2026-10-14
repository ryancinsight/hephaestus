//! CUDA implementation of [`hephaestus_core::SortOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! axis lane decodes its lane's base offsets with the padded rank-8
//! odometer (the sort axis passes as the padded `dim_lane`, `8 - N +
//! dim`), then insertion-sorts the lane streaming from the input,
//! maintaining `(value, original position)` pairs directly in the output
//! slots. The bubble step swaps only on a strict comparison, which makes
//! the sort stable: equal cells never cross. CUDA C++ advertises IEEE
//! special values, so `NaN` sorts as greater than every other value —
//! last ascending, first descending, stable among NaNs — via an `isnan`
//! test that constant-folds away for integer scalars, exactly as the
//! host reference does. The metadata crosses as a by-value kernel
//! argument, matching the strided elementwise launches. There is no
//! error flag — the op takes no indices, so no data-dependent fault
//! exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CudaC, DeviceBuffer, DialectScalar, HephaestusError, MAX_STRIDED_RANK, Result,
    SortOps, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_sort_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `SortNdMeta` kernel argument: rank-8 padded input shape,
/// input, values, and indices strides, `[in_offset, val_offset,
/// idx_offset, _]`, and `[lanes, extent, dim_lane, descending]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortNdMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    val_strides: [i32; 8],
    idx_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<SortNdMeta>() == 160);

const CUDA_SORT_ND_META: &str = r"struct SortNdMeta {
    unsigned int in_shape[8];
    int in_strides[8];
    int val_strides[8];
    int idx_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn sort_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void sort_kernel(
    SortNdMeta cmeta,
    const {ty}* input,
    {ty}* out_vals,
    unsigned int* out_idx
) {{
    unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= cmeta.params[0]) {{
        return;
    }}
    unsigned int extent = cmeta.params[1];
    unsigned int dim_lane = cmeta.params[2];
    unsigned int descending = cmeta.params[3];
    unsigned int rem = lane;
    int in_base = (int)cmeta.offsets[0];
    int val_base = (int)cmeta.offsets[1];
    int idx_base = (int)cmeta.offsets[2];
    for (int p = 7; p >= 0; --p) {{
        if (p == (int)dim_lane) {{
            continue;
        }}
        unsigned int d = cmeta.in_shape[p];
        unsigned int c = rem % d;
        rem = rem / d;
        in_base = in_base + (int)c * cmeta.in_strides[p];
        val_base = val_base + (int)c * cmeta.val_strides[p];
        idx_base = idx_base + (int)c * cmeta.idx_strides[p];
    }}
    int in_stride = cmeta.in_strides[dim_lane];
    int val_stride = cmeta.val_strides[dim_lane];
    int idx_stride = cmeta.idx_strides[dim_lane];

    for (unsigned int a = 0; a < extent; ++a) {{
        {ty} val = input[(unsigned int)(in_base + (int)a * in_stride)];
        unsigned int pos = a;
        out_vals[(unsigned int)(val_base + (int)pos * val_stride)] = val;
        out_idx[(unsigned int)(idx_base + (int)pos * idx_stride)] = a;
        while (pos > 0u) {{
            unsigned int vp = (unsigned int)(val_base + (int)pos * val_stride);
            unsigned int vq = (unsigned int)(val_base + (int)(pos - 1u) * val_stride);
            {ty} pv = out_vals[vp];
            {ty} qv = out_vals[vq];
            bool q_nan = isnan((double)qv);
            bool p_nan = isnan((double)pv);
            bool swap;
            if (descending == 0u) {{
                if (!q_nan && !p_nan) {{ swap = (qv > pv); }}
                else {{ swap = (q_nan && !p_nan); }}
            }} else {{
                if (!q_nan && !p_nan) {{ swap = (qv < pv); }}
                else {{ swap = (p_nan && !q_nan); }}
            }}
            if (!swap) {{ break; }}
            unsigned int ip = (unsigned int)(idx_base + (int)pos * idx_stride);
            unsigned int iq = (unsigned int)(idx_base + (int)(pos - 1u) * idx_stride);
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
"#,
        prelude = T::PRELUDE,
        meta = CUDA_SORT_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`SortOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaSortOps;

impl<T> SortOps<CudaDevice, T> for CudaSortOps
where
    T: DialectScalar<CudaC> + Pod + PartialOrd,
{
    type Dialect = CudaC;

    fn sort_into<const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        dim: usize,
        descending: bool,
        values: StridedView<'_, CudaBuffer<T>, N>,
        indices: StridedView<'_, CudaBuffer<u32>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "sort dispatch supports rank <= 8");
        }

        validate_sort_shape(
            input.layout.shape(),
            dim,
            values.layout.shape(),
            indices.layout.shape(),
        )?;

        let in_layout = input.layout;
        let val_layout = values.layout;
        let idx_layout = indices.layout;
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
                message: "sort output layouts must be non-overlapping".to_string(),
            });
        }
        if values.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "sort values buffer must not alias the input buffer".to_string(),
            });
        }
        let len = val_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Sort {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "sort_kernel", || sort_source::<T>())?;

        let extent = in_layout.shape()[dim];
        let lanes = len / extent;
        let mut meta = SortNdMeta {
            in_shape: pad_shape(in_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            val_strides: pad_strides(val_layout.strides())?,
            idx_strides: pad_strides(idx_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(val_layout.offset(), "values offset")?,
                to_u32(idx_layout.offset(), "indices offset")?,
                0,
            ],
            params: [
                to_u32(lanes, "dispatch size")?,
                to_u32(extent, "sort extent")?,
                to_u32(8 - N + dim, "sort dim lane")?,
                u32::from(descending),
            ],
        };
        let mut in_ptr = input.buffer.raw();
        let mut vals_ptr = values.buffer.raw();
        let mut idx_ptr = indices.buffer.raw();

        // Argument list mirrors `sort_kernel(SortNdMeta, const T*, T*,
        // unsigned int*)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            (&mut meta as *mut SortNdMeta).cast(),
            (&mut in_ptr as *mut u64).cast(),
            (&mut vals_ptr as *mut u64).cast(),
            (&mut idx_ptr as *mut u64).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(lanes, width)?, width),
            &mut args,
        )?;
        Ok(())
    }
}
