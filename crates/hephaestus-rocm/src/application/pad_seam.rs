//! ROCm implementation of [`hephaestus_core::PadOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK), mirroring the
//! WGSL seam's decode core: the output's logical flat index decodes into a
//! per-axis coordinate, which maps to an input coordinate by subtracting
//! that axis's `before` pad width. An input coordinate at or past the
//! input's extent (including the wrapped value of an out-of-range unsigned
//! subtraction) selects the fill value instead of a source read — one
//! bounds check covers both pad margins without a branch per margin per
//! axis. The metadata crosses as a by-value kernel argument, matching the
//! strided elementwise launches.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, MAX_STRIDED_RANK, PadOps,
    PadWidth, Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32,
    validate_pad_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `PadMeta` kernel argument: rank-8 padded output/input shapes,
/// input and output strides, per-axis pad-before widths, and
/// `[input_offset, output_offset, len, _pad]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PadMeta {
    out_shape: [u32; 8],
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    pad_before: [u32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<PadMeta>() == 176);

const HIP_PAD_META: &str = r"struct PadMeta {
    unsigned int out_shape[8];
    unsigned int in_shape[8];
    int in_strides[8];
    int out_strides[8];
    unsigned int pad_before[8];
    unsigned int offsets[4];
};
";

fn pad_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void pad_kernel(
    PadMeta pmeta,
    const {ty}* input,
    {ty} fill,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pmeta.offsets[2]) {{
        return;
    }}
    unsigned int rem = i;
    int in_off = (int)pmeta.offsets[0];
    int out_off = (int)pmeta.offsets[1];
    bool in_bounds = true;
    for (int d = 7; d >= 0; --d) {{
        unsigned int out_dim = pmeta.out_shape[d];
        unsigned int out_idx = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + (int)out_idx * pmeta.out_strides[d];
        unsigned int before = pmeta.pad_before[d];
        unsigned int in_idx = out_idx - before;
        unsigned int in_dim = pmeta.in_shape[d];
        if (in_idx >= in_dim) {{
            in_bounds = false;
        }} else {{
            in_off = in_off + (int)in_idx * pmeta.in_strides[d];
        }}
    }}
    if (in_bounds) {{
        out[(unsigned int)out_off] = input[(unsigned int)in_off];
    }} else {{
        out[(unsigned int)out_off] = fill;
    }}
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_PAD_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`PadOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmPadOps;

impl<T> PadOps<RocmDevice, T> for RocmPadOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn pad_into<const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        width: PadWidth<N>,
        fill: T,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "pad dispatch supports rank <= 8");
        }

        let in_layout = input.layout;
        let out_layout = output.layout;

        validate_pad_shape(in_layout.shape(), width, out_layout.shape())?;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "pad output layout must be non-overlapping".to_string(),
            });
        }
        let len = out_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            return Ok(());
        }

        let mut pad_before = [0u32; 8];
        for d in 0..N {
            pad_before[8 - N + d] = to_u32(width[d].0, "pad before-width")?;
        }

        let mut meta = PadMeta {
            out_shape: pad_shape(out_layout.shape())?,
            in_shape: pad_shape(in_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            pad_before,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
                0,
            ],
        };

        let width_ = BlockWidth::DEFAULT;
        let key = PipelineKey::Pad {
            scalar: TypeId::of::<T>(),
            width: width_.get(),
        };
        let kernel = cached_kernel(device, key, "pad_kernel", || pad_source::<T>())?;

        let mut input_ptr: DevicePtr = input.buffer.raw();
        let mut out_ptr: DevicePtr = output.buffer.raw();
        let mut fill_val = fill;

        // Argument list mirrors `pad_kernel(PadMeta, const T*, T, T*)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            core::ptr::addr_of_mut!(meta).cast::<PadMeta>().cast(),
            core::ptr::addr_of_mut!(input_ptr)
                .cast::<DevicePtr>()
                .cast(),
            core::ptr::addr_of_mut!(fill_val).cast::<T>().cast(),
            core::ptr::addr_of_mut!(out_ptr).cast::<DevicePtr>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(len, width_)?, width_),
            &mut args,
        )
    }
}
