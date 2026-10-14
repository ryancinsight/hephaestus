//! ROCm implementation of [`hephaestus_core::SplitOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK), mirroring the
//! WGSL seam's block-copy core: each output dispatches one pass over its own
//! cells, and a thread decodes its output coordinate, reads the same
//! coordinate with the `axis` lane shifted by the segment's validated base
//! offset, and writes its own cell. No bounds check is needed because
//! validation proves each input window equals its output shape exactly. The
//! metadata crosses as a by-value kernel argument, matching the strided
//! elementwise launches.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, MAX_STRIDED_RANK, Result,
    SplitOps, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_split_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `SplitMeta` kernel argument: rank-8 padded output shape, input
/// and output strides, `[input_offset, output_offset, len, axis]`, and
/// `[axis_base, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SplitMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<SplitMeta>() == 128);

const HIP_SPLIT_META: &str = r"struct SplitMeta {
    unsigned int out_shape[8];
    int in_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn split_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void split_kernel(
    SplitMeta cmeta,
    const {ty}* input,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.offsets[2]) {{
        return;
    }}
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int out_off = (int)cmeta.offsets[1];
    int axis = (int)cmeta.offsets[3];
    unsigned int base = cmeta.params[0];
    for (int d = 7; d >= 0; --d) {{
        unsigned int out_dim = cmeta.out_shape[d];
        unsigned int out_idx = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + (int)out_idx * cmeta.out_strides[d];
        unsigned int in_idx = out_idx;
        if (d == axis) {{
            in_idx = in_idx + base;
        }}
        in_off = in_off + (int)in_idx * cmeta.in_strides[d];
    }}
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_SPLIT_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`SplitOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmSplitOps;

impl<T> SplitOps<RocmDevice, T> for RocmSplitOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn split_into<const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        axis: usize,
        sizes: &[usize],
        outputs: &[StridedView<'_, RocmBuffer<T>, N>],
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "split dispatch supports rank <= 8");
        }

        let shapes: Vec<[usize; N]> = outputs.iter().map(|view| view.layout.shape()).collect();
        let bases = validate_split_shape(input.layout.shape(), axis, sizes, &shapes)?;

        let in_layout = input.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        for (index, output) in outputs.iter().enumerate() {
            output
                .layout
                .validate_storage_len(output.buffer.len())
                .map_err(map_layout_err)?;
            if !output.layout.is_injective().map_err(map_layout_err)? {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("split output {index} layout must be non-overlapping"),
                });
            }
        }
        if outputs
            .iter()
            .any(|output| output.buffer.aliases(input.buffer))
        {
            return Err(HephaestusError::DispatchFailed {
                message: "split output buffer must not alias the input buffer".to_string(),
            });
        }
        for (index, first) in outputs.iter().enumerate() {
            if outputs[index + 1..]
                .iter()
                .any(|second| first.buffer.aliases(second.buffer))
            {
                return Err(HephaestusError::DispatchFailed {
                    message: "split output buffers must not alias each other".to_string(),
                });
            }
        }
        let in_len = in_layout.checked_size().map_err(map_layout_err)?;
        if in_len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Split {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "split_kernel", || split_source::<T>())?;

        // The rank-8 lane holding `axis`: shapes pack leading ones, so the
        // operation rank occupies the trailing lanes.
        let padded_axis = to_u32(8 - N + axis, "split axis lane")?;
        let in_strides = pad_strides(in_layout.strides())?;
        let in_offset = to_u32(in_layout.offset(), "input offset")?;
        let mut input_ptr: DevicePtr = input.buffer.raw();

        for (output, base) in outputs.iter().zip(bases.iter()) {
            let out_layout = output.layout;
            let len = out_layout.checked_size().map_err(map_layout_err)?;
            if len == 0 {
                // An empty segment contributes no cells; skipping its launch
                // also sidesteps zero-length dispatches.
                continue;
            }

            let mut meta = SplitMeta {
                out_shape: pad_shape(out_layout.shape())?,
                in_strides,
                out_strides: pad_strides(out_layout.strides())?,
                offsets: [
                    in_offset,
                    to_u32(out_layout.offset(), "output offset")?,
                    to_u32(len, "dispatch size")?,
                    padded_axis,
                ],
                params: [to_u32(*base, "axis base")?, 0, 0, 0],
            };
            let mut out_ptr: DevicePtr = output.buffer.raw();

            // Argument list mirrors `split_kernel(SplitMeta, const T*, T*)`.
            let mut args: [*mut core::ffi::c_void; 3] = [
                core::ptr::addr_of_mut!(meta).cast::<SplitMeta>().cast(),
                core::ptr::addr_of_mut!(input_ptr)
                    .cast::<DevicePtr>()
                    .cast(),
                core::ptr::addr_of_mut!(out_ptr).cast::<DevicePtr>().cast(),
            ];
            launch_kernel(
                device,
                &kernel,
                LaunchConfig::linear(grid_size(len, width)?, width),
                &mut args,
            )?;
        }
        Ok(())
    }
}
