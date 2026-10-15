//! ROCm implementation of [`hephaestus_core::StackOps`].
//!
//! One packed-metadata kernel serves every input rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK) minus one,
//! mirroring the WGSL seam's block-copy core: each input dispatches one
//! pass over its own cells, and a thread decodes its input coordinate,
//! reads through the input strides, and writes the same coordinate with
//! the new `axis` lane set to the input's index. No bounds check is needed
//! because validation proves each output window equals its input shape
//! exactly. The metadata crosses as a by-value kernel argument, matching
//! the strided elementwise launches.
//!
//! The decode loop walks padded *input* lanes while the output has one more
//! lane than the input, so each input lane maps to its output lane
//! arithmetically (same lane below the new axis, one higher at or above
//! it); the new-axis lane is seeded once from the input index, and padding
//! lanes are skipped so no out-of-range stride lane is ever read.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, MAX_STRIDED_RANK, Result,
    StackOps, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_stack_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `StackMeta` kernel argument: rank-8 padded input shape, input
/// and output strides, `[input_offset, output_offset, len, axis]`, and
/// `[input_index, input_pad, _, _]`.
///
/// `axis` is the logical new-axis position (`0..=N`); `input_pad` is `8 - N`,
/// the first padded lane holding a real input dimension.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StackMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<StackMeta>() == 128);

const HIP_STACK_META: &str = r"struct StackMeta {
    unsigned int in_shape[8];
    int in_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn stack_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void stack_kernel(
    StackMeta cmeta,
    const {ty}* input,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.offsets[2]) {{
        return;
    }}
    int axis = (int)cmeta.offsets[3];
    int in_pad = (int)cmeta.params[1];
    int q_axis = in_pad + axis - 1;
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int out_off = (int)cmeta.offsets[1] + (int)cmeta.params[0] * cmeta.out_strides[q_axis];
    for (int p = 7; p >= 0; --p) {{
        unsigned int in_dim = cmeta.in_shape[p];
        unsigned int in_idx = rem % in_dim;
        rem = rem / in_dim;
        in_off = in_off + (int)in_idx * cmeta.in_strides[p];
        if (p >= in_pad) {{
            int d_in = p - in_pad;
            int d_out = d_in;
            if (d_in >= axis) {{
                d_out = d_out + 1;
            }}
            int q = in_pad - 1 + d_out;
            out_off = out_off + (int)in_idx * cmeta.out_strides[q];
        }}
    }}
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_STACK_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`StackOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmStackOps;

impl<T> StackOps<RocmDevice, T> for RocmStackOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn stack_into<const N: usize, const M: usize>(
        &self,
        device: &RocmDevice,
        inputs: &[StridedView<'_, RocmBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, RocmBuffer<T>, M>,
    ) -> Result<()>
    where
        leto::RankMarker<N>: leto::InsertAxis<N, LargerShape = [usize; M]>,
    {
        const {
            assert!(
                N < MAX_STRIDED_RANK,
                "stack dispatch supports input rank <= 7"
            );
        }

        let shapes: Vec<[usize; N]> = inputs.iter().map(|view| view.layout.shape()).collect();
        validate_stack_shape(&shapes, axis, output.layout.shape())?;

        let out_layout = output.layout;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "stack output layout must be non-overlapping".to_string(),
            });
        }
        if inputs
            .iter()
            .any(|input| output.buffer.aliases(input.buffer))
        {
            return Err(HephaestusError::DispatchFailed {
                message: "stack output buffer must not alias any input buffer".to_string(),
            });
        }
        let out_len = out_layout.checked_size().map_err(map_layout_err)?;
        if out_len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Stack {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "stack_kernel", || stack_source::<T>())?;

        let logical_axis = to_u32(axis, "stack axis")?;
        let input_pad = to_u32(8 - N, "stack input pad")?;
        let out_strides = pad_strides(out_layout.strides())?;
        let out_offset = to_u32(out_layout.offset(), "output offset")?;
        let mut out_ptr: DevicePtr = output.buffer.raw();

        for (index, input) in inputs.iter().enumerate() {
            let in_layout = input.layout;
            in_layout
                .validate_storage_len(input.buffer.len())
                .map_err(map_layout_err)?;
            let len = in_layout.checked_size().map_err(map_layout_err)?;
            if len == 0 {
                // An empty input contributes no cells; skipping its launch
                // also sidesteps zero-length dispatches.
                continue;
            }

            let mut meta = StackMeta {
                in_shape: pad_shape(in_layout.shape())?,
                in_strides: pad_strides(in_layout.strides())?,
                out_strides,
                offsets: [
                    to_u32(in_layout.offset(), "input offset")?,
                    out_offset,
                    to_u32(len, "dispatch size")?,
                    logical_axis,
                ],
                params: [to_u32(index, "input index")?, input_pad, 0, 0],
            };
            let mut input_ptr: DevicePtr = input.buffer.raw();

            // Argument list mirrors `stack_kernel(StackMeta, const T*, T*)`.
            let mut args: [*mut core::ffi::c_void; 3] = [
                core::ptr::addr_of_mut!(meta).cast::<StackMeta>().cast(),
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
