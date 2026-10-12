//! CUDA implementation of [`hephaestus_core::CatOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK), mirroring the
//! WGSL seam's block-copy core: each input dispatches one pass over its own
//! cells, and a thread decodes its input coordinate, reads through the input
//! strides, and writes the same coordinate with the `axis` lane shifted by
//! the input's validated base offset. No bounds check is needed because
//! validation proves each output window equals its input shape exactly. The
//! metadata crosses as a by-value kernel argument, matching the strided
//! elementwise launches.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CatOps, CudaC, DeviceBuffer, DialectScalar, HephaestusError, MAX_STRIDED_RANK,
    Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_cat_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `CatMeta` kernel argument: rank-8 padded input shape, input and
/// output strides, `[input_offset, output_offset, len, axis]`, and
/// `[axis_base, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CatMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<CatMeta>() == 128);

const CUDA_CAT_META: &str = r"struct CatMeta {
    unsigned int in_shape[8];
    int in_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn cat_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void cat_kernel(
    CatMeta cmeta,
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
        unsigned int in_dim = cmeta.in_shape[d];
        unsigned int in_idx = rem % in_dim;
        rem = rem / in_dim;
        in_off = in_off + (int)in_idx * cmeta.in_strides[d];
        unsigned int out_idx = in_idx;
        if (d == axis) {{
            out_idx = out_idx + base;
        }}
        out_off = out_off + (int)out_idx * cmeta.out_strides[d];
    }}
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_CAT_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`CatOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCatOps;

impl<T> CatOps<CudaDevice, T> for CudaCatOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn cat_into<const N: usize>(
        &self,
        device: &CudaDevice,
        inputs: &[StridedView<'_, CudaBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "cat dispatch supports rank <= 8");
        }

        let shapes: Vec<[usize; N]> = inputs.iter().map(|view| view.layout.shape()).collect();
        let bases = validate_cat_shape(&shapes, axis, output.layout.shape())?;

        let out_layout = output.layout;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "cat output layout must be non-overlapping".to_string(),
            });
        }
        if inputs
            .iter()
            .any(|input| output.buffer.aliases(input.buffer))
        {
            return Err(HephaestusError::DispatchFailed {
                message: "cat output buffer must not alias any input buffer".to_string(),
            });
        }
        let out_len = out_layout.checked_size().map_err(map_layout_err)?;
        if out_len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Cat {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "cat_kernel", || cat_source::<T>())?;

        // The rank-8 lane holding `axis`: shapes pack leading ones, so the
        // operation rank occupies the trailing lanes.
        let padded_axis = to_u32(8 - N + axis, "cat axis lane")?;
        let out_strides = pad_strides(out_layout.strides())?;
        let out_offset = to_u32(out_layout.offset(), "output offset")?;
        let mut out_ptr = output.buffer.raw();

        for (input, base) in inputs.iter().zip(bases.iter()) {
            let in_layout = input.layout;
            in_layout
                .validate_storage_len(input.buffer.len())
                .map_err(map_layout_err)?;
            let len = in_layout.checked_size().map_err(map_layout_err)?;
            if len == 0 {
                // An empty segment contributes no cells; skipping its launch
                // also sidesteps zero-length dispatches.
                continue;
            }

            let mut meta = CatMeta {
                in_shape: pad_shape(in_layout.shape())?,
                in_strides: pad_strides(in_layout.strides())?,
                out_strides,
                offsets: [
                    to_u32(in_layout.offset(), "input offset")?,
                    out_offset,
                    to_u32(len, "dispatch size")?,
                    padded_axis,
                ],
                params: [to_u32(*base, "axis base")?, 0, 0, 0],
            };
            let mut input_ptr = input.buffer.raw();

            // Argument list mirrors `cat_kernel(CatMeta, const T*, T*)`.
            let mut args: [*mut core::ffi::c_void; 3] = [
                (&mut meta as *mut CatMeta).cast(),
                (&mut input_ptr as *mut u64).cast(),
                (&mut out_ptr as *mut u64).cast(),
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
