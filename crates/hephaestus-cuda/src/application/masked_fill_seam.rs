//! CUDA implementation of [`hephaestus_core::MaskedFillOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell decodes its coordinate, reads the input and mask cells at
//! that coordinate, and writes the fill value or the input cell depending
//! on the mask test. The mask test is scalar inequality against zero:
//! CUDA C++ advertises IEEE special values, so a `NaN` mask fills
//! (unordered compares unequal) while a negative-zero mask keeps the
//! input, exactly as the host reference does. The metadata crosses as a
//! by-value kernel argument, matching the strided elementwise launches,
//! and the fill value crosses as a second by-value argument, the way the
//! pad seam's fill does. There is no error flag — the op takes no
//! indices, so no data-dependent fault exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CudaC, DeviceBuffer, DialectScalar, HephaestusError, MAX_STRIDED_RANK,
    MaskedFillOps, Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32,
    validate_masked_fill_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `MaskedFillNdMeta` kernel argument: rank-8 padded output
/// shape, input, mask, and output strides, and `[input_offset,
/// mask_offset, output_offset, len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskedFillNdMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    mask_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<MaskedFillNdMeta>() == 144);

const CUDA_MASKED_FILL_ND_META: &str = r"struct MaskedFillNdMeta {
    unsigned int out_shape[8];
    int in_strides[8];
    int mask_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
};
";

fn masked_fill_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void masked_fill_kernel(
    MaskedFillNdMeta cmeta,
    const {ty}* input,
    const {ty}* mask,
    {ty} value,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.offsets[3]) {{
        return;
    }}
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int mask_off = (int)cmeta.offsets[1];
    int out_off = (int)cmeta.offsets[2];
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + (int)c * cmeta.in_strides[p];
        mask_off = mask_off + (int)c * cmeta.mask_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
    }}
    if (mask[(unsigned int)mask_off] != ({ty})0) {{
        out[(unsigned int)out_off] = value;
    }} else {{
        out[(unsigned int)out_off] = input[(unsigned int)in_off];
    }}
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_MASKED_FILL_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`MaskedFillOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMaskedFillOps;

impl<T> MaskedFillOps<CudaDevice, T> for CudaMaskedFillOps
where
    T: DialectScalar<CudaC> + Pod,
{
    type Dialect = CudaC;

    fn masked_fill_into<const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        mask: StridedView<'_, CudaBuffer<T>, N>,
        value: T,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(
                N <= MAX_STRIDED_RANK,
                "masked_fill dispatch supports rank <= 8"
            );
        }

        validate_masked_fill_shape(
            input.layout.shape(),
            mask.layout.shape(),
            output.layout.shape(),
        )?;

        let in_layout = input.layout;
        let mask_layout = mask.layout;
        let out_layout = output.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        mask_layout
            .validate_storage_len(mask.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "masked_fill output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "masked_fill output buffer must not alias the input buffer".to_string(),
            });
        }
        if output.buffer.aliases(mask.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "masked_fill output buffer must not alias the mask buffer".to_string(),
            });
        }
        let len = out_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::MaskedFill {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "masked_fill_kernel", || {
            masked_fill_source::<T>()
        })?;

        let mut meta = MaskedFillNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            mask_strides: pad_strides(mask_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(mask_layout.offset(), "mask offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
            ],
        };
        let mut value_arg = value;
        let mut input_ptr = input.buffer.raw();
        let mut mask_ptr = mask.buffer.raw();
        let mut out_ptr = output.buffer.raw();

        // Argument list mirrors `masked_fill_kernel(MaskedFillNdMeta,
        // const T*, const T*, T, T*)`.
        let mut args: [*mut core::ffi::c_void; 5] = [
            core::ptr::addr_of_mut!(meta)
                .cast::<MaskedFillNdMeta>()
                .cast(),
            core::ptr::addr_of_mut!(input_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(mask_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(value_arg).cast::<T>().cast(),
            core::ptr::addr_of_mut!(out_ptr).cast::<u64>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(len, width)?, width),
            &mut args,
        )?;
        Ok(())
    }
}
