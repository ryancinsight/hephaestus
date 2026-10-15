//! CUDA implementation of [`hephaestus_core::IndexSelectOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell decodes its output coordinate, reads the `u32` index at
//! its own `dim` coordinate through the rank-1 index stride, substitutes
//! it into the `dim` lane, and copies the addressed input cell. The
//! metadata crosses as a by-value kernel argument, matching the strided
//! elementwise launches and the gather seam. The index values are
//! device-resident data, so an out-of-range value is reported through a
//! one-`u32` flag written with `atomicExch` and downloaded after the
//! launch completes, rather than clamped or wrapped — the same contract
//! the embedding seam keeps. The faulting lane's copy is skipped; its
//! output cell is left at whatever `output` held on entry, which is
//! irrelevant once the call returns an error.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, IndexSelectOps,
    MAX_STRIDED_RANK, Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32,
    validate_index_select_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `IndexSelectNdMeta` kernel argument: rank-8 padded output
/// shape, input, index, and output strides, `[input_offset, index_offset,
/// output_offset, len]`, and `[sel_lane, dim_extent, _, _]`.
///
/// `sel_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes
/// pack leading ones, so the operation rank occupies the trailing lanes.
/// The rank-1 index stride packs into the trailing lane (`7`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexSelectNdMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    idx_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexSelectNdMeta>() == 160);

const CUDA_INDEX_SELECT_ND_META: &str = r"struct IndexSelectNdMeta {
    unsigned int out_shape[8];
    int in_strides[8];
    int idx_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn index_select_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void index_select_kernel(
    IndexSelectNdMeta cmeta,
    const {ty}* input,
    const unsigned int* indices,
    {ty}* out,
    unsigned int* error_flag
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.offsets[3]) {{
        return;
    }}
    int sel_lane = (int)cmeta.params[0];
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int out_off = (int)cmeta.offsets[2];
    unsigned int sel_coord = 0u;
    unsigned int dim_coord = 0u;
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + (int)c * cmeta.in_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
        if (p == sel_lane) {{
            sel_coord = c;
            dim_coord = c;
        }}
    }}
    int idx_off = (int)cmeta.offsets[1] + (int)sel_coord * cmeta.idx_strides[7];
    unsigned int g = indices[(unsigned int)idx_off];
    if (g >= cmeta.params[1]) {{
        atomicExch(error_flag, 1u);
        return;
    }}
    int dim_stride = cmeta.in_strides[sel_lane];
    in_off = in_off - (int)dim_coord * dim_stride + (int)g * dim_stride;
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_INDEX_SELECT_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`IndexSelectOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaIndexSelectOps;

impl<T> IndexSelectOps<CudaDevice, T> for CudaIndexSelectOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn index_select_into<const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, CudaBuffer<u32>, 1>,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(
                N <= MAX_STRIDED_RANK,
                "index_select dispatch supports rank <= 8"
            );
        }

        validate_index_select_shape(
            input.layout.shape(),
            dim,
            index.layout.shape()[0],
            output.layout.shape(),
        )?;

        let in_layout = input.layout;
        let idx_layout = index.layout;
        let out_layout = output.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        idx_layout
            .validate_storage_len(index.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "index_select output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "index_select output buffer must not alias the input buffer".to_string(),
            });
        }
        let len = out_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            return Ok(());
        }

        let dim_extent = in_layout.shape()[dim];
        if dim_extent == 0 {
            // No input cell exists for any thread to read, so the kernel
            // could not even bind its source — but every index is
            // trivially out of range for an empty selection dimension.
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "index_select encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::IndexSelect {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "index_select_kernel", || {
            index_select_source::<T>()
        })?;

        let mut meta = IndexSelectNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            idx_strides: pad_strides(idx_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(idx_layout.offset(), "index offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
            ],
            params: [
                to_u32(8 - N + dim, "index_select dim lane")?,
                to_u32(dim_extent, "index_select dim extent")?,
                0,
                0,
            ],
        };
        let error_flag = device.alloc_zeroed::<u32>(1)?;
        let mut input_ptr = input.buffer.raw();
        let mut index_ptr = index.buffer.raw();
        let mut out_ptr = output.buffer.raw();
        let mut flag_ptr = error_flag.raw();

        // Argument list mirrors `index_select_kernel(IndexSelectNdMeta,
        // const T*, const unsigned int*, T*, unsigned int*)`.
        let mut args: [*mut core::ffi::c_void; 5] = [
            core::ptr::addr_of_mut!(meta)
                .cast::<IndexSelectNdMeta>()
                .cast(),
            core::ptr::addr_of_mut!(input_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(index_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(out_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(flag_ptr).cast::<u64>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(len, width)?, width),
            &mut args,
        )?;

        let mut flag_host = [0u32; 1];
        device.download(&error_flag, &mut flag_host)?;
        if flag_host[0] != 0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "index_select encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }
        Ok(())
    }
}
