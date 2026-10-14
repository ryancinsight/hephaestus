//! CUDA implementation of [`hephaestus_core::ScatterOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell copies its input cell, then scans every source cell in
//! ascending row-major order and adds the ones whose index maps to it. The
//! scan is output-parallel because repeated indices would race an
//! input-parallel scatter; the ascending scan keeps every backend
//! bitwise-identical on floating-point sums, exactly as the embedding
//! scatter-add does. The metadata crosses as a by-value kernel argument,
//! matching the strided elementwise launches.
//!
//! Unlike the WGPU backend, CUDA has no four-buffer binding limit, so the
//! copy and the scan share one launch with five pointers; the WGPU
//! copy/scan split is a backend-local workaround the seam permits, not a
//! contract. A source cell maps to its thread's cell exactly when the
//! output offset recomputed from its coordinates equals the thread's own
//! offset, which is sound only because validation proves the output
//! layout injective. The index *values* are device-resident data, so an
//! out-of-range value is reported through a one-`u32` flag written with
//! `atomicExch` and downloaded after the launch completes, rather than
//! clamped or wrapped. A faulting update maps to no cell and is skipped;
//! the output is left partially accumulated, which is irrelevant once the
//! call returns an error.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    MAX_STRIDED_RANK, Result, ScatterOps, StridedView, map_layout_err, pad_shape, pad_strides,
    to_u32, validate_scatter_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `ScatterNdMeta` kernel argument: rank-8 padded output and index
/// shapes, input, index, source, and output strides, `[input_offset,
/// index_offset, src_offset, output_offset]`, and `[output_len, src_len,
/// dim_lane, dim_extent]`.
///
/// `dim_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterNdMeta {
    out_shape: [u32; 8],
    idx_shape: [u32; 8],
    in_strides: [i32; 8],
    idx_strides: [i32; 8],
    src_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<ScatterNdMeta>() == 224);

const CUDA_SCATTER_ND_META: &str = r"struct ScatterNdMeta {
    unsigned int out_shape[8];
    unsigned int idx_shape[8];
    int in_strides[8];
    int idx_strides[8];
    int src_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn scatter_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void scatter_kernel(
    ScatterNdMeta cmeta,
    const {ty}* input,
    const unsigned int* indices,
    const {ty}* src,
    {ty}* out,
    unsigned int* error_flag
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.params[0]) {{
        return;
    }}
    int dim_lane = (int)cmeta.params[2];
    unsigned int dim_extent = cmeta.params[3];
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int out_off = (int)cmeta.offsets[3];
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + (int)c * cmeta.in_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
    }}
    {ty} acc = input[(unsigned int)in_off];
    unsigned int src_len = cmeta.params[1];
    for (unsigned int j = 0u; j < src_len; ++j) {{
        unsigned int srem = j;
        int idx_off = (int)cmeta.offsets[1];
        int src_off = (int)cmeta.offsets[2];
        int mapped = (int)cmeta.offsets[3];
        for (int p = 7; p >= 0; --p) {{
            unsigned int idx_dim = cmeta.idx_shape[p];
            unsigned int c = srem % idx_dim;
            srem = srem / idx_dim;
            idx_off = idx_off + (int)c * cmeta.idx_strides[p];
            src_off = src_off + (int)c * cmeta.src_strides[p];
            if (p != dim_lane) {{
                mapped = mapped + (int)c * cmeta.out_strides[p];
            }}
        }}
        unsigned int g = indices[(unsigned int)idx_off];
        if (g >= dim_extent) {{
            atomicExch(error_flag, 1u);
            continue;
        }}
        mapped = mapped + (int)g * cmeta.out_strides[dim_lane];
        if (mapped == out_off) {{
            acc = acc + src[(unsigned int)src_off];
        }}
    }}
    out[(unsigned int)out_off] = acc;
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_SCATTER_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`ScatterOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaScatterOps;

impl<T> ScatterOps<CudaDevice, T> for CudaScatterOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn scatter_add_into<const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, CudaBuffer<u32>, N>,
        src: StridedView<'_, CudaBuffer<T>, N>,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "scatter dispatch supports rank <= 8");
        }

        validate_scatter_shape(
            input.layout.shape(),
            dim,
            index.layout.shape(),
            src.layout.shape(),
            output.layout.shape(),
        )?;

        let in_layout = input.layout;
        let idx_layout = index.layout;
        let src_layout = src.layout;
        let out_layout = output.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        idx_layout
            .validate_storage_len(index.buffer.len())
            .map_err(map_layout_err)?;
        src_layout
            .validate_storage_len(src.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "scatter output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "scatter output buffer must not alias the input buffer".to_string(),
            });
        }
        if output.buffer.aliases(src.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "scatter output buffer must not alias the source buffer".to_string(),
            });
        }
        let out_len = out_layout.checked_size().map_err(map_layout_err)?;
        let src_len = idx_layout.checked_size().map_err(map_layout_err)?;
        if out_len == 0 {
            // An empty output holds no cells to fault on only when the
            // index is empty too: a nonempty index against an empty gather
            // dimension is trivially out of range for every update.
            if src_len == 0 {
                return Ok(());
            }
            let dim_extent = in_layout.shape()[dim];
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "scatter encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }

        let dim_extent = in_layout.shape()[dim];
        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Scatter {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "scatter_kernel", || scatter_source::<T>())?;

        let mut meta = ScatterNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            idx_shape: pad_shape(idx_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            idx_strides: pad_strides(idx_layout.strides())?,
            src_strides: pad_strides(src_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(idx_layout.offset(), "index offset")?,
                to_u32(src_layout.offset(), "src offset")?,
                to_u32(out_layout.offset(), "output offset")?,
            ],
            params: [
                to_u32(out_len, "dispatch size")?,
                to_u32(src_len, "scan size")?,
                to_u32(8 - N + dim, "scatter dim lane")?,
                to_u32(dim_extent, "scatter dim extent")?,
            ],
        };
        let error_flag = device.alloc_zeroed::<u32>(1)?;
        let mut input_ptr = input.buffer.raw();
        let mut index_ptr = index.buffer.raw();
        let mut src_ptr = src.buffer.raw();
        let mut out_ptr = output.buffer.raw();
        let mut flag_ptr = error_flag.raw();

        // Argument list mirrors `scatter_kernel(ScatterNdMeta, const T*,
        // const unsigned int*, const T*, T*, unsigned int*)`. Empty
        // index/source buffers pass null pointers the kernel never
        // dereferences: with `src_len == 0` the scan loop runs zero times.
        let mut args: [*mut core::ffi::c_void; 6] = [
            core::ptr::addr_of_mut!(meta).cast::<ScatterNdMeta>().cast(),
            core::ptr::addr_of_mut!(input_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(index_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(src_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(out_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(flag_ptr).cast::<u64>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(out_len, width)?, width),
            &mut args,
        )?;

        let mut flag_host = [0u32; 1];
        device.download(&error_flag, &mut flag_host)?;
        if flag_host[0] != 0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "scatter encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }
        Ok(())
    }
}
