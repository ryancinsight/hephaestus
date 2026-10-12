//! CUDA implementation of [`hephaestus_core::IndexPutOps`].
//!
//! One packed-metadata kernel per `accumulate` mode serves every rank up
//! to [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread
//! per output cell copies its base input cell, then scans every selection
//! in ascending order. Accumulate mode adds the values slices whose index
//! maps to the cell; replace mode keeps the last matching slice, so
//! duplicate indices resolve deterministically to last-wins — a valid
//! choice because `torch.index_put` leaves duplicate replace order
//! undefined and no clause pins it. Both scans are output-parallel because
//! repeated indices would race an input-parallel scatter; the ascending
//! scan keeps every backend bitwise-identical on floating-point sums,
//! exactly as the scatter seam does. The metadata crosses as a by-value
//! kernel argument, matching the strided elementwise launches.
//!
//! Unlike the WGPU backend, CUDA has no four-buffer binding limit, so the
//! copy and the scan share one launch with five pointers; the WGPU
//! copy/update split is a backend-local workaround the seam permits, not
//! a contract. The index *values* are device-resident data, so an
//! out-of-range value is reported through a one-`u32` flag written with
//! `atomicExch` and downloaded after the launch completes, rather than
//! clamped or wrapped. A faulting update maps to no cell and is skipped;
//! the output is left partially written, which is irrelevant once the call
//! returns an error.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, IndexPutOps,
    MAX_STRIDED_RANK, Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32,
    validate_index_put_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `IndexPutNdMeta` kernel argument: rank-8 padded output shape,
/// input, index, values, and output strides, `[input_offset, index_offset,
/// values_offset, output_offset]`, and `[output_len, sel_len, sel_lane,
/// dim_extent]`.
///
/// `sel_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes. The
/// rank-1 index stride packs into the trailing lane (`7`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexPutNdMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    idx_strides: [i32; 8],
    vals_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexPutNdMeta>() == 192);

const CUDA_INDEX_PUT_ND_META: &str = r"struct IndexPutNdMeta {
    unsigned int out_shape[8];
    int in_strides[8];
    int idx_strides[8];
    int vals_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn index_put_source<T: DialectScalar<CudaC>>(accumulate: bool) -> String {
    let update = if accumulate {
        "acc = acc + values[(unsigned int)vals_off];"
    } else {
        "acc = values[(unsigned int)vals_off];"
    };
    let kernel = if accumulate {
        "index_put_accumulate_kernel"
    } else {
        "index_put_replace_kernel"
    };
    format!(
        r#"{prelude}{meta}extern "C" __global__ void {kernel}(
    IndexPutNdMeta cmeta,
    const {ty}* input,
    const unsigned int* indices,
    const {ty}* values,
    {ty}* out,
    unsigned int* error_flag
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.params[0]) {{
        return;
    }}
    int sel_lane = (int)cmeta.params[2];
    unsigned int dim_extent = cmeta.params[3];
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int out_off = (int)cmeta.offsets[3];
    int vals_base = (int)cmeta.offsets[2];
    unsigned int wanted = 0u;
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + (int)c * cmeta.in_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
        if (p != sel_lane) {{
            vals_base = vals_base + (int)c * cmeta.vals_strides[p];
        }} else {{
            wanted = c;
        }}
    }}
    {ty} acc = input[(unsigned int)in_off];
    unsigned int sel_len = cmeta.params[1];
    int idx_base = (int)cmeta.offsets[1];
    int idx_stride = cmeta.idx_strides[7];
    int vals_dim_stride = cmeta.vals_strides[sel_lane];
    for (unsigned int s = 0u; s < sel_len; ++s) {{
        unsigned int g = indices[(unsigned int)(idx_base + (int)s * idx_stride)];
        if (g >= dim_extent) {{
            atomicExch(error_flag, 1u);
            continue;
        }}
        if (g == wanted) {{
            int vals_off = vals_base + (int)s * vals_dim_stride;
            {update}
        }}
    }}
    out[(unsigned int)out_off] = acc;
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_INDEX_PUT_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`IndexPutOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaIndexPutOps;

impl<T> IndexPutOps<CudaDevice, T> for CudaIndexPutOps
where
    T: DialectScalar<CudaC> + Pod,
{
    #[allow(clippy::too_many_arguments)]
    fn index_put_into<const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, CudaBuffer<u32>, 1>,
        values: StridedView<'_, CudaBuffer<T>, N>,
        accumulate: bool,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(
                N <= MAX_STRIDED_RANK,
                "index_put dispatch supports rank <= 8"
            );
        }

        validate_index_put_shape(
            input.layout.shape(),
            dim,
            index.layout.shape()[0],
            values.layout.shape(),
            output.layout.shape(),
        )?;

        let in_layout = input.layout;
        let idx_layout = index.layout;
        let vals_layout = values.layout;
        let out_layout = output.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        idx_layout
            .validate_storage_len(index.buffer.len())
            .map_err(map_layout_err)?;
        vals_layout
            .validate_storage_len(values.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "index_put output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "index_put output buffer must not alias the input buffer".to_string(),
            });
        }
        if output.buffer.aliases(values.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "index_put output buffer must not alias the values buffer".to_string(),
            });
        }
        let out_len = out_layout.checked_size().map_err(map_layout_err)?;
        let sel_len = idx_layout.checked_size().map_err(map_layout_err)?;
        if out_len == 0 {
            // An empty output holds no cells to fault on only when the
            // update is empty too — but `sel_len > 0` against an empty
            // non-`dim` extent is still vacuous (the values hold no cells),
            // matching the host's early-out. Only an empty `dim` extent
            // with selections proves every index trivially out of range.
            if sel_len == 0 {
                return Ok(());
            }
            let vacuous = vals_layout
                .shape()
                .iter()
                .enumerate()
                .any(|(d, &extent)| d != dim && extent == 0);
            if vacuous {
                return Ok(());
            }
            let dim_extent = in_layout.shape()[dim];
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "index_put encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }

        let dim_extent = in_layout.shape()[dim];
        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::IndexPut {
            scalar: TypeId::of::<T>(),
            width: width.get(),
            accumulate,
        };
        let kernel_name = if accumulate {
            "index_put_accumulate_kernel"
        } else {
            "index_put_replace_kernel"
        };
        let kernel = cached_kernel(device, key, kernel_name, || {
            index_put_source::<T>(accumulate)
        })?;

        let mut meta = IndexPutNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            idx_strides: pad_strides(idx_layout.strides())?,
            vals_strides: pad_strides(vals_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(idx_layout.offset(), "index offset")?,
                to_u32(vals_layout.offset(), "values offset")?,
                to_u32(out_layout.offset(), "output offset")?,
            ],
            params: [
                to_u32(out_len, "dispatch size")?,
                to_u32(sel_len, "selection count")?,
                to_u32(8 - N + dim, "index_put dim lane")?,
                to_u32(dim_extent, "index_put dim extent")?,
            ],
        };
        let error_flag = device.alloc_zeroed::<u32>(1)?;
        let mut input_ptr = input.buffer.raw();
        let mut index_ptr = index.buffer.raw();
        let mut values_ptr = values.buffer.raw();
        let mut out_ptr = output.buffer.raw();
        let mut flag_ptr = error_flag.raw();

        // Argument list mirrors `index_put_*_kernel(IndexPutNdMeta,
        // const T*, const unsigned int*, const T*, T*, unsigned int*)`.
        // Empty index/values buffers pass null pointers the kernel never
        // dereferences: with `sel_len == 0` the scan loop runs zero times.
        let mut args: [*mut core::ffi::c_void; 6] = [
            (&mut meta as *mut IndexPutNdMeta).cast(),
            (&mut input_ptr as *mut u64).cast(),
            (&mut index_ptr as *mut u64).cast(),
            (&mut values_ptr as *mut u64).cast(),
            (&mut out_ptr as *mut u64).cast(),
            (&mut flag_ptr as *mut u64).cast(),
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
                    "index_put encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }
        Ok(())
    }
}
