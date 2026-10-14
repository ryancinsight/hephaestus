//! ROCm implementation of [`hephaestus_core::GatherOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell decodes its output coordinate, reads the `u32` index at its
//! own position, substitutes it into the `dim` lane, and copies the
//! addressed input cell. The metadata crosses as a by-value kernel
//! argument, matching the strided elementwise launches. The index values
//! are device-resident data, so an out-of-range value is reported through
//! a one-`u32` flag written with `atomicExch` and downloaded after the
//! launch completes, rather than clamped or wrapped — the same contract
//! the embedding seam keeps. The faulting lane's copy is skipped; its
//! output cell is left at whatever `output` held on entry, which is
//! irrelevant once the call returns an error.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, GatherOps, HephaestusError, HipC,
    MAX_STRIDED_RANK, Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32,
    validate_gather_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `GatherNdMeta` kernel argument: rank-8 padded output/index
/// shape, input, index, and output strides, `[input_offset, index_offset,
/// output_offset, len]`, and `[dim_lane, dim_extent, _, _]`.
///
/// `dim_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherNdMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    idx_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<GatherNdMeta>() == 160);

const HIP_GATHER_ND_META: &str = r"struct GatherNdMeta {
    unsigned int out_shape[8];
    int in_strides[8];
    int idx_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn gather_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void gather_kernel(
    GatherNdMeta cmeta,
    const {ty}* input,
    const unsigned int* indices,
    {ty}* out,
    unsigned int* error_flag
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.offsets[3]) {{
        return;
    }}
    int dim_lane = (int)cmeta.params[0];
    unsigned int rem = i;
    int in_off = (int)cmeta.offsets[0];
    int idx_off = (int)cmeta.offsets[1];
    int out_off = (int)cmeta.offsets[2];
    unsigned int dim_coord = 0u;
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + (int)c * cmeta.in_strides[p];
        idx_off = idx_off + (int)c * cmeta.idx_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
        if (p == dim_lane) {{
            dim_coord = c;
        }}
    }}
    unsigned int g = indices[(unsigned int)idx_off];
    if (g >= cmeta.params[1]) {{
        atomicExch(error_flag, 1u);
        return;
    }}
    int dim_stride = cmeta.in_strides[dim_lane];
    in_off = in_off - (int)dim_coord * dim_stride + (int)g * dim_stride;
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_GATHER_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`GatherOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmGatherOps;

impl<T> GatherOps<RocmDevice, T> for RocmGatherOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn gather_into<const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, RocmBuffer<u32>, N>,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "gather dispatch supports rank <= 8");
        }

        validate_gather_shape(
            input.layout.shape(),
            dim,
            index.layout.shape(),
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
                message: "gather output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "gather output buffer must not alias the input buffer".to_string(),
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
            // trivially out of range for an empty gather dimension.
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "gather encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Gather {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "gather_kernel", || gather_source::<T>())?;

        let mut meta = GatherNdMeta {
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
                to_u32(8 - N + dim, "gather dim lane")?,
                to_u32(dim_extent, "gather dim extent")?,
                0,
                0,
            ],
        };
        let error_flag = device.alloc_zeroed::<u32>(1)?;
        let mut input_ptr: DevicePtr = input.buffer.raw();
        let mut index_ptr: DevicePtr = index.buffer.raw();
        let mut out_ptr: DevicePtr = output.buffer.raw();
        let mut flag_ptr: DevicePtr = error_flag.raw();

        // Argument list mirrors `gather_kernel(GatherNdMeta, const T*,
        // const unsigned int*, T*, unsigned int*)`.
        let mut args: [*mut core::ffi::c_void; 5] = [
            (&mut meta as *mut GatherNdMeta).cast(),
            (&mut input_ptr as *mut DevicePtr).cast(),
            (&mut index_ptr as *mut DevicePtr).cast(),
            (&mut out_ptr as *mut DevicePtr).cast(),
            (&mut flag_ptr as *mut DevicePtr).cast(),
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
                    "gather encountered an index out of range for dim {dim} extent {dim_extent}",
                ),
            });
        }
        Ok(())
    }
}
