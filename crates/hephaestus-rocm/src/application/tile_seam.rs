//! ROCm implementation of [`hephaestus_core::TileOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell decodes its coordinate, takes each lane modulo the input
//! extent, and copies the addressed input cell. The rep counts never reach
//! the kernel: validation proves each output extent is its input extent
//! times its reps, so the modulo alone reproduces the tiling. The metadata
//! crosses as a by-value kernel argument, matching the strided elementwise
//! launches. There is no error flag — the op takes no indices, so no
//! data-dependent fault exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, MAX_STRIDED_RANK, Result,
    StridedView, TileOps, TileReps, map_layout_err, pad_shape, pad_strides, to_u32,
    validate_tile_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `TileNdMeta` kernel argument: rank-8 padded output and input
/// shapes, input and output strides, and `[input_offset, output_offset,
/// len, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileNdMeta {
    out_shape: [u32; 8],
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<TileNdMeta>() == 144);

const HIP_TILE_ND_META: &str = r"struct TileNdMeta {
    unsigned int out_shape[8];
    unsigned int in_shape[8];
    int in_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
};
";

fn tile_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void tile_kernel(
    TileNdMeta cmeta,
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
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        unsigned int src_c = c % cmeta.in_shape[p];
        in_off = in_off + (int)src_c * cmeta.in_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
    }}
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_TILE_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`TileOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmTileOps;

impl<T> TileOps<RocmDevice, T> for RocmTileOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn tile_into<const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        reps: TileReps<N>,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "tile dispatch supports rank <= 8");
        }

        validate_tile_shape(input.layout.shape(), reps, output.layout.shape())?;

        let in_layout = input.layout;
        let out_layout = output.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "tile output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "tile output buffer must not alias the input buffer".to_string(),
            });
        }
        let len = out_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Tile {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "tile_kernel", || tile_source::<T>())?;

        let mut meta = TileNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            in_shape: pad_shape(in_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
                0,
            ],
        };
        let mut input_ptr: DevicePtr = input.buffer.raw();
        let mut out_ptr: DevicePtr = output.buffer.raw();

        // Argument list mirrors `tile_kernel(TileNdMeta, const T*, T*)`.
        let mut args: [*mut core::ffi::c_void; 3] = [
            core::ptr::addr_of_mut!(meta).cast::<TileNdMeta>().cast(),
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
        Ok(())
    }
}
