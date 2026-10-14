//! ROCm implementation of [`hephaestus_core::NonzeroOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread
//! per coordinates row rescans the input's logical cells in row-major
//! order, latches its row's flat index, and writes that flat's decoded
//! coordinates — the output-parallel scan that keeps row order
//! deterministic without atomics or scratch storage. Thread 0
//! additionally writes the total count, so even a zero-cell input
//! reports through the single launch (the dispatch is at least one
//! thread). The zero test is scalar inequality against zero: HIP C++
//! advertises IEEE special values, so a `NaN` input cell counts as
//! nonzero (unordered compares unequal) while a negative-zero cell
//! counts as zero, exactly as the host reference does. The metadata
//! crosses as a by-value kernel argument, matching the strided
//! elementwise launches. There is no error flag — the op takes no
//! indices, so no data-dependent fault exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, MAX_STRIDED_RANK, NonzeroOps,
    Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_nonzero_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `NonzeroNdMeta` kernel argument: rank-8 padded input shape,
/// input and coordinates strides, `[in_offset, coords_offset,
/// count_offset, _]`, and `[dispatch, numel, ndim, _]`.
///
/// The coordinates are always rank 2, so their padded strides hold the
/// row stride in lane 6 and the column stride in lane 7. The count is a
/// single cell, so only its offset crosses — its stride is irrelevant.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NonzeroNdMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    coords_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<NonzeroNdMeta>() == 128);

const HIP_NONZERO_ND_META: &str = r"struct NonzeroNdMeta {
    unsigned int in_shape[8];
    int in_strides[8];
    int coords_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn nonzero_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void nonzero_kernel(
    NonzeroNdMeta cmeta,
    const {ty}* input,
    unsigned int* coords,
    unsigned int* count
) {{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int numel = cmeta.params[1];
    if (r > 0u && r >= numel) {{
        return;
    }}
    unsigned int ndim = cmeta.params[2];
    unsigned int seen = 0u;
    unsigned int wanted = numel;
    for (unsigned int flat = 0u; flat < numel; ++flat) {{
        unsigned int rem = flat;
        int in_off = (int)cmeta.offsets[0];
        for (int p = 7; p >= 0; --p) {{
            unsigned int d = cmeta.in_shape[p];
            unsigned int c = rem % d;
            rem = rem / d;
            in_off = in_off + (int)c * cmeta.in_strides[p];
        }}
        if (input[(unsigned int)in_off] != ({ty})0) {{
            if (seen == r) {{
                wanted = flat;
            }}
            seen = seen + 1u;
        }}
    }}
    if (r == 0u) {{
        count[cmeta.offsets[2]] = seen;
    }}
    if (wanted < numel) {{
        unsigned int first_lane = 8u - ndim;
        unsigned int rem = wanted;
        for (int p = 7; p >= 0; --p) {{
            unsigned int d = cmeta.in_shape[p];
            unsigned int c = rem % d;
            rem = rem / d;
            if ((unsigned int)p >= first_lane) {{
                unsigned int col = (unsigned int)p - first_lane;
                unsigned int cell = (unsigned int)((int)cmeta.offsets[1] + (int)r * cmeta.coords_strides[6] + (int)col * cmeta.coords_strides[7]);
                coords[cell] = c;
            }}
        }}
    }}
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_NONZERO_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`NonzeroOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmNonzeroOps;

impl<T> NonzeroOps<RocmDevice, T> for RocmNonzeroOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;

    fn nonzero_into<const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        coords: StridedView<'_, RocmBuffer<u32>, 2>,
        count: StridedView<'_, RocmBuffer<u32>, 1>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "nonzero dispatch supports rank <= 8");
        }

        validate_nonzero_shape(
            input.layout.shape(),
            coords.layout.shape(),
            count.layout.shape(),
        )?;

        let in_layout = input.layout;
        let coords_layout = coords.layout;
        let count_layout = count.layout;
        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        coords_layout
            .validate_storage_len(coords.buffer.len())
            .map_err(map_layout_err)?;
        count_layout
            .validate_storage_len(count.buffer.len())
            .map_err(map_layout_err)?;
        if !coords_layout.is_injective().map_err(map_layout_err)?
            || !count_layout.is_injective().map_err(map_layout_err)?
        {
            return Err(HephaestusError::DispatchFailed {
                message: "nonzero output layouts must be non-overlapping".to_string(),
            });
        }
        if coords.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "nonzero coords buffer must not alias the input buffer".to_string(),
            });
        }
        if count.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "nonzero count buffer must not alias the input buffer".to_string(),
            });
        }
        if coords.buffer.aliases(count.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "nonzero coords buffer must not alias the count buffer".to_string(),
            });
        }

        // Validated above: the coordinates rows are the input numel, so
        // the dispatch derives from them rather than re-deriving the
        // product. At least one thread launches so thread 0 reports the
        // count of an empty input.
        let numel = coords_layout.shape()[0];
        let dispatch = numel.max(1);

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Nonzero {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "nonzero_kernel", || nonzero_source::<T>())?;

        let mut meta = NonzeroNdMeta {
            in_shape: pad_shape(in_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            coords_strides: pad_strides(coords_layout.strides())?,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(coords_layout.offset(), "coords offset")?,
                to_u32(count_layout.offset(), "count offset")?,
                0,
            ],
            params: [
                to_u32(dispatch, "dispatch size")?,
                to_u32(numel, "scan size")?,
                to_u32(N, "input rank")?,
                0,
            ],
        };
        let mut in_ptr: DevicePtr = input.buffer.raw();
        let mut coords_ptr: DevicePtr = coords.buffer.raw();
        let mut count_ptr: DevicePtr = count.buffer.raw();

        // Argument list mirrors `nonzero_kernel(NonzeroNdMeta, const
        // T*, unsigned int*, unsigned int*)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            core::ptr::addr_of_mut!(meta).cast::<NonzeroNdMeta>().cast(),
            core::ptr::addr_of_mut!(in_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(coords_ptr)
                .cast::<DevicePtr>()
                .cast(),
            core::ptr::addr_of_mut!(count_ptr)
                .cast::<DevicePtr>()
                .cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(dispatch, width)?, width),
            &mut args,
        )?;
        Ok(())
    }
}
