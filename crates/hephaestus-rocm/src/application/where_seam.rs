//! ROCm implementation of [`hephaestus_core::WhereOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell decodes its coordinate, reads the condition and both branch
//! cells at that coordinate, and writes the selected branch. The condition
//! test is scalar inequality against zero: HIP C++ advertises IEEE
//! special values, so a `NaN` condition selects `on_true` (unordered
//! compares unequal) while a negative-zero condition selects `on_false`,
//! exactly as the host reference does. The metadata crosses as a by-value
//! kernel argument, matching the strided elementwise launches. There is no
//! error flag — the op takes no indices, so no data-dependent fault
//! exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HephaestusError, HipC, MAX_STRIDED_RANK, Result,
    StridedView, WhereOps, map_layout_err, pad_shape, pad_strides, to_u32, validate_where_shape,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// By-value `WhereNdMeta` kernel argument: rank-8 padded output shape,
/// condition, `on_true`, `on_false`, and output strides, `[cond_offset,
/// true_offset, false_offset, output_offset]`, and `[len, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WhereNdMeta {
    out_shape: [u32; 8],
    cond_strides: [i32; 8],
    true_strides: [i32; 8],
    false_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<WhereNdMeta>() == 192);

const HIP_WHERE_ND_META: &str = r"struct WhereNdMeta {
    unsigned int out_shape[8];
    int cond_strides[8];
    int true_strides[8];
    int false_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn where_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void where_kernel(
    WhereNdMeta cmeta,
    const {ty}* cond,
    const {ty}* on_true,
    const {ty}* on_false,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cmeta.params[0]) {{
        return;
    }}
    unsigned int rem = i;
    int cond_off = (int)cmeta.offsets[0];
    int true_off = (int)cmeta.offsets[1];
    int false_off = (int)cmeta.offsets[2];
    int out_off = (int)cmeta.offsets[3];
    for (int p = 7; p >= 0; --p) {{
        unsigned int out_dim = cmeta.out_shape[p];
        unsigned int c = rem % out_dim;
        rem = rem / out_dim;
        cond_off = cond_off + (int)c * cmeta.cond_strides[p];
        true_off = true_off + (int)c * cmeta.true_strides[p];
        false_off = false_off + (int)c * cmeta.false_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
    }}
    if (cond[(unsigned int)cond_off] != ({ty})0) {{
        out[(unsigned int)out_off] = on_true[(unsigned int)true_off];
    }} else {{
        out[(unsigned int)out_off] = on_false[(unsigned int)false_off];
    }}
}}
"#,
        prelude = T::PRELUDE,
        meta = HIP_WHERE_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// ROCm device marker implementing [`WhereOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmWhereOps;

impl<T> WhereOps<RocmDevice, T> for RocmWhereOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;

    fn where_into<const N: usize>(
        &self,
        device: &RocmDevice,
        cond: StridedView<'_, RocmBuffer<T>, N>,
        on_true: StridedView<'_, RocmBuffer<T>, N>,
        on_false: StridedView<'_, RocmBuffer<T>, N>,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "where dispatch supports rank <= 8");
        }

        validate_where_shape(
            cond.layout.shape(),
            on_true.layout.shape(),
            on_false.layout.shape(),
            output.layout.shape(),
        )?;

        let cond_layout = cond.layout;
        let true_layout = on_true.layout;
        let false_layout = on_false.layout;
        let out_layout = output.layout;
        cond_layout
            .validate_storage_len(cond.buffer.len())
            .map_err(map_layout_err)?;
        true_layout
            .validate_storage_len(on_true.buffer.len())
            .map_err(map_layout_err)?;
        false_layout
            .validate_storage_len(on_false.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "where output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(cond.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "where output buffer must not alias the cond buffer".to_string(),
            });
        }
        if output.buffer.aliases(on_true.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "where output buffer must not alias the on_true buffer".to_string(),
            });
        }
        if output.buffer.aliases(on_false.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "where output buffer must not alias the on_false buffer".to_string(),
            });
        }
        let len = out_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Where {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "where_kernel", || where_source::<T>())?;

        let mut meta = WhereNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            cond_strides: pad_strides(cond_layout.strides())?,
            true_strides: pad_strides(true_layout.strides())?,
            false_strides: pad_strides(false_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(cond_layout.offset(), "cond offset")?,
                to_u32(true_layout.offset(), "on_true offset")?,
                to_u32(false_layout.offset(), "on_false offset")?,
                to_u32(out_layout.offset(), "output offset")?,
            ],
            params: [to_u32(len, "dispatch size")?, 0, 0, 0],
        };
        let mut cond_ptr: DevicePtr = cond.buffer.raw();
        let mut true_ptr: DevicePtr = on_true.buffer.raw();
        let mut false_ptr: DevicePtr = on_false.buffer.raw();
        let mut out_ptr: DevicePtr = output.buffer.raw();

        // Argument list mirrors `where_kernel(WhereNdMeta, const T*,
        // const T*, const T*, T*)`.
        let mut args: [*mut core::ffi::c_void; 5] = [
            core::ptr::addr_of_mut!(meta).cast::<WhereNdMeta>().cast(),
            core::ptr::addr_of_mut!(cond_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(true_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(false_ptr)
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
