//! CUDA implementation of [`hephaestus_core::RollOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): one thread per
//! output cell decodes its coordinate, moves each rolled lane back by its
//! effective shift modulo the shared extent, and copies the addressed
//! input cell. Rolls along distinct dims commute and rolls along one dim
//! add, so the host folds every `(shift, dim)` pair into one effective
//! per-axis displacement before launch — duplicate dims compose by
//! summation, and arbitrarily large or negative shifts reduce once, in
//! exact integer arithmetic, rather than per thread. The metadata crosses
//! as a by-value kernel argument, matching the strided elementwise
//! launches. There is no error flag — the op takes no indices, so no
//! data-dependent fault exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CudaC, DeviceBuffer, DialectScalar, HephaestusError, MAX_STRIDED_RANK, Result,
    RollOps, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_roll_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `RollNdMeta` kernel argument: rank-8 padded output shape,
/// input and output strides, per-axis effective back-shifts, and
/// `[input_offset, output_offset, len, _]`.
///
/// Each back-shift is already reduced modulo its dim extent, so the
/// kernel subtracts without underflow; unrolled lanes hold zero. Shapes
/// pack leading ones, so the operation rank occupies the trailing lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RollNdMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    back_shifts: [u32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<RollNdMeta>() == 144);

const CUDA_ROLL_ND_META: &str = r"struct RollNdMeta {
    unsigned int out_shape[8];
    int in_strides[8];
    int out_strides[8];
    unsigned int back_shifts[8];
    unsigned int offsets[4];
};
";

fn roll_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void roll_kernel(
    RollNdMeta cmeta,
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
        unsigned int src_c = (c + out_dim - cmeta.back_shifts[p]) % out_dim;
        in_off = in_off + (int)src_c * cmeta.in_strides[p];
        out_off = out_off + (int)c * cmeta.out_strides[p];
    }}
    out[(unsigned int)out_off] = input[(unsigned int)in_off];
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_ROLL_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`RollOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRollOps;

impl<T> RollOps<CudaDevice, T> for CudaRollOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn roll_into<const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        shifts: &[isize],
        dims: &[usize],
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "roll dispatch supports rank <= 8");
        }

        validate_roll_shape(input.layout.shape(), shifts, dims, output.layout.shape())?;

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
                message: "roll output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "roll output buffer must not alias the input buffer".to_string(),
            });
        }
        let len = out_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            // Every extent is nonzero below this point, so no back-shift
            // normalizes against a zero modulus.
            return Ok(());
        }

        // Fold every pair into one effective per-axis shift in i128, then
        // reduce modulo the extent: exact, once per axis, never per
        // thread. The reduced shift is below the extent, hence below
        // `usize::MAX`; the fallible conversion keeps the function total
        // regardless.
        let mut totals = [0i128; N];
        for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
            totals[dim] += shift as i128;
        }
        let mut back_shifts = [0u32; 8];
        for (d, &total) in totals.iter().enumerate() {
            let extent = in_layout.shape()[d];
            let reduced = total.rem_euclid(extent as i128);
            let reduced_usize =
                usize::try_from(reduced).map_err(|_| HephaestusError::InvalidConfiguration {
                    message: format!("roll back-shift {reduced} exceeds the usize range"),
                })?;
            back_shifts[8 - N + d] = to_u32(reduced_usize, "roll back-shift")?;
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Roll {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "roll_kernel", || roll_source::<T>())?;

        let mut meta = RollNdMeta {
            out_shape: pad_shape(out_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            back_shifts,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
                0,
            ],
        };
        let mut input_ptr = input.buffer.raw();
        let mut out_ptr = output.buffer.raw();

        // Argument list mirrors `roll_kernel(RollNdMeta, const T*, T*)`.
        let mut args: [*mut core::ffi::c_void; 3] = [
            (&mut meta as *mut RollNdMeta).cast(),
            (&mut input_ptr as *mut u64).cast(),
            (&mut out_ptr as *mut u64).cast(),
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
