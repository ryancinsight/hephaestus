//! CUDA implementation of [`hephaestus_core::DotOps`].
//!
//! One packed-metadata kernel serves every rank pair up to
//! [`MAX_STRIDED_RANK`](hephaestus_core::MAX_STRIDED_RANK): thread 0
//! folds the whole elementwise product in ascending flat order,
//! decoding each side's logical coordinates with the padded rank-8
//! odometer. One thread is the whole parallel strategy, and that is
//! deliberate: any tree or tiling would reassociate the sum and break
//! the bitwise float agreement the seam pins, so the kernel is exactly
//! the host's single-pass fold running on the device. The kernel is
//! pure arithmetic with no value comparison, so it carries no dialect
//! branch. The metadata crosses as a by-value kernel argument,
//! matching the strided elementwise launches. There is no error flag —
//! the op takes no indices, so no data-dependent fault exists.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CudaC, DeviceBuffer, DialectScalar, DotOps, HephaestusError, MAX_STRIDED_RANK,
    Result, StridedView, map_layout_err, pad_shape, pad_strides, to_u32, validate_dot_shape,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// By-value `DotNdMeta` kernel argument: rank-8 padded shapes and
/// strides of both inputs, `[a_offset, b_offset, out_offset, _]`, and
/// `[numel, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DotNdMeta {
    a_shape: [u32; 8],
    a_strides: [i32; 8],
    b_shape: [u32; 8],
    b_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<DotNdMeta>() == 160);

const CUDA_DOT_ND_META: &str = r"struct DotNdMeta {
    unsigned int a_shape[8];
    int a_strides[8];
    unsigned int b_shape[8];
    int b_strides[8];
    unsigned int offsets[4];
    unsigned int params[4];
};
";

fn dot_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}{meta}extern "C" __global__ void dot_kernel(
    DotNdMeta cmeta,
    const {ty}* a,
    const {ty}* b,
    {ty}* out
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0u) {{
        return;
    }}
    unsigned int numel = cmeta.params[0];
    {ty} acc = ({ty})0;
    for (unsigned int flat = 0u; flat < numel; ++flat) {{
        unsigned int rem = flat;
        int a_off = (int)cmeta.offsets[0];
        for (int p = 7; p >= 0; --p) {{
            unsigned int d = cmeta.a_shape[p];
            unsigned int c = rem % d;
            rem = rem / d;
            a_off = a_off + (int)c * cmeta.a_strides[p];
        }}
        rem = flat;
        int b_off = (int)cmeta.offsets[1];
        for (int p = 7; p >= 0; --p) {{
            unsigned int d = cmeta.b_shape[p];
            unsigned int c = rem % d;
            rem = rem / d;
            b_off = b_off + (int)c * cmeta.b_strides[p];
        }}
        acc = acc + a[(unsigned int)a_off] * b[(unsigned int)b_off];
    }}
    out[cmeta.offsets[2]] = acc;
}}
"#,
        prelude = T::PRELUDE,
        meta = CUDA_DOT_ND_META,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`DotOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDotOps;

impl<T> DotOps<CudaDevice, T> for CudaDotOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn dot_into<const N: usize, const M: usize>(
        &self,
        device: &CudaDevice,
        a: StridedView<'_, CudaBuffer<T>, N>,
        b: StridedView<'_, CudaBuffer<T>, M>,
        out: StridedView<'_, CudaBuffer<T>, 1>,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_STRIDED_RANK, "dot dispatch supports rank <= 8");
            assert!(M <= MAX_STRIDED_RANK, "dot dispatch supports rank <= 8");
        }

        validate_dot_shape(a.layout.shape(), b.layout.shape(), out.layout.shape())?;

        let a_layout = a.layout;
        let b_layout = b.layout;
        let out_layout = out.layout;
        a_layout
            .validate_storage_len(a.buffer.len())
            .map_err(map_layout_err)?;
        b_layout
            .validate_storage_len(b.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(out.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "dot output layout must be non-overlapping".to_string(),
            });
        }
        if out.buffer.aliases(a.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "dot output buffer must not alias the first input buffer".to_string(),
            });
        }
        if out.buffer.aliases(b.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "dot output buffer must not alias the second input buffer".to_string(),
            });
        }

        // Validated above: both numels agree, so the fold bound derives
        // from `a` rather than re-deriving either product. One thread
        // folds the whole product, so the dispatch is one block.
        let numel: usize = a_layout.shape().iter().product();

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Dot {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "dot_kernel", || dot_source::<T>())?;

        let mut meta = DotNdMeta {
            a_shape: pad_shape(a_layout.shape())?,
            a_strides: pad_strides(a_layout.strides())?,
            b_shape: pad_shape(b_layout.shape())?,
            b_strides: pad_strides(b_layout.strides())?,
            offsets: [
                to_u32(a_layout.offset(), "a offset")?,
                to_u32(b_layout.offset(), "b offset")?,
                to_u32(out_layout.offset(), "out offset")?,
                0,
            ],
            params: [to_u32(numel, "fold size")?, 0, 0, 0],
        };
        let mut a_ptr = a.buffer.raw();
        let mut b_ptr = b.buffer.raw();
        let mut out_ptr = out.buffer.raw();

        // Argument list mirrors `dot_kernel(DotNdMeta, const T*, const
        // T*, T*)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            core::ptr::addr_of_mut!(meta).cast::<DotNdMeta>().cast(),
            core::ptr::addr_of_mut!(a_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(b_ptr).cast::<u64>().cast(),
            core::ptr::addr_of_mut!(out_ptr).cast::<u64>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(1, width)?, width),
            &mut args,
        )?;
        Ok(())
    }
}
