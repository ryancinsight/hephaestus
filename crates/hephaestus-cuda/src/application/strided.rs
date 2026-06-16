use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};
use leto::Layout;

use crate::application::cuda_type::CudaScalar;
use crate::application::elementwise::binary::BinaryCudaOp;
use crate::application::elementwise::unary::UnaryCudaOp;
use crate::application::pipeline::{cached_kernel, grid_size};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

/// Maximum rank the packed rank-4 metadata covers.
pub const MAX_STRIDED_RANK: usize = 4;

/// A device buffer paired with the leto layout describing its logical view.
#[derive(Clone, Copy)]
pub struct StridedOperand<'a, T, const N: usize> {
    /// The device buffer.
    pub buffer: &'a CudaBuffer<T>,
    /// The logical layout over that buffer.
    pub layout: &'a Layout<N>,
}

/// Metadata passed to strided kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StridedMeta {
    shape: [u32; 4],
    a_strides: [i32; 4],
    b_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
}

const CUDA_META: &str = r#"
struct Meta {
    unsigned int shape[4];
    int a_strides[4];
    int b_strides[4];
    int out_strides[4];
    unsigned int offsets[4];
};
"#;

const CUDA_DECODE: &str = r#"
    unsigned int rem = i;
    int a_off = lmeta.offsets[0];
    int b_off = lmeta.offsets[1];
    int o_off = lmeta.offsets[2];
    for (int d = 3; d >= 0; d--) {
        unsigned int dim = lmeta.shape[d];
        int idx = (int)(rem % dim);
        rem = rem / dim;
        a_off += idx * lmeta.a_strides[d];
        b_off += idx * lmeta.b_strides[d];
        o_off += idx * lmeta.out_strides[d];
    }
"#;

#[inline]
pub(crate) fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

#[inline]
fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

#[inline]
fn pad_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 4]> {
    let mut out = [1u32; 4];
    for (d, &dim) in shape.iter().enumerate() {
        out[4 - N + d] = to_u32(dim, "dimension")?;
    }
    Ok(out)
}

#[inline]
fn pad_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 4]> {
    let mut out = [0i32; 4];
    for (d, &stride) in strides.iter().enumerate() {
        out[4 - N + d] = i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("stride {stride} exceeds i32 range"),
        })?;
    }
    Ok(out)
}

fn validate_out<T, const N: usize>(out: &CudaBuffer<T>, out_layout: &Layout<N>) -> Result<usize> {
    if out_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    out_layout
        .validate_storage_len(out.len())
        .map_err(map_layout_err)?;
    out_layout.checked_size().map_err(map_layout_err)
}

fn binary_shader<Op: BinaryCudaOp, T: CudaScalar>() -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void binary_strided_kernel(
    Meta lmeta,
    const {ty}* lhs_ptr,
    const {ty}* rhs_ptr,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} a = lhs_ptr[a_off];
    {ty} b = rhs_ptr[b_off];
    out[o_off] = {expr};
}}
"#,
        meta = CUDA_META,
        ty = T::CUDA_TYPE,
        decode = CUDA_DECODE,
        expr = Op::CUDA_EXPR,
    )
}

fn unary_shader<Op: UnaryCudaOp, T: CudaScalar>() -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void unary_strided_kernel(
    Meta lmeta,
    const {ty}* a,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} x = a[a_off];
    out[o_off] = {expr};
}}
"#,
        meta = CUDA_META,
        ty = T::CUDA_TYPE,
        decode = CUDA_DECODE,
        expr = Op::CUDA_EXPR,
    )
}

/// Run `out[idx] = op(a[idx], b[idx])` over logical indices of `out_layout`.
pub fn binary_elementwise_strided_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    let b_layout = b
        .layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len())
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.buffer.len())
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        b_strides: pad_strides(b_layout.strides)?,
        out_strides: pad_strides(out_layout.strides)?,
        offsets: [
            to_u32(a_layout.offset, "input offset")?,
            to_u32(b_layout.offset, "input offset")?,
            to_u32(out_layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    let grid_size_val = grid_size(len, width)?;

    let key = format!(
        "strided_binary_{}_{}_{}",
        std::any::type_name::<Op>(),
        std::any::type_name::<T>(),
        width.get()
    );

    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<Op, T>()
    })?;

    #[cfg(feature = "cuda")]
    {
        let mut meta_val = meta;
        let mut a_ptr = a.buffer.raw();
        let mut b_ptr = b.buffer.raw();
        let mut out_ptr = out.buffer.raw();

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];

        // SAFETY: Buffers are valid, dimensions match.
        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                grid_size_val,
                1,
                1,
                width.get(),
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuLaunchKernel failed with code: {res}"),
                });
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, grid_size_val, meta);
    }

    Ok(())
}

/// Run `out = op(a, b)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// Inputs are broadcast to `output_shape` through the same layout contract as
/// [`binary_elementwise_strided_into`].
pub fn binary_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(len)?;
    binary_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        b,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}

/// Run `out[idx] = op(a[idx])` over logical indices of `out_layout`.
pub fn unary_elementwise_strided_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryCudaOp,
    T: CudaScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape)
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len())
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(());
    }

    let meta = StridedMeta {
        shape: pad_shape(out_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        b_strides: [0; 4],
        out_strides: pad_strides(out_layout.strides)?,
        offsets: [
            to_u32(a_layout.offset, "input offset")?,
            0,
            to_u32(out_layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    let grid_size_val = grid_size(len, width)?;

    let key = format!(
        "strided_unary_{}_{}_{}",
        std::any::type_name::<Op>(),
        std::any::type_name::<T>(),
        width.get()
    );

    let kernel = cached_kernel(device, key, "unary_strided_kernel", || {
        unary_shader::<Op, T>()
    })?;

    #[cfg(feature = "cuda")]
    {
        let mut meta_val = meta;
        let mut a_ptr = a.buffer.raw();
        let mut out_ptr = out.buffer.raw();

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];

        // SAFETY: Buffers are valid, dimensions match.
        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                grid_size_val,
                1,
                1,
                width.get(),
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuLaunchKernel failed with code: {res}"),
                });
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, grid_size_val, meta);
    }

    Ok(())
}

/// Run `out = op(a)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// The input is broadcast to `output_shape` through the same layout contract as
/// [`unary_elementwise_strided_into`].
pub fn unary_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: UnaryCudaOp,
    T: CudaScalar + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(len)?;
    unary_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}

/// Run `out[idx] = op(a[idx], scalar)` over logical indices of `out_layout`.
pub fn scalar_elementwise_strided_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    let scalar_buffer = device.upload(core::slice::from_ref(&scalar))?;
    let scalar_layout = Layout::new([1usize; N], [0isize; N], 0);
    binary_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        StridedOperand {
            buffer: &scalar_buffer,
            layout: &scalar_layout,
        },
        out,
        width,
    )
}

/// Run `out = op(a, scalar)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// The scalar path delegates through [`binary_elementwise_strided`] with a
/// one-element broadcast operand, preserving scalar/binary semantic identity.
pub fn scalar_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    let scalar_buffer = device.upload(core::slice::from_ref(&scalar))?;
    let scalar_layout = Layout::new([1usize; N], [0isize; N], 0);
    binary_elementwise_strided::<Op, T, N>(
        device,
        a,
        StridedOperand {
            buffer: &scalar_buffer,
            layout: &scalar_layout,
        },
        output_shape,
        width,
    )
}
