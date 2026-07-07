use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    Result, UnaryExpr,
};
use leto::Layout;

use crate::application::pipeline::{
    cached_kernel, grid_size, launch_kernel, LaunchConfig, PipelineKey,
};
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

/// Borrowed dynamic-rank layout metadata for runtime-shaped consumers.
///
/// This is the layout-neutral counterpart to [`leto::Layout<N>`]. It lets
/// consumers pass runtime tensor layouts without materializing a fixed-rank Leto
/// layout. `shape` and `strides` must have equal length, and rank must be at
/// most [`MAX_STRIDED_RANK`].
#[derive(Clone, Copy)]
pub struct StridedLayout<'a> {
    /// Logical shape, right-aligned into the rank-4 CUDA metadata.
    pub shape: &'a [usize],
    /// Physical element strides, one per logical dimension.
    pub strides: &'a [usize],
    /// Base element offset into the storage buffer.
    pub offset: usize,
}

/// A device buffer paired with dynamic-rank layout metadata.
#[derive(Clone, Copy)]
pub struct StridedOperandDyn<'a, T> {
    /// The device buffer.
    pub buffer: &'a CudaBuffer<T>,
    /// The logical layout over that buffer.
    pub layout: StridedLayout<'a>,
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
fn pad_shape_dyn(shape: &[usize]) -> Result<[u32; 4]> {
    let mut out = [1u32; 4];
    for (d, &dim) in shape.iter().enumerate() {
        out[4 - shape.len() + d] = to_u32(dim, "dimension")?;
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

#[inline]
fn pad_usize_strides_dyn(strides: &[usize]) -> Result<[i32; 4]> {
    let mut out = [0i32; 4];
    for (d, &stride) in strides.iter().enumerate() {
        out[4 - strides.len() + d] =
            i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
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

fn validate_dyn_layout_shape(layout: StridedLayout<'_>) -> Result<()> {
    if layout.shape.len() != layout.strides.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "layout rank mismatch: shape rank {}, stride rank {}",
                layout.shape.len(),
                layout.strides.len()
            ),
        });
    }
    if layout.shape.len() > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "strided dispatch supports rank <= {MAX_STRIDED_RANK}, got {}",
                layout.shape.len()
            ),
        });
    }
    Ok(())
}

fn validate_dyn_layout<T>(
    what: &str,
    buffer: &CudaBuffer<T>,
    layout: StridedLayout<'_>,
) -> Result<usize> {
    validate_dyn_layout_shape(layout)?;
    let len = layout
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("{what} logical size overflows usize"),
        })?;
    if len == 0 {
        return Ok(0);
    }

    let max_offset = layout
        .shape
        .iter()
        .zip(layout.strides)
        .try_fold(layout.offset, |acc, (&dim, &stride)| {
            dim.checked_sub(1)
                .and_then(|span| span.checked_mul(stride))
                .and_then(|span| acc.checked_add(span))
        })
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("{what} layout address range overflows usize"),
        })?;
    if max_offset >= buffer.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: max_offset + 1,
            device_len: buffer.len(),
        });
    }
    Ok(len)
}

fn validate_dyn_out<T>(out: &CudaBuffer<T>, layout: StridedLayout<'_>) -> Result<usize> {
    if layout
        .shape
        .iter()
        .zip(layout.strides)
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
    {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    validate_dyn_layout("output", out, layout)
}

#[derive(Clone, Copy)]
struct BroadcastDyn {
    strides: [i32; 4],
    offset: u32,
}

fn broadcast_dyn_layout(layout: StridedLayout<'_>, out_shape: &[usize]) -> Result<BroadcastDyn> {
    validate_dyn_layout_shape(layout)?;
    if out_shape.len() > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "strided dispatch supports rank <= {MAX_STRIDED_RANK}, got {}",
                out_shape.len()
            ),
        });
    }
    if layout.shape.len() > out_shape.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "cannot broadcast rank {} layout to rank {} output",
                layout.shape.len(),
                out_shape.len()
            ),
        });
    }

    let mut strides = [0i32; 4];
    let rank_delta = out_shape.len() - layout.shape.len();
    for (out_axis, &out_dim) in out_shape.iter().enumerate() {
        let dst = 4 - out_shape.len() + out_axis;
        if out_axis < rank_delta {
            strides[dst] = 0;
            continue;
        }

        let in_axis = out_axis - rank_delta;
        let in_dim = layout.shape[in_axis];
        if in_dim == out_dim {
            strides[dst] = i32::try_from(layout.strides[in_axis]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("stride {} exceeds i32 range", layout.strides[in_axis]),
                }
            })?;
        } else if in_dim == 1 {
            strides[dst] = 0;
        } else {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "incompatible broadcast: input shape {:?} to output shape {:?}",
                    layout.shape, out_shape
                ),
            });
        }
    }

    Ok(BroadcastDyn {
        strides,
        offset: to_u32(layout.offset, "input offset")?,
    })
}

fn binary_shader<Op: BinaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
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
    {ty} lhs = lhs_ptr[a_off];
    {ty} rhs = rhs_ptr[b_off];
    out[o_off] = {expr};
}}
"#,
        meta = CUDA_META,
        ty = T::TYPE_TOKEN,
        decode = CUDA_DECODE,
        expr = Op::EXPR,
    )
}

fn unary_shader<Op: UnaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
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
        ty = T::TYPE_TOKEN,
        decode = CUDA_DECODE,
        expr = Op::EXPR,
    )
}

fn scalar_shader<Op: BinaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void scalar_strided_kernel(
    Meta lmeta,
    const {ty}* input,
    {ty} scalar,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} lhs = input[a_off];
    {ty} rhs = scalar;
    out[o_off] = {expr};
}}
"#,
        meta = CUDA_META,
        ty = T::TYPE_TOKEN,
        decode = CUDA_DECODE,
        expr = Op::EXPR,
    )
}

fn launch_binary_strided<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    b: &CudaBuffer<T>,
    out: &CudaBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let grid_size_val = grid_size(len, width)?;

    let key = PipelineKey::StridedBinary {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<Op, T>()
    })?;

    let mut meta_val = meta;
    let mut a_ptr = a.raw();
    let mut b_ptr = b.raw();
    let mut out_ptr = out.raw();

    // Argument list mirrors `binary_strided_kernel(Meta, const T*, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size_val, width),
        &mut args,
    )
}

fn launch_unary_strided<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    out: &CudaBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let grid_size_val = grid_size(len, width)?;

    let key = PipelineKey::StridedUnary {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "unary_strided_kernel", || {
        unary_shader::<Op, T>()
    })?;

    let mut meta_val = meta;
    let mut a_ptr = a.raw();
    let mut out_ptr = out.raw();

    // Argument list mirrors `unary_strided_kernel(Meta, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size_val, width),
        &mut args,
    )
}

fn launch_scalar_strided<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    scalar: T,
    out: &CudaBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let grid_size_val = grid_size(len, width)?;

    let key = PipelineKey::StridedScalar {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "scalar_strided_kernel", || {
        scalar_shader::<Op, T>()
    })?;

    let mut meta_val = meta;
    let mut a_ptr = a.raw();
    let mut scalar_val = scalar;
    let mut out_ptr = out.raw();

    // Argument list mirrors `scalar_strided_kernel(Meta, const T*, T, T*)`.
    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut scalar_val as *mut T as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size_val, width),
        &mut args,
    )
}

/// Run `out[idx] = op(a[idx], b[idx])` over dynamic-rank logical output indices.
pub fn binary_elementwise_strided_dyn_into<Op, T>(
    device: &CudaDevice,
    a: StridedOperandDyn<'_, T>,
    b: StridedOperandDyn<'_, T>,
    out: StridedOperandDyn<'_, T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    validate_dyn_layout("binary left", a.buffer, a.layout)?;
    validate_dyn_layout("binary right", b.buffer, b.layout)?;
    let len = validate_dyn_out(out.buffer, out.layout)?;
    if len == 0 {
        return Ok(());
    }

    let a_layout = broadcast_dyn_layout(a.layout, out.layout.shape)?;
    let b_layout = broadcast_dyn_layout(b.layout, out.layout.shape)?;
    let meta = StridedMeta {
        shape: pad_shape_dyn(out.layout.shape)?,
        a_strides: a_layout.strides,
        b_strides: b_layout.strides,
        out_strides: pad_usize_strides_dyn(out.layout.strides)?,
        offsets: [
            a_layout.offset,
            b_layout.offset,
            to_u32(out.layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    launch_binary_strided::<Op, T>(device, a.buffer, b.buffer, out.buffer, meta, width, len)
}

/// Run `out[idx] = op(a[idx])` over dynamic-rank logical output indices.
pub fn unary_elementwise_strided_dyn_into<Op, T>(
    device: &CudaDevice,
    a: StridedOperandDyn<'_, T>,
    out: StridedOperandDyn<'_, T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    validate_dyn_layout("unary input", a.buffer, a.layout)?;
    let len = validate_dyn_out(out.buffer, out.layout)?;
    if len == 0 {
        return Ok(());
    }

    let a_layout = broadcast_dyn_layout(a.layout, out.layout.shape)?;
    let meta = StridedMeta {
        shape: pad_shape_dyn(out.layout.shape)?,
        a_strides: a_layout.strides,
        b_strides: [0; 4],
        out_strides: pad_usize_strides_dyn(out.layout.strides)?,
        offsets: [
            a_layout.offset,
            0,
            to_u32(out.layout.offset, "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    launch_unary_strided::<Op, T>(device, a.buffer, out.buffer, meta, width, len)
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
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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

    launch_binary_strided::<Op, T>(device, a.buffer, b.buffer, out.buffer, meta, width, len)
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
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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

    launch_unary_strided::<Op, T>(device, a.buffer, out.buffer, meta, width, len)
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
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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

    launch_scalar_strided::<Op, T>(device, a.buffer, scalar, out.buffer, meta, width, len)
}

/// Run `out = op(a, scalar)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// The scalar path delegates through [`scalar_elementwise_strided_into`], passing
/// the scalar as a kernel argument instead of allocating a one-element device
/// buffer.
pub fn scalar_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_zeroed::<T>(len)?;
    scalar_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        scalar,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}
