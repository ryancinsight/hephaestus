//! Rank-≤4 strided elementwise dispatch over Leto layouts.
//!
//! One packed metadata contract serves binary, unary, and scalar kernels. The
//! device decodes each logical output index into per-operand offsets, so
//! transposed, sliced, and broadcast inputs execute directly from their device
//! storage without a host materialization copy.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC,
    Result, UnaryExpr,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

/// Maximum rank represented by the packed strided metadata.
pub const MAX_STRIDED_RANK: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StridedMeta {
    shape: [u32; 4],
    a_strides: [i32; 4],
    b_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<StridedMeta>() == 80);

const HIP_META: &str = r#"
struct Meta {
    unsigned int shape[4];
    int a_strides[4];
    int b_strides[4];
    int out_strides[4];
    unsigned int offsets[4];
};
"#;

const HIP_DECODE: &str = r#"
    unsigned int rem = i;
    int a_offset = (int)lmeta.offsets[0];
    int b_offset = (int)lmeta.offsets[1];
    int out_offset = (int)lmeta.offsets[2];
    for (int dimension = 3; dimension >= 0; dimension--) {
        unsigned int extent = lmeta.shape[dimension];
        int index = (int)(rem % extent);
        rem = rem / extent;
        a_offset += index * lmeta.a_strides[dimension];
        b_offset += index * lmeta.b_strides[dimension];
        out_offset += index * lmeta.out_strides[dimension];
    }
"#;

fn binary_shader<Op: BinaryExpr<HipC>, T: DialectScalar<HipC>>() -> String {
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
    {ty} lhs = lhs_ptr[a_offset];
    {ty} rhs = rhs_ptr[b_offset];
    out[out_offset] = {expr};
}}
"#,
        meta = HIP_META,
        decode = HIP_DECODE,
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
    )
}

fn unary_shader<Op: UnaryExpr<HipC>, T: DialectScalar<HipC>>() -> String {
    format!(
        r#"
{meta}
extern "C" __global__ void unary_strided_kernel(
    Meta lmeta,
    const {ty}* input,
    {ty}* out
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lmeta.offsets[3]) {{
        return;
    }}
{decode}
    {ty} x = input[a_offset];
    out[out_offset] = {expr};
}}
"#,
        meta = HIP_META,
        decode = HIP_DECODE,
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
    )
}

fn scalar_shader<Op: BinaryExpr<HipC>, T: DialectScalar<HipC>>() -> String {
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
    {ty} lhs = input[a_offset];
    {ty} rhs = scalar;
    out[out_offset] = {expr};
}}
"#,
        meta = HIP_META,
        decode = HIP_DECODE,
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
    )
}

fn pad_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 4]> {
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }
    let mut padded = [1_u32; 4];
    for (axis, &extent) in shape.iter().enumerate() {
        padded[4 - N + axis] =
            u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dimension {extent} exceeds u32 range"),
            })?;
    }
    Ok(padded)
}

fn pad_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 4]> {
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }
    let mut padded = [0_i32; 4];
    for (axis, &stride) in strides.iter().enumerate() {
        padded[4 - N + axis] =
            i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("stride {stride} exceeds i32 range"),
            })?;
    }
    Ok(padded)
}

fn validate_output<T, const N: usize>(output: StridedOperand<'_, T, N>) -> Result<usize> {
    if output.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    output
        .layout
        .validate_storage_len(output.buffer.len())
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("layout rejected: {error}"),
        })?;
    output
        .layout
        .checked_size()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("layout rejected: {error}"),
        })
}

fn dispatch_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("strided dispatch length {len} exceeds u32 range"),
    })
}

fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {error}"),
    }
}

fn launch_binary<Op, T>(
    device: &RocmDevice,
    lhs: &RocmBuffer<T>,
    rhs: &RocmBuffer<T>,
    output: &RocmBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let key = PipelineKey::StridedBinary {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<Op, T>()
    })?;
    let mut meta = meta;
    let mut lhs_ptr: DevicePtr = lhs.raw();
    let mut rhs_ptr: DevicePtr = rhs.raw();
    let mut output_ptr: DevicePtr = output.raw();
    let mut args: [*mut core::ffi::c_void; 4] = [
        (&mut meta as *mut StridedMeta).cast(),
        (&mut lhs_ptr as *mut DevicePtr).cast(),
        (&mut rhs_ptr as *mut DevicePtr).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size(len, width)?, width),
        &mut args,
    )
}

fn launch_unary<Op, T>(
    device: &RocmDevice,
    input: &RocmBuffer<T>,
    output: &RocmBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: UnaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let key = PipelineKey::StridedUnary {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "unary_strided_kernel", || {
        unary_shader::<Op, T>()
    })?;
    let mut meta = meta;
    let mut input_ptr: DevicePtr = input.raw();
    let mut output_ptr: DevicePtr = output.raw();
    let mut args: [*mut core::ffi::c_void; 3] = [
        (&mut meta as *mut StridedMeta).cast(),
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size(len, width)?, width),
        &mut args,
    )
}

fn launch_scalar<Op, T>(
    device: &RocmDevice,
    input: &RocmBuffer<T>,
    scalar: T,
    output: &RocmBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
) -> Result<()>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let key = PipelineKey::StridedScalar {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "scalar_strided_kernel", || {
        scalar_shader::<Op, T>()
    })?;
    let mut meta = meta;
    let mut input_ptr: DevicePtr = input.raw();
    let mut scalar = scalar;
    let mut output_ptr: DevicePtr = output.raw();
    let mut args: [*mut core::ffi::c_void; 4] = [
        (&mut meta as *mut StridedMeta).cast(),
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut scalar as *mut T).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size(len, width)?, width),
        &mut args,
    )
}

/// Run `out[idx] = op(lhs[idx], rhs[idx])` over a strided rank-`N` output.
pub fn binary_elementwise_strided_into<Op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, N>,
    rhs: StridedOperand<'_, T, N>,
    output: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }
    let lhs_layout = lhs
        .layout
        .broadcast(output.layout.shape)
        .map_err(map_layout_err)?;
    let rhs_layout = rhs
        .layout
        .broadcast(output.layout.shape)
        .map_err(map_layout_err)?;
    lhs_layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(map_layout_err)?;
    rhs_layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    if lhs.buffer.aliases(output.buffer) || rhs.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }
    let len = validate_output(output)?;
    if len == 0 {
        return Ok(());
    }
    let meta = StridedMeta {
        shape: pad_shape(output.layout.shape)?,
        a_strides: pad_strides(lhs_layout.strides)?,
        b_strides: pad_strides(rhs_layout.strides)?,
        out_strides: pad_strides(output.layout.strides)?,
        offsets: [
            u32::try_from(lhs_layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "input offset exceeds u32 range".to_string(),
            })?,
            u32::try_from(rhs_layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "input offset exceeds u32 range".to_string(),
            })?,
            u32::try_from(output.layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "output offset exceeds u32 range".to_string(),
            })?,
            dispatch_len(len)?,
        ],
    };
    launch_binary::<Op, T>(
        device,
        lhs.buffer,
        rhs.buffer,
        output.buffer,
        meta,
        width,
        len,
    )
}

/// Allocate a C-contiguous output and run a strided binary operation.
pub fn binary_elementwise_strided<Op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, N>,
    rhs: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    binary_elementwise_strided_into::<Op, T, N>(
        device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

/// Run `out[idx] = op(input[idx])` over a strided rank-`N` output.
pub fn unary_elementwise_strided_into<Op, T, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, N>,
    output: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }
    let input_layout = input
        .layout
        .broadcast(output.layout.shape)
        .map_err(map_layout_err)?;
    input_layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_output(output)?;
    if len == 0 {
        return Ok(());
    }
    let meta = StridedMeta {
        shape: pad_shape(output.layout.shape)?,
        a_strides: pad_strides(input_layout.strides)?,
        b_strides: [0; 4],
        out_strides: pad_strides(output.layout.strides)?,
        offsets: [
            u32::try_from(input_layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "input offset exceeds u32 range".to_string(),
            })?,
            0,
            u32::try_from(output.layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "output offset exceeds u32 range".to_string(),
            })?,
            dispatch_len(len)?,
        ],
    };
    launch_unary::<Op, T>(device, input.buffer, output.buffer, meta, width, len)
}

/// Allocate a C-contiguous output and run a strided unary operation.
pub fn unary_elementwise_strided<Op, T, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: UnaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    unary_elementwise_strided_into::<Op, T, N>(
        device,
        input,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

/// Run `out[idx] = op(input[idx], scalar)` over a strided rank-`N` output.
pub fn scalar_elementwise_strided_into<Op, T, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, N>,
    scalar: T,
    output: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "strided dispatch supports rank <= 4");
    }
    let input_layout = input
        .layout
        .broadcast(output.layout.shape)
        .map_err(map_layout_err)?;
    input_layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_output(output)?;
    if len == 0 {
        return Ok(());
    }
    let meta = StridedMeta {
        shape: pad_shape(output.layout.shape)?,
        a_strides: pad_strides(input_layout.strides)?,
        b_strides: [0; 4],
        out_strides: pad_strides(output.layout.strides)?,
        offsets: [
            u32::try_from(input_layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "input offset exceeds u32 range".to_string(),
            })?,
            0,
            u32::try_from(output.layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: "output offset exceeds u32 range".to_string(),
            })?,
            dispatch_len(len)?,
        ],
    };
    launch_scalar::<Op, T>(
        device,
        input.buffer,
        scalar,
        output.buffer,
        meta,
        width,
        len,
    )
}

/// Allocate a C-contiguous output and run a strided scalar operation.
pub fn scalar_elementwise_strided<Op, T, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, N>,
    scalar: T,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    scalar_elementwise_strided_into::<Op, T, N>(
        device,
        input,
        scalar,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{binary_shader, scalar_shader, unary_shader};

    #[test]
    fn sources_share_rank_four_decode_and_operation_contracts() {
        let binary = binary_shader::<hephaestus_core::AddOp, i32>();
        let unary = unary_shader::<hephaestus_core::IdentityOp, i32>();
        let scalar = scalar_shader::<hephaestus_core::MulOp, i32>();
        for source in [&binary, &unary, &scalar] {
            assert!(source.contains("unsigned int shape[4]"));
            assert!(source.contains("for (int dimension = 3; dimension >= 0"));
            assert!(source.contains("lmeta.out_strides[dimension]"));
        }
        assert!(binary.contains("out[out_offset] = lhs + rhs"));
        assert!(unary.contains("out[out_offset] = x"));
        assert!(scalar.contains("out[out_offset] = lhs * rhs"));
    }
}
