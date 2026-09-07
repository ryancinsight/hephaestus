//! Rank-≤8 strided elementwise dispatch over Leto layouts.
//!
//! One packed metadata contract serves binary, unary, and scalar kernels. The
//! device decodes each logical output index into per-operand offsets, so
//! transposed, sliced, and broadcast inputs execute directly from their device
//! storage without a host materialization copy.

use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, DialectScalar, HipC, Result, TypedBinaryExpr, UnaryExpr,
};
use leto::Layout;

pub(crate) mod kernel;
pub(crate) mod metadata;

use kernel::{binary_shader, scalar_shader, unary_shader};
pub use metadata::MAX_STRIDED_RANK;
use metadata::{
    StridedMeta, binary_strided_meta, map_layout_err, scalar_strided_meta, unary_strided_meta,
};

use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

struct BinaryKernelLaunch<'a, T> {
    lhs: &'a RocmBuffer<T>,
    rhs: &'a RocmBuffer<T>,
    output: &'a RocmBuffer<T>,
    meta: StridedMeta,
    width: BlockWidth,
    len: usize,
    operation: core::any::TypeId,
    expr: &'static str,
}

fn launch_binary_expression<T>(
    device: &RocmDevice,
    request: BinaryKernelLaunch<'_, T>,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod,
{
    let BinaryKernelLaunch {
        lhs,
        rhs,
        output,
        meta,
        width,
        len,
        operation,
        expr,
    } = request;
    let key = PipelineKey::StridedBinary {
        op: operation,
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<T>(expr)
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
fn binary_elementwise_strided_into_expression<T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, N>,
    rhs: StridedOperand<'_, T, N>,
    output: StridedOperand<'_, T, N>,
    width: BlockWidth,
    operation: core::any::TypeId,
    expr: &'static str,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod,
{
    let Some((meta, len)) = binary_strided_meta(&lhs, &rhs, &output)? else {
        return Ok(());
    };
    launch_binary_expression(
        device,
        BinaryKernelLaunch {
            lhs: lhs.buffer,
            rhs: rhs.buffer,
            output: output.buffer,
            meta,
            width,
            len,
            operation,
            expr,
        },
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
    binary_elementwise_strided_into_expression::<T, N>(
        device,
        lhs,
        rhs,
        output,
        width,
        core::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<HipC>>::EXPR,
    )
}

/// Run a scalar-aware binary operation over a strided rank-`N` output.
pub fn binary_elementwise_strided_typed_into<Op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, N>,
    rhs: StridedOperand<'_, T, N>,
    output: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: TypedBinaryExpr<HipC, T>,
    T: DialectScalar<HipC> + Pod,
{
    binary_elementwise_strided_into_expression::<T, N>(
        device,
        lhs,
        rhs,
        output,
        width,
        core::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<HipC, T>>::EXPR,
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
    let output =
        device.alloc_uninitialized::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
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

/// Allocate a C-contiguous output and run a scalar-aware strided binary
/// operation.
pub fn binary_elementwise_strided_typed<Op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'_, T, N>,
    rhs: StridedOperand<'_, T, N>,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: TypedBinaryExpr<HipC, T>,
    T: DialectScalar<HipC> + Pod,
{
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output =
        device.alloc_uninitialized::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
    binary_elementwise_strided_typed_into::<Op, T, N>(
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
    let Some((meta, len)) = unary_strided_meta(&input, &output)? else {
        return Ok(());
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
    let output =
        device.alloc_uninitialized::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
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
    let Some((meta, len)) = scalar_strided_meta(&input, &output)? else {
        return Ok(());
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
    let output =
        device.alloc_uninitialized::<T>(output_layout.checked_size().map_err(map_layout_err)?)?;
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
