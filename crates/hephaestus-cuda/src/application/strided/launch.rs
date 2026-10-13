//! Strided kernel launch helpers.

use super::StridedMeta;
use super::shaders::{binary_shader, scalar_shader, unary_shader};
use crate::CudaDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use eunomia::Pod;
use hephaestus_core::{BinaryExpr, BlockWidth, CudaC, DialectScalar, Result, UnaryExpr};

pub(crate) struct BinaryStridedLaunch<'a, T> {
    pub(crate) a: &'a CudaBuffer<T>,
    pub(crate) b: &'a CudaBuffer<T>,
    pub(crate) out: &'a CudaBuffer<T>,
    pub(crate) meta: StridedMeta,
    pub(crate) width: BlockWidth,
    pub(crate) len: usize,
    pub(crate) operation: std::any::TypeId,
    pub(crate) expr: &'static str,
}

pub(crate) fn launch_binary_strided_expression<T>(
    device: &CudaDevice,
    request: BinaryStridedLaunch<'_, T>,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod,
{
    let BinaryStridedLaunch {
        a,
        b,
        out,
        meta,
        width,
        len,
        operation,
        expr,
    } = request;
    let grid_size_val = grid_size(len, width)?;

    let key = PipelineKey::StridedBinary {
        op: operation,
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<T>(expr)
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

pub(crate) fn launch_unary_strided<Op, T>(
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

pub(crate) fn launch_scalar_strided<Op, T>(
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
