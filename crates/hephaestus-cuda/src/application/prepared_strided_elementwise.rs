//! Reusable strided elementwise plans bound to fixed operand pairs.
//!
//! Each plan validates its operands and resolves its kernel once at
//! preparation, then re-dispatches over the *bound allocations*: the launch
//! re-reads the device pointers at dispatch time, so a write to a bound
//! operand between dispatches is observed (the seam's rebind contract, the
//! elementwise counterpart of [`crate::application::prepared_axis_reduction`]).

use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, CudaC, DialectScalar, Result, TypedBinaryExpr, UnaryExpr,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::{
    StridedMeta, StridedOperand, binary_shader, binary_strided_meta, scalar_shader, unary_shader,
    unary_strided_meta,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Kernel and geometry shared by every prepared strided elementwise plan.
struct StridedPlan {
    meta: StridedMeta,
    kernel: Arc<SafeCachedKernel>,
    grid: u32,
    width: BlockWidth,
}

/// A unary strided elementwise dispatch bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound input between dispatches are observed. An empty
/// dispatch (zero-length output) prepares to a no-op.
pub struct PreparedStridedUnary<'op, T> {
    a: &'op CudaBuffer<T>,
    out: &'op CudaBuffer<T>,
    plan: Option<StridedPlan>,
}

impl<T> PreparedStridedUnary<'_, T>
where
    T: DialectScalar<CudaC> + Pod,
{
    /// Re-run the bound unary dispatch.
    ///
    /// # Errors
    ///
    /// Returns the native launch failure.
    pub fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        let Some(plan) = &self.plan else {
            return Ok(());
        };
        let mut meta_val = plan.meta;
        let mut a_ptr = self.a.raw();
        let mut out_ptr = self.out.raw();
        // Argument list mirrors `unary_strided_kernel(Meta, const T*, T*)`.
        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];
        launch_kernel(
            device,
            &plan.kernel,
            LaunchConfig::linear(plan.grid, plan.width),
            &mut args,
        )
    }
}

/// A binary (expression- or scalar-typed) strided elementwise dispatch bound
/// to one lhs/rhs/output triple.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound inputs between dispatches are observed. An empty
/// dispatch (zero-length output) prepares to a no-op.
pub struct PreparedStridedBinary<'op, T> {
    a: &'op CudaBuffer<T>,
    b: &'op CudaBuffer<T>,
    out: &'op CudaBuffer<T>,
    plan: Option<StridedPlan>,
}

impl<T> PreparedStridedBinary<'_, T>
where
    T: DialectScalar<CudaC> + Pod,
{
    /// Re-run the bound binary dispatch.
    ///
    /// # Errors
    ///
    /// Returns the native launch failure.
    pub fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        let Some(plan) = &self.plan else {
            return Ok(());
        };
        let mut meta_val = plan.meta;
        let mut a_ptr = self.a.raw();
        let mut b_ptr = self.b.raw();
        let mut out_ptr = self.out.raw();
        // Argument list mirrors `binary_strided_kernel(Meta, const T*, const T*, T*)`.
        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut b_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];
        launch_kernel(
            device,
            &plan.kernel,
            LaunchConfig::linear(plan.grid, plan.width),
            &mut args,
        )
    }
}

/// Prepare a unary strided elementwise dispatch bound to `a` and `out`.
///
/// # Errors
///
/// Returns a layout validation failure or the kernel compilation failure.
pub fn prepare_unary_elementwise_strided_into<'op, Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'op, T, N>,
    out: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedUnary<'op, T>>
where
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let Some((meta, len)) = unary_strided_meta(&a, &out)? else {
        return Ok(PreparedStridedUnary {
            a: a.buffer,
            out: out.buffer,
            plan: None,
        });
    };
    let key = PipelineKey::StridedUnary {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "unary_strided_kernel", || {
        unary_shader::<Op, T>()
    })?;
    Ok(PreparedStridedUnary {
        a: a.buffer,
        out: out.buffer,
        plan: Some(StridedPlan {
            meta,
            kernel,
            grid: grid_size(len, width)?,
            width,
        }),
    })
}

/// Prepare a binary strided elementwise dispatch bound to `a`, `b`, and
/// `out` under the dialect expression of `Op`.
///
/// # Errors
///
/// Returns a layout validation failure or the kernel compilation failure.
pub fn prepare_binary_elementwise_strided_into<'op, Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'op, T, N>,
    b: StridedOperand<'op, T, N>,
    out: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedBinary<'op, T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    prepare_binary_expression(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<CudaC>>::EXPR,
    )
}

/// Prepare a scalar-typed binary strided elementwise dispatch bound to `a`,
/// `b`, and `out` under the scalar-aware expression of `Op`.
///
/// # Errors
///
/// Returns a layout validation failure or the kernel compilation failure.
pub fn prepare_binary_elementwise_strided_typed_into<'op, Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'op, T, N>,
    b: StridedOperand<'op, T, N>,
    out: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedBinary<'op, T>>
where
    Op: TypedBinaryExpr<CudaC, T>,
    T: DialectScalar<CudaC> + Pod,
{
    prepare_binary_expression(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<CudaC, T>>::EXPR,
    )
}

fn prepare_binary_expression<'op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'op, T, N>,
    b: StridedOperand<'op, T, N>,
    out: StridedOperand<'op, T, N>,
    width: BlockWidth,
    operation: std::any::TypeId,
    expr: &'static str,
) -> Result<PreparedStridedBinary<'op, T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let Some((meta, len)) = binary_strided_meta(&a, &b, &out)? else {
        return Ok(PreparedStridedBinary {
            a: a.buffer,
            b: b.buffer,
            out: out.buffer,
            plan: None,
        });
    };
    let key = PipelineKey::StridedBinary {
        op: operation,
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<T>(expr)
    })?;
    Ok(PreparedStridedBinary {
        a: a.buffer,
        b: b.buffer,
        out: out.buffer,
        plan: Some(StridedPlan {
            meta,
            kernel,
            grid: grid_size(len, width)?,
            width,
        }),
    })
}

/// A broadcast-scalar strided dispatch bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound input between dispatches are observed. The scalar is
/// dispatch data captured at preparation. An empty dispatch prepares to a
/// no-op.
pub struct PreparedStridedScalar<'op, T> {
    a: &'op CudaBuffer<T>,
    out: &'op CudaBuffer<T>,
    scalar: T,
    plan: Option<StridedPlan>,
}

impl<T> PreparedStridedScalar<'_, T>
where
    T: DialectScalar<CudaC> + Pod,
{
    /// Re-run the bound broadcast-scalar dispatch.
    ///
    /// # Errors
    ///
    /// Returns the native launch failure.
    pub fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        let Some(plan) = &self.plan else {
            return Ok(());
        };
        let mut meta_val = plan.meta;
        let mut a_ptr = self.a.raw();
        let mut scalar_val = self.scalar;
        let mut out_ptr = self.out.raw();
        // Argument list mirrors `scalar_strided_kernel(Meta, const T*, T, T*)`.
        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut meta_val as *mut StridedMeta as *mut std::ffi::c_void,
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            (&mut scalar_val as *mut T).cast(),
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];
        launch_kernel(
            device,
            &plan.kernel,
            LaunchConfig::linear(plan.grid, plan.width),
            &mut args,
        )
    }
}

/// Validate and bind a broadcast-scalar strided dispatch.
///
/// # Errors
///
/// Returns a layout validation failure, an aliasing violation, or the kernel
/// compilation failure.
pub fn prepare_scalar_elementwise_strided_into<'op, Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'op, T, N>,
    scalar: T,
    out: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedScalar<'op, T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let meta = crate::application::strided::scalar_strided_meta(&a, &out)?;
    let Some((meta, len)) = meta else {
        return Ok(PreparedStridedScalar {
            a: a.buffer,
            out: out.buffer,
            scalar,
            plan: None,
        });
    };
    let key = PipelineKey::StridedScalar {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "scalar_strided_kernel", || {
        scalar_shader::<Op, T>()
    })?;
    Ok(PreparedStridedScalar {
        a: a.buffer,
        out: out.buffer,
        scalar,
        plan: Some(StridedPlan {
            meta,
            kernel,
            grid: grid_size(len, width)?,
            width,
        }),
    })
}
