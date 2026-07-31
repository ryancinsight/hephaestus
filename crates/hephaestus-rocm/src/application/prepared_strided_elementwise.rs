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
    BinaryExpr, BlockWidth, DialectScalar, HipC, Result, TypedBinaryExpr, UnaryExpr,
};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::{
    StridedMeta, binary_shader, binary_strided_meta, unary_shader, unary_strided_meta,
};
use crate::infrastructure::DevicePtr;

/// Kernel and geometry shared by every prepared strided elementwise plan.
struct StridedPlan {
    meta: StridedMeta,
    kernel: Arc<RocmKernel>,
    grid: u32,
    width: BlockWidth,
}

/// A unary strided elementwise dispatch bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound input between dispatches are observed. An empty
/// dispatch (zero-length output) prepares to a no-op.
pub struct PreparedStridedUnary<'op, T> {
    input: &'op RocmBuffer<T>,
    output: &'op RocmBuffer<T>,
    plan: Option<StridedPlan>,
}

impl<T> PreparedStridedUnary<'_, T>
where
    T: DialectScalar<HipC> + Pod,
{
    /// Re-run the bound unary dispatch.
    ///
    /// # Errors
    ///
    /// Returns the native launch failure.
    pub fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        let Some(plan) = &self.plan else {
            return Ok(());
        };
        let mut meta = plan.meta;
        let mut input_ptr: DevicePtr = self.input.raw();
        let mut output_ptr: DevicePtr = self.output.raw();
        // Argument list mirrors `unary_strided_kernel(Meta, const T*, T*)`.
        let mut args: [*mut core::ffi::c_void; 3] = [
            (&mut meta as *mut StridedMeta).cast(),
            (&mut input_ptr as *mut DevicePtr).cast(),
            (&mut output_ptr as *mut DevicePtr).cast(),
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
    lhs: &'op RocmBuffer<T>,
    rhs: &'op RocmBuffer<T>,
    output: &'op RocmBuffer<T>,
    plan: Option<StridedPlan>,
}

impl<T> PreparedStridedBinary<'_, T>
where
    T: DialectScalar<HipC> + Pod,
{
    /// Re-run the bound binary dispatch.
    ///
    /// # Errors
    ///
    /// Returns the native launch failure.
    pub fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        let Some(plan) = &self.plan else {
            return Ok(());
        };
        let mut meta = plan.meta;
        let mut lhs_ptr: DevicePtr = self.lhs.raw();
        let mut rhs_ptr: DevicePtr = self.rhs.raw();
        let mut output_ptr: DevicePtr = self.output.raw();
        // Argument list mirrors `binary_strided_kernel(Meta, const T*, const T*, T*)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            (&mut meta as *mut StridedMeta).cast(),
            (&mut lhs_ptr as *mut DevicePtr).cast(),
            (&mut rhs_ptr as *mut DevicePtr).cast(),
            (&mut output_ptr as *mut DevicePtr).cast(),
        ];
        launch_kernel(
            device,
            &plan.kernel,
            LaunchConfig::linear(plan.grid, plan.width),
            &mut args,
        )
    }
}

/// Prepare a unary strided elementwise dispatch bound to `input` and
/// `output`.
///
/// # Errors
///
/// Returns a layout validation failure or the kernel compilation failure.
pub fn prepare_unary_elementwise_strided_into<'op, Op, T, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'op, T, N>,
    output: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedUnary<'op, T>>
where
    Op: UnaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let Some((meta, len)) = unary_strided_meta(&input, &output)? else {
        return Ok(PreparedStridedUnary {
            input: input.buffer,
            output: output.buffer,
            plan: None,
        });
    };
    let key = PipelineKey::StridedUnary {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "unary_strided_kernel", || {
        unary_shader::<Op, T>()
    })?;
    Ok(PreparedStridedUnary {
        input: input.buffer,
        output: output.buffer,
        plan: Some(StridedPlan {
            meta,
            kernel,
            grid: grid_size(len, width)?,
            width,
        }),
    })
}

/// Prepare a binary strided elementwise dispatch bound to `lhs`, `rhs`, and
/// `output` under the dialect expression of `Op`.
///
/// # Errors
///
/// Returns a layout validation failure or the kernel compilation failure.
pub fn prepare_binary_elementwise_strided_into<'op, Op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'op, T, N>,
    rhs: StridedOperand<'op, T, N>,
    output: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedBinary<'op, T>>
where
    Op: BinaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    prepare_binary_expression(
        device,
        lhs,
        rhs,
        output,
        width,
        core::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<HipC>>::EXPR,
    )
}

/// Prepare a scalar-typed binary strided elementwise dispatch bound to
/// `lhs`, `rhs`, and `output` under the scalar-aware expression of `Op`.
///
/// # Errors
///
/// Returns a layout validation failure or the kernel compilation failure.
pub fn prepare_binary_elementwise_strided_typed_into<'op, Op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'op, T, N>,
    rhs: StridedOperand<'op, T, N>,
    output: StridedOperand<'op, T, N>,
    width: BlockWidth,
) -> Result<PreparedStridedBinary<'op, T>>
where
    Op: TypedBinaryExpr<HipC, T>,
    T: DialectScalar<HipC> + Pod,
{
    prepare_binary_expression(
        device,
        lhs,
        rhs,
        output,
        width,
        core::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<HipC, T>>::EXPR,
    )
}

fn prepare_binary_expression<'op, T, const N: usize>(
    device: &RocmDevice,
    lhs: StridedOperand<'op, T, N>,
    rhs: StridedOperand<'op, T, N>,
    output: StridedOperand<'op, T, N>,
    width: BlockWidth,
    operation: core::any::TypeId,
    expr: &'static str,
) -> Result<PreparedStridedBinary<'op, T>>
where
    T: DialectScalar<HipC> + Pod,
{
    let Some((meta, len)) = binary_strided_meta(&lhs, &rhs, &output)? else {
        return Ok(PreparedStridedBinary {
            lhs: lhs.buffer,
            rhs: rhs.buffer,
            output: output.buffer,
            plan: None,
        });
    };
    let key = PipelineKey::StridedBinary {
        op: operation,
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "binary_strided_kernel", || {
        binary_shader::<T>(expr)
    })?;
    Ok(PreparedStridedBinary {
        lhs: lhs.buffer,
        rhs: rhs.buffer,
        output: output.buffer,
        plan: Some(StridedPlan {
            meta,
            kernel,
            grid: grid_size(len, width)?,
            width,
        }),
    })
}
