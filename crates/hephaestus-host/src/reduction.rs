//! Whole-operand and rank-2 axis reductions for the host reference device
//! (ADR 0061).
//!
//! `Host` never renders a kernel: [`HostFullReductionOps`] and
//! [`HostAxisReductionOps`] resolve the combining operator through
//! `<Op as CombineExpr<Host>>::value`, captured as a plain function pointer
//! at prepare time (`crate::combine::combine_fn`) together with the
//! operator's host-side identity (`OpIdentity::IDENTITY`), so the erased
//! `Prepared` associated types never need to name `Op` again. An operator
//! with no host value function is the typed
//! [`HephaestusError::DispatchFailed`] from
//! `crate::combine::unsupported_operator`, never a silent identity or no-op.
//!
//! # Reduction order
//!
//! Every reduction folds its elements as a **sequential left fold**, in the
//! operand's logical row-major order: `elements.fold(identity, combine)`,
//! one element at a time, never a tree or a parallel partial-sum
//! regrouping. This is a deliberate choice, not an incidental
//! implementation detail — it is what makes the host the one backend whose
//! result follows directly from this specification rather than from
//! hardware scheduling, and therefore the deterministic reference every
//! other backend's differential test compares against. GPU backends
//! reassociate for parallelism; the host does not need to and therefore
//! never does.
//!
//! `mean_axis_into` divides the accumulated sum by the reduced axis length
//! cast to `T`, matching the WGSL kernels' `partials[0] / T(axis_len)`: for
//! an integer `T` this is truncating integer division, safe here because an
//! empty reduced axis is rejected before any division executes.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{
    AxisReductionMeta, AxisReductionOps, BlockWidth, CombineExpr, DialectScalar, FullReductionOps,
    HephaestusError, Host, IdentityToken, OpIdentity, Result, StridedView, SumOp,
    plan_axis_reduction,
};
use leto::{ArrayView, Layout};

use crate::combine::{axis_len, combine_fn, offset_2d, unsupported_operator};
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Whole-operand reductions for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostFullReductionOps;

/// A full reduction bound to its input view and one-element output.
///
/// Holds the resolved combine function and identity (ADR 0061) rather than
/// the erased `Op`; dispatch re-reads the input view fresh on every call, so
/// a write to the bound input made after preparation is observed.
pub struct HostPreparedFullReduction<'op, T, const N: usize> {
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, 1>,
    combine: fn(T, T) -> Option<T>,
    identity: T,
    op_name: &'static str,
}

/// Validate a full-reduction output: exactly one element, and not the same
/// allocation as `input`. Mirrors `hephaestus-wgpu`'s
/// `WgpuFullReductionOps::prepare_reduce_full`.
fn validate_full_reduction_output<T>(
    input: &HostBuffer<T>,
    output: &HostBuffer<T>,
    output_layout: &Layout<1>,
) -> Result<()> {
    output_layout
        .validate_storage_len(output.read().len())
        .map_err(map_leto_error)?;
    let output_len = output_layout.checked_size().map_err(map_leto_error)?;
    if output_len != 1 {
        return Err(HephaestusError::DispatchFailed {
            message: "full reduction output must have exactly 1 element".to_string(),
        });
    }
    if output.aliases(input) {
        return Err(HephaestusError::DispatchFailed {
            message: "full reduction output buffer must not alias input buffer".to_string(),
        });
    }
    Ok(())
}

impl<T> FullReductionOps<HostDevice, T> for HostFullReductionOps
where
    T: DialectScalar<Host> + Pod + NumericElement,
{
    type Dialect = Host;
    type Prepared<'op, const N: usize>
        = HostPreparedFullReduction<'op, T, N>
    where
        T: 'op;

    fn prepare_reduce_full<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, 1>,
    ) -> Result<Self::Prepared<'op, N>>
    where
        Op: CombineExpr<Host>,
        T: OpIdentity<Op> + IdentityToken<Op, Host>,
    {
        validate_full_reduction_output(input.buffer, output.buffer, output.layout)?;
        Ok(HostPreparedFullReduction {
            input,
            output,
            combine: combine_fn::<Op, T>(),
            identity: <T as OpIdentity<Op>>::IDENTITY,
            op_name: core::any::type_name::<Op>(),
        })
    }

    fn dispatch_full<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::Prepared<'_, N>,
    ) -> Result<()> {
        let input_cells = prepared.input.buffer.read();
        let view =
            ArrayView::try_new(*prepared.input.layout, &input_cells).map_err(map_leto_error)?;
        let mut acc = prepared.identity;
        for &value in &view {
            acc = (prepared.combine)(acc, value)
                .ok_or_else(|| unsupported_operator(prepared.op_name))?;
        }
        let output_index = prepared.output.layout.offset();
        let mut output_cells = prepared.output.buffer.write();
        output_cells[output_index] = acc;
        Ok(())
    }
}

/// Rank-2 axis reductions for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostAxisReductionOps;

/// An axis reduction bound to its input/output views and a validated
/// dispatch plan; `meta` is `None` for an empty output (a no-op dispatch,
/// matching [`hephaestus_core::plan_axis_reduction`]'s contract).
pub struct HostPreparedAxisReduction<'op, T> {
    input: StridedView<'op, HostBuffer<T>, 2>,
    output: StridedView<'op, HostBuffer<T>, 2>,
    meta: Option<AxisReductionMeta>,
    combine: fn(T, T) -> Option<T>,
    identity: T,
    op_name: &'static str,
}

/// Fold each output position's lane under `combine`, in the sequential
/// left-fold order this module documents, applying `finalize` to the
/// accumulated value before it is written — identity for a plain reduction,
/// division by the reduced length for `mean_axis_into`.
fn fold_axis_reduction<T: Copy>(
    meta: &AxisReductionMeta,
    input: &[T],
    output: &mut [T],
    combine: fn(T, T) -> Option<T>,
    identity: T,
    op_name: &str,
    finalize: impl Fn(T) -> T,
) -> Result<()> {
    let axis = meta.offsets[2] as usize;
    let output_len = meta.offsets[3] as usize;
    let axis_len = meta.input_shape[axis] as usize;
    for i in 0..output_len {
        let (out_row, out_col) = if axis == 0 { (0, i) } else { (i, 0) };
        let mut acc = identity;
        for idx in 0..axis_len {
            let (in_row, in_col) = if axis == 0 {
                (idx, out_col)
            } else {
                (out_row, idx)
            };
            let in_off = offset_2d(meta.offsets[0], meta.input_strides, in_row, in_col);
            acc = combine(acc, input[in_off]).ok_or_else(|| unsupported_operator(op_name))?;
        }
        let out_off = offset_2d(meta.offsets[1], meta.output_strides, out_row, out_col);
        output[out_off] = finalize(acc);
    }
    Ok(())
}

impl<T> AxisReductionOps<HostDevice, T> for HostAxisReductionOps
where
    T: DialectScalar<Host> + Pod + NumericElement,
{
    type Dialect = Host;
    type Prepared<'op>
        = HostPreparedAxisReduction<'op, T>
    where
        T: 'op;

    fn mean_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()>
    where
        SumOp: CombineExpr<Host>,
        T: OpIdentity<SumOp> + IdentityToken<SumOp, Host>,
    {
        let reduced_len = axis_len(input.layout.shape(), axis)?;
        if reduced_len == 0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("mean_axis is undefined for empty axis {axis}"),
            });
        }
        let Some(dispatch) = plan_axis_reduction(
            input.layout,
            input.buffer.read().len(),
            output.layout,
            output.buffer.read().len(),
            axis,
            BlockWidth::DEFAULT,
            input.buffer.aliases(output.buffer),
        )?
        else {
            return Ok(());
        };
        let reduced_len_i32 =
            i32::try_from(reduced_len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("reduced axis length {reduced_len} exceeds i32 range"),
            })?;
        let divisor = T::cast_from(reduced_len_i32);
        let input_cells = input.buffer.read();
        let mut output_cells = output.buffer.write();
        fold_axis_reduction(
            &dispatch.meta,
            &input_cells,
            &mut output_cells,
            combine_fn::<SumOp, T>(),
            <T as OpIdentity<SumOp>>::IDENTITY,
            core::any::type_name::<SumOp>(),
            |sum| {
                sum.checked_div(divisor)
                    .expect("invariant: reduced axis length is non-zero (rejected above)")
            },
        )
    }

    fn prepare_reduce_axis_into<'op, Op>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'op, HostBuffer<T>, 2>,
    ) -> Result<Self::Prepared<'op>>
    where
        Op: CombineExpr<Host>,
        T: OpIdentity<Op> + IdentityToken<Op, Host>,
    {
        let dispatch = plan_axis_reduction(
            input.layout,
            input.buffer.read().len(),
            output.layout,
            output.buffer.read().len(),
            axis,
            BlockWidth::DEFAULT,
            input.buffer.aliases(output.buffer),
        )?;
        Ok(HostPreparedAxisReduction {
            input,
            output,
            meta: dispatch.map(|d| d.meta),
            combine: combine_fn::<Op, T>(),
            identity: <T as OpIdentity<Op>>::IDENTITY,
            op_name: core::any::type_name::<Op>(),
        })
    }

    fn dispatch_prepared(&self, _device: &HostDevice, prepared: &Self::Prepared<'_>) -> Result<()> {
        let Some(meta) = prepared.meta else {
            return Ok(());
        };
        let input_cells = prepared.input.buffer.read();
        let mut output_cells = prepared.output.buffer.write();
        fold_axis_reduction(
            &meta,
            &input_cells,
            &mut output_cells,
            prepared.combine,
            prepared.identity,
            prepared.op_name,
            |acc| acc,
        )
    }
}
