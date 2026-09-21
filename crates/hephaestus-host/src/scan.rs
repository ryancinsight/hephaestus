//! Rank-2 prefix/suffix scans for the host reference device (ADR 0061).
//!
//! Mirrors [`crate::reduction`]'s combine-dispatch plumbing: the operator is
//! resolved to a `fn(T, T) -> Option<T>` at prepare time via
//! `<Op as CombineExpr<Host>>::value`, and an operator without a host value
//! function is the typed [`HephaestusError::DispatchFailed`] from
//! `crate::combine::unsupported_operator`.
//!
//! # Scan order
//!
//! Each scan line folds **sequentially and inclusively**, in traversal
//! order: [`hephaestus_core::ScanDirection::Forward`] accumulates index `0`
//! upward, writing `combine(combine(..combine(identity, x0)..), xk)` at
//! position `k`; [`hephaestus_core::ScanDirection::Reverse`] accumulates the
//! last index downward, writing the same inclusive fold over the reversed
//! traversal. This is the same
//! deterministic left-fold discipline [`crate::reduction`] documents for
//! reductions, and is what makes the host the reference every other
//! backend's scan differential test compares against.
//!
//! The seam is generic over rank `N`, but [`hephaestus_core::plan_axis_scan`]
//! — shared with every other backend — validates rank-2 layouts only; a
//! rank other than 2 is a typed rejection, matching `hephaestus-wgpu`'s
//! `WgpuScanOps`.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{
    AxisScanMeta, BlockWidth, CombineExpr, DialectScalar, HephaestusError, Host, IdentityToken,
    OpIdentity, Result, ScanDirection, ScanOps, StridedView, plan_axis_scan,
};
use leto::Layout;

use crate::combine::{combine_fn, offset_2d, unsupported_operator};
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Rank-2 prefix/suffix scans for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostScanOps;

/// A scan bound to its input/output views and a validated dispatch plan;
/// `meta` is `None` for an empty output (a no-op dispatch, matching
/// [`hephaestus_core::plan_axis_scan`]'s contract).
pub struct HostPreparedScan<'op, T, const N: usize> {
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    meta: Option<AxisScanMeta>,
    combine: fn(T, T) -> Option<T>,
    identity: T,
    op_name: &'static str,
}

/// Rebuild a rank-`N` layout already verified to have `N == 2` as a rank-2
/// layout.
///
/// `Layout<N>` cannot itself carry that runtime fact in its type (`N` is a
/// caller-chosen const), so this copies the same shape, strides, and offset
/// into a fresh `Layout<2>` rather than transmuting — the host has no
/// performance reason to take on the unsafe reinterpret `hephaestus-wgpu`
/// uses for the same problem.
fn rank2_layout<const N: usize>(layout: &Layout<N>) -> Result<Layout<2>> {
    debug_assert_eq!(N, 2, "precondition: caller already rejected rank != 2");
    let shape = layout.shape();
    let strides = layout.strides();
    Layout::try_new(
        [shape[0], shape[1]],
        [strides[0], strides[1]],
        layout.offset(),
    )
    .map_err(map_leto_error)
}

/// Compute one scan line's inclusive sequential fold and write every
/// partial along the way.
fn fold_scan_line<T: Copy>(
    meta: &AxisScanMeta,
    input: &[T],
    output: &mut [T],
    combine: fn(T, T) -> Option<T>,
    identity: T,
    op_name: &str,
    line: usize,
) -> Result<()> {
    let axis = (meta.offsets[2] & 1) as usize;
    let reverse = meta.offsets[2] & 2 != 0;
    let len = meta.input_shape[axis] as usize;
    let mut acc = identity;
    for s in 0..len {
        let idx = if reverse { len - 1 - s } else { s };
        let (row, col) = if axis == 0 { (idx, line) } else { (line, idx) };
        let in_off = offset_2d(meta.offsets[0], meta.input_strides, row, col);
        let out_off = offset_2d(meta.offsets[1], meta.output_strides, row, col);
        acc = combine(acc, input[in_off]).ok_or_else(|| unsupported_operator(op_name))?;
        output[out_off] = acc;
    }
    Ok(())
}

/// Fold every scan line of a validated dispatch.
fn run_axis_scan<T: Copy>(
    meta: &AxisScanMeta,
    input: &[T],
    output: &mut [T],
    combine: fn(T, T) -> Option<T>,
    identity: T,
    op_name: &str,
) -> Result<()> {
    let line_count = meta.offsets[3] as usize;
    for line in 0..line_count {
        fold_scan_line(meta, input, output, combine, identity, op_name, line)?;
    }
    Ok(())
}

impl<T> ScanOps<HostDevice, T> for HostScanOps
where
    T: DialectScalar<Host> + Pod + NumericElement,
{
    type Dialect = Host;
    type PreparedScan<'op, const N: usize>
        = HostPreparedScan<'op, T, N>
    where
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Host>,
        T: OpIdentity<Op> + IdentityToken<Op, Host>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("scan currently supports only rank-2 operands, got rank {N}"),
            });
        }
        let input_layout_2 = rank2_layout(input.layout)?;
        let output_layout_2 = rank2_layout(output.layout)?;
        let dispatch = plan_axis_scan(
            &input_layout_2,
            input.buffer.read().len(),
            &output_layout_2,
            output.buffer.read().len(),
            axis,
            direction,
            BlockWidth::DEFAULT,
            input.buffer.aliases(output.buffer),
        )?;
        Ok(HostPreparedScan {
            input,
            output,
            meta: dispatch.map(|d| d.meta),
            combine: combine_fn::<Op, T>(),
            identity: <T as OpIdentity<Op>>::IDENTITY,
            op_name: core::any::type_name::<Op>(),
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        let Some(meta) = prepared.meta else {
            return Ok(());
        };
        let input_cells = prepared.input.buffer.read();
        let mut output_cells = prepared.output.buffer.write();
        run_axis_scan(
            &meta,
            &input_cells,
            &mut output_cells,
            prepared.combine,
            prepared.identity,
            prepared.op_name,
        )
    }
}
