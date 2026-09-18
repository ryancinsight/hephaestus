//! Unary and binary elementwise operations for the host reference device
//! (ADR 0061).
//!
//! `Host` never renders a kernel: [`HostElementwiseOps`] resolves each
//! operator through its ADR 0061 value function, captured as a plain
//! function pointer at prepare time (mirroring `crate::combine`'s
//! `combine_fn`/`unsupported_operator` pattern for the reduction and scan
//! seams). The bridge from the generic `T: Pod` seam to the two differently
//! bounded value-method families (`UnaryExpr::value`/`BinaryExpr::real_value`
//! need `T: RealField`; `BinaryExpr::value`/`TypedBinaryExpr::value` need
//! only `T: NumericElement`) is `ElementwiseDispatch` (ADR 0061
//! Decision 5): `f32`/`f64` resolve through the `RealField` methods,
//! `u32`/`i32`/`F16`/`Bf16` through the `NumericElement` methods, and a
//! real-only unary operator on a non-real scalar reports no application at
//! all — the seam turns that into the typed
//! [`HephaestusError::DispatchFailed`] naming the operator, identically to an
//! operator with no host value function whatsoever.
//!
//! # Broadcasting, aliasing, and output semantics
//!
//! Mirrors `hephaestus-wgpu`'s `WgpuElementwiseOps`: each input broadcasts to
//! the output's shape with leto's own broadcast rules (arbitrary rank `N`),
//! the output buffer must not alias any input buffer, and the output layout
//! must be non-overlapping (validated once at prepare time; re-validated by
//! [`leto::ArrayViewMut::try_iter_mut`] at dispatch, since the prepared form
//! borrows the buffers rather than copying them). A prepared dispatch re-reads
//! its bound buffers fresh on every call, so writes made after preparation are
//! observed; the broadcast shapes themselves are fixed at preparation, since
//! rebind semantics apply to buffer contents only.
//!
//! # Aliased operands
//!
//! `binary_into`/`typed_binary_into` may name the same buffer as both `lhs`
//! and `rhs` (e.g. `mul(a, a)`) with independent layouts (e.g. `a` against
//! its own transpose): [`HostBuffer`] is one `RwLock`, so a second `read()`
//! on the same lock is undefined by `std::sync::RwLock`'s own contract
//! (`crate::operands` documents this for the other host seams). Dispatch
//! detects the alias and reads the shared cells once, viewing them through
//! each operand's own broadcast layout.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{
    BinaryExpr, DialectScalar, ElementwiseOps, HephaestusError, Host, Result, StridedView,
    TypedBinaryExpr, UnaryExpr,
};
use leto::{ArrayView, ArrayViewMut, ElementIterMut, Layout};

use crate::combine::unsupported_operator;
use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice};

/// Host-local per-scalar dispatch (ADR 0061 Decision 5): bridges the two
/// value-method bounds the source traits carry behind one interface the
/// host's generic `T: Pod` seam can call uniformly.
///
/// `f32`/`f64` resolve `unary_fn`/`binary_fn` through the `RealField`-bound
/// methods (`UnaryExpr::value`, `BinaryExpr::real_value`); `u32`/`i32`/`F16`/
/// `Bf16` resolve `binary_fn` through the `NumericElement`-bound
/// `BinaryExpr::value` (valid for every admitting operator, `None` for the
/// real-only [`hephaestus_core::PowOp`]) and `unary_fn` to a function with no
/// application at all, since every unary operator is real-only (ADR 0061
/// Decision 2). Both branches funnel into the same `None` ->
/// [`unsupported_operator`] translation at the call site, so "operator has no
/// host value function" and "scalar admits no real application" are one
/// diagnosable error.
trait ElementwiseDispatch: NumericElement + DialectScalar<Host> + Sized {
    /// Resolve `Op`'s host unary value function for this scalar.
    fn unary_fn<Op: UnaryExpr<Host>>() -> fn(Self) -> Option<Self>;
    /// Resolve `Op`'s host binary value function for this scalar.
    fn binary_fn<Op: BinaryExpr<Host>>() -> fn(Self, Self) -> Option<Self>;
    /// Resolve `Op`'s host typed-comparison value function for this scalar.
    fn typed_binary_fn<Op: TypedBinaryExpr<Host, Self>>() -> fn(Self, Self) -> Option<Self>;
}

macro_rules! impl_real_elementwise_dispatch {
    ($($t:ty),+ $(,)?) => {
        $(
            impl ElementwiseDispatch for $t {
                fn unary_fn<Op: UnaryExpr<Host>>() -> fn(Self) -> Option<Self> {
                    <Op as UnaryExpr<Host>>::value::<Self>
                }
                fn binary_fn<Op: BinaryExpr<Host>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as BinaryExpr<Host>>::real_value::<Self>
                }
                fn typed_binary_fn<Op: TypedBinaryExpr<Host, Self>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as TypedBinaryExpr<Host, Self>>::value
                }
            }
        )+
    };
}

impl_real_elementwise_dispatch!(f32, f64);

macro_rules! impl_numeric_only_elementwise_dispatch {
    ($($t:ty),+ $(,)?) => {
        $(
            impl ElementwiseDispatch for $t {
                fn unary_fn<Op: UnaryExpr<Host>>() -> fn(Self) -> Option<Self> {
                    |_| None
                }
                fn binary_fn<Op: BinaryExpr<Host>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as BinaryExpr<Host>>::value::<Self>
                }
                fn typed_binary_fn<Op: TypedBinaryExpr<Host, Self>>() -> fn(Self, Self) -> Option<Self> {
                    <Op as TypedBinaryExpr<Host, Self>>::value
                }
            }
        )+
    };
}

impl_numeric_only_elementwise_dispatch!(u32, i32, eunomia::F16, eunomia::Bf16);

/// Unary and binary elementwise operations for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostElementwiseOps;

/// A unary dispatch bound to its input/output views and a resolved value
/// function.
pub struct HostPreparedUnary<'op, T, const N: usize> {
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    input_layout: Layout<N>,
    apply: fn(T) -> Option<T>,
    op_name: &'static str,
}

/// A binary dispatch bound to its operand/output views and a resolved value
/// function. Shared by [`ElementwiseOps::PreparedBinary`] and
/// [`ElementwiseOps::PreparedTypedBinary`]: both prepare an identical shape
/// (two broadcast operand layouts, one output, one resolved
/// `fn(T, T) -> Option<T>`) and differ only in which source trait resolved
/// that function.
pub struct HostPreparedBinary<'op, T, const N: usize> {
    lhs: StridedView<'op, HostBuffer<T>, N>,
    rhs: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    lhs_layout: Layout<N>,
    rhs_layout: Layout<N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
}

/// A broadcast-scalar dispatch bound to its input/output views, the captured
/// scalar, and a resolved value function.
pub struct HostPreparedScalar<'op, T, const N: usize> {
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    input_layout: Layout<N>,
    scalar: T,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
}

/// Map a leto layout error the same way `hephaestus-wgpu`'s
/// `application::strided::map_layout_err` does: the shared elementwise
/// conformance clauses (`hephaestus-conformance`) assert this exact
/// `"layout rejected: "` wording for a rejected broadcast, so every backend's
/// diagnostic names the violated constraint identically.
///
/// `pub(crate)`: shared verbatim with [`crate::parameterized`], whose
/// runtime-parameter unary seam broadcasts and iterates the same way.
pub(crate) fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

/// Broadcast `layout` to `target_shape` and validate it against `storage_len`.
/// Mirrors `hephaestus-wgpu`'s per-operand broadcast + `validate_storage_len`
/// step in `prepare_*_inner`.
///
/// `pub(crate)`: shared with [`crate::parameterized`] (see [`map_layout_err`]).
pub(crate) fn broadcast_operand<const N: usize>(
    layout: &Layout<N>,
    target_shape: [usize; N],
    storage_len: usize,
) -> Result<Layout<N>> {
    let broadcast = layout.broadcast(target_shape).map_err(map_layout_err)?;
    broadcast
        .validate_storage_len(storage_len)
        .map_err(map_layout_err)?;
    Ok(broadcast)
}

/// Validate an elementwise output layout: matches its buffer's storage and is
/// non-overlapping (no two logical positions writing the same physical
/// element). Mirrors `hephaestus-wgpu`'s `validate_out`.
fn validate_elementwise_output<T, const N: usize>(
    output: &HostBuffer<T>,
    out_layout: &Layout<N>,
) -> Result<()> {
    out_layout
        .validate_storage_len(output.read().len())
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must be non-overlapping".to_string(),
        });
    }
    Ok(())
}

fn prepare_unary<'op, Op, T, const N: usize>(
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
) -> Result<HostPreparedUnary<'op, T, N>>
where
    Op: UnaryExpr<Host>,
    T: ElementwiseDispatch,
{
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let input_layout = broadcast_operand(
        input.layout,
        output.layout.shape(),
        input.buffer.read().len(),
    )?;
    validate_elementwise_output(output.buffer, output.layout)?;
    Ok(HostPreparedUnary {
        input,
        output,
        input_layout,
        apply: T::unary_fn::<Op>(),
        op_name: core::any::type_name::<Op>(),
    })
}

fn dispatch_unary<T: Copy, const N: usize>(prepared: &HostPreparedUnary<'_, T, N>) -> Result<()> {
    let input_cells = prepared.input.buffer.read();
    let in_view =
        ArrayView::try_new(prepared.input_layout, &input_cells).map_err(map_layout_err)?;
    let mut output_cells = prepared.output.buffer.write();
    let out_view = ArrayViewMut::try_new(*prepared.output.layout, &mut output_cells)
        .map_err(map_layout_err)?;
    let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;
    for &x in &in_view {
        write_next(&mut out_iter, (prepared.apply)(x), prepared.op_name)?;
    }
    Ok(())
}

/// Write the next logical output element from an applied value, or the typed
/// dispatch failure when the operator reported no application.
///
/// `pub(crate)`: shared with [`crate::parameterized`] (see [`map_layout_err`]).
pub(crate) fn write_next<T, const N: usize>(
    out_iter: &mut ElementIterMut<'_, T, N>,
    applied: Option<T>,
    op_name: &str,
) -> Result<()> {
    let slot = out_iter
        .next()
        .expect("invariant: broadcast operands and output share element count");
    *slot = applied.ok_or_else(|| unsupported_operator(op_name))?;
    Ok(())
}

/// Apply `apply` across two same-shape views into `out_iter`, in the shared
/// logical row-major order every leto element iterator walks.
fn apply_zipped<T: Copy, const N: usize>(
    lhs_view: &ArrayView<'_, T, N>,
    rhs_view: &ArrayView<'_, T, N>,
    out_iter: &mut ElementIterMut<'_, T, N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &str,
) -> Result<()> {
    for (&l, &r) in lhs_view.iter().zip(rhs_view) {
        write_next(out_iter, apply(l, r), op_name)?;
    }
    Ok(())
}

fn prepare_binary<'op, T, const N: usize>(
    lhs: StridedView<'op, HostBuffer<T>, N>,
    rhs: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
) -> Result<HostPreparedBinary<'op, T, N>>
where
    T: Pod,
{
    require_disjoint_output(lhs.buffer, rhs.buffer, output.buffer)?;
    let out_shape = output.layout.shape();
    let lhs_layout = broadcast_operand(lhs.layout, out_shape, lhs.buffer.read().len())?;
    let rhs_layout = broadcast_operand(rhs.layout, out_shape, rhs.buffer.read().len())?;
    validate_elementwise_output(output.buffer, output.layout)?;
    Ok(HostPreparedBinary {
        lhs,
        rhs,
        output,
        lhs_layout,
        rhs_layout,
        apply,
        op_name,
    })
}

fn dispatch_binary<T: Copy, const N: usize>(prepared: &HostPreparedBinary<'_, T, N>) -> Result<()> {
    let mut output_cells = prepared.output.buffer.write();
    let out_view = ArrayViewMut::try_new(*prepared.output.layout, &mut output_cells)
        .map_err(map_layout_err)?;
    let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;

    let lhs_cells = prepared.lhs.buffer.read();
    let lhs_view = ArrayView::try_new(prepared.lhs_layout, &lhs_cells).map_err(map_layout_err)?;

    if prepared.rhs.buffer.aliases(prepared.lhs.buffer) {
        // One RwLock guard serves both operands (`crate::operands`): a
        // second `read()` on the same lock is unsound to rely on.
        let rhs_view =
            ArrayView::try_new(prepared.rhs_layout, &lhs_cells).map_err(map_layout_err)?;
        apply_zipped(
            &lhs_view,
            &rhs_view,
            &mut out_iter,
            prepared.apply,
            prepared.op_name,
        )
    } else {
        let rhs_cells = prepared.rhs.buffer.read();
        let rhs_view =
            ArrayView::try_new(prepared.rhs_layout, &rhs_cells).map_err(map_layout_err)?;
        apply_zipped(
            &lhs_view,
            &rhs_view,
            &mut out_iter,
            prepared.apply,
            prepared.op_name,
        )
    }
}

fn prepare_scalar<'op, T, const N: usize>(
    input: StridedView<'op, HostBuffer<T>, N>,
    scalar: T,
    output: StridedView<'op, HostBuffer<T>, N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
) -> Result<HostPreparedScalar<'op, T, N>>
where
    T: Pod,
{
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let input_layout = broadcast_operand(
        input.layout,
        output.layout.shape(),
        input.buffer.read().len(),
    )?;
    validate_elementwise_output(output.buffer, output.layout)?;
    Ok(HostPreparedScalar {
        input,
        output,
        input_layout,
        scalar,
        apply,
        op_name,
    })
}

fn dispatch_scalar<T: Copy, const N: usize>(prepared: &HostPreparedScalar<'_, T, N>) -> Result<()> {
    let input_cells = prepared.input.buffer.read();
    let in_view =
        ArrayView::try_new(prepared.input_layout, &input_cells).map_err(map_layout_err)?;
    let mut output_cells = prepared.output.buffer.write();
    let out_view = ArrayViewMut::try_new(*prepared.output.layout, &mut output_cells)
        .map_err(map_layout_err)?;
    let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;
    for &x in &in_view {
        write_next(
            &mut out_iter,
            (prepared.apply)(x, prepared.scalar),
            prepared.op_name,
        )?;
    }
    Ok(())
}

impl<T> ElementwiseOps<HostDevice, T> for HostElementwiseOps
where
    T: Pod + ElementwiseDispatch,
{
    type Dialect = Host;
    type PreparedUnary<'op, const N: usize>
        = HostPreparedUnary<'op, T, N>
    where
        T: 'op;
    type PreparedBinary<'op, const N: usize>
        = HostPreparedBinary<'op, T, N>
    where
        T: 'op;
    type PreparedScalar<'op, const N: usize>
        = HostPreparedScalar<'op, T, N>
    where
        T: 'op;
    type PreparedTypedBinary<'op, const N: usize>
        = HostPreparedBinary<'op, T, N>
    where
        T: 'op;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        prepare_unary::<Op, T, N>(input, output)
    }

    fn dispatch_unary<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        dispatch_unary::<T, N>(prepared)
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'op, HostBuffer<T>, N>,
        rhs: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_binary(
            lhs,
            rhs,
            output,
            T::binary_fn::<Op>(),
            core::any::type_name::<Op>(),
        )
    }

    fn dispatch_binary<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_binary::<T, N>(prepared)
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, N>,
        scalar: T,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_scalar(
            input,
            scalar,
            output,
            T::binary_fn::<Op>(),
            core::any::type_name::<Op>(),
        )
    }

    fn dispatch_scalar<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()> {
        dispatch_scalar::<T, N>(prepared)
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'op, HostBuffer<T>, N>,
        rhs: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
        T: DialectScalar<Self::Dialect>,
    {
        prepare_binary(
            lhs,
            rhs,
            output,
            T::typed_binary_fn::<Op>(),
            core::any::type_name::<Op>(),
        )
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_binary::<T, N>(prepared)
    }
}
