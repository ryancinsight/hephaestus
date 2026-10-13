#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! Backend-neutral axis-reduction planning and the device-neutral seam.
//!
//! Both GPU backends reduce a rank-2 strided operand along one axis with the
//! identical host-side contract: validate the shapes/strides, pack the launch
//! metadata into a `#[repr(C)]` block mirroring the shader's uniform struct,
//! and derive the workgroup count from the output length. That logic is
//! dialect-independent — only the shader source and raw dispatch differ — so
//! it lives here once and both backends call [`plan_axis_reduction`](crate::plan_axis_reduction).
//!
//! The one backend-specific input is whether the input and output buffers
//! alias (a device-pointer identity check); callers compute it and pass it in.
//!
//! [`crate::AxisReductionOps`] completes the family: the planning above was already
//! shared, but every backend still exposed `prod_axis_into` and
//! `prepare_reduce_axis_into` as free functions over its own device and operand
//! types, so a consumer — or a conformance suite — had to bind to one device API
//! to call them. The seam takes the same shape as [`crate::DenseVectorOps`] and
//! [`crate::SparseOperatorOps`]: generic over the device, monomorphized at every call
//! site, no `dyn` anywhere.

use crate::domain::device::ComputeDevice;
use crate::domain::dialect::KernelDialect;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, IdentityToken, OpIdentity};
use crate::domain::planning::{map_layout_err, to_i32, to_u32};
use crate::domain::vector::DenseVectorOps;
use crate::domain::view::StridedView;
use eunomia::{Pod, Zeroable};
use leto::Layout;

mod axis_ops;
pub use axis_ops::AxisReductionOps;

/// Device-neutral full-array reduction over a strided n-D operand to a
/// single scalar stored in a one-element output buffer.
///
/// This complements [`crate::AxisReductionOps`] by reducing the entire operand to
/// one value rather than along one axis. The output is a rank-1 view of length
/// one, which keeps the same `StridedView` output shape and lets consumers
/// place the scalar where they need it.
///
/// Implementors are zero-sized per-backend markers. A bound of
/// `R: FullReductionOps<D, T>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`]
/// (ADR 0043) for what is and is not promised per dialect.
pub trait FullReductionOps<D: ComputeDevice, T: Pod> {
    /// Kernel dialect this backend authors reductions in.
    type Dialect: KernelDialect;

    /// Prepared resources for a full reduction bound to fixed input/output
    /// views.
    ///
    /// Lifetime-parameterized so backends whose prepared form borrows the
    /// operand pair implement the seam without erasing that borrow;
    /// handle-based backends simply ignore `'op`.
    type Prepared<'op, const N: usize>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    /// Reduce the entire `input` view into the one-element `output` under the
    /// combining operator `Op`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch when `output` does not have length one, an
    /// aliasing violation, a layout validation failure, or the backend dispatch
    /// failure.
    fn reduce_full_into<Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<T>, N>,
        output: StridedView<'_, D::Buffer<T>, 1>,
    ) -> Result<()>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        let prepared = self.prepare_reduce_full::<Op, N>(device, input, output)?;
        self.dispatch_full::<N>(device, &prepared)
    }

    /// Bind dispatch resources for reducing the entire `input` view into the
    /// one-element `output` under the combining operator `Op`.
    ///
    /// # Errors
    ///
    /// Returns a shape mismatch when `output` does not have length one, an
    /// aliasing violation, a layout validation failure, or the backend
    /// preparation failure.
    fn prepare_reduce_full<'op, Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, N>,
        output: StridedView<'op, D::Buffer<T>, 1>,
    ) -> Result<Self::Prepared<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>;

    /// Re-dispatch a prepared full reduction over its bound operands.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dispatch_full<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::Prepared<'_, N>,
    ) -> Result<()>;
}

/// Marker asserting that a backend supplies both halves of the accelerator
/// operand surface a solver needs: dense vector recurrences and axis
/// reductions over strided views.
///
/// Consumers bind to this instead of restating both bounds, and a backend that
/// implements both acquires it through the blanket implementation below.
pub trait StridedComputeBackend<D: ComputeDevice, T: Pod>:
    DenseVectorOps<D, T> + AxisReductionOps<D, T>
{
}

impl<B, D, T> StridedComputeBackend<D, T> for B
where
    D: ComputeDevice,
    T: Pod,
    B: DenseVectorOps<D, T> + AxisReductionOps<D, T>,
{
}

/// Validate that a reduction block width can be halved into a workgroup tree.
///
/// Both WGSL and CUDA scalar reductions reduce by repeatedly halving active
/// lanes. A non-power-of-two width would drop lanes, so it is rejected before
/// backend dispatch.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when `width` is not a power of
/// two.
pub fn validate_reduction_width(width: BlockWidth) -> Result<()> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "reduction block width {} must be a power of two",
                width.get()
            ),
        });
    }
    Ok(())
}

/// Return the number of tree-reduction passes needed to reduce `len` elements.
///
/// The value is backend-neutral host planning: each pass emits
/// `ceil(current_len / width)` partials until one element remains.
#[must_use]
pub fn reduction_pass_count(mut len: usize, width: BlockWidth) -> usize {
    // `width.get()` is `u32`; this is a lossless widening on all supported
    // Rust targets (`usize >= 32` bits).
    let width = usize::try_from(width.get()).expect("reduction width fits supported host usize");
    let mut passes = 0;
    while len > 1 {
        len = len.div_ceil(width);
        passes += 1;
    }
    passes
}

/// Launch metadata for the axis-reduction kernel.
///
/// `#[repr(C)]` so it is bitwise-compatible with the WGSL/CUDA uniform struct
/// the shaders declare. `offsets` packs, in order: input element offset, output
/// element offset, the reduced axis, and the output element count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AxisReductionMeta {
    /// Input shape `[rows, cols]`.
    pub input_shape: [u32; 2],
    /// Input strides `[row, col]` in elements.
    pub input_strides: [i32; 2],
    /// Output strides `[row, col]` in elements.
    pub output_strides: [i32; 2],
    /// Padding keeping `offsets` 16-byte aligned in the uniform block.
    pub _pre_offsets_pad: [u32; 2],
    /// `[input_offset, output_offset, axis, output_len]`.
    pub offsets: [u32; 4],
}

/// A validated axis-reduction dispatch: launch metadata plus the workgroup count.
#[derive(Clone, Copy, Debug)]
pub struct AxisReductionDispatch {
    /// Launch metadata for the shader uniform.
    pub meta: AxisReductionMeta,
    /// Workgroup (block) count covering the output length.
    pub groups: u32,
}

/// Validate a rank-2 axis reduction and build its dispatch plan.
///
/// Returns `Ok(None)` when the output is empty (a no-op dispatch). The output
/// shape must equal the input shape with the reduced `axis` collapsed to 1.
/// The caller supplies `buffers_alias` (the backend's device-pointer identity
/// check) and the two buffer element counts.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when the block width is not a
/// power of two, the axis is out of range, the output shape does not match the
/// reduced input shape, the buffers alias, the output layout has zero-stride
/// aliasing, a layout does not fit its buffer, or an extent/stride exceeds the
/// shader's `u32`/`i32` range.
pub fn plan_axis_reduction(
    input_layout: &Layout<2>,
    input_buf_len: usize,
    output_layout: &Layout<2>,
    output_buf_len: usize,
    axis: usize,
    width: BlockWidth,
    buffers_alias: bool,
) -> Result<Option<AxisReductionDispatch>> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "reduction block width {} must be a power of two",
                width.get()
            ),
        });
    }
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        });
    }

    let mut expected_shape = input_layout.shape();
    expected_shape[axis] = 1;
    if output_layout.shape() != expected_shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "axis reduction output shape mismatch: input {:?}, axis {axis}, out {:?}",
                input_layout.shape(),
                output_layout.shape()
            ),
        });
    }
    if buffers_alias {
        return Err(HephaestusError::DispatchFailed {
            message: "axis reduction output buffer must not alias input buffer".to_string(),
        });
    }
    if output_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "axis reduction output layout must not contain zero-stride aliasing"
                .to_string(),
        });
    }

    input_layout
        .validate_storage_len(input_buf_len)
        .map_err(map_layout_err)?;
    output_layout
        .validate_storage_len(output_buf_len)
        .map_err(map_layout_err)?;
    let output_len = output_layout.checked_size().map_err(map_layout_err)?;
    if output_len == 0 {
        return Ok(None);
    }

    let output_len_u64 =
        u64::try_from(output_len).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("output length {output_len} exceeds u64 range"),
        })?;
    let groups = width
        .checked_covering_blocks(output_len_u64)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("output length {output_len} exceeds u32 workgroup range"),
        })?;

    let meta = AxisReductionMeta {
        input_shape: [
            to_u32(input_layout.shape()[0], "input rows")?,
            to_u32(input_layout.shape()[1], "input columns")?,
        ],
        input_strides: [
            to_i32(input_layout.strides()[0], "input row stride")?,
            to_i32(input_layout.strides()[1], "input column stride")?,
        ],
        output_strides: [
            to_i32(output_layout.strides()[0], "output row stride")?,
            to_i32(output_layout.strides()[1], "output column stride")?,
        ],
        _pre_offsets_pad: [0; 2],
        offsets: [
            to_u32(input_layout.offset(), "input offset")?,
            to_u32(output_layout.offset(), "output offset")?,
            to_u32(axis, "axis")?,
            to_u32(output_len, "output length")?,
        ],
    };

    Ok(Some(AxisReductionDispatch { meta, groups }))
}

#[cfg(test)]
#[path = "tests_reduction.rs"]
mod tests;
