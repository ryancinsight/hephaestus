//! Reusable CUDA dot-product and L2-norm plans.
//!
//! Prepared plans retain the fused first-pass workgroup partials and the
//! optional reduction tree. They do not allocate a full logical-length
//! product or square buffer per dispatch.

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, IdentityToken, OpIdentity,
    Result, SumOp, map_layout_err,
};
use leto::Layout;

use crate::CudaDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::linalg::{
    DotMap, PreparedMapReduction, RetainedMapReduction, SquareMap,
    prepare_map_reduction_with_layouts,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;

/// A reusable CUDA vector dot-product plan over fixed strided inputs.
pub struct PreparedDot<'a, T> {
    inner: PreparedMapReduction<'a, DotMap, T, 1>,
    left: &'a CudaBuffer<T>,
    right: &'a CudaBuffer<T>,
}

impl<T> PreparedDot<'_, T>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    /// Dispatch the prepared dot product and reuse its device-resident output.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a native elementwise or reduction launch
    /// fails.
    pub fn dispatch(&self) -> Result<()> {
        self.inner.dispatch()
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> &CudaBuffer<T> {
        self.inner.output()
    }

    pub(crate) fn matches(&self, left: &CudaBuffer<T>, right: &CudaBuffer<T>) -> bool {
        self.left.raw() == left.raw()
            && self.left.len() == left.len()
            && self.right.raw() == right.raw()
            && self.right.len() == right.len()
    }

    /// Forget the operand borrows into `Copy` provenance, keeping the
    /// dispatch resources.
    #[must_use]
    pub fn into_retained(self) -> RetainedDot<T> {
        RetainedDot {
            inner: self.inner.into_retained(),
        }
    }
}

/// Prepare `Σᵢ lhs[i] * rhs[i]` over fixed strided device buffers.
///
/// # Errors
///
/// Returns an error when the logical shapes differ, either input layout is
/// invalid for its buffer, or the fixed device resources cannot be allocated.
pub fn prepare_dot<'a, T>(
    device: &CudaDevice,
    lhs: StridedOperand<'a, T, 1>,
    rhs: StridedOperand<'a, T, 1>,
) -> Result<PreparedDot<'a, T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    if lhs.layout.shape() != rhs.layout.shape() {
        return Err(hephaestus_core::HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape(),
                rhs.layout.shape()
            ),
        });
    }
    lhs.layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    let inner = prepare_map_reduction_with_layouts::<DotMap, T, 1>(
        device, lhs.buffer, lhs.layout, rhs.buffer, rhs.layout,
    )?;
    Ok(PreparedDot {
        inner,
        left: lhs.buffer,
        right: rhs.buffer,
    })
}

pub(crate) fn prepare_dense_dot<'a, T>(
    device: &CudaDevice,
    left: &'a CudaBuffer<T>,
    right: &'a CudaBuffer<T>,
) -> Result<PreparedDot<'a, T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    if left.len() != right.len() {
        return Err(hephaestus_core::HephaestusError::LengthMismatch {
            host_len: left.len(),
            device_len: right.len(),
        });
    }
    let layout = Layout::c_contiguous([left.len()]).map_err(|error| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("dense vector layout failed: {error}"),
        }
    })?;
    let inner =
        prepare_map_reduction_with_layouts::<DotMap, T, 1>(device, left, &layout, right, &layout)?;
    Ok(PreparedDot { inner, left, right })
}

/// An owned CUDA vector dot-product plan over fixed dense inputs.
///
/// The [`RetainedReductions`](hephaestus_core::RetainedReductions) form of
/// [`PreparedDot`]: identical dispatch resources, but the operand borrows
/// are forgotten into provenance, so the plan can live in a workspace
/// beside the vectors it measures. Every dispatch re-presents the live
/// buffers and verifies them before launching.
pub struct RetainedDot<T> {
    inner: RetainedMapReduction<DotMap, T, 1>,
}

impl<T> RetainedDot<T>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    /// Dispatch over `left` and `right` after verifying they are the
    /// retained allocations, reusing the device-resident output.
    ///
    /// # Errors
    ///
    /// Returns a typed error when either buffer names a different device
    /// allocation than the plan was retained against, or when a native
    /// launch fails.
    pub fn dispatch_with(&self, left: &CudaBuffer<T>, right: &CudaBuffer<T>) -> Result<()> {
        self.inner.dispatch_with(left, right)
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> &CudaBuffer<T> {
        self.inner.output()
    }
}

pub(crate) fn retain_dense_dot<T>(
    device: &CudaDevice,
    left: &CudaBuffer<T>,
    right: &CudaBuffer<T>,
) -> Result<RetainedDot<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    Ok(prepare_dense_dot(device, left, right)?.into_retained())
}

/// A reusable CUDA L2-norm plan over a fixed strided input.
pub struct PreparedL2Norm<'a, T, const N: usize> {
    device: CudaDevice,
    inner: PreparedMapReduction<'a, SquareMap, T, N>,
    output: CudaBuffer<T>,
    input: &'a CudaBuffer<T>,
}

impl<T, const N: usize> PreparedL2Norm<'_, T, N>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    /// Dispatch the prepared `sqrt(Σ x²)` operation.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a native elementwise, reduction, or square
    /// root launch fails.
    pub fn dispatch(&self) -> Result<()> {
        self.inner.dispatch()?;
        unary_elementwise_into::<SqrtOp, T>(
            &self.device,
            self.inner.output(),
            &self.output,
            BlockWidth::DEFAULT,
        )
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> &CudaBuffer<T> {
        &self.output
    }

    pub(crate) fn matches(&self, input: &CudaBuffer<T>) -> bool {
        self.input.raw() == input.raw() && self.input.len() == input.len()
    }

    /// Forget the operand borrow into `Copy` provenance, keeping the
    /// dispatch resources.
    #[must_use]
    pub fn into_retained(self) -> RetainedL2Norm<T, N> {
        RetainedL2Norm {
            device: self.device,
            inner: self.inner.into_retained(),
            output: self.output,
        }
    }
}

/// Prepare `sqrt(Σ x²)` over a fixed strided device buffer.
///
/// # Errors
///
/// Returns an error when the input layout is invalid for its buffer or the
/// fixed device resources cannot be allocated.
pub fn prepare_norm_l2<'a, T, const N: usize>(
    device: &CudaDevice,
    input: StridedOperand<'a, T, N>,
) -> Result<PreparedL2Norm<'a, T, N>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    input
        .layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    let inner = prepare_map_reduction_with_layouts::<SquareMap, T, N>(
        device,
        input.buffer,
        input.layout,
        input.buffer,
        input.layout,
    )?;
    let output = device.alloc_zeroed::<T>(1)?;
    Ok(PreparedL2Norm {
        device: device.clone(),
        inner,
        output,
        input: input.buffer,
    })
}

pub(crate) fn prepare_dense_norm_l2<'a, T>(
    device: &CudaDevice,
    input: &'a CudaBuffer<T>,
) -> Result<PreparedL2Norm<'a, T, 1>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    let layout = Layout::c_contiguous([input.len()]).map_err(|error| {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("dense vector layout failed: {error}"),
        }
    })?;
    let inner = prepare_map_reduction_with_layouts::<SquareMap, T, 1>(
        device, input, &layout, input, &layout,
    )?;
    Ok(PreparedL2Norm {
        device: device.clone(),
        inner,
        output: device.alloc_zeroed::<T>(1)?,
        input,
    })
}

/// An owned CUDA L2-norm plan over a fixed input.
///
/// The [`RetainedReductions`](hephaestus_core::RetainedReductions) form of
/// [`PreparedL2Norm`]: identical dispatch resources, but the operand borrow
/// is forgotten into provenance, so the plan can live in a workspace
/// beside the vector it measures. Every dispatch re-presents the live
/// buffer and verifies it before launching.
pub struct RetainedL2Norm<T, const N: usize> {
    device: CudaDevice,
    inner: RetainedMapReduction<SquareMap, T, N>,
    output: CudaBuffer<T>,
}

impl<T, const N: usize> RetainedL2Norm<T, N>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    /// Dispatch the retained `sqrt(Σ x²)` operation over `input` after
    /// verifying it is the retained allocation.
    ///
    /// # Errors
    ///
    /// Returns a typed error when `input` names a different device
    /// allocation than the plan was retained against, or when a native
    /// elementwise, reduction, or square root launch fails.
    pub fn dispatch_with(&self, input: &CudaBuffer<T>) -> Result<()> {
        self.inner.dispatch_with(input, input)?;
        unary_elementwise_into::<SqrtOp, T>(
            &self.device,
            self.inner.output(),
            &self.output,
            BlockWidth::DEFAULT,
        )
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> &CudaBuffer<T> {
        &self.output
    }
}

pub(crate) fn retain_dense_norm_l2<T>(
    device: &CudaDevice,
    input: &CudaBuffer<T>,
) -> Result<RetainedL2Norm<T, 1>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    Ok(prepare_dense_norm_l2(device, input)?.into_retained())
}
