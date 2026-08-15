//! Reusable ROCm dot-product and L2-norm plans.
//!
//! Prepared plans retain the fused first-pass workgroup partials and the
//! optional reduction tree. They do not allocate a full logical-length
//! product or square buffer per dispatch.

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HipC, IdentityToken, OpIdentity,
    Result, SumOp,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::linalg::{
    DotMap, PreparedMapReduction, SquareMap, prepare_map_reduction_with_layouts,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// A reusable ROCm vector dot-product plan over fixed strided inputs.
pub struct PreparedDot<'a, T> {
    inner: PreparedMapReduction<'a, DotMap, T, 1>,
    left: &'a RocmBuffer<T>,
    right: &'a RocmBuffer<T>,
}

impl<T> PreparedDot<'_, T>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
    pub fn output(&self) -> &RocmBuffer<T> {
        self.inner.output()
    }

    pub(crate) fn matches(&self, left: &RocmBuffer<T>, right: &RocmBuffer<T>) -> bool {
        self.left.raw() == left.raw()
            && self.left.len() == left.len()
            && self.right.raw() == right.raw()
            && self.right.len() == right.len()
    }
}

/// Prepare `Σᵢ lhs[i] * rhs[i]` over fixed strided device buffers.
///
/// # Errors
///
/// Returns an error when the logical shapes differ, either input layout is
/// invalid for its buffer, or the fixed device resources cannot be allocated.
pub fn prepare_dot<'a, T>(
    device: &RocmDevice,
    lhs: StridedOperand<'a, T, 1>,
    rhs: StridedOperand<'a, T, 1>,
) -> Result<PreparedDot<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
        .map_err(crate::application::linalg::map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(crate::application::linalg::map_layout_err)?;
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
    device: &RocmDevice,
    left: &'a RocmBuffer<T>,
    right: &'a RocmBuffer<T>,
) -> Result<PreparedDot<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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

/// A reusable ROCm L2-norm plan over a fixed strided input.
pub struct PreparedL2Norm<'a, T, const N: usize> {
    device: RocmDevice,
    inner: PreparedMapReduction<'a, SquareMap, T, N>,
    output: RocmBuffer<T>,
    input: &'a RocmBuffer<T>,
}

impl<T, const N: usize> PreparedL2Norm<'_, T, N>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
    pub fn output(&self) -> &RocmBuffer<T> {
        &self.output
    }

    pub(crate) fn matches(&self, input: &RocmBuffer<T>) -> bool {
        self.input.raw() == input.raw() && self.input.len() == input.len()
    }
}

/// Prepare `sqrt(Σ x²)` over a fixed strided device buffer.
///
/// # Errors
///
/// Returns an error when the input layout is invalid for its buffer or the
/// fixed device resources cannot be allocated.
pub fn prepare_norm_l2<'a, T, const N: usize>(
    device: &RocmDevice,
    input: StridedOperand<'a, T, N>,
) -> Result<PreparedL2Norm<'a, T, N>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    input
        .layout
        .validate_storage_len(input.buffer.len())
        .map_err(crate::application::linalg::map_layout_err)?;
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
    device: &RocmDevice,
    input: &'a RocmBuffer<T>,
) -> Result<PreparedL2Norm<'a, T, 1>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
