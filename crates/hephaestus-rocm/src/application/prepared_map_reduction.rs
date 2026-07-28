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

use crate::RocmDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::linalg::{DotMap, PreparedMapReduction, SquareMap, prepare_map_reduction};
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// A reusable ROCm vector dot-product plan over fixed strided inputs.
pub struct PreparedDot<'a, T> {
    inner: PreparedMapReduction<'a, DotMap, T, 1>,
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
}

/// Prepare `Σᵢ lhs[i] * rhs[i]` over fixed strided device buffers.
///
/// # Errors
///
/// Returns an error when the logical shapes differ, either input layout is
/// invalid for its buffer, or the fixed device resources cannot be allocated.
pub fn prepare_dot<'a, T>(
    device: &'a RocmDevice,
    lhs: StridedOperand<'a, T, 1>,
    rhs: StridedOperand<'a, T, 1>,
) -> Result<PreparedDot<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    if lhs.layout.shape != rhs.layout.shape {
        return Err(hephaestus_core::HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                lhs.layout.shape, rhs.layout.shape
            ),
        });
    }
    lhs.layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(crate::application::linalg::map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(crate::application::linalg::map_layout_err)?;
    let inner = prepare_map_reduction::<DotMap, T, 1>(device, lhs, rhs)?;
    Ok(PreparedDot { inner })
}

/// A reusable ROCm L2-norm plan over a fixed strided input.
pub struct PreparedL2Norm<'a, T, const N: usize> {
    device: &'a RocmDevice,
    inner: PreparedMapReduction<'a, SquareMap, T, N>,
    output: RocmBuffer<T>,
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
            self.device,
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
}

/// Prepare `sqrt(Σ x²)` over a fixed strided device buffer.
///
/// # Errors
///
/// Returns an error when the input layout is invalid for its buffer or the
/// fixed device resources cannot be allocated.
pub fn prepare_norm_l2<'a, T, const N: usize>(
    device: &'a RocmDevice,
    input: StridedOperand<'a, T, N>,
) -> Result<PreparedL2Norm<'a, T, N>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    input
        .layout
        .validate_storage_len(input.buffer.len())
        .map_err(crate::application::linalg::map_layout_err)?;
    let inner = prepare_map_reduction::<SquareMap, T, N>(device, input, input)?;
    let output = device.alloc_zeroed::<T>(1)?;
    Ok(PreparedL2Norm {
        device,
        inner,
        output,
    })
}
