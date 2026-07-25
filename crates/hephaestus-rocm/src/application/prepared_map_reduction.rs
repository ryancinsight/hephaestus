//! Reusable ROCm dot-product and L2-norm plans.

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, DeviceBuffer, DialectScalar, HipC, IdentityToken, OpIdentity, Result, SumOp,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::elementwise::{MulOp, SqrtOp, unary_elementwise_into};
use crate::application::prepared_reduction::PreparedReductionPlan;
use crate::application::strided::{StridedOperand, binary_elementwise_strided_into};
use crate::infrastructure::RocmBuffer;

/// A reusable ROCm vector dot-product plan over fixed strided inputs.
pub struct PreparedDot<'a, T> {
    device: &'a RocmDevice,
    lhs: StridedOperand<'a, T, 1>,
    rhs: StridedOperand<'a, T, 1>,
    product: RocmBuffer<T>,
    product_layout: Layout<1>,
    reduction: PreparedReductionPlan<'a, T>,
}

impl<T> PreparedDot<'_, T>
where
    T: DialectScalar<HipC> + Pod,
{
    /// Dispatch the prepared dot product and reuse its device-resident output.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a native elementwise or reduction launch
    /// fails.
    pub fn dispatch(&self) -> Result<()> {
        let product = StridedOperand {
            buffer: &self.product,
            layout: &self.product_layout,
        };
        binary_elementwise_strided_into::<MulOp, T, 1>(
            self.device,
            self.lhs,
            self.rhs,
            product,
            BlockWidth::DEFAULT,
        )?;
        self.reduction.dispatch(&self.product)
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> &RocmBuffer<T> {
        self.reduction.output()
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
    let len = lhs
        .layout
        .checked_size()
        .map_err(crate::application::linalg::map_layout_err)?;
    let product = device.alloc_zeroed::<T>(len)?;
    let product_layout =
        Layout::c_contiguous([len]).map_err(crate::application::linalg::map_layout_err)?;
    let reduction = PreparedReductionPlan::prepare::<SumOp>(device, len, BlockWidth::DEFAULT)?;
    Ok(PreparedDot {
        device,
        lhs,
        rhs,
        product,
        product_layout,
        reduction,
    })
}

/// A reusable ROCm L2-norm plan over a fixed strided input.
pub struct PreparedL2Norm<'a, T, const N: usize> {
    device: &'a RocmDevice,
    input: StridedOperand<'a, T, N>,
    squares: RocmBuffer<T>,
    squares_layout: Layout<N>,
    reduction: PreparedReductionPlan<'a, T>,
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
        let squares = StridedOperand {
            buffer: &self.squares,
            layout: &self.squares_layout,
        };
        binary_elementwise_strided_into::<MulOp, T, N>(
            self.device,
            self.input,
            self.input,
            squares,
            BlockWidth::DEFAULT,
        )?;
        self.reduction.dispatch(&self.squares)?;
        unary_elementwise_into::<SqrtOp, T>(
            self.device,
            self.reduction.output(),
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
    let len = input
        .layout
        .checked_size()
        .map_err(crate::application::linalg::map_layout_err)?;
    let squares = device.alloc_zeroed::<T>(len)?;
    let squares_layout = Layout::c_contiguous(input.layout.shape)
        .map_err(crate::application::linalg::map_layout_err)?;
    let reduction = PreparedReductionPlan::prepare::<SumOp>(device, len, BlockWidth::DEFAULT)?;
    let output = device.alloc_zeroed::<T>(1)?;
    Ok(PreparedL2Norm {
        device,
        input,
        squares,
        squares_layout,
        reduction,
        output,
    })
}
