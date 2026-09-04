//! Prepared dot-product and L2-norm delegation for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, IdentityToken, OpIdentity, Result, SumOp, Wgsl};
use hephaestus_wgpu as wgpu_backend;

use crate::application::strided::{StridedOperand, to_wgpu_strided};
use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// A reusable Metal vector dot-product plan over fixed strided inputs.
pub struct PreparedDot<T> {
    inner: wgpu_backend::PreparedDot<T>,
}

impl<T> PreparedDot<T> {
    /// Dispatch the prepared dot product on the Metal-selected WGPU device.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when command encoding or submission
    /// fails.
    pub fn dispatch(&self, device: &MetalDevice) -> Result<()> {
        self.inner.dispatch(&device.inner)
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> MetalBuffer<T>
    where
        T: Clone,
    {
        MetalBuffer {
            inner: self.inner.output().clone(),
        }
    }
}

/// Prepare `Σᵢ lhs[i] * rhs[i]` over fixed strided device buffers.
///
/// # Errors
///
/// Returns an error when the logical shapes or input layouts are invalid, or
/// when Metal/WGPU resources cannot be allocated.
pub fn prepare_dot<'a, T>(
    device: &MetalDevice,
    lhs: StridedOperand<'a, T, 1>,
    rhs: StridedOperand<'a, T, 1>,
) -> Result<PreparedDot<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    let inner =
        wgpu_backend::prepare_dot(&device.inner, to_wgpu_strided(lhs), to_wgpu_strided(rhs))?;
    Ok(PreparedDot { inner })
}

/// A reusable Metal L2-norm plan over a fixed strided input.
pub struct PreparedL2Norm<T> {
    inner: wgpu_backend::PreparedL2Norm<T>,
}

impl<T> PreparedL2Norm<T> {
    /// Dispatch the prepared `sqrt(Σ x²)` operation on Metal.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when command encoding or submission
    /// fails.
    pub fn dispatch(&self, device: &MetalDevice) -> Result<()> {
        self.inner.dispatch(&device.inner)
    }

    /// Return the stable one-element output buffer.
    #[must_use]
    pub fn output(&self) -> MetalBuffer<T>
    where
        T: Clone,
    {
        MetalBuffer {
            inner: self.inner.output().clone(),
        }
    }
}

/// Prepare `sqrt(Σ x²)` over a fixed strided device buffer.
///
/// # Errors
///
/// Returns an error when the input layout is invalid, or when Metal/WGPU
/// resources cannot be allocated.
pub fn prepare_norm_l2<'a, T, const N: usize>(
    device: &MetalDevice,
    input: StridedOperand<'a, T, N>,
) -> Result<PreparedL2Norm<T>>
where
    T: wgpu_backend::L2NormScalar,
{
    let inner = wgpu_backend::prepare_norm_l2(&device.inner, to_wgpu_strided(input))?;
    Ok(PreparedL2Norm { inner })
}
