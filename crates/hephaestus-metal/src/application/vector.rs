//! Metal implementation of the dense vector-operation seam.

use hephaestus_core::{
    DenseVectorOps, IdentityToken, MaxOp, OpIdentity, Result, VectorFactor, VectorScalar, Wgsl,
};
use hephaestus_wgpu::{L2NormScalar, WgpuPreparedDot, WgpuPreparedNorm, WgpuVectorOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Prepared in-place vector kernels for one Metal-selected device.
///
/// Metal uses the existing WGPU dispatch substrate configured with the native
/// Metal backend. The bundle is constructed once so shader preparation stays
/// outside iterative solver operations.
pub struct MetalVectorOps<T> {
    inner: WgpuVectorOps<T>,
}

/// Prepared dot product bound to the allocations used during preparation.
pub struct MetalPreparedDot<T> {
    inner: WgpuPreparedDot<T>,
}

/// Prepared Euclidean norm bound to the allocation used during preparation.
pub struct MetalPreparedNorm<T> {
    inner: WgpuPreparedNorm<T>,
}

impl<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>> MetalVectorOps<T> {
    /// Compile the dense vector kernels on the Metal-selected device.
    ///
    /// # Errors
    ///
    /// Returns a shader preparation or device failure.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        Ok(Self {
            inner: WgpuVectorOps::new(device.wgpu_device())?,
        })
    }
}

impl<T> DenseVectorOps<MetalDevice, T> for MetalVectorOps<T>
where
    T: VectorScalar<Wgsl>
        + VectorFactor<Wgsl>
        + L2NormScalar
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Wgsl>,
{
    type PreparedDot<'a>
        = MetalPreparedDot<T>
    where
        Self: 'a;
    type PreparedNorm<'a>
        = MetalPreparedNorm<T>
    where
        Self: 'a;

    fn copy_vector(
        &self,
        device: &MetalDevice,
        source: &MetalBuffer<T>,
        target: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner
            .copy_vector(device.wgpu_device(), &source.inner, &target.inner)
    }

    fn scale_vector(&self, device: &MetalDevice, target: &MetalBuffer<T>, factor: T) -> Result<()> {
        self.inner
            .scale_vector(device.wgpu_device(), &target.inner, factor)
    }

    fn axpy(
        &self,
        device: &MetalDevice,
        target: &MetalBuffer<T>,
        source: &MetalBuffer<T>,
        factor: T,
    ) -> Result<()> {
        self.inner
            .axpy(device.wgpu_device(), &target.inner, &source.inner, factor)
    }

    fn xpay(
        &self,
        device: &MetalDevice,
        target: &MetalBuffer<T>,
        source: &MetalBuffer<T>,
        factor: T,
    ) -> Result<()> {
        self.inner
            .xpay(device.wgpu_device(), &target.inner, &source.inner, factor)
    }

    fn subtract_into(
        &self,
        device: &MetalDevice,
        left: &MetalBuffer<T>,
        right: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.subtract_into(
            device.wgpu_device(),
            &left.inner,
            &right.inner,
            &output.inner,
        )
    }

    fn add_into(
        &self,
        device: &MetalDevice,
        left: &MetalBuffer<T>,
        right: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.add_into(
            device.wgpu_device(),
            &left.inner,
            &right.inner,
            &output.inner,
        )
    }

    fn multiply_into(
        &self,
        device: &MetalDevice,
        left: &MetalBuffer<T>,
        right: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.multiply_into(
            device.wgpu_device(),
            &left.inner,
            &right.inner,
            &output.inner,
        )
    }

    fn divide_into(
        &self,
        device: &MetalDevice,
        left: &MetalBuffer<T>,
        right: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.divide_into(
            device.wgpu_device(),
            &left.inner,
            &right.inner,
            &output.inner,
        )
    }

    fn prepare_dot<'a>(
        &self,
        device: &MetalDevice,
        left: &'a MetalBuffer<T>,
        right: &'a MetalBuffer<T>,
    ) -> Result<Self::PreparedDot<'a>> {
        Ok(MetalPreparedDot::<T> {
            inner: self
                .inner
                .prepare_dot(device.wgpu_device(), &left.inner, &right.inner)?,
        })
    }

    fn dot_prepared<'a>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &MetalBuffer<T>,
        right: &MetalBuffer<T>,
    ) -> Result<T> {
        self.inner.dot_prepared(
            device.wgpu_device(),
            &prepared.inner,
            &left.inner,
            &right.inner,
        )
    }

    fn prepare_norm_l2<'a>(
        &self,
        device: &MetalDevice,
        vector: &'a MetalBuffer<T>,
    ) -> Result<Self::PreparedNorm<'a>> {
        Ok(MetalPreparedNorm::<T> {
            inner: self
                .inner
                .prepare_norm_l2(device.wgpu_device(), &vector.inner)?,
        })
    }

    fn norm_l1(&self, device: &MetalDevice, vector: &MetalBuffer<T>) -> Result<T> {
        self.inner.norm_l1(device.wgpu_device(), &vector.inner)
    }

    fn norm_max(&self, device: &MetalDevice, vector: &MetalBuffer<T>) -> Result<T> {
        self.inner.norm_max(device.wgpu_device(), &vector.inner)
    }

    fn norm_l2_prepared<'a>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &MetalBuffer<T>,
    ) -> Result<T> {
        self.inner
            .norm_l2_prepared(device.wgpu_device(), &prepared.inner, &vector.inner)
    }
}
