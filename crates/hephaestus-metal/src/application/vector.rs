//! Metal implementation of the dense vector-operation seam.

use hephaestus_core::{DenseVectorOps, Result};
use hephaestus_wgpu::{WgpuPreparedDot, WgpuPreparedNorm, WgpuVectorOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Prepared in-place vector kernels for one Metal-selected device.
///
/// Metal uses the existing WGPU dispatch substrate configured with the native
/// Metal backend. The bundle is constructed once so shader preparation stays
/// outside iterative solver operations.
pub struct MetalVectorOps {
    inner: WgpuVectorOps,
}

/// Prepared dot product bound to the allocations used during preparation.
pub struct MetalPreparedDot {
    inner: WgpuPreparedDot<f32>,
}

/// Prepared Euclidean norm bound to the allocation used during preparation.
pub struct MetalPreparedNorm {
    inner: WgpuPreparedNorm<f32>,
}

impl MetalVectorOps {
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

impl DenseVectorOps<MetalDevice, f32> for MetalVectorOps {
    type PreparedDot<'a>
        = MetalPreparedDot
    where
        Self: 'a;
    type PreparedNorm<'a>
        = MetalPreparedNorm
    where
        Self: 'a;

    fn copy_vector(
        &self,
        device: &MetalDevice,
        source: &MetalBuffer<f32>,
        target: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.inner
            .copy_vector(device.wgpu_device(), &source.inner, &target.inner)
    }

    fn scale_vector(
        &self,
        device: &MetalDevice,
        target: &MetalBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        self.inner
            .scale_vector(device.wgpu_device(), &target.inner, factor)
    }

    fn axpy(
        &self,
        device: &MetalDevice,
        target: &MetalBuffer<f32>,
        source: &MetalBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        self.inner
            .axpy(device.wgpu_device(), &target.inner, &source.inner, factor)
    }

    fn xpay(
        &self,
        device: &MetalDevice,
        target: &MetalBuffer<f32>,
        source: &MetalBuffer<f32>,
        factor: f32,
    ) -> Result<()> {
        self.inner
            .xpay(device.wgpu_device(), &target.inner, &source.inner, factor)
    }

    fn subtract_into(
        &self,
        device: &MetalDevice,
        left: &MetalBuffer<f32>,
        right: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
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
        left: &MetalBuffer<f32>,
        right: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
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
        left: &MetalBuffer<f32>,
        right: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
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
        left: &MetalBuffer<f32>,
        right: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
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
        left: &'a MetalBuffer<f32>,
        right: &'a MetalBuffer<f32>,
    ) -> Result<Self::PreparedDot<'a>> {
        Ok(MetalPreparedDot {
            inner: self
                .inner
                .prepare_dot(device.wgpu_device(), &left.inner, &right.inner)?,
        })
    }

    fn dot_prepared<'a>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &MetalBuffer<f32>,
        right: &MetalBuffer<f32>,
    ) -> Result<f32> {
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
        vector: &'a MetalBuffer<f32>,
    ) -> Result<Self::PreparedNorm<'a>> {
        Ok(MetalPreparedNorm {
            inner: self
                .inner
                .prepare_norm_l2(device.wgpu_device(), &vector.inner)?,
        })
    }

    fn norm_l2_prepared<'a>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &MetalBuffer<f32>,
    ) -> Result<f32> {
        self.inner
            .norm_l2_prepared(device.wgpu_device(), &prepared.inner, &vector.inner)
    }
}
