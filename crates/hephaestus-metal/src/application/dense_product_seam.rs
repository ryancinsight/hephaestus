//! Provider-owned dense product, composition, and matrix-function seams for
//! Metal.
//!
//! Metal delegates wholly to the WGPU implementation, matching the crate's
//! other seam adapters: buffers unwrap to their inner WGPU handles and every
//! method forwards to [`hephaestus_wgpu::WgpuDenseProductOps`] on the device's
//! WGPU handle.

use eunomia::Pod;
use hephaestus_core::{
    DenseCompositionOps, DenseMatrixFunctionOps, DenseProductOps, DialectScalar, Result,
    StridedView, Wgsl,
};
use hephaestus_wgpu::{MatmulZero, WgpuDenseProductOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned dense product, composition, and matrix-function implementation
/// for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalDenseProductOps {
    inner: WgpuDenseProductOps,
}

impl<T> DenseProductOps<MetalDevice, T> for MetalDenseProductOps
where
    T: DialectScalar<Wgsl> + Pod + MatmulZero,
{
    fn matmul_into(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'_, MetalBuffer<T>, 2>,
        rhs: StridedView<'_, MetalBuffer<T>, 2>,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()> {
        self.inner.matmul_into(
            device.wgpu_device(),
            StridedView::new(&lhs.buffer.inner, lhs.layout),
            StridedView::new(&rhs.buffer.inner, rhs.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn batched_matmul_into(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'_, MetalBuffer<T>, 3>,
        rhs: StridedView<'_, MetalBuffer<T>, 3>,
        output: StridedView<'_, MetalBuffer<T>, 3>,
    ) -> Result<()> {
        self.inner.batched_matmul_into(
            device.wgpu_device(),
            StridedView::new(&lhs.buffer.inner, lhs.layout),
            StridedView::new(&rhs.buffer.inner, rhs.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn kron_into(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'_, MetalBuffer<T>, 2>,
        rhs: StridedView<'_, MetalBuffer<T>, 2>,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()> {
        self.inner.kron_into(
            device.wgpu_device(),
            StridedView::new(&lhs.buffer.inner, lhs.layout),
            StridedView::new(&rhs.buffer.inner, rhs.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}

impl DenseCompositionOps<MetalDevice> for MetalDenseProductOps {
    fn matpow(
        &self,
        device: &MetalDevice,
        matrix: StridedView<'_, MetalBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .matpow(
                device.wgpu_device(),
                StridedView::new(&matrix.buffer.inner, matrix.layout),
                exponent,
            )
            .map(|inner| MetalBuffer { inner })
    }

    fn det(
        &self,
        device: &MetalDevice,
        matrix: StridedView<'_, MetalBuffer<f32>, 2>,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .det(
                device.wgpu_device(),
                StridedView::new(&matrix.buffer.inner, matrix.layout),
            )
            .map(|inner| MetalBuffer { inner })
    }

    fn matrix_rank_with_tolerance(
        &self,
        device: &MetalDevice,
        matrix: StridedView<'_, MetalBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        self.inner.matrix_rank_with_tolerance(
            device.wgpu_device(),
            StridedView::new(&matrix.buffer.inner, matrix.layout),
            relative_tolerance,
        )
    }
}

impl DenseMatrixFunctionOps<MetalDevice> for MetalDenseProductOps {
    fn matexp(
        &self,
        device: &MetalDevice,
        matrix: StridedView<'_, MetalBuffer<f32>, 2>,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .matexp(
                device.wgpu_device(),
                StridedView::new(&matrix.buffer.inner, matrix.layout),
            )
            .map(|inner| MetalBuffer { inner })
    }

    fn pinv(
        &self,
        device: &MetalDevice,
        matrix: StridedView<'_, MetalBuffer<f32>, 2>,
    ) -> Result<MetalBuffer<f32>> {
        self.inner
            .pinv(
                device.wgpu_device(),
                StridedView::new(&matrix.buffer.inner, matrix.layout),
            )
            .map(|inner| MetalBuffer { inner })
    }
}
