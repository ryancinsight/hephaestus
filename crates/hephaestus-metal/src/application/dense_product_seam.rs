//! Provider-owned dense product seam for Metal.
//!
//! Metal delegates wholly to the WGPU implementation, matching the crate's
//! other seam adapters: buffers unwrap to their inner WGPU handles and every
//! method forwards to [`hephaestus_wgpu::WgpuDenseProductOps`] on the device's
//! WGPU handle.

use eunomia::Pod;
use hephaestus_core::{DenseProductOps, DialectScalar, Result, StridedView, Wgsl};
use hephaestus_wgpu::{MatmulZero, WgpuDenseProductOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`DenseProductOps`] for Metal.
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
