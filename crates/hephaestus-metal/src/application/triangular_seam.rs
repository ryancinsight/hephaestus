//! Provider-owned rank-2 triangular masking seam for Metal.

use hephaestus_core::{Result, StridedView, TriangularMode, TriangularOps};
use hephaestus_wgpu::{TriangularStorage, WgpuTriangularOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`TriangularOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalTriangularOps {
    inner: WgpuTriangularOps,
}

impl<T> TriangularOps<MetalDevice, T> for MetalTriangularOps
where
    T: TriangularStorage,
{
    fn triangular_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        mode: TriangularMode,
        diagonal: i64,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()> {
        self.inner.triangular_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            mode,
            diagonal,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
