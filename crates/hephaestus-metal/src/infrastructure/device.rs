use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, Result};
use hephaestus_wgpu::WgpuDevice;

use crate::infrastructure::buffer::MetalBuffer;

/// An acquired Metal compute device.
///
/// Implemented by wrapping a WGPU device configured to run only on the Metal backend.
#[derive(Clone, Debug)]
pub struct MetalDevice {
    pub(crate) inner: WgpuDevice,
}

impl MetalDevice {
    /// Acquire the default Metal compute device.
    ///
    /// # Errors
    ///
    /// [`hephaestus_core::HephaestusError::AdapterUnavailable`] if no Metal adapter can be acquired.
    pub fn try_default() -> Result<Self> {
        let inner = WgpuDevice::try_metal("hephaestus-metal-device")?;
        Ok(Self { inner })
    }

    /// Access the underlying `WgpuDevice` context.
    #[must_use]
    #[inline]
    pub fn wgpu_device(&self) -> &WgpuDevice {
        &self.inner
    }

    /// Access the underlying `themis::GpuTopology` snapshot.
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.inner.topology()
    }
}

impl ComputeDevice for MetalDevice {
    type Buffer<T: Pod> = MetalBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "metal"
    }

    #[inline]
    fn alloc_zeroed<T: Pod>(&self, len: usize) -> Result<Self::Buffer<T>> {
        let inner = self.inner.alloc_zeroed(len)?;
        Ok(MetalBuffer { inner })
    }

    #[inline]
    fn upload<T: Pod>(&self, host: &[T]) -> Result<Self::Buffer<T>> {
        let inner = self.inner.upload(host)?;
        Ok(MetalBuffer { inner })
    }

    #[inline]
    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()> {
        self.inner.download(&buffer.inner, out)
    }

    #[inline]
    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()> {
        self.inner.write_buffer(&buffer.inner, host)
    }
}
