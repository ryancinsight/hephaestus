use bytemuck::Pod;
use hephaestus_core::{
    ComputeDevice, ComputeDeviceCapabilities, DeviceFeature, DeviceLimits, Result,
};
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
    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let inner = self.inner.alloc_zeroed_with_hint(len, hint)?;
        Ok(MetalBuffer { inner })
    }

    #[inline]
    fn alloc_uninitialized_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let inner = self.inner.alloc_uninitialized_with_hint(len, hint)?;
        Ok(MetalBuffer { inner })
    }

    #[inline]
    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let inner = self.inner.upload_with_hint(host, hint)?;
        Ok(MetalBuffer { inner })
    }

    #[inline]
    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()> {
        self.inner.download(&buffer.inner, out)
    }

    #[inline]
    fn download_owned<T: Pod>(&self, buffer: &Self::Buffer<T>) -> Result<Vec<T>> {
        self.inner.download_owned(&buffer.inner)
    }

    #[inline]
    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()> {
        self.inner.write_buffer(&buffer.inner, host)
    }

    #[inline]
    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        self.inner.write_sub_buffer(&buffer.inner, offset, host)
    }

    #[inline]
    fn copy_buffer<T: Pod>(&self, src: &Self::Buffer<T>, dst: &Self::Buffer<T>) -> Result<()> {
        self.inner.copy_buffer(&src.inner, &dst.inner)
    }

    #[inline]
    fn topology(&self) -> Option<&themis::GpuTopology> {
        MetalDevice::topology(self)
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }
}

impl ComputeDeviceCapabilities for MetalDevice {
    #[inline]
    fn device_limits(&self) -> DeviceLimits {
        ComputeDeviceCapabilities::device_limits(&self.inner)
    }

    #[inline]
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        ComputeDeviceCapabilities::supports_device_feature(&self.inner, feature)
    }
}
