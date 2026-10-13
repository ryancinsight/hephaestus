use hephaestus_core::Result;
use hephaestus_wgpu as wgpu_backend;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// A reusable Metal scalar reduction plan delegated to WGPU's native Metal path.
pub struct PreparedReduction<T> {
    pub(crate) inner: wgpu_backend::PreparedReduction<T>,
}

impl<T> PreparedReduction<T> {
    /// Dispatch the prepared reduction and reuse its device-resident outputs.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when command encoding or submission
    /// fails on the Metal-selected device.
    pub fn dispatch(&self, device: &MetalDevice) -> Result<()> {
        self.inner.dispatch(&device.inner)
    }

    /// Return a handle to the one-element output buffer holding the latest result.
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

/// Submit several prepared Metal reductions in one native WGPU command batch.
///
/// # Errors
///
/// Returns a typed dispatch error when command encoding or submission fails.
pub fn submit_prepared_reduction_batch<T>(
    device: &MetalDevice,
    reductions: &[&PreparedReduction<T>],
) -> Result<()> {
    let inner = reductions
        .iter()
        .map(|reduction| &reduction.inner)
        .collect::<Vec<_>>();
    wgpu_backend::submit_prepared_reduction_batch(&device.inner, &inner)
}

/// A reusable Metal rank-2 axis reduction plan delegated to WGPU's native
/// Metal path.
pub struct PreparedAxisReduction<T> {
    pub(crate) inner: wgpu_backend::PreparedAxisReduction<T>,
}

impl<T> PreparedAxisReduction<T> {
    /// Dispatch the prepared axis reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when command encoding or submission
    /// fails on the Metal-selected device.
    pub fn dispatch(&self, device: &MetalDevice) -> Result<()> {
        self.inner.dispatch(&device.inner)
    }
}

/// Submit several prepared Metal axis reductions in one native WGPU command
/// batch.
///
/// # Errors
///
/// Returns a typed dispatch error when command encoding or submission fails.
pub fn submit_prepared_axis_reduction_batch<T>(
    device: &MetalDevice,
    reductions: &[&PreparedAxisReduction<T>],
) -> Result<()> {
    let inner = reductions
        .iter()
        .map(|reduction| &reduction.inner)
        .collect::<Vec<_>>();
    wgpu_backend::submit_prepared_axis_reduction_batch(&device.inner, &inner)
}
