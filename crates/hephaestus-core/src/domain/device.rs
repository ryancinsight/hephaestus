use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;
use bytemuck::Pod;

/// The compute-device seam every accelerator backend implements.
///
/// This trait is a **deliberate extension seam** (atlas ADR 0001): the wgpu
/// backend and the CUDA backend (cuda-oxide + cutile composed) substitute here
/// without consumers changing. It is intentionally *not* sealed — new backend
/// crates in the hephaestus workspace implement it. Consumers bind generically
/// (`<D: ComputeDevice>`) so dispatch is monomorphized; no `dyn` on hot paths.
///
/// Element types are bounded by [`bytemuck::Pod`]: device transfer is a
/// byte-level copy, so only plain-old-data layouts are admissible, and the
/// bound makes that a compile-time contract instead of a runtime invariant.
pub trait ComputeDevice {
    /// The backend's typed device-buffer handle.
    type Buffer<T: Pod>: DeviceBuffer<T>;

    /// Human-readable backend identifier (e.g. `"wgpu"`).
    fn backend_name(&self) -> &'static str;

    /// Allocate a zero-initialized device buffer of `len` elements.
    fn alloc_zeroed<T: Pod>(&self, len: usize) -> Result<Self::Buffer<T>>;

    /// Allocate a device buffer initialized from a host slice (host→device).
    fn upload<T: Pod>(&self, host: &[T]) -> Result<Self::Buffer<T>>;

    /// Copy a device buffer's contents into a host slice (device→host).
    ///
    /// `out.len()` must equal the buffer's element count; backends reject a
    /// mismatch with [`HephaestusError::LengthMismatch`] before any transfer.
    ///
    /// [`HephaestusError::LengthMismatch`]: crate::domain::error::HephaestusError::LengthMismatch
    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()>;
}
