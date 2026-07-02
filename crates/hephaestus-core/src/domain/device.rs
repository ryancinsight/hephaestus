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
    #[inline]
    fn alloc_zeroed<T: Pod>(&self, len: usize) -> Result<Self::Buffer<T>> {
        self.alloc_zeroed_with_hint(len, themis::PlacementHint::default())
    }

    /// Allocate a zero-initialized device buffer of `len` elements with a placement hint.
    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>>;

    /// Allocate a device buffer initialized from a host slice (host→device).
    #[inline]
    fn upload<T: Pod>(&self, host: &[T]) -> Result<Self::Buffer<T>> {
        self.upload_with_hint(host, themis::PlacementHint::default())
    }

    /// Allocate a device buffer initialized from a host slice with a placement hint (host→device).
    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>>;

    /// Copy a device buffer's contents into a host slice (device→host).
    ///
    /// `out.len()` must equal the buffer's element count; backends reject a
    /// mismatch with [`HephaestusError::LengthMismatch`] before any transfer.
    ///
    /// [`HephaestusError::LengthMismatch`]: crate::domain::error::HephaestusError::LengthMismatch
    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()>;

    /// Overwrite an existing device buffer with host data (host→device).
    ///
    /// Unlike [`upload`](Self::upload) which allocates a new buffer, this
    /// writes into an already-allocated buffer.  `host.len()` must equal the
    /// buffer's element count.
    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()>;

    /// Wait until previously submitted work and transfers visible to this
    /// device context have completed.
    ///
    /// Backends map this to their real synchronization primitive (`Device::poll`
    /// for WGPU, `cuCtxSynchronize` for CUDA). Consumers use this for explicit
    /// blocking semantics without depending on a concrete GPU API.
    fn synchronize(&self) -> Result<()>;
}

/// Validate that a buffer size is a multiple of 4 bytes.
#[inline]
pub fn validate_buffer_size<T>(len: usize) -> Result<()> {
    let byte_len = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        crate::domain::error::HephaestusError::AllocationFailed {
            message: format!(
                "Buffer byte size calculation overflows (elements: {}, element size: {})",
                len,
                core::mem::size_of::<T>()
            ),
        }
    })?;
    if !byte_len.is_multiple_of(4) {
        return Err(crate::domain::error::HephaestusError::AllocationFailed {
            message: format!(
                "Buffer byte size {} (elements: {}, element size: {}) must be a multiple of 4 bytes for GPU compatibility",
                byte_len, len, core::mem::size_of::<T>()
            ),
        });
    }
    Ok(())
}

/// Validate that a host slice satisfies both size and pointer 4-byte alignment.
#[inline]
pub fn validate_slice_alignment<T>(slice: &[T]) -> Result<()> {
    let byte_len = core::mem::size_of_val(slice);
    if !byte_len.is_multiple_of(4) {
        return Err(crate::domain::error::HephaestusError::TransferFailed {
            message: format!(
                "Transfer byte length {} (elements: {}, element size: {}) must be a multiple of 4 bytes for GPU compatibility",
                byte_len, slice.len(), core::mem::size_of::<T>()
            ),
        });
    }
    let addr = slice.as_ptr() as usize;
    if !addr.is_multiple_of(4) {
        return Err(crate::domain::error::HephaestusError::TransferFailed {
            message: format!(
                "Transfer host memory address 0x{:x} is not 4-byte aligned",
                addr
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_buffer_size_overflow() {
        assert!(validate_buffer_size::<f32>(usize::MAX).is_err());
    }

    #[test]
    fn test_validate_buffer_size_alignment() {
        assert!(validate_buffer_size::<u8>(4).is_ok());
        assert!(validate_buffer_size::<u8>(3).is_err());
    }
}
