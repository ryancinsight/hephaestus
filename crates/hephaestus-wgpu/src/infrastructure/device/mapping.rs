//! Mapped-buffer lifetime and validated readback region metadata.

/// Cancels a pending map or releases an active mapping before pooled staging
/// storage can be recycled, including error and unwind exits.
pub(super) struct MappingLifecycle<'a>(pub(super) &'a wgpu::Buffer);

#[derive(Clone, Copy)]
pub(super) struct ReadbackRegion {
    pub(super) byte_offset: u64,
    pub(super) padded: u64,
    pub(super) byte_len: u64,
}

impl Drop for MappingLifecycle<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.unmap();
    }
}
