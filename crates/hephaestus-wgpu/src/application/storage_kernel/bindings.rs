use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::MultiStorageDevice;

/// Storage-buffer access declared in a WGSL bind-group layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WgslStorageAccess {
    /// Read-only storage buffer.
    ReadOnly,
    /// Read-write storage buffer.
    ReadWrite,
}

impl WgslStorageAccess {
    #[inline]
    pub(crate) const fn read_only(self) -> bool {
        match self {
            Self::ReadOnly => true,
            Self::ReadWrite => false,
        }
    }
}

/// One storage binding in a WGSL kernel layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WgslStorageBindingLayout {
    pub(crate) binding: u32,
    pub(crate) access: WgslStorageAccess,
}

impl WgslStorageBindingLayout {
    /// Construct a read-only storage binding layout entry.
    #[must_use]
    #[inline]
    pub const fn read_only(binding: u32) -> Self {
        Self {
            binding,
            access: WgslStorageAccess::ReadOnly,
        }
    }

    /// Construct a read-write storage binding layout entry.
    #[must_use]
    #[inline]
    pub const fn read_write(binding: u32) -> Self {
        Self {
            binding,
            access: WgslStorageAccess::ReadWrite,
        }
    }
}

/// One typed WGPU storage-buffer binding for a multi-storage WGSL kernel.
#[derive(Clone, Copy, Debug)]
pub struct WgslStorageBinding<'a> {
    pub(crate) binding: u32,
    pub(crate) buffer: &'a wgpu::Buffer,
}

impl<'a> WgslStorageBinding<'a> {
    /// Bind `buffer` to a WGSL storage binding.
    #[must_use]
    #[inline]
    pub fn new<T: Pod>(binding: u32, buffer: &'a WgpuBuffer<T>) -> Self {
        Self {
            binding,
            buffer: buffer.raw(),
        }
    }
}

impl MultiStorageDevice for WgpuDevice {
    type StorageBinding<'a> = WgslStorageBinding<'a>;

    fn storage_binding<T: Pod>(binding: u32, buffer: &Self::Buffer<T>) -> Self::StorageBinding<'_> {
        WgslStorageBinding::new(binding, buffer)
    }
}
