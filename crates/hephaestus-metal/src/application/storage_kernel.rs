//! Metal-selected storage-kernel dispatch.

use eunomia::Pod;
use hephaestus_core::{
    BinaryStorageKernel, DispatchGrid, MultiStorageDevice, MultiStorageKernel, Result,
    UnaryStorageKernel,
};
use hephaestus_wgpu::{
    WgslBinaryStorageKernel, WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
    WgslUnaryStorageKernel,
};

use crate::{MetalBuffer, MetalDevice};

/// Storage-buffer access declared in a Metal-selected WGSL layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalStorageAccess {
    /// Read-only storage buffer.
    ReadOnly,
    /// Read-write storage buffer.
    ReadWrite,
}

/// One storage binding in a Metal-selected WGSL kernel layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetalStorageBindingLayout {
    binding: u32,
    access: MetalStorageAccess,
}

impl MetalStorageBindingLayout {
    /// Construct a read-only storage binding layout entry.
    #[must_use]
    #[inline]
    pub const fn read_only(binding: u32) -> Self {
        Self {
            binding,
            access: MetalStorageAccess::ReadOnly,
        }
    }

    /// Construct a read-write storage binding layout entry.
    #[must_use]
    #[inline]
    pub const fn read_write(binding: u32) -> Self {
        Self {
            binding,
            access: MetalStorageAccess::ReadWrite,
        }
    }
}

/// One typed Metal storage-buffer binding for a multi-storage kernel.
#[derive(Clone, Copy, Debug)]
pub struct MetalStorageBinding<'a> {
    inner: WgslStorageBinding<'a>,
}

impl<'a> MetalStorageBinding<'a> {
    /// Bind a Metal buffer to a storage slot.
    #[must_use]
    #[inline]
    pub fn new<T: Pod>(binding: u32, buffer: &'a MetalBuffer<T>) -> Self {
        Self {
            inner: WgslStorageBinding::new(binding, buffer.wgpu_buffer()),
        }
    }
}

impl MultiStorageDevice for MetalDevice {
    type StorageBinding<'a> = MetalStorageBinding<'a>;

    fn storage_binding<T: Pod>(binding: u32, buffer: &Self::Buffer<T>) -> Self::StorageBinding<'_> {
        MetalStorageBinding::new(binding, buffer)
    }
}

/// Compiled WGSL storage kernel selected for the native Metal adapter.
#[derive(Debug)]
pub struct MetalMultiStorageKernel {
    inner: WgslMultiStorageKernel,
}

impl MetalMultiStorageKernel {
    /// Compile a Metal-selected multi-storage-buffer kernel.
    ///
    /// `storage_layouts` describes every storage binding in `@group(0)`;
    /// `params_binding` is the `@group(0)` uniform parameter binding.
    ///
    /// # Errors
    /// Returns [`hephaestus_core::HephaestusError::DispatchFailed`] when the entry point is
    /// empty, no storage bindings are declared, or binding numbers collide.
    pub fn new(
        device: &MetalDevice,
        label: &'static str,
        source: &'static str,
        entry_point: &'static str,
        storage_layouts: &[MetalStorageBindingLayout],
        params_binding: u32,
    ) -> Result<Self> {
        let wgpu_layouts: Vec<_> = storage_layouts
            .iter()
            .map(|layout| match layout.access {
                MetalStorageAccess::ReadOnly => WgslStorageBindingLayout::read_only(layout.binding),
                MetalStorageAccess::ReadWrite => {
                    WgslStorageBindingLayout::read_write(layout.binding)
                }
            })
            .collect();
        Ok(Self {
            inner: WgslMultiStorageKernel::new(
                device.wgpu_device(),
                label,
                source,
                entry_point,
                &wgpu_layouts,
                params_binding,
            )?,
        })
    }
}

impl<'a, P: Pod, const N: usize> MultiStorageKernel<MetalDevice, P, [MetalStorageBinding<'a>; N]>
    for MetalMultiStorageKernel
{
    fn dispatch(
        &self,
        device: &MetalDevice,
        bindings: [MetalStorageBinding<'a>; N],
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        self.inner.dispatch(
            device.wgpu_device(),
            bindings.map(|binding| binding.inner),
            params,
            grid,
        )
    }
}

/// Compiled WGSL unary storage kernel selected for the native Metal adapter.
#[derive(Debug)]
pub struct MetalUnaryStorageKernel {
    inner: WgslUnaryStorageKernel,
}

impl MetalUnaryStorageKernel {
    /// Compile a Metal-selected unary storage kernel.
    ///
    /// The WGSL module must expose read-only input, read-write output, and
    /// uniform parameter bindings at `@group(0)` bindings `0`, `1`, and `2`.
    ///
    /// # Errors
    /// Returns [`hephaestus_core::HephaestusError::DispatchFailed`] if `entry_point` is empty.
    pub fn new(
        device: &MetalDevice,
        label: &'static str,
        source: &'static str,
        entry_point: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            inner: WgslUnaryStorageKernel::new(device.wgpu_device(), label, source, entry_point)?,
        })
    }
}

impl<T: Pod, P: Pod> UnaryStorageKernel<MetalDevice, T, P> for MetalUnaryStorageKernel {
    fn dispatch(
        &self,
        device: &MetalDevice,
        input: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        self.inner.dispatch(
            device.wgpu_device(),
            input.wgpu_buffer(),
            output.wgpu_buffer(),
            params,
            grid,
        )
    }
}

/// Compiled WGSL binary storage kernel selected for the native Metal adapter.
#[derive(Debug)]
pub struct MetalBinaryStorageKernel {
    inner: WgslBinaryStorageKernel,
}

impl MetalBinaryStorageKernel {
    /// Compile a Metal-selected binary storage kernel.
    ///
    /// The WGSL module must expose read-only input bindings `0` and `1`, a
    /// read-write output binding `2`, and the parameter block at group `1`,
    /// binding `0`.
    ///
    /// # Errors
    /// Returns [`hephaestus_core::HephaestusError::DispatchFailed`] if `entry_point` is empty.
    pub fn new(
        device: &MetalDevice,
        label: &'static str,
        source: &'static str,
        entry_point: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            inner: WgslBinaryStorageKernel::new(device.wgpu_device(), label, source, entry_point)?,
        })
    }
}

impl<T: Pod, P: Pod> BinaryStorageKernel<MetalDevice, T, P> for MetalBinaryStorageKernel {
    fn dispatch(
        &self,
        device: &MetalDevice,
        left: &MetalBuffer<T>,
        right: &MetalBuffer<T>,
        output: &MetalBuffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        self.inner.dispatch(
            device.wgpu_device(),
            left.wgpu_buffer(),
            right.wgpu_buffer(),
            output.wgpu_buffer(),
            params,
            grid,
        )
    }
}
