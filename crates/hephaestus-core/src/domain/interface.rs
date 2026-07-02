//! Backend-neutral kernel authoring: interface and source declarations.
//!
//! A custom kernel is authored as a type implementing [`KernelInterface`](crate::KernelInterface)
//! (its binding layout and POD parameter block — dialect-free, declared once)
//! plus [`KernelSource<L>`](crate::KernelSource) for each [`KernelDialect`](crate::KernelDialect) it targets (entry
//! point and source text). A backend accepts only kernels implementing its
//! own dialect, so dispatching a WGSL-only kernel on the CUDA backend is a
//! compile error rather than a runtime failure. Dispatch mechanics live in
//! [`stream`](crate::domain::stream).

use super::dialect::KernelDialect;
use bytemuck::Pod;
use std::borrow::Cow;

/// Storage-binding access mode declared by a kernel interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// Read-only storage binding.
    ReadOnly,
    /// Read-write storage binding.
    ReadWrite,
}

/// One storage binding declared by a [`KernelInterface`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingDecl {
    /// Access mode the kernel source declares for this binding.
    pub access: Access,
    /// Element size in bytes of the buffer bound here.
    pub elem_size: usize,
}

impl BindingDecl {
    /// Declare a read-only binding of element type `T`.
    #[must_use]
    pub const fn read_only<T: Pod>() -> Self {
        Self {
            access: Access::ReadOnly,
            elem_size: core::mem::size_of::<T>(),
        }
    }

    /// Declare a read-write binding of element type `T`.
    #[must_use]
    pub const fn read_write<T: Pod>() -> Self {
        Self {
            access: Access::ReadWrite,
            elem_size: core::mem::size_of::<T>(),
        }
    }
}

/// Dialect-free kernel interface: binding layout, parameter block, and
/// launch tile shape. Declared once per kernel; dialect sources implement
/// [`KernelSource`] on the same type.
pub trait KernelInterface {
    /// POD parameter block uploaded per dispatch (uniform buffer, push
    /// constants, or native kernel arguments — backend's choice).
    type Params: Pod;

    /// Diagnostic label (also the backend pipeline label).
    const LABEL: &'static str;

    /// Ordered storage-binding declarations. Binding order in the kernel
    /// source must match slice order; the parameter block occupies whatever
    /// slot the backend's ABI assigns and is not listed here.
    const BINDINGS: &'static [BindingDecl];

    /// Workgroup/thread-block tile shape the source declares. Grids from
    /// [`DispatchGrid::covering_domain`](super::kernel::DispatchGrid::covering_domain)
    /// must be derived with this shape; CUDA launches use it as block
    /// dimensions.
    const WORKGROUP: [u32; 3];

    /// Dynamic shared-memory bytes per workgroup (CUDA `extern __shared__`;
    /// WGSL declares workgroup memory statically in source, so this stays 0
    /// for WGSL-only kernels).
    const SHARED_BYTES: u32 = 0;
}

/// Kernel source text in dialect `L` for a [`KernelInterface`].
pub trait KernelSource<L: KernelDialect>: KernelInterface {
    /// Entry-point (function) name in the source.
    const ENTRY: &'static str;

    /// The kernel source. `Cow` so static sources allocate nothing and
    /// generated sources (scalar-token substitution) build once per cache
    /// key.
    fn source(&self) -> Cow<'static, str>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_decl_captures_access_and_elem_size() {
        let read = BindingDecl::read_only::<f32>();
        assert_eq!(read.access, Access::ReadOnly);
        assert_eq!(read.elem_size, 4);
        let write = BindingDecl::read_write::<u32>();
        assert_eq!(write.access, Access::ReadWrite);
        assert_eq!(write.elem_size, 4);
    }
}
