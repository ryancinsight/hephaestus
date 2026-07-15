//! Backend dispatch seam for authored kernels: prepared pipelines, typed
//! bindings, and multi-pass command streams.
//!
//! [`KernelDevice`](crate::KernelDevice) extends [`ComputeDevice`](crate::ComputeDevice) with the compile/dispatch
//! surface for consumer-authored kernels ([`KernelInterface`](crate::KernelInterface) +
//! [`KernelSource`](crate::KernelSource)): `prepare` compiles-and-caches, [`CommandStream`](crate::CommandStream)
//! records ordered dispatches, and `submit` queues the recorded work. Backends
//! expose both barrier-separated command streams and grouped same-region
//! sequencing for kernels that must remain inside one backend dispatch region
//! (for example one WGPU compute pass or one CUDA stream launch sequence).
//! Completion is observed through
//! [`ComputeDevice::synchronize`](crate::ComputeDevice::synchronize) or a synchronizing transfer; no async
//! surface leaks into core (async-contagion rule).
//!
//! Bindings are typed at construction ([`Binding::read`](crate::Binding::read) /
//! [`Binding::read_write`](crate::Binding::read_write) borrow a `D::Buffer<T>`) and erase to the
//! backend's [`KernelDevice::BindingHandle`](crate::KernelDevice::BindingHandle) plus element-size/length
//! metadata, so one homogeneous `&[Binding<'_, D>]` slice carries
//! heterogeneous element types with no trait-object indirection. Arity,
//! access, and element size are validated value-semantically against the
//! kernel's [`KernelInterface::BINDINGS`](crate::KernelInterface::BINDINGS) declaration at encode
//! ([`validate_bindings`](crate::validate_bindings)).

use super::device::ComputeDevice;
use super::dialect::KernelDialect;
use super::error::{HephaestusError, Result};
use super::interface::{BindingDecl, GroupedBindingDecl, GroupedKernelSource, KernelSource};
use super::kernel::DispatchGrid;
use bytemuck::Pod;

pub use super::interface::Access;

/// A compute device that compiles and dispatches consumer-authored kernels
/// in its source dialect.
pub trait KernelDevice: ComputeDevice {
    /// The kernel-source dialect this backend compiles.
    type Dialect: KernelDialect;

    /// Erased per-binding buffer handle (e.g. a `&wgpu::Buffer`, a CUDA
    /// device pointer). `Copy` so binding slices stay by-value cheap.
    type BindingHandle<'a>: Copy
    where
        Self: 'a;

    /// Compiled kernel handle for `K`, cached by the backend so `prepare`
    /// is cache-hit cheap after first use. Cheap to clone.
    type Prepared<K: KernelSource<Self::Dialect>>: Clone;

    /// Multi-pass recording stream. Encoded passes execute in order with
    /// inter-pass barrier semantics.
    type Stream<'d>: CommandStream<'d, Self>
    where
        Self: 'd;

    /// Erase a typed buffer borrow to this backend's binding handle.
    fn binding_handle<T: Pod>(buffer: &Self::Buffer<T>) -> Self::BindingHandle<'_>;

    /// Compile (or fetch from cache) the pipeline/module for `kernel`.
    ///
    /// # Errors
    /// Returns the backend's typed failure when source compilation or
    /// pipeline creation fails; failed compilations are not cached.
    fn prepare<K: KernelSource<Self::Dialect>>(&self, kernel: &K) -> Result<Self::Prepared<K>>;

    /// Open a command stream for multi-pass recording.
    ///
    /// # Errors
    /// Returns the backend's typed failure when command-recording resources
    /// cannot be acquired.
    fn stream(&self) -> Result<Self::Stream<'_>>;

    /// One-shot dispatch: open a stream, encode one pass, submit.
    ///
    /// # Errors
    /// Propagates encode/submit failures ([`HephaestusError::DispatchFailed`]
    /// on binding-layout mismatch).
    fn dispatch<K: KernelSource<Self::Dialect>>(
        &self,
        prepared: &Self::Prepared<K>,
        bindings: &[Binding<'_, Self>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()>
    where
        Self: Sized,
    {
        let mut stream = self.stream()?;
        stream.encode(prepared, bindings, params, grid)?;
        stream.submit()
    }
}

/// A compute device that compiles and dispatches grouped consumer-authored
/// kernels in its source dialect.
pub trait GroupedKernelDevice: KernelDevice {
    /// Compiled grouped kernel handle for `K`, cached by the backend so
    /// `prepare_grouped` is cache-hit cheap after first use. Cheap to clone.
    type GroupedPrepared<K: GroupedKernelSource<Self::Dialect>>: Clone;

    /// Multi-pass grouped recording stream.
    type GroupedStream<'d>: GroupedCommandStream<'d, Self>
    where
        Self: 'd;

    /// Compile (or fetch from cache) the grouped pipeline/module for `kernel`.
    ///
    /// # Errors
    /// Returns the backend's typed failure when source compilation, grouped
    /// layout construction, or pipeline creation fails.
    fn prepare_grouped<K: GroupedKernelSource<Self::Dialect>>(
        &self,
        kernel: &K,
    ) -> Result<Self::GroupedPrepared<K>>;

    /// Open a grouped command stream for multi-pass recording.
    ///
    /// # Errors
    /// Returns the backend's typed failure when command-recording resources
    /// cannot be acquired.
    fn grouped_stream(&self) -> Result<Self::GroupedStream<'_>>;

    /// One-shot grouped dispatch: open a stream, encode one pass, submit.
    ///
    /// # Errors
    /// Propagates encode/submit failures.
    fn dispatch_grouped<K: GroupedKernelSource<Self::Dialect>>(
        &self,
        prepared: &Self::GroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, Self>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()>
    where
        Self: Sized,
    {
        let mut stream = self.grouped_stream()?;
        stream.encode_grouped(prepared, bindings, params, grid)?;
        stream.submit_grouped()
    }
}

/// One typed storage binding, erased to the backend's handle plus metadata.
#[derive(Clone, Copy, Debug)]
pub struct Binding<'a, D: KernelDevice + ?Sized + 'a> {
    /// Access mode this binding was constructed with.
    pub access: Access,
    /// Element size in bytes of the borrowed buffer.
    pub elem_size: usize,
    /// Element count of the borrowed buffer.
    pub len: usize,
    /// Backend-specific binding handle.
    pub handle: D::BindingHandle<'a>,
}

impl<'a, D: KernelDevice> Binding<'a, D> {
    /// Bind `buffer` read-only.
    #[must_use]
    pub fn read<T: Pod>(buffer: &'a D::Buffer<T>) -> Self {
        use super::buffer::DeviceBuffer;
        Self {
            access: Access::ReadOnly,
            elem_size: core::mem::size_of::<T>(),
            len: buffer.len(),
            handle: D::binding_handle(buffer),
        }
    }

    /// Bind `buffer` read-write.
    #[must_use]
    pub fn read_write<T: Pod>(buffer: &'a D::Buffer<T>) -> Self {
        use super::buffer::DeviceBuffer;
        Self {
            access: Access::ReadWrite,
            elem_size: core::mem::size_of::<T>(),
            len: buffer.len(),
            handle: D::binding_handle(buffer),
        }
    }
}

/// One typed grouped storage binding, erased to the backend's handle plus
/// metadata.
#[derive(Clone, Copy, Debug)]
pub struct GroupedBinding<'a, D: GroupedKernelDevice + ?Sized + 'a> {
    /// WGPU bind group number; CUDA flattens bindings in slice order.
    pub group: u32,
    /// Binding slot within `group`.
    pub binding: u32,
    /// Access mode this binding was constructed with.
    pub access: Access,
    /// Element size in bytes of the borrowed buffer.
    pub elem_size: usize,
    /// Element count of the borrowed buffer.
    pub len: usize,
    /// Backend-specific binding handle.
    pub handle: D::BindingHandle<'a>,
}

impl<'a, D: GroupedKernelDevice> GroupedBinding<'a, D> {
    /// Bind `buffer` read-only at `group` / `binding`.
    #[must_use]
    pub fn read<T: Pod>(group: u32, binding: u32, buffer: &'a D::Buffer<T>) -> Self {
        use super::buffer::DeviceBuffer;
        Self {
            group,
            binding,
            access: Access::ReadOnly,
            elem_size: core::mem::size_of::<T>(),
            len: buffer.len(),
            handle: D::binding_handle(buffer),
        }
    }

    /// Bind `buffer` read-write at `group` / `binding`.
    #[must_use]
    pub fn read_write<T: Pod>(group: u32, binding: u32, buffer: &'a D::Buffer<T>) -> Self {
        use super::buffer::DeviceBuffer;
        Self {
            group,
            binding,
            access: Access::ReadWrite,
            elem_size: core::mem::size_of::<T>(),
            len: buffer.len(),
            handle: D::binding_handle(buffer),
        }
    }
}

/// Ordered multi-pass command recording over a [`KernelDevice`].
pub trait CommandStream<'d, D: KernelDevice + ?Sized> {
    /// Encode one kernel pass. Passes execute in encode order with
    /// inter-pass barrier semantics.
    ///
    /// # Errors
    /// [`HephaestusError::DispatchFailed`] when bindings do not match the
    /// kernel's declared layout (arity, access, element size) or encoding
    /// fails.
    fn encode<K: KernelSource<D::Dialect>>(
        &mut self,
        prepared: &D::Prepared<K>,
        bindings: &[Binding<'_, D>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()>;

    /// Encode a device-to-device copy (whole buffer, equal lengths).
    ///
    /// # Errors
    /// [`HephaestusError::LengthMismatch`] on unequal lengths; the backend's
    /// typed failure on encoding errors.
    fn copy<T: Pod>(&mut self, src: &D::Buffer<T>, dst: &D::Buffer<T>) -> Result<()>;

    /// Encode a device-to-device copy of the first `elements` values.
    ///
    /// The source and destination retain all values outside the copied prefix.
    ///
    /// # Errors
    /// [`HephaestusError::LengthMismatch`] when `elements` exceeds either
    /// buffer's length; the backend's typed failure on encoding errors.
    fn copy_prefix<T: Pod>(
        &mut self,
        src: &D::Buffer<T>,
        dst: &D::Buffer<T>,
        elements: usize,
    ) -> Result<()>;

    /// Encode a device-side zero fill of `dst`.
    ///
    /// # Errors
    /// The backend's typed failure on encoding errors.
    fn fill_zero<T: Pod>(&mut self, dst: &D::Buffer<T>) -> Result<()>;

    /// Submit the recorded passes for execution. Completion is observed via
    /// [`ComputeDevice::synchronize`] or a synchronizing transfer.
    ///
    /// # Errors
    /// The backend's typed submission failure.
    fn submit(self) -> Result<()>
    where
        Self: Sized;
}

/// Ordered grouped-kernel recording region.
///
/// WGPU backends implement this as one live compute pass; CUDA backends
/// implement it as ordered launches on the bound stream. Consumers use this
/// when splitting dispatches into barrier-separated passes would change the
/// algorithmic or performance contract.
pub trait GroupedKernelSequence<'s, D: GroupedKernelDevice + ?Sized> {
    /// Encode one grouped kernel dispatch inside the active sequence.
    ///
    /// # Errors
    /// [`HephaestusError::DispatchFailed`] when bindings do not match the
    /// kernel's declared grouped layout or encoding fails.
    fn encode_grouped<K: GroupedKernelSource<D::Dialect>>(
        &mut self,
        prepared: &D::GroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, D>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()>;
}

/// Ordered multi-pass command recording over a [`GroupedKernelDevice`].
pub trait GroupedCommandStream<'d, D: GroupedKernelDevice + ?Sized> {
    /// Backend-specific active grouped sequence.
    type Sequence<'s>: GroupedKernelSequence<'s, D>
    where
        D: 's;

    /// Encode one grouped kernel pass. Passes execute in encode order with
    /// inter-pass barrier semantics.
    ///
    /// # Errors
    /// [`HephaestusError::DispatchFailed`] when bindings do not match the
    /// kernel's declared grouped layout or encoding fails.
    fn encode_grouped<K: GroupedKernelSource<D::Dialect>>(
        &mut self,
        prepared: &D::GroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, D>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()>;

    /// Encode an ordered sequence of grouped kernels in one backend dispatch
    /// region.
    ///
    /// WGPU implements the sequence as one compute pass. CUDA implements it as
    /// ordered launches on the bound CUDA stream. Use this method for consumers
    /// that require adjacent dispatches without backend pass-boundary barriers.
    ///
    /// # Errors
    /// Propagates validation and backend encoding errors from each encoded
    /// dispatch.
    fn encode_grouped_sequence<F>(&mut self, label: &str, encode: F) -> Result<()>
    where
        Self: Sized,
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>;

    /// Submit the recorded grouped passes for execution.
    ///
    /// # Errors
    /// The backend's typed submission failure.
    fn submit_grouped(self) -> Result<()>
    where
        Self: Sized;
}

/// Validate a binding slice against a kernel's declared layout.
///
/// Backends call this at encode; exposed so contract tests can assert the
/// exact failure vocabulary.
///
/// # Errors
/// [`HephaestusError::DispatchFailed`] naming the violated invariant
/// (arity, access mode, or element size per position).
pub fn validate_bindings<D: KernelDevice>(
    label: &str,
    decls: &[BindingDecl],
    bindings: &[Binding<'_, D>],
) -> Result<()> {
    if decls.len() != bindings.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "{label}: kernel declares {} storage bindings, got {}",
                decls.len(),
                bindings.len()
            ),
        });
    }
    for (i, (decl, binding)) in decls.iter().zip(bindings).enumerate() {
        if decl.access != binding.access {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{label}: binding {i} declared {:?}, bound {:?}",
                    decl.access, binding.access
                ),
            });
        }
        if decl.elem_size != binding.elem_size {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{label}: binding {i} declared element size {} B, bound {} B",
                    decl.elem_size, binding.elem_size
                ),
            });
        }
    }
    Ok(())
}

/// Validate a grouped binding slice against a grouped kernel's declared layout.
///
/// # Errors
/// [`HephaestusError::DispatchFailed`] naming the violated invariant (arity,
/// group/binding slot, access mode, or element size per position).
pub fn validate_grouped_bindings<D: GroupedKernelDevice>(
    label: &str,
    decls: &[GroupedBindingDecl],
    bindings: &[GroupedBinding<'_, D>],
) -> Result<()> {
    if decls.len() != bindings.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "{label}: grouped kernel declares {} storage bindings, got {}",
                decls.len(),
                bindings.len()
            ),
        });
    }
    for (i, (decl, binding)) in decls.iter().zip(bindings).enumerate() {
        if decl.group != binding.group || decl.binding != binding.binding {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{label}: binding {i} declared group {} binding {}, bound group {} binding {}",
                    decl.group, decl.binding, binding.group, binding.binding
                ),
            });
        }
        if decl.access != binding.access {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{label}: binding {i} declared {:?}, bound {:?}",
                    decl.access, binding.access
                ),
            });
        }
        if decl.elem_size != binding.elem_size {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{label}: binding {i} declared element size {} B, bound {} B",
                    decl.elem_size, binding.elem_size
                ),
            });
        }
    }
    Ok(())
}
