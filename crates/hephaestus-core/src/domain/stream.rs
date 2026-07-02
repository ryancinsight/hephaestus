//! Backend dispatch seam for authored kernels: prepared pipelines, typed
//! bindings, and multi-pass command streams.
//!
//! [`KernelDevice`] extends [`ComputeDevice`] with the compile/dispatch
//! surface for consumer-authored kernels ([`KernelInterface`] +
//! [`KernelSource`]): `prepare` compiles-and-caches, [`CommandStream`]
//! records ordered passes (inter-pass barrier semantics guaranteed by the
//! backend — wgpu compute-pass boundaries, CUDA stream order), and `submit`
//! queues the recorded work. Completion is observed through
//! [`ComputeDevice::synchronize`] or a synchronizing transfer; no async
//! surface leaks into core (async-contagion rule).
//!
//! Bindings are typed at construction ([`Binding::read`] /
//! [`Binding::read_write`] borrow a `D::Buffer<T>`) and erase to the
//! backend's [`KernelDevice::BindingHandle`] plus element-size/length
//! metadata, so one homogeneous `&[Binding<'_, D>]` slice carries
//! heterogeneous element types with no trait-object indirection. Arity,
//! access, and element size are validated value-semantically against the
//! kernel's [`KernelInterface::BINDINGS`] declaration at encode
//! ([`validate_bindings`]).

use super::device::ComputeDevice;
use super::dialect::KernelDialect;
use super::error::{HephaestusError, Result};
use super::interface::{BindingDecl, KernelSource};
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
