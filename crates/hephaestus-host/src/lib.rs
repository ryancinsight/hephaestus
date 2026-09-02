//! Host reference device for the Hephaestus seams (ADR 0046).
//!
//! [`HostDevice`] implements [`ComputeDevice`] over plain host memory so
//! CPU reference implementors — leto adapters first — can join the same
//! role traits the GPU backends implement, and the conformance suite can
//! instantiate a CPU pair for every clause. This crate is the
//! **reference substrate**: correctness and conformance first, never a
//! performance path — consumers wanting fast CPU execution use leto
//! directly.

/// Leto as a decomposition-seam implementor.
pub mod decomposition;
/// Leto-backed pooling seam implementor.
pub mod pooling;
/// Leto-backed unfold/fold seam implementor.
pub mod sliding_window;

pub use decomposition::HostDecompositionOps;
pub use pooling::{HostPoolingBackward, HostPoolingForward, HostPoolingOps};
pub use sliding_window::{HostSlidingWindowFold, HostSlidingWindowOps, HostSlidingWindowUnfold};

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bytemuck::Pod;
use hephaestus_core::{
    ComputeDevice, ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature, DeviceLimits,
    HephaestusError, Result,
};

/// Shared handle over host memory.
///
/// Interior mutability is required by the seams' shared-reference
/// mutation contract (`write_buffer(&self, &Buffer, ..)`); a lock is the
/// honest host analog of a device queue serializing access.
pub struct HostBuffer<T> {
    cells: Arc<RwLock<Vec<T>>>,
}

impl<T> Clone for HostBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            cells: Arc::clone(&self.cells),
        }
    }
}

impl<T> HostBuffer<T> {
    fn new(cells: Vec<T>) -> Self {
        Self {
            cells: Arc::new(RwLock::new(cells)),
        }
    }

    /// Read access to the underlying host memory.
    ///
    /// # Panics
    ///
    /// Panics when a prior holder poisoned the lock by panicking — the
    /// reference substrate treats poisoned state as unrecoverable test
    /// wreckage, not a runtime condition.
    pub fn read(&self) -> RwLockReadGuard<'_, Vec<T>> {
        self.cells
            .read()
            .expect("invariant: host buffer lock is never poisoned")
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, Vec<T>> {
        self.cells
            .write()
            .expect("invariant: host buffer lock is never poisoned")
    }

    pub(crate) fn aliases(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cells, &other.cells)
    }
}

impl<T> DeviceBuffer<T> for HostBuffer<T> {
    fn len(&self) -> usize {
        self.read().len()
    }

    fn tier(&self) -> themis::MemoryTier {
        themis::MemoryTier::Dram
    }
}

/// The host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDevice;

impl HostDevice {
    /// Acquire the host device; infallible and allocation-free.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn require_matching_len(expected: usize, actual: usize) -> Result<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(HephaestusError::LengthMismatch {
                host_len: actual,
                device_len: expected,
            })
        }
    }
}

pub(crate) fn map_leto_error<E: core::fmt::Display>(error: E) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("host window operation failed: {error}"),
    }
}

impl ComputeDevice for HostDevice {
    type Buffer<T: Pod> = HostBuffer<T>;

    fn backend_name(&self) -> &'static str {
        "host"
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let _ = hint;
        Ok(HostBuffer::new(vec![T::zeroed(); len]))
    }

    fn alloc_uninitialized_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        // Host vectors have no cheap uninitialized form worth unsafe here:
        // the reference substrate zero-fills, which satisfies the
        // must-overwrite-before-read contract trivially.
        self.alloc_zeroed_with_hint(len, hint)
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let _ = hint;
        Ok(HostBuffer::new(host.to_vec()))
    }

    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()> {
        let cells = buffer.read();
        Self::require_matching_len(cells.len(), out.len())?;
        out.copy_from_slice(&cells);
        Ok(())
    }

    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()> {
        let mut cells = buffer.write();
        Self::require_matching_len(cells.len(), host.len())?;
        cells.copy_from_slice(host);
        Ok(())
    }

    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        let mut cells = buffer.write();
        let end = offset
            .checked_add(host.len())
            .ok_or(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: cells.len(),
            })?;
        if end > cells.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: cells.len(),
            });
        }
        cells[offset..end].copy_from_slice(host);
        Ok(())
    }

    fn copy_buffer<T: Pod>(&self, src: &Self::Buffer<T>, dst: &Self::Buffer<T>) -> Result<()> {
        if Arc::ptr_eq(&src.cells, &dst.cells) {
            return Ok(());
        }
        let source = src.read();
        let mut destination = dst.write();
        Self::require_matching_len(destination.len(), source.len())?;
        destination.copy_from_slice(&source);
        Ok(())
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        // The host is not a GPU; there is no accelerator topology to
        // report, and fabricating one would defeat the reference role.
        None
    }

    fn synchronize(&self) -> Result<()> {
        // Host operations complete before returning; nothing is in flight.
        Ok(())
    }
}

impl ComputeDeviceCapabilities for HostDevice {
    fn device_limits(&self) -> DeviceLimits {
        DeviceLimits {
            max_buffer_size: u64::MAX,
            max_compute_workgroup_size_x: 1,
            max_compute_workgroup_size_y: 1,
            max_compute_workgroup_size_z: 1,
            max_compute_invocations_per_workgroup: 1,
            max_compute_workgroup_storage_size: 0,
            max_storage_buffers_per_shader_stage: None,
            max_buffers_and_acceleration_structures_per_shader_stage: None,
            max_immediate_size: 0,
        }
    }

    fn supports_device_feature(&self, feature: DeviceFeature) -> bool {
        // Host arithmetic is native Rust: f64 always; the other features
        // describe GPU shader machinery the host does not model.
        matches!(feature, DeviceFeature::ShaderF64)
    }
}
