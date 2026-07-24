use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;
use bytemuck::Pod;

/// Backend-neutral device selection preference.
///
/// This is an acquisition policy hint, not a capability claim. Backends map it
/// onto their concrete adapter/device selection APIs while preserving the same
/// consumer-facing contract across WGPU, CUDA, ROCm, Metal, or future
/// providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePreference {
    /// Prefer the highest-throughput device available.
    HighPerformance,
    /// Prefer a lower-power device when one is available.
    LowPower,
}

/// Backend-neutral optional device feature.
///
/// Backends map these capability names to their concrete feature flags. A
/// false result means the feature is not enabled on the acquired device; it is
/// not a statement about all adapters in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceFeature {
    /// GPU timestamp queries for profiling.
    TimestampQuery,
    /// 64-bit floating-point shader arithmetic.
    ShaderF64,
    /// 16-bit floating-point shader arithmetic.
    ShaderF16,
    /// Host-mappable primary buffers.
    MappablePrimaryBuffers,
    /// Immediate shader-data support.
    ImmediateData,
}

/// Backend-neutral compute-resource limits.
///
/// Fields are the common compute limits used by Atlas consumers. Backend
/// crates map from their concrete device descriptors into this representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceLimits {
    /// Maximum single buffer size in bytes.
    pub max_buffer_size: u64,
    /// Maximum compute workgroup size along x.
    pub max_compute_workgroup_size_x: u32,
    /// Maximum compute workgroup size along y.
    pub max_compute_workgroup_size_y: u32,
    /// Maximum compute workgroup size along z.
    pub max_compute_workgroup_size_z: u32,
    /// Maximum total invocations per compute workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Maximum workgroup-local storage size in bytes.
    pub max_compute_workgroup_storage_size: u32,
    /// Maximum storage buffers available to one shader stage, when the backend
    /// has shader-stage storage binding slots.
    ///
    /// CUDA-style flat kernel arguments do not expose an equivalent per-stage
    /// storage-buffer slot limit, so CUDA reports `None` instead of fabricating
    /// a WGPU-shaped value.
    pub max_storage_buffers_per_shader_stage: Option<u32>,
    /// Maximum combined buffer + acceleration-structure bindings available to
    /// one shader stage.
    ///
    /// WGPU 30 folds storage buffers, uniform buffers, and acceleration
    /// structures into a single per-stage budget
    /// (`max_buffers_and_acceleration_structures_per_shader_stage`), so a kernel
    /// binding N storage buffers *and* a uniform buffer needs this limit at
    /// `N + 1`, independent of `max_storage_buffers_per_shader_stage`. `None`
    /// when the backend has no equivalent combined slot limit (CUDA flat
    /// kernel arguments).
    pub max_buffers_and_acceleration_structures_per_shader_stage: Option<u32>,
    /// Maximum immediate shader-data byte count.
    pub max_immediate_size: u32,
}

/// The compute-device seam every accelerator backend implements.
///
/// This trait is a **deliberate extension seam** (atlas ADR 0001): the wgpu
/// backend, the CUDA backend (cuda-oxide + cutile composed), and the ROCm
/// backend substitute here without consumers changing. It is intentionally
/// *not* sealed — new backend crates in the hephaestus workspace implement it.
/// Consumers bind generically
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

    /// Overwrite a subrange of an existing device buffer with host data.
    ///
    /// Writes `host.len()` elements starting at element `offset`.
    /// The range `offset..offset + host.len()` must fit inside `buffer`.
    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()>;

    /// Wait until previously submitted work and transfers visible to this
    /// device context have completed.
    ///
    /// Backends map this to their real synchronization primitive (`Device::poll`
    /// for WGPU, `cuCtxSynchronize` for CUDA, `hipDeviceSynchronize` for ROCm).
    /// Consumers use this for explicit blocking semantics without depending on a
    /// concrete GPU API.
    fn synchronize(&self) -> Result<()>;
}

/// Capability-query seam for accelerator backends.
///
/// Consumers that need device limits or optional feature checks bind to this
/// trait instead of a concrete backend API. WGPU, CUDA, ROCm, and future
/// providers map their native descriptors into the shared Hephaestus
/// vocabulary.
pub trait ComputeDeviceCapabilities: ComputeDevice {
    /// Return the enabled device limits in backend-neutral form.
    fn device_limits(&self) -> DeviceLimits;

    /// Return true when the acquired device has `feature` enabled.
    fn supports_device_feature(&self, feature: DeviceFeature) -> bool;
}

/// Backend-neutral device acquisition seam.
///
/// Consumers that need to acquire devices bind to this trait instead of a
/// concrete provider constructor. Backend crates map the shared feature and
/// limit request onto their native adapter/device APIs; unsupported backends
/// report an error instead of fabricating a device.
pub trait ComputeDeviceAcquisition: ComputeDeviceCapabilities + Sized {
    /// Acquire one device matching `device_preference`.
    ///
    /// `optional_features` are enabled only when the selected backend device
    /// supports them. `required_limits` are lower bounds on the compute
    /// resources the consumer needs.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific acquisition error when no matching device is
    /// available or when the requested limits cannot be satisfied.
    fn try_acquire_device(
        label: &str,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Self>;

    /// Acquire up to `max_devices` devices matching `device_preference`.
    ///
    /// Backends that expose only one usable device return at most one entry.
    /// A backend with no matching devices returns an empty vector or an
    /// acquisition error when probing itself fails.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific acquisition error when device probing or
    /// logical-device creation fails.
    fn try_acquire_devices(
        label_prefix: &str,
        max_devices: usize,
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> Result<Vec<Self>>;
}

/// Validate that a typed buffer's byte-size calculation cannot overflow.
///
/// Device implementations own physical alignment requirements. The logical
/// element count remains valid even when its byte length needs provider-local
/// padding for storage or transfer.
#[inline]
pub fn validate_buffer_size<T>(len: usize) -> Result<()> {
    len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
        crate::domain::error::HephaestusError::AllocationFailed {
            message: format!(
                "Buffer byte size calculation overflows (elements: {}, element size: {})",
                len,
                core::mem::size_of::<T>()
            ),
        }
    })?;
    Ok(())
}

/// Validate that a host slice's byte-size calculation cannot overflow.
///
/// Device implementations own physical transfer alignment and must preserve
/// the slice's exact logical element count.
#[inline]
pub fn validate_slice_alignment<T>(slice: &[T]) -> Result<()> {
    slice
        .len()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| crate::domain::error::HephaestusError::TransferFailed {
            message: format!(
                "Transfer byte size calculation overflows (elements: {}, element size: {})",
                slice.len(),
                core::mem::size_of::<T>()
            ),
        })?;
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
    fn validates_odd_logical_lengths_without_provider_alignment() {
        assert!(validate_buffer_size::<u8>(4).is_ok());
        assert!(validate_buffer_size::<u8>(3).is_ok());
        assert!(validate_buffer_size::<u16>(27).is_ok());
        assert!(validate_slice_alignment(&[0_u16; 27]).is_ok());
    }
}
