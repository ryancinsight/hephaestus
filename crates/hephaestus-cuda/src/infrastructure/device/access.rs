use super::properties::current_memory_info;
use super::{CudaContext, CudaDevice};
use crate::infrastructure::buffer::{CudaBuffer, DevicePtr};
use crate::infrastructure::driver::Driver;
use core::ffi::c_void;
use eunomia::Pod;
use hephaestus_core::{HephaestusError, Result, validate_slice_alignment};
use std::sync::Arc;

impl CudaDevice {
    pub(crate) fn driver(&self) -> &'static Driver {
        self.context.driver
    }

    pub(crate) fn same_context(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.context, &other.context)
    }

    /// Resolve caller placement hints to CUDA's implemented allocation tier.
    ///
    /// This backend deliberately allocates primary buffers with
    /// `cuMemAllocAsync`, or `cuMemAlloc_v2` where the device lacks the
    /// stream-ordered allocator. Both are non-managed device memory: host
    /// access happens only through explicit copies. Host-visible hints are
    /// therefore normalized to [`themis::MemoryTier::Device`] instead of being
    /// recorded as mappable or managed storage.
    pub(super) fn allocation_tier(hint: themis::PlacementHint) -> Result<themis::MemoryTier> {
        match hint {
            themis::PlacementHint::Tier(tier) if !tier.is_host_allocatable() => {
                Err(HephaestusError::AllocationFailed {
                    message: format!(
                        "CUDA primary buffers cannot be allocated from budget-only tier {tier:?}"
                    ),
                })
            }
            themis::PlacementHint::Tier(_) | themis::PlacementHint::Current => {
                Ok(themis::MemoryTier::Device)
            }
            themis::PlacementHint::Numa(_)
            | themis::PlacementHint::Domain(_)
            | themis::PlacementHint::Any => Ok(themis::MemoryTier::Device),
        }
    }

    /// The device topology snapshot captured at acquisition, when available.
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.topology.as_deref()
    }

    /// The owning CUDA context (module lifetime management).
    #[inline]
    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Bind the device context to the current thread before a driver call.
    ///
    /// Transfers, allocations, module loads, and kernel launches execute
    /// against the thread's current context; binding makes this device's
    /// context current (CUDA contexts are thread-affine), so calls from any
    /// thread target the right device.
    pub fn bind(&self) -> Result<()> {
        self.context.bind()
    }

    /// Device memory not currently allocated, in bytes, as the driver reports
    /// it at this call.
    ///
    /// A point-in-time reading that other contexts and later allocations
    /// move; the stable per-device capacity is `ComputeDevice::device_limits`
    /// (`max_buffer_size`), and this value never exceeds it.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::DeviceUnavailable`] when the context cannot be bound
    /// or the driver query fails.
    pub fn free_memory_bytes(&self) -> Result<u64> {
        self.bind()?;
        let (free_bytes, _) = current_memory_info(self.driver())?;
        u64::try_from(free_bytes).map_err(|_| HephaestusError::DeviceUnavailable {
            message: "CUDA free memory byte count exceeds u64".to_string(),
        })
    }

    /// Allocate `bytes` of device memory according to the tier.
    pub(super) fn alloc_bytes(&self, bytes: usize) -> Result<DevicePtr> {
        self.bind()?;

        let mut ptr = 0;
        // SAFETY: this device's context is current after `bind`; `ptr` is a
        // valid out-pointer for one `CUdeviceptr`, and `bytes > 0` at call
        // sites.
        // Stream-ordered allocation where the device supports it. The
        // synchronous pair costs 5.5x-36.8x on this path (see
        // HEPH-CUDA-STREAM-ORDERED-ALLOC): `cuMemFree_v2` synchronizes the
        // whole device, so every buffer drop drained in-flight work.
        //
        // The pool's release threshold is deliberately left at its default of
        // zero, so memory still returns to the driver at synchronization
        // points exactly as before and this introduces no retained pool to
        // grow unbounded. The win is the ordering, not the retention: measured
        // with the threshold raised to `u64::MAX` the numbers are the same
        // within noise (4M-element reduction 44.3 us vs 45.0 us).
        let status = if self.context.stream_ordered {
            // SAFETY: this device's context is current after `bind`; `ptr` is a
            // valid out-pointer for one `CUdeviceptr`, and `bytes > 0` at call
            // sites. The null stream is the same stream every kernel and copy
            // in this backend uses, so the allocation is ordered against them.
            unsafe {
                (self.driver().memory.allocate_ordered)(&mut ptr, bytes, std::ptr::null_mut())
            }
        } else {
            // SAFETY: as above; the synchronous allocator carries no stream.
            unsafe { (self.driver().memory.allocate)(&mut ptr, bytes) }
        };
        if status != 0 {
            let function = if self.context.stream_ordered {
                "cuMemAllocAsync"
            } else {
                "cuMemAlloc_v2"
            };
            return Err(HephaestusError::AllocationFailed {
                message: format!("the CUDA driver {function}({bytes} bytes) -> {status}"),
            });
        }
        Ok(ptr)
    }

    /// Copy a subset of a device buffer's contents into a host slice (device→host).
    pub fn download_sub_buffer<T: Pod>(
        &self,
        buffer: &CudaBuffer<T>,
        offset: usize,
        out: &mut [T],
    ) -> Result<()> {
        validate_slice_alignment(out)?;
        let end =
            offset
                .checked_add(out.len())
                .ok_or_else(|| HephaestusError::AllocationFailed {
                    message: format!("offset {offset} + out.len() {} overflows usize", out.len()),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        if out.is_empty() || core::mem::size_of::<T>() == 0 {
            return Ok(());
        }
        self.bind()?;
        let element_size = std::mem::size_of::<T>();
        let byte_offset = (offset as u64)
            .checked_mul(element_size as u64)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!("byte offset calculation overflows u64 for offset {offset}"),
            })?;
        let bytes = std::mem::size_of_val(out);

        let src_ptr = buffer.raw() + byte_offset;
        // SAFETY: `src_ptr` is a valid device pointer offset from a pointer allocated by this device;
        // `out` is `bytes` of writable host memory (`T: Pod`).
        let res = unsafe {
            (self.driver().memory.download)(out.as_mut_ptr() as *mut c_void, src_ptr, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("download_sub_buffer cuMemcpyDtoH_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }

    /// Overwrite a subset of a device buffer with host data (host→device).
    pub fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &CudaBuffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        validate_slice_alignment(host)?;
        let end =
            offset
                .checked_add(host.len())
                .ok_or_else(|| HephaestusError::AllocationFailed {
                    message: format!(
                        "offset {offset} + host.len() {} overflows usize",
                        host.len()
                    ),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        if host.is_empty() || core::mem::size_of::<T>() == 0 {
            return Ok(());
        }
        self.bind()?;
        let element_size = std::mem::size_of::<T>();
        let byte_offset = (offset as u64)
            .checked_mul(element_size as u64)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!("byte offset calculation overflows u64 for offset {offset}"),
            })?;
        let bytes = std::mem::size_of_val(host);

        let dest_ptr = buffer.raw() + byte_offset;
        // SAFETY: `dest_ptr` is a valid device pointer offset from a pointer allocated by this device;
        // `host` is `bytes` of readable host memory (`T: Pod`).
        let res = unsafe {
            (self.driver().memory.upload)(dest_ptr, host.as_ptr() as *const c_void, bytes)
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!("write_sub_buffer cuMemcpyHtoD_v2({bytes} bytes) -> {res}"),
            });
        }
        Ok(())
    }
}
