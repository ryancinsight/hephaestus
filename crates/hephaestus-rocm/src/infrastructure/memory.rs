use core::{ffi::c_void, ptr};

use bytemuck::Pod;
use hephaestus_core::{
    CommandStream, ComputeDevice, HephaestusError, KernelDevice, Result, validate_buffer_size,
    validate_slice_alignment,
};

use super::device::RocmDevice;
use super::{DevicePtr, buffer::RocmBuffer};

fn checked_bytes<T>(len: usize) -> Result<usize> {
    validate_buffer_size::<T>(len)?;
    len.checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| HephaestusError::AllocationFailed {
            message: format!(
                "ROCm buffer byte size calculation overflows (elements: {len}, element size: {})",
                core::mem::size_of::<T>()
            ),
        })
}

fn checked_transfer_bytes<T>(slice: &[T]) -> Result<usize> {
    validate_slice_alignment(slice)?;
    slice
        .len()
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: format!(
                "ROCm transfer byte size calculation overflows (elements: {}, element size: {})",
                slice.len(),
                core::mem::size_of::<T>()
            ),
        })
}

impl ComputeDevice for RocmDevice {
    type Buffer<T: Pod> = RocmBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "rocm"
    }

    fn alloc_zeroed_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let buffer = self.alloc_uninitialized_with_hint::<T>(len, hint)?;
        if len == 0 {
            return Ok(buffer);
        }
        let bytes = checked_bytes::<T>(len)?;
        self.context.set_current()?;
        // SAFETY: `buffer.raw()` is the live HIP allocation created above and
        // `bytes` is its checked size.
        let status = unsafe { cubecl_hip_sys::hipMemset(buffer.raw(), 0, bytes) };
        if status != super::device::HIP_SUCCESS {
            return Err(HephaestusError::AllocationFailed {
                message: super::device::status_message(status, "hipMemset"),
            });
        }
        Ok(buffer)
    }

    fn alloc_uninitialized_with_hint<T: Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let tier = Self::allocation_tier(hint)?;
        let bytes = checked_bytes::<T>(len)?;
        self.context.set_current()?;
        let ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            let mut ptr: DevicePtr = ptr::null_mut();
            // SAFETY: `ptr` is a valid out-pointer and `bytes` was checked for
            // overflow before crossing the HIP allocation boundary.
            let status = unsafe { cubecl_hip_sys::hipMalloc(&mut ptr, bytes) };
            if status != super::device::HIP_SUCCESS {
                return Err(HephaestusError::AllocationFailed {
                    message: super::device::status_message(status, "hipMalloc"),
                });
            }
            ptr
        };
        Ok(RocmBuffer::new(ptr, len, tier, self.context.clone()))
    }

    fn upload_with_hint<T: Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> Result<Self::Buffer<T>> {
        let buffer = self.alloc_uninitialized_with_hint(host.len(), hint)?;
        self.write_buffer(&buffer, host)?;
        Ok(buffer)
    }

    fn download<T: Pod>(&self, buffer: &Self::Buffer<T>, out: &mut [T]) -> Result<()> {
        if out.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: out.len(),
                device_len: buffer.len,
            });
        }
        let bytes = checked_transfer_bytes(out)?;
        if bytes == 0 {
            return Ok(());
        }
        self.context.set_current()?;
        // SAFETY: `buffer.ptr` is owned HIP device memory for the current
        // device; `out` is a valid writable host slice of exactly `bytes`.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpy(
                out.as_mut_ptr().cast::<c_void>(),
                buffer.ptr.cast_const(),
                bytes,
                cubecl_hip_sys::hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        };
        if status == super::device::HIP_SUCCESS {
            Ok(())
        } else {
            Err(HephaestusError::TransferFailed {
                message: super::device::status_message(status, "hipMemcpy device-to-host"),
            })
        }
    }

    fn download_owned<T: Pod>(&self, buffer: &Self::Buffer<T>) -> Result<Vec<T>> {
        let len = buffer.len;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|error| HephaestusError::AllocationFailed {
                message: format!(
                    "ROCm host download allocation for {len} elements failed: {error}"
                ),
            })?;
        if core::mem::size_of::<T>() == 0 {
            out.resize(len, bytemuck::Zeroable::zeroed());
            return Ok(out);
        }
        let bytes = checked_bytes::<T>(len)?;
        if bytes != 0 {
            self.context.set_current()?;
            // SAFETY: `try_reserve_exact` established capacity for `len`
            // elements, so the spare-capacity pointer addresses `bytes`
            // writable host bytes. `hipMemcpy` is synchronous and initializes
            // every byte before vector length is published below.
            let status = unsafe {
                cubecl_hip_sys::hipMemcpy(
                    out.spare_capacity_mut().as_mut_ptr().cast::<c_void>(),
                    buffer.ptr.cast_const(),
                    bytes,
                    cubecl_hip_sys::hipMemcpyKind_hipMemcpyDeviceToHost,
                )
            };
            if status != super::device::HIP_SUCCESS {
                return Err(HephaestusError::TransferFailed {
                    message: super::device::status_message(
                        status,
                        "owned hipMemcpy device-to-host",
                    ),
                });
            }
        }
        // SAFETY: `T` is non-zero-sized here and the successful synchronous
        // copy above initialized all `len` elements before publication.
        unsafe { out.set_len(len) };
        Ok(out)
    }

    fn write_buffer<T: Pod>(&self, buffer: &Self::Buffer<T>, host: &[T]) -> Result<()> {
        if host.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: host.len(),
                device_len: buffer.len,
            });
        }
        let bytes = checked_transfer_bytes(host)?;
        if bytes == 0 {
            return Ok(());
        }
        self.context.set_current()?;
        // SAFETY: `buffer.ptr` is owned HIP device memory for the current
        // device; `host` is a valid readable host slice of exactly `bytes`.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpy(
                buffer.ptr,
                host.as_ptr().cast::<c_void>(),
                bytes,
                cubecl_hip_sys::hipMemcpyKind_hipMemcpyHostToDevice,
            )
        };
        if status == super::device::HIP_SUCCESS {
            Ok(())
        } else {
            Err(HephaestusError::TransferFailed {
                message: super::device::status_message(status, "hipMemcpy host-to-device"),
            })
        }
    }

    fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> Result<()> {
        let end =
            offset
                .checked_add(host.len())
                .ok_or_else(|| HephaestusError::TransferFailed {
                    message: format!(
                        "ROCm sub-buffer range overflows: offset {offset}, length {}",
                        host.len()
                    ),
                })?;
        if end > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: end,
                device_len: buffer.len,
            });
        }
        let bytes = checked_transfer_bytes(host)?;
        if bytes == 0 {
            return Ok(());
        }
        let offset_bytes = offset
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| HephaestusError::TransferFailed {
                message: format!("ROCm sub-buffer byte offset overflows: offset {offset}"),
            })?;
        self.context.set_current()?;
        // SAFETY: `end <= buffer.len` and the checked byte arithmetic prove
        // that the HIP allocation contains the complete destination range.
        let destination = unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast::<c_void>() };
        // SAFETY: `destination` is the in-bounds HIP device subrange above;
        // `host` is a valid readable host slice of exactly `bytes`.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpy(
                destination,
                host.as_ptr().cast::<c_void>(),
                bytes,
                cubecl_hip_sys::hipMemcpyKind_hipMemcpyHostToDevice,
            )
        };
        if status == super::device::HIP_SUCCESS {
            Ok(())
        } else {
            Err(HephaestusError::TransferFailed {
                message: super::device::status_message(status, "hipMemcpy host-to-device subrange"),
            })
        }
    }

    fn copy_buffer<T: Pod>(&self, src: &RocmBuffer<T>, dst: &RocmBuffer<T>) -> Result<()> {
        let mut stream = self.stream()?;
        stream.copy(src, dst)?;
        stream.submit()?;
        self.synchronize()
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        RocmDevice::topology(self)
    }

    fn synchronize(&self) -> Result<()> {
        self.context.set_current()?;
        // SAFETY: no pointers are passed; HIP synchronizes the current device.
        let status = unsafe { cubecl_hip_sys::hipDeviceSynchronize() };
        if status == super::device::HIP_SUCCESS {
            Ok(())
        } else {
            Err(HephaestusError::TransferFailed {
                message: super::device::status_message(status, "hipDeviceSynchronize"),
            })
        }
    }
}
