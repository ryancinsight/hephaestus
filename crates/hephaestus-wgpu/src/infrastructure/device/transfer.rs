use std::borrow::Cow;
use std::time::Duration;

use eunomia::Pod;
use hephaestus_core::{HephaestusError, Result, validate_slice_alignment};

use crate::infrastructure::buffer::WgpuBuffer;

use super::{WgpuDevice, device_wait_deadline, poll_failure};

/// Cancels a pending map or releases an active mapping before pooled staging
/// storage can be recycled, including error and unwind exits.
struct MappingLifecycle<'a>(&'a wgpu::Buffer);

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

impl WgpuDevice {
    /// Exact byte size of `len` elements of `T`.
    pub(crate) fn byte_size<T>(len: usize) -> Result<u64> {
        len.checked_mul(core::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!(
                    "buffer length {len} overflows byte size for {}-byte elements",
                    core::mem::size_of::<T>()
                ),
            })
    }

    /// Size in bytes of `len` elements of `T`, padded to wgpu copy alignment.
    pub(super) fn padded_size<T>(len: usize) -> Result<u64> {
        let bytes = Self::byte_size::<T>(len)?;
        Self::aligned_size(bytes, wgpu::COPY_BUFFER_ALIGNMENT)
    }

    /// Borrow host bytes when they already meet WGPU's copy alignment; otherwise
    /// append zeroed physical padding outside the logical typed range.
    pub(super) fn padded_host_bytes<T: Pod>(host: &[T]) -> Result<Cow<'_, [u8]>> {
        let bytes: &[u8] = eunomia::layout::cast_slice(host);
        let padded_len = usize::try_from(Self::padded_size::<T>(host.len())?).map_err(|_| {
            HephaestusError::AllocationFailed {
                message: format!(
                    "padded byte size does not fit usize for {} elements of {} bytes",
                    host.len(),
                    core::mem::size_of::<T>()
                ),
            }
        })?;
        if bytes.len() == padded_len {
            return Ok(Cow::Borrowed(bytes));
        }

        let mut padded = Vec::new();
        padded.try_reserve_exact(padded_len).map_err(|error| {
            HephaestusError::AllocationFailed {
                message: format!(
                    "WGPU host padding allocation for {padded_len} bytes failed: {error}"
                ),
            }
        })?;
        padded.extend_from_slice(bytes);
        padded.resize(padded_len, 0);
        Ok(Cow::Owned(padded))
    }

    /// Ensure a WGPU copy starts at a valid byte boundary.
    fn validate_copy_offset(byte_offset: u64) -> Result<()> {
        if !byte_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "sub-buffer offset {byte_offset} must be a multiple of {} bytes",
                    wgpu::COPY_BUFFER_ALIGNMENT
                ),
            });
        }
        Ok(())
    }

    /// Align `size` upward to `alignment`.
    pub(super) fn aligned_size(size: u64, alignment: u64) -> Result<u64> {
        size.checked_add(alignment - 1)
            .map(|bytes| (bytes / alignment) * alignment)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!("buffer byte size {size} cannot be aligned to {alignment} bytes"),
            })
    }
    pub(super) fn download_into<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        out: &mut [T],
        timeout: Duration,
    ) -> Result<()> {
        validate_slice_alignment(out)?;
        if out.len() != buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: out.len(),
                device_len: buffer.len,
            });
        }
        if out.is_empty() || core::mem::size_of::<T>() == 0 {
            return Ok(());
        }

        let byte_len = Self::byte_size::<T>(buffer.len)?;
        let padded = Self::padded_size::<T>(buffer.len)?;
        self.stage_and_read(
            &buffer.buffer,
            ReadbackRegion {
                byte_offset: 0,
                padded,
                byte_len,
            },
            timeout,
            "hephaestus-download",
            |bytes| eunomia::layout::cast_slice_mut(out).copy_from_slice(bytes),
        )
    }

    /// Copy a device buffer into a host slice with an explicit completion deadline.
    ///
    /// `ComputeDevice::download` already carries a deadline; use this form
    /// only when the caller's own bound differs from the backend default.
    ///
    /// # Errors
    ///
    /// Returns the same validation and transfer errors as
    /// `ComputeDevice::download`, plus
    /// [`HephaestusError::DeviceWaitTimeout`] when the device does not
    /// complete before `timeout`.
    pub fn download_with_timeout<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        out: &mut [T],
        timeout: std::time::Duration,
    ) -> Result<()> {
        self.download_into(buffer, out, timeout)
    }

    /// Copy a subset of a device buffer's contents into a host slice (device→host).
    ///
    /// The transfer starts at element `offset` in the device buffer and copies
    /// `out.len()` elements. The range `offset..offset + out.len()` must be within
    /// `buffer.len`.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::LengthMismatch`] if the requested range falls outside the buffer bounds.
    /// [`HephaestusError::AllocationFailed`] if element byte conversion overflows `u64`.
    pub fn download_sub_buffer<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
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

        // byte_size::<T>(offset) = offset * size_of::<T>() with checked overflow → u64.
        let byte_offset = Self::byte_size::<T>(offset)?;
        Self::validate_copy_offset(byte_offset)?;
        let byte_len = Self::byte_size::<T>(out.len())?;
        let padded = Self::padded_size::<T>(out.len())?;
        self.stage_and_read(
            &buffer.buffer,
            ReadbackRegion {
                byte_offset,
                padded,
                byte_len,
            },
            device_wait_deadline(),
            "hephaestus-download-sub",
            |bytes| eunomia::layout::cast_slice_mut(out).copy_from_slice(bytes),
        )
    }

    /// Overwrite a subset of a device buffer with host data (host→device).
    ///
    /// Writes `host.len()` elements starting at element `offset` in the device buffer.
    /// The range `offset..offset + host.len()` must be within `buffer.len`.
    ///
    /// # Errors
    ///
    /// [`HephaestusError::LengthMismatch`] if the requested range falls outside the buffer bounds.
    /// [`HephaestusError::AllocationFailed`] if byte conversion, host padding, or
    /// queue staging allocation fails. [`HephaestusError::TransferFailed`] if
    /// the requested interior range is not copy-aligned.
    /// [`HephaestusError::DispatchFailed`] if the destination belongs to another
    /// device or WGPU rejects the write. Rejected requests preserve its values.
    pub fn write_sub_buffer<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
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
        if host.is_empty() {
            return Ok(());
        }

        // Byte offset reuses the existing checked multiplication.
        let byte_offset = Self::byte_size::<T>(offset)?;
        Self::validate_copy_offset(byte_offset)?;
        let bytes = eunomia::layout::cast_slice(host);
        let payload = if end == buffer.len {
            Self::padded_host_bytes(host)?
        } else if bytes.len() % (wgpu::COPY_BUFFER_ALIGNMENT as usize) == 0 {
            Cow::Borrowed(bytes)
        } else {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "interior sub-buffer byte length {} must be a multiple of {} bytes",
                    bytes.len(),
                    wgpu::COPY_BUFFER_ALIGNMENT
                ),
            });
        };
        self.write_bytes(buffer, byte_offset, payload.as_ref())
    }

    pub(super) fn write_bytes<T: Pod>(
        &self,
        buffer: &WgpuBuffer<T>,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        // WGPU resource IDs belong to an instance. A foreign handle can fail
        // resource lookup before WGPU reaches its scoped validation path.
        if !buffer.belongs_to(&self.pipeline_cache) {
            return Err(HephaestusError::DispatchFailed {
                message: "WGPU transfer destination belongs to a different device".to_owned(),
            });
        }
        self.buffer_operation("hephaestus-buffer-write", || {
            self.queue.write_buffer(buffer.raw(), offset, bytes);
            Ok(())
        })
    }
    /// Core GPU→host transfer: copy `padded` bytes from `src_buf[byte_offset..]`
    /// into a staging buffer, map it, and pass exactly `byte_len` bytes to
    /// `consume` before unmapping the staging allocation.
    ///
    /// `byte_len` ≤ `padded` must hold; `padded` must fit the alignment required by
    /// `wgpu::COPY_BUFFER_ALIGNMENT`. This is the SSOT for all synchronous
    /// device→host readback paths.
    pub(super) fn stage_and_read<R>(
        &self,
        src_buf: &wgpu::Buffer,
        region: ReadbackRegion,
        timeout: Duration,
        label: &str,
        consume: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        let ReadbackRegion {
            byte_offset,
            padded,
            byte_len,
        } = region;
        let staging = self.get_staging_buffer(padded)?;
        let staging_size = staging.size();
        // Capacity overflow is resolved before queue submission. The two
        // owners keep retained state quarantined until both this reader and
        // the WGPU callback have terminated, including error and unwind paths.
        let (completion, callback_completion) = self.map_completion_pool.acquire();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_buffer_to_buffer(src_buf, byte_offset, &staging, 0, padded);
        let submission_index = self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..staging_size);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            callback_completion.complete(result);
        });
        // Constructed before the poll so an elapsed deadline unmaps the
        // staging allocation on the way out, aborting the pending `map_async`
        // rather than recycling a buffer with a live mapping request.
        let mapping = MappingLifecycle(&staging);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(timeout),
            })
            .map_err(|error| poll_failure(label, timeout, &error))?;
        completion.result()?;

        // byte_len comes from byte_size::<T>(n) = n * size_of::<T>(), which fits usize.
        let byte_len_usize = usize::try_from(byte_len)
            .expect("invariant: byte_len fits usize (derived from element count * size_of::<T>())");
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| HephaestusError::TransferFailed {
                message: format!("mapped-range acquisition failed: {error}"),
            })?;
        let Some(bytes) = mapped.get(..byte_len_usize) else {
            drop(mapped);
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "mapped staging range is shorter than requested readback: {staging_size} < {byte_len} bytes"
                ),
            });
        };
        let result = consume(bytes);
        drop(mapped);
        drop(mapping);

        Ok(result)
    }
}
