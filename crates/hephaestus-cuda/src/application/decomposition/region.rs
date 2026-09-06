#![cfg(feature = "cuda")]

//! Rectangular transfers preserve device pitch and compact host storage.
//! Downloads own pinned storage until stream completion. Uploads borrow host
//! storage and use the synchronous driver contract so no DMA outlives the borrow.

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use crate::infrastructure::driver::{CopyRegion, Endpoint};
use crate::infrastructure::pinned::PinnedHostBuffer;
use hephaestus_core::{HephaestusError, Result};

#[derive(Clone, Copy)]
pub(crate) struct MatrixRegion {
    pub(crate) stride: usize,
    pub(crate) row_start: usize,
    pub(crate) col_start: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

struct RegionExtent {
    pointer: u64,
    pitch: usize,
    row_bytes: usize,
    compact_len: usize,
}

impl MatrixRegion {
    fn extent(self, buffer: &CudaBuffer<f32>) -> Result<RegionExtent> {
        let overflow = || HephaestusError::TransferFailed {
            message: "matrix region extent overflows its address space".to_owned(),
        };
        let row_end = self.col_start.checked_add(self.cols).ok_or_else(overflow)?;
        if row_end > self.stride {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "matrix region column end {row_end} exceeds stride {}",
                    self.stride
                ),
            });
        }
        let start = self
            .row_start
            .checked_mul(self.stride)
            .and_then(|start| start.checked_add(self.col_start))
            .ok_or_else(overflow)?;
        let needed = self
            .rows
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(self.stride))
            .and_then(|tail| start.checked_add(tail))
            .and_then(|tail| tail.checked_add(self.cols))
            .ok_or_else(overflow)?;
        if needed > buffer.len {
            return Err(HephaestusError::LengthMismatch {
                host_len: needed,
                device_len: buffer.len,
            });
        }
        let element_bytes = core::mem::size_of::<f32>();
        let offset = start.checked_mul(element_bytes).ok_or_else(overflow)?;
        let offset = u64::try_from(offset).map_err(|_| overflow())?;
        Ok(RegionExtent {
            pointer: buffer.raw().checked_add(offset).ok_or_else(overflow)?,
            pitch: self
                .stride
                .checked_mul(element_bytes)
                .ok_or_else(overflow)?,
            row_bytes: self.cols.checked_mul(element_bytes).ok_or_else(overflow)?,
            compact_len: self.rows.checked_mul(self.cols).ok_or_else(overflow)?,
        })
    }
}

pub(crate) fn download_matrix_region_compact(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
    region: MatrixRegion,
) -> Result<PinnedHostBuffer<f32>> {
    if region.rows == 0 || region.cols == 0 {
        // SAFETY: an empty allocation has no elements requiring initialization.
        return unsafe { PinnedHostBuffer::uninitialized(device.cuda_context().clone(), 0) };
    }
    let extent = region.extent(buffer)?;
    device.bind()?;
    // SAFETY: the entire compact allocation is written by the rectangular
    // transfer and synchronized before any slice reference or host read exists.
    let mut compact = unsafe {
        PinnedHostBuffer::<f32>::uninitialized(device.cuda_context().clone(), extent.compact_len)?
    };
    let copy = CopyRegion::new(
        Endpoint::device(extent.pointer, extent.pitch),
        Endpoint::host(compact.as_mut_ptr(), extent.row_bytes),
        extent.row_bytes,
        region.rows,
    );
    // SAFETY: both endpoints cover the validated rectangle; the destination
    // is pinned and exclusively owned. The descriptor is consumed at enqueue;
    // compact remains alive until stream completion, including error paths.
    let status =
        unsafe { (device.driver().memory.copy_region_async)(&copy, core::ptr::null_mut()) };
    if let Err(drain) = device.synchronize_default_stream() {
        // A failed barrier cannot prove DMA completion. Keeping the owning
        // allocation alive prevents freeing a possibly active DMA destination.
        core::mem::forget(compact);
        return Err(HephaestusError::TransferFailed {
            message: format!(
                "cuMemcpy2DAsync_v2 -> {status}; stream drain failed ({drain}); pinned destination retained"
            ),
        });
    }
    if status != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("cuMemcpy2DAsync_v2 -> {status}"),
        });
    }
    Ok(compact)
}

pub(crate) fn write_matrix_region_compact(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
    compact_host: &[f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }
    let extent = region.extent(buffer)?;
    if compact_host.len() != extent.compact_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: compact_host.len(),
            device_len: extent.compact_len,
        });
    }
    device.bind()?;
    let copy = CopyRegion::new(
        Endpoint::host(compact_host.as_ptr().cast(), extent.row_bytes),
        Endpoint::device(extent.pointer, extent.pitch),
        extent.row_bytes,
        region.rows,
    );
    // SAFETY: the length-checked host slice and bounds-checked device rectangle
    // remain live during this synchronous copy. CUDA consumes host storage
    // before return, so a driver error cannot leave a borrowed DMA source live.
    let status = unsafe { (device.driver().memory.copy_region)(&copy) };
    if status != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("cuMemcpy2D_v2 -> {status}"),
        });
    }
    device.synchronize_default_stream()
}

#[cfg(test)]
mod tests {
    use super::{MatrixRegion, download_matrix_region_compact, write_matrix_region_compact};
    use crate::CudaDevice;
    use hephaestus_core::{ComputeDevice, HephaestusError};

    #[test]
    fn rectangular_transfers_preserve_pitch_offsets_and_surrounding_values() {
        let device = match CudaDevice::try_default() {
            Ok(device) => device,
            Err(HephaestusError::AdapterUnavailable { .. })
                if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
            {
                return;
            }
            Err(error) => panic!("CUDA rectangular transfer requires a working device: {error}"),
        };
        // Exact small integers require no floating-point tolerance: transfers
        // preserve bits and perform no arithmetic on the matrix elements.
        let original: Vec<f32> = (0_u16..35).map(f32::from).collect();
        let buffer = device
            .upload(&original)
            .expect("upload five rows with pitch seven");
        let region = MatrixRegion {
            stride: 7,
            row_start: 1,
            col_start: 2,
            rows: 3,
            cols: 2,
        };
        let compact = download_matrix_region_compact(&device, &buffer, region)
            .expect("download offset rectangle");
        assert_eq!(&*compact, &[9.0, 10.0, 16.0, 17.0, 23.0, 24.0]);
        let replacement = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        write_matrix_region_compact(&device, &buffer, &replacement, region)
            .expect("upload compact rectangle");
        let mut actual = vec![0.0; original.len()];
        device
            .download(&buffer, &mut actual)
            .expect("read entire matrix");
        for (index, (&actual, &original)) in actual.iter().zip(&original).enumerate() {
            let expected = match index {
                9 => -1.0,
                10 => -2.0,
                16 => -3.0,
                17 => -4.0,
                23 => -5.0,
                24 => -6.0,
                _ => original,
            };
            assert_eq!(actual, expected, "matrix element {index}");
        }

        let oversized = MatrixRegion {
            stride: 7,
            row_start: 4,
            col_start: 2,
            rows: 2,
            cols: 2,
        };
        match download_matrix_region_compact(&device, &buffer, oversized) {
            Err(HephaestusError::LengthMismatch {
                host_len,
                device_len,
            }) => {
                assert_eq!(host_len, 39);
                assert_eq!(device_len, 35);
            }
            Err(error) => panic!("wrong oversized extent error: {error}"),
            Ok(_) => panic!("oversized rectangle was accepted"),
        }
        let overflow = MatrixRegion {
            stride: 7,
            row_start: usize::MAX,
            col_start: 2,
            rows: 1,
            cols: 2,
        };
        match write_matrix_region_compact(&device, &buffer, &[1.0, 2.0], overflow) {
            Err(HephaestusError::TransferFailed { message }) => {
                assert_eq!(message, "matrix region extent overflows its address space")
            }
            other => panic!("wrong overflowing extent result: {other:?}"),
        }
        let mut after_rejection = vec![0.0; original.len()];
        device
            .download(&buffer, &mut after_rejection)
            .expect("read after rejected transfers");
        assert_eq!(
            after_rejection, actual,
            "rejected transfers do not write memory"
        );
    }
}
