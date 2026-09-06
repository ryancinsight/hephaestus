//! CUDA_MEMCPY2D_v2 layout from CUDA 13.3 cuda.h (lines 3456–3476).

use crate::infrastructure::buffer::DevicePtr;
use core::ffi::c_void;

/// The identical source/destination field sequence in CUDA_MEMCPY2D_v2.
#[repr(C)]
pub(crate) struct Endpoint {
    x_bytes: usize,
    y: usize,
    memory_type: i32,
    host: *const c_void,
    device: DevicePtr,
    array: *mut c_void,
    pitch: usize,
}

impl Endpoint {
    pub(crate) fn host(pointer: *const c_void, pitch: usize) -> Self {
        Self {
            x_bytes: 0,
            y: 0,
            memory_type: 1,
            host: pointer,
            device: 0,
            array: core::ptr::null_mut(),
            pitch,
        }
    }

    pub(crate) fn device(pointer: DevicePtr, pitch: usize) -> Self {
        Self {
            x_bytes: 0,
            y: 0,
            memory_type: 2,
            host: core::ptr::null(),
            device: pointer,
            array: core::ptr::null_mut(),
            pitch,
        }
    }
}

#[repr(C)]
pub(crate) struct CopyRegion {
    source: Endpoint,
    destination: Endpoint,
    width_bytes: usize,
    height: usize,
}

impl CopyRegion {
    pub(crate) fn new(
        source: Endpoint,
        destination: Endpoint,
        width_bytes: usize,
        height: usize,
    ) -> Self {
        Self {
            source,
            destination,
            width_bytes,
            height,
        }
    }
}

// CUDA's 64-bit ABI uses 8-byte size_t, addresses, pitches and handles;
// CUmemorytype is a 4-byte enum followed by pointer-alignment padding.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(core::mem::size_of::<Endpoint>() == 56);
    assert!(core::mem::align_of::<CopyRegion>() == 8);
    assert!(core::mem::size_of::<CopyRegion>() == 128);
    assert!(core::mem::offset_of!(Endpoint, host) == 24);
    assert!(core::mem::offset_of!(Endpoint, device) == 32);
    assert!(core::mem::offset_of!(Endpoint, pitch) == 48);
    assert!(core::mem::offset_of!(CopyRegion, destination) == 56);
    assert!(core::mem::offset_of!(CopyRegion, width_bytes) == 112);
    assert!(core::mem::offset_of!(CopyRegion, height) == 120);
};
