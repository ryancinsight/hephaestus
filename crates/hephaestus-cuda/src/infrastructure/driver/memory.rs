//! CUDA memory entry points from the CUDA 13.3 driver header.

use super::loader::LoadError;
use crate::infrastructure::buffer::DevicePtr;
use core::ffi::c_void;
use libloading::Library;

#[derive(Debug)]
pub(crate) struct Functions {
    pub(crate) allocate: unsafe extern "system" fn(*mut DevicePtr, usize) -> i32,
    pub(crate) allocate_ordered:
        unsafe extern "system" fn(*mut DevicePtr, usize, *mut c_void) -> i32,
    pub(crate) free: unsafe extern "system" fn(DevicePtr) -> i32,
    pub(crate) free_ordered: unsafe extern "system" fn(DevicePtr, *mut c_void) -> i32,
    pub(crate) allocate_host: unsafe extern "system" fn(*mut *mut c_void, usize) -> i32,
    pub(crate) free_host: unsafe extern "system" fn(*mut c_void) -> i32,
    pub(crate) info: unsafe extern "system" fn(*mut usize, *mut usize) -> i32,
    pub(crate) fill: unsafe extern "system" fn(DevicePtr, u8, usize) -> i32,
    pub(crate) copy: unsafe extern "system" fn(DevicePtr, DevicePtr, usize) -> i32,
    pub(crate) download: unsafe extern "system" fn(*mut c_void, DevicePtr, usize) -> i32,
    pub(crate) upload: unsafe extern "system" fn(DevicePtr, *const c_void, usize) -> i32,
    pub(crate) copy_region: unsafe extern "system" fn(*const super::CopyRegion) -> i32,
    pub(crate) copy_region_async:
        unsafe extern "system" fn(*const super::CopyRegion, *mut c_void) -> i32,
}

impl Functions {
    pub(super) fn load(library: &Library) -> Result<Self, LoadError> {
        // SAFETY: each symbol is a CUDA driver export with the exact CUDAAPI
        // calling convention and signature declared in CUDA 13.3 cuda.h.
        // Driver retains the library for every copied function pointer.
        unsafe {
            Ok(Self {
                allocate: *library
                    .get(b"cuMemAlloc_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuMemAlloc_v2",
                        source,
                    })?,
                allocate_ordered: *library.get(b"cuMemAllocAsync\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemAllocAsync",
                        source,
                    }
                })?,
                free: *library
                    .get(b"cuMemFree_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuMemFree_v2",
                        source,
                    })?,
                free_ordered: *library.get(b"cuMemFreeAsync\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemFreeAsync",
                        source,
                    }
                })?,
                allocate_host: *library.get(b"cuMemAllocHost_v2\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemAllocHost_v2",
                        source,
                    }
                })?,
                free_host: *library.get(b"cuMemFreeHost\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemFreeHost",
                        source,
                    }
                })?,
                info: *library
                    .get(b"cuMemGetInfo_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuMemGetInfo_v2",
                        source,
                    })?,
                fill: *library
                    .get(b"cuMemsetD8_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuMemsetD8_v2",
                        source,
                    })?,
                copy: *library
                    .get(b"cuMemcpyDtoD_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuMemcpyDtoD_v2",
                        source,
                    })?,
                download: *library.get(b"cuMemcpyDtoH_v2\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemcpyDtoH_v2",
                        source,
                    }
                })?,
                upload: *library
                    .get(b"cuMemcpyHtoD_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuMemcpyHtoD_v2",
                        source,
                    })?,
                copy_region: *library.get(b"cuMemcpy2D_v2\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemcpy2D_v2",
                        source,
                    }
                })?,
                copy_region_async: *library.get(b"cuMemcpy2DAsync_v2\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuMemcpy2DAsync_v2",
                        source,
                    }
                })?,
            })
        }
    }
}
