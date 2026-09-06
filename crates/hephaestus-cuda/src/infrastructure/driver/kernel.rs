//! CUDA kernel entry points from the CUDA 13.3 driver header.

use super::loader::LoadError;
use core::ffi::c_char;
use core::ffi::c_void;
use libloading::Library;

#[derive(Debug)]
pub(crate) struct Functions {
    pub(crate) load: unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> i32,
    pub(crate) function:
        unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32,
    pub(crate) unload: unsafe extern "system" fn(*mut c_void) -> i32,
    pub(crate) launch: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
}

impl Functions {
    pub(super) fn load(library: &Library) -> Result<Self, LoadError> {
        // SAFETY: each symbol is a CUDA driver export with the exact CUDAAPI
        // calling convention and signature declared in CUDA 13.3 cuda.h.
        // Driver retains the library for every copied function pointer.
        unsafe {
            Ok(Self {
                load: *library
                    .get(b"cuModuleLoadData\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuModuleLoadData",
                        source,
                    })?,
                function: *library.get(b"cuModuleGetFunction\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuModuleGetFunction",
                        source,
                    }
                })?,
                unload: *library
                    .get(b"cuModuleUnload\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuModuleUnload",
                        source,
                    })?,
                launch: *library
                    .get(b"cuLaunchKernel\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuLaunchKernel",
                        source,
                    })?,
            })
        }
    }
}
