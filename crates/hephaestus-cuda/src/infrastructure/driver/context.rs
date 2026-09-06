//! CUDA context entry points from the CUDA 13.3 driver header.

use super::loader::LoadError;
use core::ffi::c_void;
use libloading::Library;

#[derive(Debug)]
pub(crate) struct Functions {
    pub(crate) create: unsafe extern "system" fn(*mut *mut c_void, u32, i32) -> i32,
    pub(crate) destroy: unsafe extern "system" fn(*mut c_void) -> i32,
    pub(crate) current: unsafe extern "system" fn(*mut *mut c_void) -> i32,
    pub(crate) bind: unsafe extern "system" fn(*mut c_void) -> i32,
    pub(crate) synchronize: unsafe extern "system" fn() -> i32,
    pub(crate) synchronize_stream: unsafe extern "system" fn(*mut c_void) -> i32,
}

impl Functions {
    pub(super) fn load(library: &Library) -> Result<Self, LoadError> {
        // SAFETY: each symbol is a CUDA driver export with the exact CUDAAPI
        // calling convention and signature declared in CUDA 13.3 cuda.h.
        // Driver retains the library for every copied function pointer.
        unsafe {
            Ok(Self {
                create: *library
                    .get(b"cuCtxCreate_v2\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuCtxCreate_v2",
                        source,
                    })?,
                destroy: *library.get(b"cuCtxDestroy_v2\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuCtxDestroy_v2",
                        source,
                    }
                })?,
                current: *library.get(b"cuCtxGetCurrent\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuCtxGetCurrent",
                        source,
                    }
                })?,
                bind: *library
                    .get(b"cuCtxSetCurrent\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuCtxSetCurrent",
                        source,
                    })?,
                synchronize: *library.get(b"cuCtxSynchronize\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuCtxSynchronize",
                        source,
                    }
                })?,
                synchronize_stream: *library.get(b"cuStreamSynchronize\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuStreamSynchronize",
                        source,
                    }
                })?,
            })
        }
    }
}
