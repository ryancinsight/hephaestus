//! CUDA device entry points from the CUDA 13.3 driver header.

use super::Attribute;
use super::loader::LoadError;
use libloading::Library;

#[derive(Debug)]
pub(crate) struct Functions {
    pub(crate) initialize: unsafe extern "system" fn(u32) -> i32,
    pub(crate) get: unsafe extern "system" fn(*mut i32, i32) -> i32,
    pub(crate) count: unsafe extern "system" fn(*mut i32) -> i32,
    pub(crate) attribute: unsafe extern "system" fn(*mut i32, Attribute, i32) -> i32,
}

impl Functions {
    pub(super) fn load(library: &Library) -> Result<Self, LoadError> {
        // SAFETY: each symbol is a CUDA driver export with the exact CUDAAPI
        // calling convention and signature declared in CUDA 13.3 cuda.h.
        // Driver retains the library for every copied function pointer.
        unsafe {
            Ok(Self {
                initialize: *library
                    .get(b"cuInit\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuInit",
                        source,
                    })?,
                get: *library
                    .get(b"cuDeviceGet\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuDeviceGet",
                        source,
                    })?,
                count: *library
                    .get(b"cuDeviceGetCount\0")
                    .map_err(|source| LoadError::Symbol {
                        name: "cuDeviceGetCount",
                        source,
                    })?,
                attribute: *library.get(b"cuDeviceGetAttribute\0").map_err(|source| {
                    LoadError::Symbol {
                        name: "cuDeviceGetAttribute",
                        source,
                    }
                })?,
            })
        }
    }
}
