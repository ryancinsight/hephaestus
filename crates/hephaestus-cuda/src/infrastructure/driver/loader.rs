//! Process-lifetime ownership of driver exports and initialization failures.

use super::{context, device, kernel, memory};
use hephaestus_core::{HephaestusError, Result};
use libloading::Library;
use std::sync::OnceLock;

#[cfg(target_os = "windows")]
const LIBRARY_NAME: &str = "nvcuda.dll";
#[cfg(not(target_os = "windows"))]
const LIBRARY_NAME: &str = "libcuda.so.1";

#[derive(Debug)]
pub(super) enum LoadError {
    Library {
        name: String,
        source: libloading::Error,
        absent: bool,
    },
    Symbol {
        name: &'static str,
        source: libloading::Error,
    },
    Initialize(i32),
}

impl LoadError {
    pub(super) fn report(&self) -> HephaestusError {
        match self {
            Self::Library {
                name,
                source,
                absent,
            } => {
                let message = format!("CUDA driver library {name}: {source:?}");
                if *absent {
                    HephaestusError::AdapterUnavailable { message }
                } else {
                    HephaestusError::DeviceUnavailable { message }
                }
            }
            Self::Symbol { name, source } => HephaestusError::DeviceUnavailable {
                message: format!("CUDA driver required symbol {name}: {source:?}"),
            },
            // CUDA's CUresult contract defines 34 as a loaded stub library:
            // no real driver is available through this process's library.
            Self::Initialize(34) => HephaestusError::AdapterUnavailable {
                message: "cuInit -> 34 (CUDA_ERROR_STUB_LIBRARY)".to_owned(),
            },
            Self::Initialize(100) => HephaestusError::AdapterUnavailable {
                message: "cuInit -> 100 (CUDA_ERROR_NO_DEVICE)".to_owned(),
            },
            Self::Initialize(status) => HephaestusError::DeviceUnavailable {
                message: format!("cuInit -> {status}"),
            },
        }
    }
}

#[derive(Debug)]
pub(crate) struct Driver {
    pub(crate) context: context::Functions,
    pub(crate) device: device::Functions,
    pub(crate) memory: memory::Functions,
    pub(crate) kernel: kernel::Functions,
    // The process-owned table cannot outlive this library. Contexts borrow the
    // table, and allocations/modules retain their context through Arc.
    _library: Library,
}

static DRIVER: OnceLock<std::result::Result<Driver, LoadError>> = OnceLock::new();

impl Driver {
    pub(crate) fn get() -> Result<&'static Self> {
        DRIVER
            .get_or_init(|| {
                let driver = Self::load(LIBRARY_NAME)?;
                // SAFETY: the retained library provides cuInit with its verified
                // CUDAAPI signature; zero is the only supported initialization flag.
                let status = unsafe { (driver.device.initialize)(0) };
                if status != 0 {
                    return Err(LoadError::Initialize(status));
                }
                Ok(driver)
            })
            .as_ref()
            .map_err(LoadError::report)
    }

    pub(super) fn load(name: &str) -> std::result::Result<Self, LoadError> {
        let library = super::library::open(name).map_err(|source| LoadError::Library {
            absent: super::library::absent(name, &source),
            name: name.to_owned(),
            source,
        })?;
        Ok(Self {
            context: context::Functions::load(&library)?,
            device: device::Functions::load(&library)?,
            memory: memory::Functions::load(&library)?,
            kernel: kernel::Functions::load(&library)?,
            _library: library,
        })
    }
}
