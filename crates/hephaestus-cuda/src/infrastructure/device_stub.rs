use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::infrastructure::buffer::CudaBuffer;

/// Stub CUDA device for builds without the `cuda` feature.
///
/// Acquisition always reports [`HephaestusError::AdapterUnavailable`], so a
/// consumer that binds generically (`<D: ComputeDevice>`) still compiles
/// against the CUDA backend but observes an honest "unavailable" failure
/// instead of a fabricated device. Rebuild with `--features cuda` to enable
/// the real backend. The transfer methods are unreachable (no instance is ever
/// constructed) but defined so the [`ComputeDevice`] contract is satisfied.
#[derive(Clone, Debug)]
pub struct CudaDevice {
    _private: (),
    #[allow(dead_code)]
    pub(crate) pipeline_cache: std::sync::Arc<moirai_sync::sync::ConcurrentHashMap<String, ()>>,
    #[allow(dead_code)]
    topology: Option<std::sync::Arc<themis::GpuTopology>>,
}

impl CudaDevice {
    /// Report the CUDA backend unavailable in a `cuda`-feature-less build.
    pub fn try_default() -> Result<Self> {
        Err(HephaestusError::AdapterUnavailable {
            message: "hephaestus-cuda built without the `cuda` feature; \
                      rebuild with --features cuda"
                .to_string(),
        })
    }

    /// The device topology snapshot captured at acquisition, when available.
    #[must_use]
    #[inline]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        None
    }

    fn unavailable<R>() -> Result<R> {
        Err(HephaestusError::AdapterUnavailable {
            message: "hephaestus-cuda built without the `cuda` feature".to_string(),
        })
    }
}

impl ComputeDevice for CudaDevice {
    type Buffer<T: Pod> = CudaBuffer<T>;

    #[inline]
    fn backend_name(&self) -> &'static str {
        "cuda"
    }

    fn alloc_zeroed<T: Pod>(&self, _len: usize) -> Result<Self::Buffer<T>> {
        Self::unavailable()
    }

    fn upload<T: Pod>(&self, _host: &[T]) -> Result<Self::Buffer<T>> {
        Self::unavailable()
    }

    fn download<T: Pod>(&self, _buffer: &Self::Buffer<T>, _out: &mut [T]) -> Result<()> {
        Self::unavailable()
    }

    fn write_buffer<T: Pod>(&self, _buffer: &Self::Buffer<T>, _host: &[T]) -> Result<()> {
        Self::unavailable()
    }
}
