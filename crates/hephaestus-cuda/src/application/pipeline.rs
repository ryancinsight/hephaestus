use crate::CudaDevice;
use hephaestus_core::{BlockWidth, HephaestusError, Result};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::infrastructure::compiler::SafeCachedKernel;

#[cfg(not(feature = "cuda"))]
/// Stub cached kernel.
pub struct SafeCachedKernel;

/// Retrieve a cached kernel, compiling the source if it is not present in the cache.
pub fn cached_kernel(
    device: &CudaDevice,
    key: String,
    func_name: &str,
    source: impl FnOnce() -> String,
) -> Result<Arc<SafeCachedKernel>> {
    #[cfg(feature = "cuda")]
    {
        if let Some(cached) = device
            .pipeline_cache
            .get(&key)
            .expect("invariant: pipeline cache is not poisoned")
        {
            return Ok(cached);
        }

        let src = source();
        let ptx = crate::infrastructure::compiler::compile_cuda_to_ptx(&src).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("CUDA compilation failed for {func_name}: {e}"),
            }
        })?;

        let ptx_c = std::ffi::CString::new(ptx).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("PTX is not a valid CString: {e}"),
        })?;

        let func_name_c =
            std::ffi::CString::new(func_name).map_err(|e| HephaestusError::DispatchFailed {
                message: format!("kernel name is not a valid CString: {e}"),
            })?;

        let mut module: cuda_core::sys::CUmodule = std::ptr::null_mut();
        // SAFETY: The driver and context must be initialized. CudaDevice::try_default() has bound the context.
        // `module` is a valid out-pointer.
        unsafe {
            let res = cuda_core::sys::cuModuleLoadData(
                &mut module as *mut cuda_core::sys::CUmodule,
                ptx_c.as_ptr() as *const std::ffi::c_void,
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuModuleLoadData failed with code: {res}"),
                });
            }

            let mut func: cuda_core::sys::CUfunction = std::ptr::null_mut();
            let res = cuda_core::sys::cuModuleGetFunction(
                &mut func as *mut cuda_core::sys::CUfunction,
                module as *mut _,
                func_name_c.as_ptr(),
            );
            if res != 0 {
                let _ = cuda_core::sys::cuModuleUnload(module as *mut _);
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuModuleGetFunction('{func_name}') failed with code: {res}"),
                });
            }

            let kernel = Arc::new(SafeCachedKernel { module, func });
            if let Some(cached) = device
                .pipeline_cache
                .insert(key, kernel.clone())
                .expect("invariant: pipeline cache is not poisoned")
            {
                Ok(cached)
            } else {
                Ok(kernel)
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, key, func_name, source);
        Err(HephaestusError::AdapterUnavailable {
            message: "hephaestus-cuda built without the `cuda` feature".to_string(),
        })
    }
}

/// Convert a logical work-item count into CUDA grid size (block count).
pub fn grid_size(len: usize, width: BlockWidth) -> Result<u32> {
    let len_u64 = u64::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("dispatch size {len} exceeds u64 range"),
    })?;
    let checked =
        width
            .checked_covering_blocks(len_u64)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("dispatch size {len} exceeds u32 grid range"),
            })?;
    let budget = mnemosyne_core::KernelResourceBudget::new(0, 0, width.get())
        .expect("invariant: BlockWidth is non-zero, so budget threads are non-zero");
    let planned = moirai_gpu::plan_launch(budget, len_u64);
    debug_assert_eq!(planned.threads_per_block, width.get());
    debug_assert_eq!(planned.grid_blocks, checked);
    Ok(planned.grid_blocks)
}
