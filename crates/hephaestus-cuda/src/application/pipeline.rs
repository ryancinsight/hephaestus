use crate::CudaDevice;
use hephaestus_core::{BlockWidth, HephaestusError, Result};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

mod key;

pub(crate) use key::PipelineKey;

#[cfg(feature = "cuda")]
pub(crate) use crate::infrastructure::compiler::SafeCachedKernel;

#[cfg(not(feature = "cuda"))]
/// Stub cached kernel.
pub(crate) struct SafeCachedKernel;

/// Hash a runtime-authored kernel's complete cache identity once at preparation.
pub(crate) fn source_hash(label: &str, entry: &str, source: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    entry.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

/// Retrieve a cached kernel, compiling the source if it is not present in the cache.
///
/// Only successful compilations are cached: a failed NVRTC compile or module
/// load returns the error and leaves the cache slot empty, so a transient
/// driver failure (out-of-memory, TDR reset) does not poison the key for the
/// device's lifetime. Two threads racing on a cold key may both compile and
/// one module is dropped — bounded first-use-only waste, preferred over
/// caching failures or holding a lock across a 10–100 ms NVRTC compile.
pub(crate) fn cached_kernel(
    device: &CudaDevice,
    key: PipelineKey,
    func_name: &str,
    source: impl FnOnce() -> String,
) -> Result<Arc<SafeCachedKernel>> {
    #[cfg(feature = "cuda")]
    {
        let cell = device
            .pipeline_cache
            .get_or_insert_with(key, || std::sync::Arc::new(std::sync::OnceLock::new()))
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("pipeline cache segment poisoned: {e}"),
            })?;
        if let Some(kernel) = cell.get() {
            return Ok(kernel.clone());
        }

        // Compile outside any cache lock. Module loading requires this
        // device's context current on the calling thread.
        device.bind()?;
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

        let mut module: cuda_oxide::sys::CUmodule = std::ptr::null_mut();
        // SAFETY: this device's context is current on this thread (`bind`
        // above); `ptx_c` is a NUL-terminated PTX image kept alive across the
        // call; `module` is a valid out-pointer for one `CUmodule`.
        let compiled = unsafe {
            let res = cuda_oxide::sys::cuModuleLoadData(
                &mut module as *mut cuda_oxide::sys::CUmodule,
                ptx_c.as_ptr() as *const std::ffi::c_void,
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuModuleLoadData failed with code: {res}"),
                });
            }

            let mut func: cuda_oxide::sys::CUfunction = std::ptr::null_mut();
            let res = cuda_oxide::sys::cuModuleGetFunction(
                &mut func as *mut cuda_oxide::sys::CUfunction,
                module as *mut _,
                func_name_c.as_ptr(),
            );
            if res != 0 {
                let unload = cuda_oxide::sys::cuModuleUnload(module as *mut _);
                debug_assert_eq!(unload, 0, "cuModuleUnload during error cleanup");
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuModuleGetFunction('{func_name}') failed with code: {res}"),
                });
            }

            Arc::new(SafeCachedKernel::new(
                module,
                func,
                device.cuda_context().clone(),
            ))
        };

        // Another thread may have won the race; its kernel is kept and ours
        // drops (module unload via SafeCachedKernel::drop).
        Ok(cell.get_or_init(|| compiled).clone())
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, key, func_name, source);
        Err(HephaestusError::AdapterUnavailable {
            message: "hephaestus-cuda built without the `cuda` feature".to_string(),
        })
    }
}

/// Grid/block launch configuration for [`launch_kernel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchConfig {
    /// Grid dimensions in blocks (x, y, z).
    pub grid: (u32, u32, u32),
    /// Block dimensions in threads (x, y, z).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory bytes per block.
    pub shared_bytes: u32,
}

impl LaunchConfig {
    /// One-dimensional launch: `grid_x` blocks of `width` threads.
    #[must_use]
    pub(crate) const fn linear(grid_x: u32, width: BlockWidth) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (width.get(), 1, 1),
            shared_bytes: 0,
        }
    }

    /// One-dimensional launch with dynamic shared memory per block.
    #[must_use]
    pub(crate) const fn linear_shared(grid_x: u32, width: BlockWidth, shared_bytes: u32) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (width.get(), 1, 1),
            shared_bytes,
        }
    }

    /// Two-dimensional launch: `grid_x` × `grid_y` blocks of
    /// `block_x` × `block_y` threads.
    #[must_use]
    pub(crate) const fn planar(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_bytes: 0,
        }
    }

    /// Three-dimensional launch: `grid_x` × `grid_y` × `grid_z` blocks of
    /// `block_x` × `block_y` threads (the batch/z dimension carries one
    /// thread block's worth of work per z-slice, not per-thread depth).
    #[must_use]
    pub(crate) const fn batched_planar(
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
    ) -> Self {
        Self {
            grid: (grid_x, grid_y, grid_z),
            block: (block_x, block_y, 1),
            shared_bytes: 0,
        }
    }
}

/// Launch a cached kernel on this device (single source of truth for
/// `cuLaunchKernel`).
///
/// Binds the device's context to the calling thread first: CUDA contexts are
/// thread-affine and `CudaDevice` is `Clone + Send`, so the caller's thread
/// may not be the acquiring thread. Launches on the legacy null stream.
///
/// The launch is asynchronous on every target: errors from kernel *execution*
/// surface at the next synchronizing operation, not from this call. Launch
/// rejection is synchronous and is reported here.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when the driver rejects the
/// launch (bad handle, invalid configuration, resource exhaustion).
#[cfg(feature = "cuda")]
pub(crate) fn launch_kernel(
    device: &CudaDevice,
    kernel: &SafeCachedKernel,
    config: LaunchConfig,
    args: &mut [*mut core::ffi::c_void],
) -> Result<()> {
    device.bind()?;
    // SAFETY: this device's context is current on this thread (`bind` above);
    // `kernel.func` is a live function handle whose module the caller keeps
    // alive (Arc) for at least the duration of this call; `args` mirrors the
    // kernel's `extern "C"` parameter list in order and type, each entry
    // pointing to a live caller local that outlives this call (the driver
    // copies argument VALUES at launch). Device pointers passed as arguments
    // stay valid until the asynchronous kernel completes: buffer deallocation
    // routes through `cuMemFree`-family calls, which the driver orders after
    // in-flight work on the default stream (implicit synchronization on free).
    let res = unsafe {
        cuda_oxide::sys::cuLaunchKernel(
            kernel.func,
            config.grid.0,
            config.grid.1,
            config.grid.2,
            config.block.0,
            config.block.1,
            config.block.2,
            config.shared_bytes,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if res != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "cuLaunchKernel failed with code {res} (grid {:?}, block {:?}, shared {} B)",
                config.grid, config.block, config.shared_bytes
            ),
        });
    }

    // A Windows-only `cuCtxSynchronize` drain stood here. It guarded WDDM's
    // lack of concurrent host/device access to `cuMemAllocManaged` ranges: a
    // host touchpoint issued while a kernel was in flight faulted with
    // STATUS_IN_PAGE_ERROR (0xc0000006).
    //
    // That premise no longer holds. The backend allocates only `cuMemAlloc_v2`
    // (`infrastructure/device.rs`, which documents the choice as deliberately
    // non-managed), so no managed range exists for WDDM to fault on. The drain
    // outlived the allocation strategy it was written for.
    //
    // Removed on run evidence, on a WDDM RTX 5080 (driver 610.47, CUDA 13.3,
    // compute 12.0 — a GeForce part, so WDDM is the only available driver
    // model and this is the exact configuration the drain targeted):
    //
    // - Launch throughput, 2000 back-to-back launches over 1024 f32, best of 7
    //   interleaved blocks in one process: 29.750 us/launch with the drain,
    //   6.070 us/launch without — 4.9x. The drain serialized every launch.
    // - `cuda_launch_survives_host_allocation_in_flight` reproduces the exact
    //   cited scenario — host alloc/upload/free with a launch outstanding and
    //   no intervening sync — and does not fault.
    // - The package suite is green without the drain (152/152).
    //
    // Error attribution, the drain's second effect, is deliberately given up:
    // kernel *execution* faults now surface at the next synchronizing
    // operation, as they already did on every non-Windows target. Launch
    // rejection is unaffected — `cuLaunchKernel` reports that synchronously
    // and is checked above.
    Ok(())
}

/// Stub launch for builds without the `cuda` feature: reports the backend
/// unavailable instead of silently succeeding. Unreachable in practice — the
/// stub device cannot be constructed and [`cached_kernel`] errors first — but
/// kept honest so no call path fabricates success.
#[cfg(not(feature = "cuda"))]
pub(crate) fn launch_kernel(
    device: &CudaDevice,
    _kernel: &SafeCachedKernel,
    _config: LaunchConfig,
    _args: &mut [*mut core::ffi::c_void],
) -> Result<()> {
    let _ = device;
    Err(HephaestusError::AdapterUnavailable {
        message: "hephaestus-cuda built without the `cuda` feature".to_string(),
    })
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
