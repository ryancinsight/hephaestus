use crate::CudaDevice;
use hephaestus_core::{BlockWidth, HephaestusError, Result, ScanDirection};
use std::any::TypeId;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::infrastructure::compiler::SafeCachedKernel;

#[cfg(not(feature = "cuda"))]
/// Stub cached kernel.
pub struct SafeCachedKernel;

/// Pipeline-cache key for a compiled CUDA kernel.
///
/// One variant per distinct shader source / `extern "C"` entry point in this
/// crate, so two call sites can never collide even when they share the same
/// `Op`/`T` type parameters (e.g. `binary_elementwise_into` and
/// `scalar_elementwise_into` are both generic over `Op: BinaryExpr`, but
/// compile to different kernels — a bare `(TypeId::of::<Op>(), ..)` tuple
/// shared across both would alias the wrong compiled function). Mirrors
/// `hephaestus-wgpu`'s `(TypeId, TypeId, u32)` pipeline key (CU-P9/P10):
/// `Copy`, no heap allocation, no `format!`/`type_name` string building on
/// every dispatch (including cache hits).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PipelineKey {
    Binary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    Scalar {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    Unary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    Reduction {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    AxisReduction {
        op: TypeId,
        scalar: TypeId,
        axis: usize,
        width: u32,
    },
    MeanAxis {
        scalar: TypeId,
        axis: usize,
        width: u32,
    },
    AxisScan {
        marker: TypeId,
        scalar: TypeId,
        direction: ScanDirection,
        axis: usize,
        width: u32,
    },
    Kron {
        marker: TypeId,
        scalar: TypeId,
    },
    Matmul {
        marker: TypeId,
        scalar: TypeId,
    },
    MatrixRank {
        marker: TypeId,
        scalar: TypeId,
    },
    Spmm {
        marker: TypeId,
        scalar: TypeId,
    },
    Spmv {
        marker: TypeId,
        scalar: TypeId,
    },
    StridedBinary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    StridedUnary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    StridedScalar {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Fixed non-generic decomposition kernels: one f32 shader each, no
    /// `Op`/`T` type parameter to key on. Only constructed by the
    /// `cuda`-feature decomposition modules.
    #[cfg(feature = "cuda")]
    CholeskySyrk,
    #[cfg(feature = "cuda")]
    LuGemm,
    #[cfg(feature = "cuda")]
    QrHouseholder,
    /// Runtime-authored kernels (the ADR-0004 `KernelSource<L>` seam and the
    /// legacy multi-storage API): `K::LABEL`/`K::ENTRY`/the compiled source
    /// text vary per value, not per Rust type, so `TypeId` cannot key these —
    /// `source_hash(label, entry, source)` (already computed once at
    /// prepare/construction time) is the uniqueness signal. Each variant
    /// still gets its own tag so a stream kernel, a grouped-stream kernel,
    /// and a legacy multi-storage kernel sharing identical label/entry/source
    /// text never alias.
    Stream(u64),
    GroupedStream(u64),
    MultiStorage(u64),
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
}

/// Launch a cached kernel on this device (single source of truth for
/// `cuLaunchKernel`).
///
/// Binds the device's context to the calling thread first: CUDA contexts are
/// thread-affine and `CudaDevice` is `Clone + Send`, so the caller's thread
/// may not be the acquiring thread. Launches on the legacy null stream. On
/// non-Windows targets, the launch remains asynchronous and errors from kernel
/// *execution* surface at the next synchronizing operation. On Windows, the
/// WDDM managed-memory drain below makes kernel completion and execution
/// errors observable before returning.
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

    // WDDM managed-memory safety: on Windows the default driver model (WDDM)
    // does not support concurrent host/device access to `cuMemAllocManaged`
    // ranges. A subsequent host touchpoint — allocating the next intermediate
    // buffer, or the driver's own managed-heap bookkeeping — issued while this
    // kernel is still in flight on the null stream faults with
    // STATUS_IN_PAGE_ERROR (0xc0000006). Draining the context after the launch
    // makes the completion explicit before any such host access. The backend
    // is already null-stream-serial, so on Windows this only surfaces the
    // existing serialization; Linux/UVM handles async managed access natively
    // and keeps the launch asynchronous. This also converts an
    // asynchronously-reported kernel-execution fault into an error attributed
    // to the launching operation rather than the next unrelated transfer.
    #[cfg(target_os = "windows")]
    {
        // SAFETY: this device's context is current on this thread (`bind`
        // above); draining reports this kernel's execution status.
        let sync = unsafe { cuda_oxide::sys::cuCtxSynchronize() };
        if sync != 0 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("cuCtxSynchronize after launch -> {sync}"),
            });
        }
    }
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
