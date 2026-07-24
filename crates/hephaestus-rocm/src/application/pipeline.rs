use hephaestus_core::{BlockWidth, HephaestusError, Result};
use std::any::TypeId;

/// Pipeline-cache key for a runtime-compiled ROCm kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PipelineKey {
    /// Binary elementwise operation keyed by operation, scalar, and block width.
    Binary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Scalar elementwise operation keyed by operation, scalar, and block width.
    Scalar {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Unary elementwise operation keyed by operation, scalar, and block width.
    Unary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Strided binary operation keyed by operation, scalar, and block width.
    StridedBinary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Strided unary operation keyed by operation, scalar, and block width.
    StridedUnary {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Strided scalar operation keyed by operation, scalar, and block width.
    StridedScalar {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Contiguous reduction operation keyed by operation, scalar, and block width.
    Reduction {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Strided map-reduction operation keyed by map, scalar, and block width.
    MapReduction {
        op: TypeId,
        scalar: TypeId,
        width: u32,
    },
    /// Rank-2 axis reduction keyed by operation, scalar, axis, and block width.
    AxisReduction {
        op: TypeId,
        scalar: TypeId,
        axis: usize,
        width: u32,
    },
    /// Rank-2 mean reduction keyed by scalar, axis, and block width.
    MeanAxis {
        scalar: TypeId,
        axis: usize,
        width: u32,
    },
    /// Rank-2 axis scan keyed by operation, scalar, direction, axis, and block width.
    AxisScan {
        marker: TypeId,
        scalar: TypeId,
        direction: hephaestus_core::ScanDirection,
        axis: usize,
        width: u32,
    },
    /// Rank-2 matrix multiplication keyed by kernel marker and scalar.
    Matmul { marker: TypeId, scalar: TypeId },
    /// Batched rank-3 matrix multiplication keyed by kernel marker and scalar.
    BatchedMatmul { marker: TypeId, scalar: TypeId },
    /// Rank-2 Kronecker product keyed by kernel marker and scalar.
    Kron { marker: TypeId, scalar: TypeId },
    /// Rank-2 matrix rank and determinant keyed by kernel marker and scalar.
    MatrixRank { marker: TypeId, scalar: TypeId },
    /// Sparse-dense matrix product keyed by kernel marker and scalar.
    Spmm { marker: TypeId, scalar: TypeId },
    /// Sparse matrix-vector product keyed by kernel marker and scalar.
    Spmv { marker: TypeId, scalar: TypeId },
    /// Backend-neutral multi-storage kernel keyed by its authored source.
    MultiStorage(u64),
    /// Authored kernel stream keyed by its source.
    Stream(u64),
    /// Grouped authored kernel stream keyed by its source.
    GroupedStream(u64),
}

/// Grid/block launch configuration for a one-dimensional HIP kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchConfig {
    pub(crate) grid: (u32, u32, u32),
    pub(crate) block: (u32, u32, u32),
    pub(crate) shared_bytes: u32,
}

impl LaunchConfig {
    #[must_use]
    pub(crate) const fn linear(grid_x: u32, width: BlockWidth) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (width.get(), 1, 1),
            shared_bytes: 0,
        }
    }

    #[must_use]
    pub(crate) const fn linear_shared(grid_x: u32, width: BlockWidth, shared_bytes: u32) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (width.get(), 1, 1),
            shared_bytes,
        }
    }

    #[must_use]
    pub(crate) const fn planar(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_bytes: 0,
        }
    }

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

/// Calculate a checked one-dimensional grid size.
pub(crate) fn grid_size(work_items: usize, width: BlockWidth) -> Result<u32> {
    let work_items = u64::try_from(work_items).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("ROCm work-item count {work_items} exceeds u64 range"),
    })?;
    width
        .checked_covering_blocks(work_items)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!(
                "ROCm work-item count {work_items} exceeds the HIP grid limit at block width {}",
                width.get()
            ),
        })
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
mod native {
    use super::{LaunchConfig, PipelineKey};
    use crate::RocmDevice;
    use crate::infrastructure::device::{HIP_SUCCESS, RocmContext, check_status};
    use core::ffi::{c_char, c_void};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ffi::{CStr, CString};
    use std::ptr;
    use std::rc::Rc;
    use std::sync::Arc;

    /// Cache confined to the thread that owns the HIP current-device binding.
    pub(crate) type PipelineCache = Rc<RefCell<HashMap<PipelineKey, Rc<RocmKernel>>>>;

    pub(crate) fn new_cache() -> PipelineCache {
        Rc::new(RefCell::new(HashMap::new()))
    }

    /// A loaded HIP module and its entry point, owned by the device cache.
    pub(crate) struct RocmKernel {
        module: cubecl_hip_sys::hipModule_t,
        function: cubecl_hip_sys::hipFunction_t,
        context: Arc<RocmContext>,
    }

    impl Drop for RocmKernel {
        fn drop(&mut self) {
            if self.module.is_null() {
                return;
            }
            if self.context.set_current().is_err() {
                debug_assert!(false, "ROCm kernel drop: device selection failed");
                return;
            }
            // SAFETY: `self.module` was returned by `hipModuleLoadData` and is
            // owned by this cache entry exactly once.
            let status = unsafe { cubecl_hip_sys::hipModuleUnload(self.module) };
            debug_assert_eq!(
                status,
                HIP_SUCCESS,
                "hipModuleUnload failed: {}",
                crate::infrastructure::device::status_message(status, "hipModuleUnload")
            );
        }
    }

    struct HipRtcProgram(cubecl_hip_sys::hiprtcProgram);

    impl Drop for HipRtcProgram {
        fn drop(&mut self) {
            if self.0.is_null() {
                return;
            }
            // SAFETY: the handle was created by `hiprtcCreateProgram` and is
            // destroyed exactly once when this guard leaves scope.
            let status = unsafe { cubecl_hip_sys::hiprtcDestroyProgram(&mut self.0) };
            debug_assert_eq!(
                status,
                cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS,
                "hiprtcDestroyProgram failed"
            );
        }
    }

    fn rtc_message(status: cubecl_hip_sys::hiprtcResult, operation: &str) -> String {
        // SAFETY: hipRTC returns a process-lifetime null-terminated diagnostic
        // string for a result code, or a null pointer.
        let detail = unsafe {
            let ptr = cubecl_hip_sys::hiprtcGetErrorString(status);
            if ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
            }
        };
        match detail {
            Some(detail) => format!("{operation} -> {detail} (status {status})"),
            None => format!("{operation} -> unknown hipRTC status {status}"),
        }
    }

    fn rtc_failure(
        status: cubecl_hip_sys::hiprtcResult,
        operation: &str,
    ) -> hephaestus_core::HephaestusError {
        hephaestus_core::HephaestusError::DispatchFailed {
            message: rtc_message(status, operation),
        }
    }

    fn compile_to_code(source: &str) -> hephaestus_core::Result<Vec<u8>> {
        let source = CString::new(source).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("ROCm kernel source contains NUL: {error}"),
            }
        })?;
        let name = CString::new("hephaestus_kernel").map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("ROCm kernel name contains NUL: {error}"),
            }
        })?;
        let mut program = cubecl_hip_sys::hiprtcProgram::default();
        // SAFETY: all pointers refer to live NUL-terminated strings or valid
        // out-pointers, and no headers/options are supplied.
        let status = unsafe {
            cubecl_hip_sys::hiprtcCreateProgram(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if status != cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS {
            return Err(rtc_failure(status, "hiprtcCreateProgram"));
        }
        let program = HipRtcProgram(program);

        // SAFETY: `program.0` is a live hipRTC program and no compile options
        // are passed.
        let status = unsafe { cubecl_hip_sys::hiprtcCompileProgram(program.0, 0, ptr::null_mut()) };
        if status != cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS {
            let mut log_size = 0_usize;
            // SAFETY: the program is live and `log_size` is a valid out-pointer.
            let log_status =
                unsafe { cubecl_hip_sys::hiprtcGetProgramLogSize(program.0, &mut log_size) };
            let log = if log_status == cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS && log_size > 0 {
                let mut bytes = vec![0_i8; log_size];
                // SAFETY: `bytes` has the size reported by hipRTC and is live
                // for the duration of the call.
                let read_status =
                    unsafe { cubecl_hip_sys::hiprtcGetProgramLog(program.0, bytes.as_mut_ptr()) };
                if read_status == cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS {
                    // SAFETY: hipRTC writes a NUL-terminated log into the
                    // reported buffer.
                    Some(
                        unsafe { CStr::from_ptr(bytes.as_ptr()) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                } else {
                    None
                }
            } else {
                None
            };
            let message = match log {
                Some(log) => format!(
                    "{}; compiler log: {log}",
                    rtc_message(status, "hiprtcCompileProgram")
                ),
                None => rtc_message(status, "hiprtcCompileProgram"),
            };
            return Err(hephaestus_core::HephaestusError::DispatchFailed { message });
        }

        let mut code_size = 0_usize;
        // SAFETY: the program is live and `code_size` is a valid out-pointer.
        let status = unsafe { cubecl_hip_sys::hiprtcGetCodeSize(program.0, &mut code_size) };
        if status != cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS {
            return Err(rtc_failure(status, "hiprtcGetCodeSize"));
        }
        let mut code = vec![0_u8; code_size];
        // SAFETY: `code` has the size reported by hipRTC and remains live for
        // the duration of the call.
        let status =
            unsafe { cubecl_hip_sys::hiprtcGetCode(program.0, code.as_mut_ptr().cast::<c_char>()) };
        if status != cubecl_hip_sys::hiprtcResult_HIPRTC_SUCCESS {
            return Err(rtc_failure(status, "hiprtcGetCode"));
        }
        Ok(code)
    }

    pub(crate) fn cached_kernel(
        device: &RocmDevice,
        key: PipelineKey,
        func_name: &str,
        source: impl FnOnce() -> String,
    ) -> hephaestus_core::Result<Rc<RocmKernel>> {
        if let Some(kernel) = device.pipeline_cache.borrow().get(&key) {
            return Ok(kernel.clone());
        }

        device.context.set_current()?;
        let code = compile_to_code(&source())?;
        let func_name = CString::new(func_name).map_err(|error| {
            hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("ROCm kernel name contains NUL: {error}"),
            }
        })?;
        let mut module = ptr::null_mut();
        // SAFETY: the HIP device is current, `code` is a live code object, and
        // `module` is a valid out-pointer.
        let status = unsafe {
            cubecl_hip_sys::hipModuleLoadData(&mut module, code.as_ptr().cast::<c_void>())
        };
        if status != HIP_SUCCESS {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: crate::infrastructure::device::status_message(status, "hipModuleLoadData"),
            });
        }
        let mut function = ptr::null_mut();
        // SAFETY: `module` is live, `func_name` is NUL-terminated, and
        // `function` is a valid out-pointer.
        let status = unsafe {
            cubecl_hip_sys::hipModuleGetFunction(&mut function, module, func_name.as_ptr())
        };
        if status != HIP_SUCCESS {
            // SAFETY: `module` was successfully loaded above and has not been
            // transferred to a kernel owner on this error path.
            let unload_status = unsafe { cubecl_hip_sys::hipModuleUnload(module) };
            debug_assert_eq!(unload_status, HIP_SUCCESS);
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: crate::infrastructure::device::status_message(
                    status,
                    "hipModuleGetFunction",
                ),
            });
        }
        let compiled = Rc::new(RocmKernel {
            module,
            function,
            context: device.context.clone(),
        });
        let mut cache = device.pipeline_cache.borrow_mut();
        let kernel = cache.entry(key).or_insert_with(|| compiled.clone());
        Ok(kernel.clone())
    }

    pub(crate) fn launch_kernel(
        device: &RocmDevice,
        kernel: &RocmKernel,
        config: LaunchConfig,
        args: &mut [*mut c_void],
    ) -> hephaestus_core::Result<()> {
        device.context.set_current()?;
        // SAFETY: the module entry point belongs to this current HIP context;
        // dimensions and argument storage are validated/owned by the caller,
        // and `args` remains live through the synchronous launch call.
        let status = unsafe {
            cubecl_hip_sys::hipModuleLaunchKernel(
                kernel.function,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_bytes,
                ptr::null_mut(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        check_status(status, "hipModuleLaunchKernel")
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
pub(crate) use native::{PipelineCache, RocmKernel, cached_kernel, launch_kernel, new_cache};

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
mod unavailable {
    use super::{LaunchConfig, PipelineKey};
    use crate::RocmDevice;
    use std::rc::Rc;

    pub(crate) fn cached_kernel(
        _device: &RocmDevice,
        _key: PipelineKey,
        _func_name: &str,
        source: impl FnOnce() -> String,
    ) -> hephaestus_core::Result<Rc<RocmKernel>> {
        let _ = (source,);
        Err(hephaestus_core::HephaestusError::AdapterUnavailable {
            message:
                "ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
                    .to_string(),
        })
    }

    pub(crate) fn launch_kernel(
        _device: &RocmDevice,
        _kernel: &RocmKernel,
        _config: LaunchConfig,
        _args: &mut [*mut core::ffi::c_void],
    ) -> hephaestus_core::Result<()> {
        Err(hephaestus_core::HephaestusError::AdapterUnavailable {
            message:
                "ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
                    .to_string(),
        })
    }

    #[derive(Debug)]
    pub(crate) struct RocmKernel;
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
pub(crate) use unavailable::{RocmKernel, cached_kernel, launch_kernel};
