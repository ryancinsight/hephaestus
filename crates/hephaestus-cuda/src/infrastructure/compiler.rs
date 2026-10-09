use crate::infrastructure::device::{CudaContext, CurrentContext};

mod headers;
mod loader;

use loader::NvrtcDriver;

#[allow(non_camel_case_types)]
pub type nvrtcProgram = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type nvrtcResult = i32;

/// A loaded CUDA module and one resolved kernel function handle.
///
/// The owning device is retained so `Drop` can make the module's context
/// current before unloading — modules are context-owned, and the last `Arc`
/// may be released on any thread.
pub struct SafeCachedKernel {
    pub module: *mut core::ffi::c_void,
    pub func: *mut core::ffi::c_void,
    context: std::sync::Arc<CudaContext>,
}

impl SafeCachedKernel {
    pub(crate) fn new(
        module: *mut core::ffi::c_void,
        func: *mut core::ffi::c_void,
        context: std::sync::Arc<CudaContext>,
    ) -> Self {
        Self {
            module,
            func,
            context,
        }
    }
}

// SAFETY: `CUmodule`/`CUfunction` are opaque context-owned driver handles,
// not thread-affine pointers; the CUDA driver API is thread-safe and any
// thread may use a handle after making the owning context current (every
// launch/unload site binds first). The handles are never dereferenced on the
// host. The retained `CudaContext` binds the owning context before unload.
unsafe impl Send for SafeCachedKernel {}
// SAFETY: shared use is read-only handle passing into driver calls that
// perform their own internal synchronization; see the Send justification.
unsafe impl Sync for SafeCachedKernel {}

impl Drop for SafeCachedKernel {
    fn drop(&mut self) {
        if self.module.is_null() {
            return;
        }
        // Unloading requires the owning context current on this thread. Drop
        // cannot surface errors; a failed bind or unload leaks the module
        // (bounded: at most one per cache key per device lifetime). Drop does
        // not panic on a release fault. It can run between a
        // consumer's `device.bind()` and a raw-handle driver call; restore
        // the previously current context so the drop-time bind cannot
        // silently retarget the dropping thread.
        let Ok(previous) = CurrentContext::capture(self.context.driver) else {
            // Without the previous context, retain this one resource in
            // the driver rather than retarget the dropping thread.
            return;
        };
        if self.context.bind().is_ok() {
            // SAFETY: `module` is a live handle owned by this value, the
            // owning context is current (bind above), and no other user
            // exists — Drop runs at the last Arc release.
            let res = unsafe { (self.context.driver.kernel.unload)(self.module) };
            if res != 0 {
                tracing::error!(
                    operation = "cuModuleUnload",
                    status = res,
                    "CUDA resource release failed; resource retained"
                );
                // Preserve thread routing and leave this unreleased
                // allocation/module with CUDA after a release failure.
                previous.restore();
                return;
            }
        }
        previous.restore();
    }
}

/// Compile a CUDA C++ source code string to PTX at runtime using NVRTC.
pub fn compile_cuda_to_ptx(src: &str, device: &crate::CudaDevice) -> Result<String, String> {
    let nvrtc = NvrtcDriver::get().map_err(ToString::to_string)?;

    let src_c = std::ffi::CString::new(src).map_err(|e| e.to_string())?;
    let name_c = std::ffi::CString::new("kernel.cu").map_err(|e| e.to_string())?;

    let mut prog: nvrtcProgram = std::ptr::null_mut();
    // SAFETY: every call in this block goes through a function pointer
    // resolved from the live NVRTC library (`NvrtcDriver::get`), typed to
    // match the NVRTC C ABI. `src_c`/`name_c`/`options` are NUL-terminated
    // CStrings kept alive across the calls; `prog` is a valid out-pointer,
    // and after a successful create every subsequent call passes the same
    // live program handle, which is destroyed exactly once on every exit
    // path after the compilation result is collected. The log and PTX buffers are heap
    // allocations sized by the immediately preceding NVRTC size queries
    // before the driver writes into them.
    unsafe {
        let res = (nvrtc.nvrtcCreateProgram)(
            &mut prog,
            src_c.as_ptr(),
            name_c.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if res != 0 {
            return Err(format!("nvrtcCreateProgram failed: {}", res));
        }

        let compiled = (|| {
            let mut options = vec![
                std::ffi::CString::new("--std=c++11")
                    .expect("invariant: standard flag contains no null bytes"),
                std::ffi::CString::new(format!(
                    "--gpu-architecture=compute_{}",
                    device.compute_capability()
                ))
                .expect("invariant: decimal architecture contains no null bytes"),
            ];
            if let Some(directory) =
                headers::include_directory(nvrtc.nvrtcCreateProgram as *const core::ffi::c_void)?
            {
                options.push(
                    std::ffi::CString::new(format!("--include-path={}", directory.display()))
                        .map_err(|error| {
                            format!("CUDA header path contains a null byte: {error}")
                        })?,
                );
            }
            let options_ptr: Vec<*const std::ffi::c_char> =
                options.iter().map(|o| o.as_ptr()).collect();

            let compile_res = (nvrtc.nvrtcCompileProgram)(
                prog,
                options_ptr.len() as std::ffi::c_int,
                options_ptr.as_ptr(),
            );

            if compile_res != 0 {
                // Best-effort log retrieval: a failed size/log query yields an
                // empty log rather than reading uninitialized bytes.
                let mut log_size: usize = 0;
                let log_str =
                    if (nvrtc.nvrtcGetProgramLogSize)(prog, &mut log_size) == 0 && log_size > 0 {
                        let mut log_bytes = vec![0u8; log_size];
                        if (nvrtc.nvrtcGetProgramLog)(
                            prog,
                            log_bytes.as_mut_ptr() as *mut std::ffi::c_char,
                        ) == 0
                        {
                            while log_bytes.last() == Some(&0) {
                                log_bytes.pop();
                            }
                            String::from_utf8_lossy(&log_bytes).into_owned()
                        } else {
                            "<nvrtcGetProgramLog failed>".to_string()
                        }
                    } else {
                        "<no compile log available>".to_string()
                    };

                return Err(format!(
                    "nvrtcCompileProgram failed (code {}). Log:\n{}",
                    compile_res, log_str
                ));
            }

            let mut ptx_size: usize = 0;
            let ptx_res = (nvrtc.nvrtcGetPTXSize)(prog, &mut ptx_size);
            if ptx_res != 0 {
                return Err(format!("nvrtcGetPTXSize failed: {}", ptx_res));
            }

            let mut ptx_bytes = vec![0u8; ptx_size];
            let ptx_get_res =
                (nvrtc.nvrtcGetPTX)(prog, ptx_bytes.as_mut_ptr() as *mut std::ffi::c_char);
            if ptx_get_res != 0 {
                return Err(format!("nvrtcGetPTX failed: {}", ptx_get_res));
            }

            while ptx_bytes.last() == Some(&0) {
                ptx_bytes.pop();
            }

            let ptx_str = String::from_utf8(ptx_bytes)
                .map_err(|e| format!("PTX is not valid UTF-8: {}", e))?;
            Ok(ptx_str)
        })();
        let destroy_status = (nvrtc.nvrtcDestroyProgram)(&mut prog);
        if destroy_status != 0 {
            tracing::error!(
                operation = "nvrtcDestroyProgram",
                status = destroy_status,
                "NVRTC program release failed; program retained"
            );
            let cleanup = format!("nvrtcDestroyProgram -> {destroy_status}");
            return Err(match compiled {
                Ok(_) => cleanup,
                Err(primary) => format!("{primary}; {cleanup}"),
            });
        }
        compiled
    }
}
