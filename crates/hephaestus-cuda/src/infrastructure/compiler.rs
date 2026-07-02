use libloading::Library;
use std::sync::OnceLock;

#[allow(non_camel_case_types)]
pub type nvrtcProgram = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type nvrtcResult = i32;

#[allow(non_snake_case)]
pub struct NvrtcDriver {
    _lib: Library,
    pub nvrtcCreateProgram: unsafe extern "C" fn(
        prog: *mut nvrtcProgram,
        src: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        numHeaders: std::ffi::c_int,
        headers: *const *const std::ffi::c_char,
        includeNames: *const *const std::ffi::c_char,
    ) -> nvrtcResult,
    pub nvrtcCompileProgram: unsafe extern "C" fn(
        prog: nvrtcProgram,
        numOptions: std::ffi::c_int,
        options: *const *const std::ffi::c_char,
    ) -> nvrtcResult,
    pub nvrtcGetPTXSize:
        unsafe extern "C" fn(prog: nvrtcProgram, ptxSize: *mut usize) -> nvrtcResult,
    pub nvrtcGetPTX:
        unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut std::ffi::c_char) -> nvrtcResult,
    pub nvrtcGetProgramLogSize:
        unsafe extern "C" fn(prog: nvrtcProgram, logSize: *mut usize) -> nvrtcResult,
    pub nvrtcGetProgramLog:
        unsafe extern "C" fn(prog: nvrtcProgram, log: *mut std::ffi::c_char) -> nvrtcResult,
    pub nvrtcDestroyProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult,
}

/// A loaded CUDA module and one resolved kernel function handle.
///
/// The owning device is retained so `Drop` can make the module's context
/// current before unloading — modules are context-owned, and the last `Arc`
/// may be released on any thread.
pub struct SafeCachedKernel {
    pub module: cuda_core::sys::CUmodule,
    pub func: cuda_core::sys::CUfunction,
    device: std::sync::Arc<cuda_core::Device>,
}

impl SafeCachedKernel {
    pub(crate) fn new(
        module: cuda_core::sys::CUmodule,
        func: cuda_core::sys::CUfunction,
        device: std::sync::Arc<cuda_core::Device>,
    ) -> Self {
        Self {
            module,
            func,
            device,
        }
    }
}

// SAFETY: `CUmodule`/`CUfunction` are opaque context-owned driver handles,
// not thread-affine pointers; the CUDA driver API is thread-safe and any
// thread may use a handle after making the owning context current (every
// launch/unload site binds first). The handles are never dereferenced on the
// host. `cuda_core::Device` is itself Send + Sync.
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
        // (bounded: at most one per cache key per device lifetime) and trips
        // the debug assertion in dev/test builds.
        if self.device.bind_to_thread().is_ok() {
            // SAFETY: `module` is a live handle owned by this value, the
            // owning context is current (bind above), and no other user
            // exists — Drop runs at the last Arc release.
            let res = unsafe { cuda_core::sys::cuModuleUnload(self.module) };
            debug_assert_eq!(res, 0, "cuModuleUnload failed with code {res}");
        } else {
            debug_assert!(false, "SafeCachedKernel drop: context bind failed");
        }
    }
}

static NVRTC_DRIVER: OnceLock<Option<NvrtcDriver>> = OnceLock::new();

impl NvrtcDriver {
    #[allow(non_snake_case)]
    pub fn get() -> Option<&'static Self> {
        NVRTC_DRIVER
            .get_or_init(|| {
                let lib = find_nvrtc_library()?;
                unsafe {
                    let nvrtcCreateProgram = *lib.get(b"nvrtcCreateProgram\0").ok()?;
                    let nvrtcCompileProgram = *lib.get(b"nvrtcCompileProgram\0").ok()?;
                    let nvrtcGetPTXSize = *lib.get(b"nvrtcGetPTXSize\0").ok()?;
                    let nvrtcGetPTX = *lib.get(b"nvrtcGetPTX\0").ok()?;
                    let nvrtcGetProgramLogSize = *lib.get(b"nvrtcGetProgramLogSize\0").ok()?;
                    let nvrtcGetProgramLog = *lib.get(b"nvrtcGetProgramLog\0").ok()?;
                    let nvrtcDestroyProgram = *lib.get(b"nvrtcDestroyProgram\0").ok()?;

                    Some(Self {
                        _lib: lib,
                        nvrtcCreateProgram,
                        nvrtcCompileProgram,
                        nvrtcGetPTXSize,
                        nvrtcGetPTX,
                        nvrtcGetProgramLogSize,
                        nvrtcGetProgramLog,
                        nvrtcDestroyProgram,
                    })
                }
            })
            .as_ref()
    }
}

fn find_nvrtc_library() -> Option<Library> {
    if let Ok(lib) = unsafe { Library::new("nvrtc") } {
        return Some(lib);
    }
    if let Ok(lib) = unsafe { Library::new("nvrtc64") } {
        return Some(lib);
    }

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let paths = vec![
            format!("{}/bin/x64", cuda_path),
            format!("{}/bin", cuda_path),
            format!("{}/lib64", cuda_path),
            format!("{}/lib", cuda_path),
        ];
        for dir in paths {
            let cleaned_dir = dir.replace('\\', "/");
            if let Ok(entries) = std::fs::read_dir(&cleaned_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                            let matches = if cfg!(windows) {
                                filename.starts_with("nvrtc")
                                    && !filename.contains("builtins")
                                    && filename.ends_with(".dll")
                            } else {
                                filename.starts_with("libnvrtc")
                                    && (filename.ends_with(".so") || filename.contains(".so."))
                            };
                            if matches {
                                if let Ok(lib) = unsafe { Library::new(&path) } {
                                    return Some(lib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let fallback_names = if cfg!(windows) {
        vec![
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
        ]
    } else {
        vec!["libnvrtc.so"]
    };
    for name in fallback_names {
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Some(lib);
        }
    }
    None
}

/// Destroy an NVRTC program, checking the result (cleanup-path SSOT).
///
/// A failed destroy leaks the program object; it cannot be surfaced from
/// cleanup paths that are already returning a primary error, so it trips the
/// debug assertion in dev/test builds instead of being silently discarded.
fn destroy_program(nvrtc: &NvrtcDriver, prog: &mut nvrtcProgram) {
    // SAFETY (caller-upheld, all call sites in this module): `prog` was
    // created by `nvrtcCreateProgram` on this same dynamically-loaded NVRTC
    // instance and has not been destroyed yet.
    let res = unsafe { (nvrtc.nvrtcDestroyProgram)(prog) };
    debug_assert_eq!(res, 0, "nvrtcDestroyProgram failed with code {res}");
}

/// Compile a CUDA C++ source code string to PTX at runtime using NVRTC.
pub fn compile_cuda_to_ptx(src: &str) -> Result<String, String> {
    let nvrtc = NvrtcDriver::get().ok_or_else(|| "NVRTC driver not available".to_string())?;

    let src_c = std::ffi::CString::new(src).map_err(|e| e.to_string())?;
    let name_c = std::ffi::CString::new("kernel.cu").map_err(|e| e.to_string())?;

    let mut prog: nvrtcProgram = std::ptr::null_mut();
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

        let options = [std::ffi::CString::new("--std=c++11").unwrap()];
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
            let log_str = if (nvrtc.nvrtcGetProgramLogSize)(prog, &mut log_size) == 0
                && log_size > 0
            {
                let mut log_bytes = vec![0u8; log_size];
                if (nvrtc.nvrtcGetProgramLog)(prog, log_bytes.as_mut_ptr() as *mut std::ffi::c_char)
                    == 0
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

            destroy_program(nvrtc, &mut prog);
            return Err(format!(
                "nvrtcCompileProgram failed (code {}). Log:\n{}",
                compile_res, log_str
            ));
        }

        let mut ptx_size: usize = 0;
        let ptx_res = (nvrtc.nvrtcGetPTXSize)(prog, &mut ptx_size);
        if ptx_res != 0 {
            destroy_program(nvrtc, &mut prog);
            return Err(format!("nvrtcGetPTXSize failed: {}", ptx_res));
        }

        let mut ptx_bytes = vec![0u8; ptx_size];
        let ptx_get_res =
            (nvrtc.nvrtcGetPTX)(prog, ptx_bytes.as_mut_ptr() as *mut std::ffi::c_char);
        if ptx_get_res != 0 {
            destroy_program(nvrtc, &mut prog);
            return Err(format!("nvrtcGetPTX failed: {}", ptx_get_res));
        }

        destroy_program(nvrtc, &mut prog);

        while ptx_bytes.last() == Some(&0) {
            ptx_bytes.pop();
        }

        let ptx_str =
            String::from_utf8(ptx_bytes).map_err(|e| format!("PTX is not valid UTF-8: {}", e))?;
        Ok(ptx_str)
    }
}
