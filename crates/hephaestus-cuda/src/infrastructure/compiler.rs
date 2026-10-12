use libloading::Library;
use std::sync::OnceLock;

use crate::infrastructure::device::{CudaContext, CurrentContext};

mod headers;

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

/// How NVRTC acquisition failed, naming the failing step.
///
/// Every variant preserves the identity the loader reported: searched
/// locations for a failed search, the candidate path for a rejected file,
/// and the export name for a missing symbol. Callers surface these through
/// `Display` rather than collapsing them into "not available".
#[derive(Clone, Debug)]
pub enum NvrtcDriverError {
    /// No candidate library existed at any searched location.
    NotFound {
        /// Every loader name and toolkit directory tried, in order.
        searched: Vec<String>,
    },
    /// A candidate file existed but the platform loader rejected it.
    LoadFailed {
        /// The rejected file.
        candidate: String,
        /// The platform loader diagnostic.
        message: String,
    },
    /// The library loaded but lacked a required NVRTC export.
    MissingSymbol {
        /// The unresolved export name.
        symbol: &'static str,
        /// The platform loader diagnostic.
        message: String,
    },
}

impl std::fmt::Display for NvrtcDriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvrtcDriverError::NotFound { searched } => {
                write!(
                    f,
                    "NVRTC library not found; searched: {}",
                    searched.join(", ")
                )
            }
            NvrtcDriverError::LoadFailed { candidate, message } => {
                write!(f, "NVRTC library {candidate} failed to load: {message}")
            }
            NvrtcDriverError::MissingSymbol { symbol, message } => {
                write!(f, "NVRTC library lacks required export {symbol}: {message}")
            }
        }
    }
}

impl std::error::Error for NvrtcDriverError {}

/// Per-attempt record of one NVRTC library search.
#[derive(Clone, Debug, Default)]
struct SearchDiagnostics {
    /// Locations and loader names tried, with their failures inline.
    searched: Vec<String>,
    /// Candidate files the platform loader rejected, with diagnostics.
    load_failures: Vec<(String, String)>,
}

static NVRTC_DRIVER: OnceLock<Result<NvrtcDriver, NvrtcDriverError>> = OnceLock::new();

impl NvrtcDriver {
    #[allow(non_snake_case)]
    pub fn get() -> Result<&'static Self, NvrtcDriverError> {
        NVRTC_DRIVER
            .get_or_init(|| {
                let lib = find_nvrtc_library()?;
                resolve_symbols(lib)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

#[allow(non_snake_case)]
fn resolve_symbols(lib: Library) -> Result<NvrtcDriver, NvrtcDriverError> {
    macro_rules! resolve {
        ($name:literal) => {
            // SAFETY: the symbol resolves from the loaded library by its
            // documented NVRTC exported name, and the recorded function
            // pointer type matches the NVRTC C ABI signature. The raw
            // pointer copied out of the `Symbol` guard stays valid because
            // `_lib` keeps the library loaded for the driver's `'static`
            // lifetime.
            *unsafe { lib.get(concat!($name, "\0").as_bytes()) }.map_err(|error| {
                NvrtcDriverError::MissingSymbol {
                    symbol: $name,
                    message: error.to_string(),
                }
            })?
        };
    }
    Ok(NvrtcDriver {
        nvrtcCreateProgram: resolve!("nvrtcCreateProgram"),
        nvrtcCompileProgram: resolve!("nvrtcCompileProgram"),
        nvrtcGetPTXSize: resolve!("nvrtcGetPTXSize"),
        nvrtcGetPTX: resolve!("nvrtcGetPTX"),
        nvrtcGetProgramLogSize: resolve!("nvrtcGetProgramLogSize"),
        nvrtcGetProgramLog: resolve!("nvrtcGetProgramLog"),
        nvrtcDestroyProgram: resolve!("nvrtcDestroyProgram"),
        _lib: lib,
    })
}

fn load_named(name: &str, diagnostics: &mut SearchDiagnostics) -> Option<Library> {
    // SAFETY: loading a shared library runs its platform initialization
    // code; the probed names all resolve the NVRTC redistributable shipped
    // with the CUDA toolkit, whose initializers uphold the platform loader
    // contract. No other invariant is required at load time — symbol typing
    // is justified at the `resolve_symbols` sites.
    match unsafe { Library::new(name) } {
        Ok(lib) => Some(lib),
        Err(error) => {
            diagnostics.searched.push(format!("{name}: {error}"));
            None
        }
    }
}

fn search_directory(dir: &str, diagnostics: &mut SearchDiagnostics) -> Option<Library> {
    let cleaned_dir = dir.replace('\\', "/");
    let entries = match std::fs::read_dir(&cleaned_dir) {
        Ok(entries) => entries,
        Err(error) => {
            diagnostics
                .searched
                .push(format!("{cleaned_dir}: directory unavailable: {error}"));
            return None;
        }
    };
    diagnostics.searched.push(cleaned_dir.clone());
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                diagnostics.searched.push(format!(
                    "{cleaned_dir}: directory entry unreadable: {error}"
                ));
                continue;
            }
        };
        let path = entry.path();
        if path.is_file()
            && let Some(filename) = path.file_name().and_then(|s| s.to_str())
        {
            let extension = path.extension().and_then(|s| s.to_str());
            let matches = if cfg!(windows) {
                // Windows paths are case-insensitive, so the
                // whole comparison folds case: some toolkit
                // installers ship `NVRTC64_120_0.DLL`. Extension
                // matching goes through `Path::extension` rather
                // than `str::ends_with` for the same reason.
                let lowered = filename.to_ascii_lowercase();
                lowered.starts_with("nvrtc")
                    && !lowered.contains("builtins")
                    && extension.is_some_and(|e| e.eq_ignore_ascii_case("dll"))
            } else {
                filename.starts_with("libnvrtc")
                    && (extension == Some("so") || filename.contains(".so."))
            };
            if matches {
                // SAFETY: as in `load_named` (NVRTC library under a
                // toolkit directory).
                match unsafe { Library::new(&path) } {
                    Ok(lib) => return Some(lib),
                    Err(error) => diagnostics
                        .load_failures
                        .push((path.display().to_string(), error.to_string())),
                }
            }
        }
    }
    None
}

fn find_nvrtc_library() -> Result<Library, NvrtcDriverError> {
    let mut diagnostics = SearchDiagnostics::default();
    for name in ["nvrtc", "nvrtc64"] {
        if let Some(lib) = load_named(name, &mut diagnostics) {
            return Ok(lib);
        }
    }

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let paths = vec![
            format!("{}/bin/x64", cuda_path),
            format!("{}/bin", cuda_path),
            format!("{}/lib64", cuda_path),
            format!("{}/lib", cuda_path),
        ];
        for dir in paths {
            if let Some(lib) = search_directory(&dir, &mut diagnostics) {
                return Ok(lib);
            }
        }
    }

    let fallback_names: &[&str] = if cfg!(windows) {
        &[
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
        ]
    } else {
        &["libnvrtc.so"]
    };
    for name in fallback_names {
        if let Some(lib) = load_named(name, &mut diagnostics) {
            return Ok(lib);
        }
    }
    // A rejected file is more informative than an empty search: prefer the
    // present-but-invalid identity whenever one was recorded.
    if let Some((candidate, message)) = diagnostics.load_failures.into_iter().next() {
        return Err(NvrtcDriverError::LoadFailed { candidate, message });
    }
    Err(NvrtcDriverError::NotFound {
        searched: diagnostics.searched,
    })
}

/// Compile a CUDA C++ source code string to PTX at runtime using NVRTC.
pub fn compile_cuda_to_ptx(src: &str, device: &crate::CudaDevice) -> Result<String, String> {
    let nvrtc = NvrtcDriver::get().map_err(|error| error.to_string())?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "hephaestus-nvrtc-loader-{}-{}-{}",
            name,
            std::process::id(),
            // Test threads within one process share the pid; the thread id
            // keeps parallel scratch directories disjoint.
            format!("{:?}", std::thread::current().id()).replace(['(', ')', ' '], "_")
        ))
    }

    #[test]
    fn missing_export_names_the_symbol() {
        // The test executable is a real loadable module that is not NVRTC:
        // symbol resolution genuinely fails against it on every platform.
        let exe = std::env::current_exe().expect("test executable path resolves");
        // SAFETY: loading the running executable re-enters no
        // initializers; the platform already ran them at startup.
        let this = unsafe { Library::new(&exe) }.expect("current process image loads");
        let error = match resolve_symbols(this) {
            Err(error) => error,
            Ok(_) => panic!("non-NVRTC library must miss exports"),
        };
        assert!(
            matches!(
                error,
                NvrtcDriverError::MissingSymbol {
                    symbol: "nvrtcCreateProgram",
                    ..
                }
            ),
            "unexpected fault identity: {error:?}"
        );
        assert!(
            !error.to_string().is_empty(),
            "missing-symbol error must carry the loader message"
        );
    }

    #[test]
    fn invalid_library_file_keeps_path_identity() {
        let dir = scratch_dir("invalid");
        std::fs::create_dir_all(&dir).expect("scratch dir creates");
        let filename = if cfg!(windows) {
            "nvrtc64_999_0_test.dll"
        } else {
            "libnvrtc_fake_test.so"
        };
        let candidate = dir.join(filename);
        std::fs::write(&candidate, b"not a shared library").expect("fixture writes");
        let mut diagnostics = SearchDiagnostics::default();
        let found = search_directory(
            &candidate
                .parent()
                .expect("fixture has a parent")
                .to_string_lossy(),
            &mut diagnostics,
        );
        assert!(found.is_none(), "garbage file must not load");
        assert_eq!(
            diagnostics.load_failures.len(),
            1,
            "exactly one load failure expected: {diagnostics:?}"
        );
        let (path, message) = &diagnostics.load_failures[0];
        assert!(
            path.contains(filename),
            "load failure must name the file: {path}"
        );
        assert!(
            !message.is_empty(),
            "load failure must carry the loader message"
        );
        std::fs::remove_file(&candidate).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn absent_directory_reports_enumeration_fault() {
        let missing = scratch_dir("absent").join("no-such-dir");
        let mut diagnostics = SearchDiagnostics::default();
        let found = search_directory(&missing.to_string_lossy(), &mut diagnostics);
        assert!(found.is_none(), "absent directory yields no library");
        assert!(
            diagnostics
                .searched
                .iter()
                .any(|entry| entry.contains("directory unavailable")),
            "enumeration fault must be recorded: {diagnostics:?}"
        );
    }

    #[test]
    fn empty_directory_reports_searched_location() {
        let dir = scratch_dir("empty");
        std::fs::create_dir_all(&dir).expect("scratch dir creates");
        let mut diagnostics = SearchDiagnostics::default();
        let found = search_directory(&dir.to_string_lossy(), &mut diagnostics);
        assert!(found.is_none(), "empty directory yields no library");
        assert!(
            diagnostics.load_failures.is_empty(),
            "no load attempted: {diagnostics:?}"
        );
        assert!(
            !diagnostics.searched.is_empty(),
            "searched location must be recorded"
        );
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn not_found_lists_everywhere_searched() {
        let error = NvrtcDriverError::NotFound {
            searched: vec!["nvrtc".to_string(), "C:/toolkit/bin".to_string()],
        };
        let message = error.to_string();
        assert!(
            message.contains("nvrtc") && message.contains("C:/toolkit/bin"),
            "not-found error must list searched locations: {message}"
        );
    }

    #[test]
    fn driver_acquisition_succeeds_with_installed_toolkit() {
        NvrtcDriver::get().expect("installed NVRTC must resolve");
    }
}
