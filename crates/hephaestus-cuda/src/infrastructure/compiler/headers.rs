//! Find toolkit headers beside the NVRTC library actually loaded by the process.

use std::path::{Path, PathBuf};

pub(super) fn include_directory(
    symbol: *const core::ffi::c_void,
) -> Result<Option<PathBuf>, String> {
    let library = library_path(symbol)?;
    let library = library
        .canonicalize()
        .map_err(|error| format!("NVRTC library path {}: {error}", library.display()))?;
    // SDKs place NVRTC in lib64, lib/<target>, bin, or bin/x64. Inspect
    // those two parent levels plus the library directory itself, never a
    // separately installed toolkit selected by an unrelated PATH entry.
    for parent in library
        .parent()
        .into_iter()
        .flat_map(Path::ancestors)
        .take(3)
    {
        let directory = parent.join("include");
        match directory.join("cuda_fp16.h").try_exists() {
            Ok(true) => return Ok(Some(directory)),
            Ok(false) => {}
            Err(error) => {
                return Err(format!(
                    "CUDA header directory {}: {error}",
                    directory.display()
                ));
            }
        }
    }
    // Redistributable-only NVRTC remains sufficient for header-free kernels.
    // Header-dependent sources receive NVRTC's missing-header diagnostic.
    Ok(None)
}

#[cfg(windows)]
fn library_path(symbol: *const core::ffi::c_void) -> Result<PathBuf, String> {
    use std::os::windows::ffi::OsStringExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(
            flags: u32,
            address: *const u16,
            module: *mut *mut core::ffi::c_void,
        ) -> i32;
        fn GetModuleFileNameW(module: *mut core::ffi::c_void, path: *mut u16, capacity: u32)
        -> u32;
    }
    let mut module = core::ptr::null_mut();
    // FROM_ADDRESS | UNCHANGED_REFCOUNT: the NvrtcDriver retains this loaded
    // symbol's library for the entire call; no new loader reference is owned.
    // SAFETY: symbol is a live NVRTC function address; module is a writable handle.
    if unsafe { GetModuleHandleExW(0x4 | 0x2, symbol.cast(), &mut module) } == 0 {
        return Err(format!(
            "NVRTC module lookup: {}",
            std::io::Error::last_os_error()
        ));
    }
    // Windows extended paths are bounded at 32,767 UTF-16 code units plus NUL.
    let mut path = vec![0u16; 32_768];
    // SAFETY: module is retained by NvrtcDriver; path has exactly the supplied capacity.
    let length = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), 32_768) };
    if length == 0 || length == 32_768 {
        return Err(format!(
            "NVRTC module path: {}",
            std::io::Error::last_os_error()
        ));
    }
    path.truncate(usize::try_from(length).expect("invariant: CUDA requires a 64-bit host"));
    Ok(std::ffi::OsString::from_wide(&path).into())
}

#[cfg(unix)]
fn library_path(symbol: *const core::ffi::c_void) -> Result<PathBuf, String> {
    use core::ffi::{c_char, c_void};
    use std::os::unix::ffi::OsStrExt;
    #[repr(C)]
    struct DlInfo {
        filename: *const c_char,
        base: *mut c_void,
        symbol_name: *const c_char,
        symbol_address: *mut c_void,
    }
    #[cfg_attr(not(target_os = "macos"), link(name = "dl"))]
    unsafe extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> i32;
    }
    let mut info = DlInfo {
        filename: core::ptr::null(),
        base: core::ptr::null_mut(),
        symbol_name: core::ptr::null(),
        symbol_address: core::ptr::null_mut(),
    };
    // SAFETY: symbol is retained by NvrtcDriver; info matches the dlfcn.h Dl_info ABI.
    if unsafe { dladdr(symbol, &mut info) } == 0 || info.filename.is_null() {
        return Err("NVRTC module path is unavailable from dladdr".to_owned());
    }
    // SAFETY: successful dladdr returns a NUL-terminated pathname retained with the library.
    let bytes = unsafe { std::ffi::CStr::from_ptr(info.filename) }.to_bytes();
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}
