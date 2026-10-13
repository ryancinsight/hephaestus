//! Platform loader and conservative driver-absence classification.

use libloading::Library;

pub(super) fn open(name: &str) -> Result<Library, libloading::Error> {
    #[cfg(target_os = "windows")]
    {
        // SAFETY: only the trusted Windows system directory participates in
        // lookup. This excludes the working directory and PATH DLL shadowing.
        unsafe {
            libloading::os::windows::Library::load_with_flags(
                name,
                libloading::os::windows::LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
            .map(Into::into)
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        // SAFETY: the platform loader opens the named CUDA driver soname.
        unsafe { Library::new(name) }
    }
}

pub(super) fn absent(name: &str, error: &libloading::Error) -> bool {
    #[cfg(target_os = "windows")]
    {
        use std::error::Error;
        // ERROR_MOD_NOT_FOUND can also name a missing dependent DLL. Confirm
        // the requested system DLL itself is absent before reporting no adapter.
        if error
            .source()
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .and_then(std::io::Error::raw_os_error)
            != Some(126)
        {
            return false;
        }
        system_path(name).is_some_and(|path| match path.try_exists() {
            Ok(exists) => !exists,
            Err(_) => false, // Access/metadata failures are driver faults.
        })
    }
    #[cfg(target_os = "macos")]
    {
        // dyld distinguishes a missing requested image (a list of failed
        // search paths) from a present image with a missing dependency
        // ("Library not loaded" / "Referenced from"). Only accept the former,
        // and require every attempted path to retain the requested file suffix and
        // report absence. Unknown, invalid-image, and dependency errors remain
        // driver faults. No path existence probe can substitute for dyld:
        // system images can exist solely in its shared cache.
        let diagnostic = error.to_string();
        let prefix = format!("dlopen({name}, ");
        let Some((flags, reason)) = diagnostic
            .strip_prefix(&prefix)
            .and_then(|remaining| remaining.split_once("): "))
        else {
            return false;
        };
        if !flags.strip_prefix("0x").is_some_and(|digits| {
            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_hexdigit())
        }) {
            return false;
        }
        let Some(mut attempts) = reason.strip_prefix("tried: ") else {
            return false;
        };
        let Some(requested_file) = std::path::Path::new(name)
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
        else {
            return false;
        };
        loop {
            let Some((path, outcome)) = attempts
                .strip_prefix('\'')
                .and_then(|entry| entry.split_once("' ("))
            else {
                return false;
            };
            // dyld prepends its Cryptex namespace even to a bare soname;
            // that attempted path ends in the requested file without a slash
            // before it. Header identity still excludes dependent images.
            if !path.ends_with(requested_file) {
                return false;
            }
            let Some((reason, remaining)) = outcome.split_once(')') else {
                return false;
            };
            if !matches!(
                reason,
                "no such file" | "not in dyld cache" | "no such file, not in dyld cache"
            ) {
                return false;
            }
            if remaining.is_empty() {
                return true;
            }
            let Some(next) = remaining.strip_prefix(", ") else {
                return false;
            };
            attempts = next;
        }
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        // dlerror exposes text rather than errno. Recognize only the exact
        // requested-soname ENOENT diagnostic; a dependency name, unfamiliar
        // loader, or localized message remains a driver fault, never absence.
        error.to_string()
            == format!("{name}: cannot open shared object file: No such file or directory")
    }
}

#[cfg(target_os = "windows")]
fn system_path(name: &str) -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
    }
    // Windows documents MAX_PATH as sufficient for the system directory.
    let mut buffer = [0u16; 260];
    // SAFETY: buffer holds 260 writable WCHARs and the supplied count matches.
    let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), 260) };
    if length == 0 || length >= 260 {
        return None;
    }
    let directory = std::ffi::OsString::from_wide(&buffer[..length as usize]);
    Some(std::path::PathBuf::from(directory).join(name))
}
