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
    #[cfg(not(target_os = "windows"))]
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
