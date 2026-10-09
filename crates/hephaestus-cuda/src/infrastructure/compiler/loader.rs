//! Process-lifetime ownership of NVRTC exports and loader failures.

use super::{nvrtcProgram, nvrtcResult};
use libloading::Library;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LibraryFailureKind {
    Missing,
    Invalid,
    Loader,
}

#[derive(Debug)]
pub(super) struct LibraryFailure {
    pub(super) candidate: PathBuf,
    pub(super) kind: LibraryFailureKind,
    pub(super) source: libloading::Error,
}

#[derive(Debug)]
pub(super) struct DirectoryFailure {
    pub(super) directory: PathBuf,
    pub(super) source: std::io::Error,
}

#[derive(Debug)]
pub(super) enum LoadError {
    Search {
        libraries: Vec<LibraryFailure>,
        directories: Vec<DirectoryFailure>,
    },
    Symbol {
        name: &'static str,
        source: libloading::Error,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Search {
                libraries,
                directories,
            } => {
                write!(formatter, "NVRTC runtime compiler unavailable")?;
                for failure in libraries {
                    write!(
                        formatter,
                        "; library {} ({:?}): {}",
                        failure.candidate.display(),
                        failure.kind,
                        failure.source
                    )?;
                }
                for failure in directories {
                    write!(
                        formatter,
                        "; directory {}: {}",
                        failure.directory.display(),
                        failure.source
                    )?;
                }
                Ok(())
            }
            Self::Symbol { name, source } => {
                write!(formatter, "NVRTC required symbol {name}: {source}")
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Symbol { source, .. } => Some(source),
            Self::Search { .. } => None,
        }
    }
}

#[derive(Debug)]
#[allow(non_snake_case)]
pub(super) struct NvrtcDriver {
    _lib: Library,
    pub(super) nvrtcCreateProgram: unsafe extern "C" fn(
        prog: *mut nvrtcProgram,
        src: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        numHeaders: std::ffi::c_int,
        headers: *const *const std::ffi::c_char,
        includeNames: *const *const std::ffi::c_char,
    ) -> nvrtcResult,
    pub(super) nvrtcCompileProgram: unsafe extern "C" fn(
        prog: nvrtcProgram,
        numOptions: std::ffi::c_int,
        options: *const *const std::ffi::c_char,
    ) -> nvrtcResult,
    pub(super) nvrtcGetPTXSize:
        unsafe extern "C" fn(prog: nvrtcProgram, ptxSize: *mut usize) -> nvrtcResult,
    pub(super) nvrtcGetPTX:
        unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut std::ffi::c_char) -> nvrtcResult,
    pub(super) nvrtcGetProgramLogSize:
        unsafe extern "C" fn(prog: nvrtcProgram, logSize: *mut usize) -> nvrtcResult,
    pub(super) nvrtcGetProgramLog:
        unsafe extern "C" fn(prog: nvrtcProgram, log: *mut std::ffi::c_char) -> nvrtcResult,
    pub(super) nvrtcDestroyProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult,
}

static NVRTC_DRIVER: OnceLock<Result<NvrtcDriver, LoadError>> = OnceLock::new();

impl NvrtcDriver {
    pub(super) fn get() -> Result<&'static Self, &'static LoadError> {
        NVRTC_DRIVER
            .get_or_init(|| find_nvrtc_library().and_then(Self::from_library))
            .as_ref()
    }

    #[allow(non_snake_case)]
    fn from_library(lib: Library) -> Result<Self, LoadError> {
        // SAFETY: each symbol is resolved from the retained NVRTC library by
        // its documented exported name. The function pointer types match the
        // NVRTC C ABI, and `_lib` outlives every copied pointer.
        unsafe {
            Ok(Self {
                nvrtcCreateProgram: required_symbol(
                    &lib,
                    b"nvrtcCreateProgram\0",
                    "nvrtcCreateProgram",
                )?,
                nvrtcCompileProgram: required_symbol(
                    &lib,
                    b"nvrtcCompileProgram\0",
                    "nvrtcCompileProgram",
                )?,
                nvrtcGetPTXSize: required_symbol(&lib, b"nvrtcGetPTXSize\0", "nvrtcGetPTXSize")?,
                nvrtcGetPTX: required_symbol(&lib, b"nvrtcGetPTX\0", "nvrtcGetPTX")?,
                nvrtcGetProgramLogSize: required_symbol(
                    &lib,
                    b"nvrtcGetProgramLogSize\0",
                    "nvrtcGetProgramLogSize",
                )?,
                nvrtcGetProgramLog: required_symbol(
                    &lib,
                    b"nvrtcGetProgramLog\0",
                    "nvrtcGetProgramLog",
                )?,
                nvrtcDestroyProgram: required_symbol(
                    &lib,
                    b"nvrtcDestroyProgram\0",
                    "nvrtcDestroyProgram",
                )?,
                _lib: lib,
            })
        }
    }
}

unsafe fn required_symbol<T: Copy>(
    library: &Library,
    export: &'static [u8],
    name: &'static str,
) -> Result<T, LoadError> {
    // SAFETY: the caller supplies the exact ABI type for the named export and
    // retains `library` for at least as long as the returned function pointer.
    let symbol =
        unsafe { library.get::<T>(export) }.map_err(|source| LoadError::Symbol { name, source })?;
    Ok(*symbol)
}

fn find_nvrtc_library() -> Result<Library, LoadError> {
    let mut libraries = Vec::new();
    let mut directories = Vec::new();

    for candidate in ["nvrtc", "nvrtc64"].map(Path::new) {
        match open(candidate) {
            Ok(library) => return Ok(library),
            Err(failure) => libraries.push(failure),
        }
    }

    if let Some(cuda_path) = std::env::var_os("CUDA_PATH") {
        let root = PathBuf::from(cuda_path);
        for directory in [
            root.join("bin").join("x64"),
            root.join("bin"),
            root.join("lib64"),
            root.join("lib"),
        ] {
            if let Some(library) = scan_directory(&directory, &mut libraries, &mut directories) {
                return Ok(library);
            }
        }
    }

    #[cfg(target_os = "windows")]
    let fallback_names = [
        "nvrtc64_130_0.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_112_0.dll",
    ]
    .as_slice();
    #[cfg(not(target_os = "windows"))]
    let fallback_names = ["libnvrtc.so"].as_slice();

    for candidate in fallback_names.iter().map(Path::new) {
        match open(candidate) {
            Ok(library) => return Ok(library),
            Err(failure) => libraries.push(failure),
        }
    }

    Err(LoadError::Search {
        libraries,
        directories,
    })
}

fn scan_directory(
    directory: &Path,
    libraries: &mut Vec<LibraryFailure>,
    directories: &mut Vec<DirectoryFailure>,
) -> Option<Library> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(source) => {
            directories.push(DirectoryFailure {
                directory: directory.to_owned(),
                source,
            });
            return None;
        }
    };

    let mut candidates = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                match is_nvrtc_library(&path) {
                    Ok(true) => candidates.push(path),
                    Ok(false) => {}
                    Err(source) => directories.push(DirectoryFailure {
                        directory: path,
                        source,
                    }),
                }
            }
            Err(source) => directories.push(DirectoryFailure {
                directory: directory.to_owned(),
                source,
            }),
        }
    }
    candidates.sort_unstable();

    for candidate in candidates {
        match open(&candidate) {
            Ok(library) => return Some(library),
            Err(failure) => libraries.push(failure),
        }
    }
    None
}

fn is_nvrtc_library(path: &Path) -> std::io::Result<bool> {
    if !path.metadata()?.is_file() {
        return Ok(false);
    }
    let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    let extension = path.extension().and_then(|extension| extension.to_str());
    Ok(if cfg!(windows) {
        let lowered = filename.to_ascii_lowercase();
        lowered.starts_with("nvrtc")
            && !lowered.contains("builtins")
            && extension.is_some_and(|value| value.eq_ignore_ascii_case("dll"))
    } else {
        filename.starts_with("libnvrtc") && (extension == Some("so") || filename.contains(".so."))
    })
}

fn open(candidate: &Path) -> Result<Library, LibraryFailure> {
    // SAFETY: candidates are either NVRTC names from the platform loader or
    // entries beneath the configured CUDA toolkit. Loading may run platform
    // initialization; symbol typing is validated separately in `from_library`.
    unsafe { Library::new(candidate) }.map_err(|source| LibraryFailure {
        kind: library_failure_kind(candidate),
        candidate: candidate.to_owned(),
        source,
    })
}

fn library_failure_kind(candidate: &Path) -> LibraryFailureKind {
    if candidate.components().count() == 1 {
        return LibraryFailureKind::Loader;
    }
    match candidate.try_exists() {
        Ok(false) => LibraryFailureKind::Missing,
        Ok(true) => LibraryFailureKind::Invalid,
        Err(_) => LibraryFailureKind::Loader,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DirectoryFailure, LibraryFailure, LibraryFailureKind, LoadError, NvrtcDriver, open,
        scan_directory,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn absent_library_preserves_candidate_and_source() {
        let candidate = std::env::temp_dir().join(format!(
            "atlas-nvrtc-library-that-does-not-exist-{}",
            std::process::id()
        ));
        let failure = open(&candidate).expect_err("absent library cannot load");
        assert_eq!(failure.candidate, candidate);
        assert_eq!(failure.kind, LibraryFailureKind::Missing);
        assert!(!failure.source.to_string().is_empty());
    }

    #[test]
    fn present_non_library_is_invalid() {
        #[cfg(target_os = "windows")]
        let candidate = PathBuf::from(std::env::var_os("SystemRoot").expect("Windows system root"))
            .join("System32")
            .join("drivers")
            .join("etc")
            .join("hosts");
        #[cfg(not(target_os = "windows"))]
        let candidate = PathBuf::from("/etc/hosts");
        assert!(candidate.is_file(), "system hosts file is present");

        let failure = open(&candidate).expect_err("plain file cannot load as a library");
        assert_eq!(failure.candidate, candidate);
        assert_eq!(failure.kind, LibraryFailureKind::Invalid);
        assert!(!failure.source.to_string().is_empty());
    }

    #[test]
    fn missing_export_preserves_symbol_and_source() {
        #[cfg(target_os = "windows")]
        let candidate = Path::new("kernel32.dll");
        #[cfg(not(target_os = "windows"))]
        let candidate = Path::new("libc.so.6");
        let library = open(candidate).expect("system library is loadable");
        let error = NvrtcDriver::from_library(library).expect_err("system library is not NVRTC");
        match error {
            LoadError::Symbol { name, source } => {
                assert_eq!(name, "nvrtcCreateProgram");
                assert!(!source.to_string().is_empty());
            }
            other @ LoadError::Search { .. } => {
                panic!("wrong failure category: {other:?}")
            }
        }
    }

    #[test]
    fn directory_enumeration_preserves_path_and_source() {
        let directory = std::env::temp_dir().join(format!(
            "atlas-nvrtc-directory-that-does-not-exist-{}",
            std::process::id()
        ));
        let mut libraries: Vec<LibraryFailure> = Vec::new();
        let mut directories: Vec<DirectoryFailure> = Vec::new();

        assert!(
            scan_directory(&directory, &mut libraries, &mut directories).is_none(),
            "absent directory cannot provide NVRTC"
        );
        assert!(libraries.is_empty());
        assert_eq!(directories.len(), 1);
        assert_eq!(directories[0].directory, directory);
        assert_eq!(directories[0].source.kind(), std::io::ErrorKind::NotFound);
    }
}
