/// Build script: resolve CUDA toolkit path for the current host.
///
/// Same resolution strategy as coeus-cuda/build.rs:
///   Linux: CUDA_HOME > CUDA_PATH > /usr/local/cuda > /opt/cuda > /usr > WSL2 Windows path
///   Windows: CUDA_PATH > C:\Program Files\NVIDIA...\CUDA\v12.4
use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cuda_path = match target_os.as_str() {
        "linux" => resolve_linux(),
        "windows" => resolve_windows(),
        _ => None,
    };

    if let Some(path) = cuda_path {
        println!("cargo::rustc-env=CUDA_TOOLKIT_PATH={}", path.display());
        println!("cargo::rustc-env=CUDA_LIB_PATH={}", path.join("lib64").display());
    }
}

fn resolve_linux() -> Option<PathBuf> {
    if let Some(home) = env::var("CUDA_HOME").ok() {
        let p = PathBuf::from(&home);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    if let Some(path) = env::var("CUDA_PATH").ok() {
        let p = PathBuf::from(&path);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    let candidates = ["/usr/local/cuda", "/opt/cuda", "/usr", "/usr/bin"];
    for &cand in &candidates {
        let p = PathBuf::from(cand);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    // WSL2: Windows CUDA mounted at /mnt/c/
    let wsl2 = PathBuf::from("/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4");
    if wsl2.join("bin/nvcc").exists() {
        return Some(wsl2);
    }
    None
}

fn resolve_windows() -> Option<PathBuf> {
    if let Some(path) = env::var("CUDA_PATH").ok() {
        let p = PathBuf::from(&path);
        if p.join("bin/nvcc.exe").exists() {
            return Some(p);
        }
    }
    let candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
    ];
    for &cand in &candidates {
        let p = PathBuf::from(cand);
        if p.join("bin/nvcc.exe").exists() {
            return Some(p);
        }
    }
    None
}
