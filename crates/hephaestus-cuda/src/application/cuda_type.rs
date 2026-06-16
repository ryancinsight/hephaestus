use bytemuck::Pod;

/// Maps a host scalar type to its CUDA C++ type token at compile time.
pub trait CudaScalar: Pod {
    /// The CUDA type token (`"float"`, `"unsigned int"`, `"int"`).
    const CUDA_TYPE: &'static str;
}

impl CudaScalar for f32 {
    const CUDA_TYPE: &'static str = "float";
}

impl CudaScalar for u32 {
    const CUDA_TYPE: &'static str = "unsigned int";
}

impl CudaScalar for i32 {
    const CUDA_TYPE: &'static str = "int";
}
