//! Device-attribute discriminants from CUDA 13.3 cuda.h.

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub(crate) enum Attribute {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxSharedMemoryPerBlock = 8,
    WarpSize = 10,
    MultiprocessorCount = 16,
    L2CacheSize = 38,
    MaxThreadsPerMultiprocessor = 39,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxRegistersPerMultiprocessor = 82,
    MemoryPoolsSupported = 115,
}
