//! Typed resources and device ceilings for a static FFT plan.

use core::fmt;

/// FFT resource whose required capacity exceeded a device limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FftResource {
    /// One real or imaginary operand buffer.
    Volume,
    /// One shared real or imaginary staging buffer.
    Workspace,
    /// Roots and normalization values for staged radix transforms.
    RadixTwiddle,
    /// Roots and normalization values for fused transforms.
    FusedTwiddle,
    /// One transformed Bluestein convolution kernel component on this axis.
    ChirpKernel(usize),
    /// One direct Bluestein factor component on this axis.
    ChirpFactor(usize),
    /// One dispatch grid dimension.
    Dispatch,
    /// Invocations in a workgroup.
    Workgroup,
}

impl fmt::Display for FftResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Volume => formatter.write_str("volume"),
            Self::Workspace => formatter.write_str("workspace"),
            Self::RadixTwiddle => formatter.write_str("radix twiddle"),
            Self::FusedTwiddle => formatter.write_str("fused twiddle"),
            Self::ChirpKernel(axis) => write!(formatter, "axis {axis} Bluestein kernel"),
            Self::ChirpFactor(axis) => write!(formatter, "axis {axis} Bluestein factor"),
            Self::Dispatch => formatter.write_str("dispatch"),
            Self::Workgroup => formatter.write_str("workgroup"),
        }
    }
}

/// Inclusive limit whose units also define the reported required quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FftLimit {
    /// The device's maximum allocation size in bytes.
    BufferBytes,
    /// The device's maximum storage binding size in bytes.
    StorageBindingBytes,
    /// The shader's maximum addressable element count.
    ShaderElements,
    /// The maximum number of dispatch workgroups per dimension.
    WorkgroupsPerDimension,
    /// The maximum number of invocations along the workgroup's X dimension.
    WorkgroupSizeX,
    /// The maximum total number of invocations in one workgroup.
    InvocationsPerWorkgroup,
}

impl fmt::Display for FftLimit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::BufferBytes => "max_buffer_size (bytes)",
            Self::StorageBindingBytes => "max_storage_buffer_binding_size (bytes)",
            Self::ShaderElements => "shader address domain (elements)",
            Self::WorkgroupsPerDimension => "max_compute_workgroups_per_dimension (workgroups)",
            Self::WorkgroupSizeX => "max_compute_workgroup_size_x (invocations)",
            Self::InvocationsPerWorkgroup => "max_compute_invocations_per_workgroup (invocations)",
        })
    }
}
