//! CUDA fixed-scheme three-dimensional first-derivative dispatch.
//!
//! The CUDA counterpart of the WGPU sweep, computing the same five schemes
//! from the same parameter block. One entry point serves every scheme behind
//! a uniform branch, and the per-lane arithmetic keeps the provider's exact
//! operation order. `f32` only, matching
//! [`FixedFd3DParams`](hephaestus_core::FixedFd3DParams)'s lane layout.

use hephaestus_core::{DeviceBuffer, DispatchGrid, HephaestusError, MultiStorageKernel, Result};
pub use hephaestus_core::{FixedFd3DParams, FixedFd3DScheme};

use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

const WORKGROUP: [usize; 3] = [4, 4, 4];

const FIXED_FD_3D_KERNEL: &str = r#"
struct FixedFd3DParams {
    unsigned int dims_axis[4];
    unsigned int scheme[4];
    float scales[4];
};

__device__ __forceinline__ unsigned int input_stride(const FixedFd3DParams& p) {
    unsigned int axis = p.dims_axis[3];
    if (axis == 0u) {
        return p.dims_axis[1] * p.dims_axis[2];
    }
    if (axis == 1u) {
        return p.dims_axis[2];
    }
    return 1u;
}

__device__ __forceinline__ unsigned int output_extent(const FixedFd3DParams& p, unsigned int axis) {
    unsigned int extent = p.dims_axis[axis];
    if (p.scheme[0] == 3u && axis == p.dims_axis[3]) {
        return extent - 1u;
    }
    return extent;
}

__device__ __forceinline__ float first_forward(const float* field, unsigned int at, unsigned int stride, float inv_h) {
    return (field[at + stride] - field[at]) * inv_h;
}

__device__ __forceinline__ float first_backward(const float* field, unsigned int at, unsigned int stride, float inv_h) {
    return (field[at] - field[at - stride]) * inv_h;
}

__device__ __forceinline__ float second(const float* field, unsigned int at, unsigned int stride, float inv_2h) {
    return (field[at + stride] - field[at - stride]) * inv_2h;
}

// ((-8*m1 + 8*p1) + -p2) + m2, over 12h — the exact association the
// provider evaluates.
__device__ __forceinline__ float fourth(const float* field, unsigned int at, unsigned int stride, float inv_12h) {
    float m1 = field[at - stride];
    float m2 = field[at - 2u * stride];
    float p1 = field[at + stride];
    float p2 = field[at + 2u * stride];
    return ((-8.0f * m1) + (8.0f * p1) + (-p2) + m2) * inv_12h;
}

// ((-9*f[i+2] + 45*f[i+1]) + -45*f[i-1]) + 9*f[i-2] + -f[i-3] + f[i+3],
// over 60h — the exact chain the provider evaluates.
__device__ __forceinline__ float sixth_interior(const float* field, unsigned int at, unsigned int stride, float inv_60h) {
    return ((-9.0f * field[at + 2u * stride])
        + (45.0f * field[at + stride])
        + (-45.0f * field[at - stride])
        + (9.0f * field[at - 2u * stride])
        + (-field[at - 3u * stride])
        + field[at + 3u * stride]) * inv_60h;
}

extern "C" __global__ void fixed_fd_sweep(
    const float* field,
    float* result,
    FixedFd3DParams params
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int axis = params.dims_axis[3];
    unsigned int onx = output_extent(params, 0u);
    unsigned int ony = output_extent(params, 1u);
    unsigned int onz = output_extent(params, 2u);
    if (i >= onx || j >= ony || k >= onz) {
        return;
    }

    unsigned int nx = params.dims_axis[0];
    unsigned int ny = params.dims_axis[1];
    unsigned int nz = params.dims_axis[2];
    unsigned int c = k;
    if (axis == 0u) {
        c = i;
    } else if (axis == 1u) {
        c = j;
    }
    unsigned int n = params.dims_axis[axis];
    unsigned int stride = input_stride(params);
    // The output coordinate is a valid input coordinate on every axis: the
    // only shrunk axis is the swept one under the forward scheme, whose
    // output coordinate stays below n - 1.
    unsigned int input_flat = (i * ny + j) * nz + k;
    unsigned int base = input_flat - c * stride;
    unsigned int out_flat = (i * ony + j) * onz + k;
    (void)nx;

    float inv_h = params.scales[0];
    float inv_2h = params.scales[1];
    float inv_12h = params.scales[2];
    float inv_60h = params.scales[3];
    unsigned int at = base + c * stride;
    unsigned int id = params.scheme[0];
    float value = 0.0f;
    if (id == 0u) {
        if (c == 0u) {
            value = first_forward(field, at, stride, inv_h);
        } else if (c == n - 1u) {
            value = first_backward(field, at, stride, inv_h);
        } else {
            value = second(field, at, stride, inv_2h);
        }
    } else if (id == 1u) {
        if (n == 1u) {
            value = 0.0f;
        } else if (c == 0u) {
            value = first_forward(field, at, stride, inv_h);
        } else if (c == n - 1u) {
            value = first_backward(field, at, stride, inv_h);
        } else if (c < 2u || c >= n - 2u) {
            value = second(field, at, stride, inv_2h);
        } else {
            value = fourth(field, at, stride, inv_12h);
        }
    } else if (id == 2u) {
        if (c == 0u) {
            value = first_forward(field, at, stride, inv_h);
        } else if (c == n - 1u) {
            value = first_backward(field, at, stride, inv_h);
        } else if (c == 1u || c == n - 2u) {
            value = second(field, at, stride, inv_2h);
        } else if (c == 2u || c == n - 3u) {
            value = fourth(field, at, stride, inv_12h);
        } else {
            value = sixth_interior(field, at, stride, inv_60h);
        }
    } else if (id == 3u) {
        value = first_forward(field, at, stride, inv_h);
    } else {
        if (c == 0u) {
            value = first_forward(field, at, stride, inv_h);
        } else {
            value = first_backward(field, at, stride, inv_h);
        }
    }
    result[out_flat] = value;
}
"#;

/// Compiled CUDA fixed-scheme sweep kernel.
#[derive(Debug)]
pub struct FixedFd3DKernel {
    sweep: CudaMultiStorageKernel,
}

impl FixedFd3DKernel {
    /// Compile the sweep for a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns the runtime compiler or module-load failure.
    pub fn new(_device: &CudaDevice) -> Result<Self> {
        let block = [
            u32::try_from(WORKGROUP[0]).expect("invariant: literal block dim fits u32"),
            u32::try_from(WORKGROUP[1]).expect("invariant: literal block dim fits u32"),
            u32::try_from(WORKGROUP[2]).expect("invariant: literal block dim fits u32"),
        ];
        Ok(Self {
            sweep: CudaMultiStorageKernel::new(
                "hephaestus-fixed-fd-3d-sweep",
                FIXED_FD_3D_KERNEL,
                "fixed_fd_sweep",
                &[0, 1],
                block,
                0,
            )?,
        })
    }

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the launch
    /// failure.
    pub fn sweep(
        &self,
        device: &CudaDevice,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_storage(input.len(), output.len())?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.output_dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        MultiStorageKernel::<CudaDevice, FixedFd3DParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            &self.sweep,
            device,
            [
                CudaStorageBinding::new(0, input),
                CudaStorageBinding::new(1, output),
            ],
            params,
            grid,
        )
    }
}

/// Provider-owned implementation of [`hephaestus_core::FixedFd3DOps`] for
/// CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaFixedFd3DOps;

impl hephaestus_core::FixedFd3DOps<CudaDevice> for CudaFixedFd3DOps {
    type FixedFd3D = FixedFd3DKernel;

    fn prepare_fixed_fd_3d(&self, device: &CudaDevice) -> Result<Self::FixedFd3D> {
        FixedFd3DKernel::new(device)
    }

    fn fixed_fd_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::FixedFd3D,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.sweep(device, input, output, params)
    }
}
