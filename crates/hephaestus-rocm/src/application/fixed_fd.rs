//! ROCm/HIP fixed-scheme three-dimensional first-derivative dispatch.
//!
//! The HIP counterpart of the CUDA sweep, computing the same five schemes
//! from the same parameter block. One entry point serves every scheme behind
//! a uniform branch, and the per-lane arithmetic keeps the provider's exact
//! operation order. `f32` only, matching
//! [`FixedFd3DParams`](hephaestus_core::FixedFd3DParams)'s lane layout.

use hephaestus_core::{DeviceBuffer, DispatchGrid, HephaestusError, MultiStorageKernel, Result};
pub use hephaestus_core::{FixedFd3DParams, FixedFd3DScheme};

use crate::RocmDevice;
use crate::application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
use crate::infrastructure::RocmBuffer;

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

// ((-8*m1 + 8*p1) + -p2) + m2, over 12h â€” the exact association the
// provider evaluates.
__device__ __forceinline__ float fourth(const float* field, unsigned int at, unsigned int stride, float inv_12h) {
    float m1 = field[at - stride];
    float m2 = field[at - 2u * stride];
    float p1 = field[at + stride];
    float p2 = field[at + 2u * stride];
    return ((-8.0f * m1) + (8.0f * p1) + (-p2) + m2) * inv_12h;
}

// ((-9*f[i+2] + 45*f[i+1]) + -45*f[i-1]) + 9*f[i-2] + -f[i-3] + f[i+3],
// over 60h â€” the exact chain the provider evaluates.
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

// Transpose sweeps: each lane accumulates the provider's predicated terms in
// canonical order â€” wall taps, then second-, fourth-, then sixth-order row
// taps, each (c*u)*inv. The grid covers the input domain; upstream reads use
// the upstream strides, which shrink on the swept axis under the forward
// scheme.

__device__ __forceinline__ unsigned int upstream_stride(const FixedFd3DParams& p) {
    unsigned int axis = p.dims_axis[3];
    if (axis == 0u) {
        return output_extent(p, 1u) * output_extent(p, 2u);
    }
    if (axis == 1u) {
        return output_extent(p, 2u);
    }
    return 1u;
}

__device__ __forceinline__ bool is_central4_second(unsigned int n, unsigned int i) {
    return i >= 1u && i + 2u <= n && (i < 2u || i + 2u >= n);
}

__device__ __forceinline__ bool is_central4_fourth(unsigned int n, unsigned int i) {
    return i >= 2u && i + 3u <= n;
}

__device__ __forceinline__ float adjoint2(const float* field, const FixedFd3DParams& p, unsigned int base_up, unsigned int stride_up, unsigned int j, unsigned int n) {
    float inv_h = p.scales[0];
    float inv_2h = p.scales[1];
    float v = 0.0f;
    if (j == 0u) {
        v = v + ((-field[base_up]) * inv_h);
    }
    if (j == 1u) {
        v = v + (field[base_up] * inv_h);
    }
    if (j + 2u == n) {
        v = v + ((-field[base_up + (n - 1u) * stride_up]) * inv_h);
    }
    if (j + 1u == n) {
        v = v + (field[base_up + (n - 1u) * stride_up] * inv_h);
    }
    if (j + 3u <= n) {
        v = v + ((-field[base_up + (j + 1u) * stride_up]) * inv_2h);
    }
    if (j >= 2u) {
        v = v + (field[base_up + (j - 1u) * stride_up] * inv_2h);
    }
    return v;
}

__device__ __forceinline__ float adjoint4(const float* field, const FixedFd3DParams& p, unsigned int base_up, unsigned int stride_up, unsigned int j, unsigned int n) {
    if (n == 1u) {
        return 0.0f;
    }
    float inv_h = p.scales[0];
    float inv_2h = p.scales[1];
    float inv_12h = p.scales[2];
    float v = 0.0f;
    if (j == 0u) {
        v = v + ((-field[base_up]) * inv_h);
    }
    if (j == 1u) {
        v = v + (field[base_up] * inv_h);
    }
    if (j + 2u == n) {
        v = v + ((-field[base_up + (n - 1u) * stride_up]) * inv_h);
    }
    if (j + 1u == n) {
        v = v + (field[base_up + (n - 1u) * stride_up] * inv_h);
    }
    if (j + 1u < n && is_central4_second(n, j + 1u)) {
        v = v + ((-field[base_up + (j + 1u) * stride_up]) * inv_2h);
    }
    if (j >= 1u && is_central4_second(n, j - 1u)) {
        v = v + (field[base_up + (j - 1u) * stride_up] * inv_2h);
    }
    if (j + 2u < n && is_central4_fourth(n, j + 2u)) {
        v = v + (field[base_up + (j + 2u) * stride_up] * inv_12h);
    }
    if (j + 1u < n && is_central4_fourth(n, j + 1u)) {
        v = v + (((-8.0f) * field[base_up + (j + 1u) * stride_up]) * inv_12h);
    }
    if (j >= 1u && is_central4_fourth(n, j - 1u)) {
        v = v + ((8.0f * field[base_up + (j - 1u) * stride_up]) * inv_12h);
    }
    if (j >= 2u && is_central4_fourth(n, j - 2u)) {
        v = v + ((-field[base_up + (j - 2u) * stride_up]) * inv_12h);
    }
    return v;
}

__device__ __forceinline__ float adjoint6(const float* field, const FixedFd3DParams& p, unsigned int base_up, unsigned int stride_up, unsigned int j, unsigned int n) {
    float inv_h = p.scales[0];
    float inv_2h = p.scales[1];
    float inv_12h = p.scales[2];
    float inv_60h = p.scales[3];
    float v = 0.0f;
    if (j == 0u) {
        v = v + ((-field[base_up]) * inv_h);
    }
    if (j == 1u) {
        v = v + (field[base_up] * inv_h);
    }
    if (j + 2u == n) {
        v = v + ((-field[base_up + (n - 1u) * stride_up]) * inv_h);
    }
    if (j + 1u == n) {
        v = v + (field[base_up + (n - 1u) * stride_up] * inv_h);
    }
    if (j + 1u < n && (j + 1u == 1u || j + 1u + 2u == n)) {
        v = v + ((-field[base_up + (j + 1u) * stride_up]) * inv_2h);
    }
    if (j >= 1u && (j - 1u == 1u || j - 1u + 2u == n)) {
        v = v + (field[base_up + (j - 1u) * stride_up] * inv_2h);
    }
    if (j + 2u < n && (j + 2u == 2u || j + 2u + 3u == n)) {
        v = v + (field[base_up + (j + 2u) * stride_up] * inv_12h);
    }
    if (j + 1u < n && (j + 1u == 2u || j + 1u + 3u == n)) {
        v = v + (((-8.0f) * field[base_up + (j + 1u) * stride_up]) * inv_12h);
    }
    if (j >= 1u && (j - 1u == 2u || j - 1u + 3u == n)) {
        v = v + ((8.0f * field[base_up + (j - 1u) * stride_up]) * inv_12h);
    }
    if (j >= 2u && (j - 2u == 2u || j - 2u + 3u == n)) {
        v = v + ((-field[base_up + (j - 2u) * stride_up]) * inv_12h);
    }
    if (j + 3u < n && j + 3u >= 3u && j + 3u + 4u <= n) {
        v = v + ((-field[base_up + (j + 3u) * stride_up]) * inv_60h);
    }
    if (j + 2u < n && j + 2u >= 3u && j + 2u + 4u <= n) {
        v = v + ((9.0f * field[base_up + (j + 2u) * stride_up]) * inv_60h);
    }
    if (j + 1u < n && j + 1u >= 3u && j + 1u + 4u <= n) {
        v = v + (((-45.0f) * field[base_up + (j + 1u) * stride_up]) * inv_60h);
    }
    if (j >= 1u && j - 1u >= 3u && j - 1u + 4u <= n) {
        v = v + ((45.0f * field[base_up + (j - 1u) * stride_up]) * inv_60h);
    }
    if (j >= 2u && j - 2u >= 3u && j - 2u + 4u <= n) {
        v = v + (((-9.0f) * field[base_up + (j - 2u) * stride_up]) * inv_60h);
    }
    if (j >= 3u && j - 3u >= 3u && j - 3u + 4u <= n) {
        v = v + (field[base_up + (j - 3u) * stride_up] * inv_60h);
    }
    return v;
}

__device__ __forceinline__ float adjoint_forward(const float* field, float inv_h, unsigned int base_up, unsigned int stride_up, unsigned int j, unsigned int n) {
    float v = 0.0f;
    if (j + 1u < n) {
        v = v + ((-field[base_up + j * stride_up]) * inv_h);
    }
    if (j >= 1u) {
        v = v + (field[base_up + (j - 1u) * stride_up] * inv_h);
    }
    return v;
}

__device__ __forceinline__ float adjoint_backward(const float* field, float inv_h, unsigned int base_up, unsigned int stride_up, unsigned int j, unsigned int n) {
    float v = 0.0f;
    if (j == 0u) {
        v = v + ((-field[base_up]) * inv_h);
    }
    if (j == 1u) {
        v = v + (field[base_up] * inv_h);
    }
    if (j + 2u <= n) {
        v = v + ((-field[base_up + (j + 1u) * stride_up]) * inv_h);
    }
    if (j >= 1u) {
        v = v + (field[base_up + j * stride_up] * inv_h);
    }
    return v;
}

extern "C" __global__ void fixed_fd_adjoint(
    const float* field,
    float* result,
    FixedFd3DParams params
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int axis = params.dims_axis[3];
    unsigned int nx = params.dims_axis[0];
    unsigned int ny = params.dims_axis[1];
    unsigned int nz = params.dims_axis[2];
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    unsigned int lane = k;
    if (axis == 0u) {
        lane = i;
    } else if (axis == 1u) {
        lane = j;
    }
    unsigned int n = params.dims_axis[axis];
    unsigned int stride_up = upstream_stride(params);
    // Upstream flat of this lane's coordinates under the upstream strides:
    // arithmetic only, since lane n - 1 has no upstream lane under the
    // forward scheme â€” subtracting lane * stride_up lands back in range.
    unsigned int ony = output_extent(params, 1u);
    unsigned int onz = output_extent(params, 2u);
    unsigned int up_flat = (i * ony + j) * onz + k;
    unsigned int base_up = up_flat - lane * stride_up;
    unsigned int out_flat = (i * ny + j) * nz + k;

    unsigned int id = params.scheme[0];
    float value = 0.0f;
    if (id == 0u) {
        value = adjoint2(field, params, base_up, stride_up, lane, n);
    } else if (id == 1u) {
        value = adjoint4(field, params, base_up, stride_up, lane, n);
    } else if (id == 2u) {
        value = adjoint6(field, params, base_up, stride_up, lane, n);
    } else if (id == 3u) {
        value = adjoint_forward(field, params.scales[0], base_up, stride_up, lane, n);
    } else {
        value = adjoint_backward(field, params.scales[0], base_up, stride_up, lane, n);
    }
    result[out_flat] = value;
}
"#;

/// Compiled HIP fixed-scheme sweep kernel, with its transpose.
#[derive(Debug)]
pub struct FixedFd3DKernel {
    sweep: RocmMultiStorageKernel,
    adjoint: RocmMultiStorageKernel,
}

impl FixedFd3DKernel {
    /// Compile the sweep and its transpose for a ROCm device.
    pub fn new(_device: &RocmDevice) -> Result<Self> {
        let sweep = RocmMultiStorageKernel::new(
            "hephaestus-fixed-fd-3d-sweep",
            FIXED_FD_3D_KERNEL,
            "fixed_fd_sweep",
            &[0, 1],
            [4, 4, 4],
            0,
        )?;
        let adjoint = RocmMultiStorageKernel::new(
            "hephaestus-fixed-fd-3d-adjoint",
            FIXED_FD_3D_KERNEL,
            "fixed_fd_adjoint",
            &[0, 1],
            [4, 4, 4],
            0,
        )?;
        Ok(Self { sweep, adjoint })
    }

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    pub fn sweep(
        &self,
        device: &RocmDevice,
        input: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
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
        MultiStorageKernel::<RocmDevice, FixedFd3DParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.sweep,
            device,
            [
                RocmStorageBinding::new(0, input),
                RocmStorageBinding::new(1, output),
            ],
            params,
            grid,
        )
    }

    /// Sweep the transpose of the scheme in `params` along its axis,
    /// `upstream` into `grad`.
    pub fn adjoint(
        &self,
        device: &RocmDevice,
        upstream: &RocmBuffer<f32>,
        grad: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        params.validate_adjoint_storage(upstream.len(), grad.len())?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(params.dims()) {
            *slot =
                usize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid extent does not fit usize: {error}"),
                })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        MultiStorageKernel::<RocmDevice, FixedFd3DParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.adjoint,
            device,
            [
                RocmStorageBinding::new(0, upstream),
                RocmStorageBinding::new(1, grad),
            ],
            params,
            grid,
        )
    }
}

/// Provider-owned implementation of [`FixedFd3DOps`](hephaestus_core::FixedFd3DOps) for Rocm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmFixedFd3DOps;

impl hephaestus_core::FixedFd3DOps<RocmDevice> for RocmFixedFd3DOps {
    type FixedFd3D = FixedFd3DKernel;

    fn prepare_fixed_fd_3d(&self, device: &RocmDevice) -> Result<Self::FixedFd3D> {
        FixedFd3DKernel::new(device)
    }

    fn fixed_fd_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::FixedFd3D,
        input: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.sweep(device, input, output, params)
    }

    fn fixed_fd_adjoint_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::FixedFd3D,
        upstream: &RocmBuffer<f32>,
        grad: &RocmBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        kernel.adjoint(device, upstream, grad, params)
    }
}
