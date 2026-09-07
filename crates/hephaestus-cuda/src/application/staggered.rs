//! CUDA three-dimensional staggered gradient/divergence dispatch.
//!
//! The CUDA counterpart of the WGPU pair, computing the same stencil from the
//! same provider-derived taps. Both entry points are compiled from one source
//! and share its indexing and reflection helpers, so the pair cannot drift
//! apart. `f32` only, matching [`Staggered3DParams`](hephaestus_core::Staggered3DParams)'s lane layout.
//!
//! # The divergence gathers
//!
//! Leto scatters `-Gᵀ` directly, which makes the adjoint identity true by
//! construction; a GPU cannot scatter without atomics, so this kernel gathers
//! the transpose. The derivation and its wall closure are recorded once, in
//! `hephaestus-wgpu`'s `staggered3d` module and ADR 0057 — this kernel is the
//! same formula in CUDA C, and the shared conformance clauses judge both
//! backends against the same three oracles.

use hephaestus_core::{DeviceBuffer, DispatchGrid, HephaestusError, MultiStorageKernel, Result};
pub use hephaestus_core::{Staggered3DParams, StaggeredAxis};

use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

const WORKGROUP: [usize; 3] = [4, 4, 4];

const STAGGERED_3D_KERNEL: &str = r#"
struct Staggered3DParams {
    unsigned int dims_axis[4];
    unsigned int order[4];
    float inv_spacing[4];
    float taps_low[4];
    float taps_high[4];
};

__device__ __forceinline__ float tap(const Staggered3DParams& p, unsigned int n) {
    // n is 0-based; two four-lane groups carry c_1..c_8.
    return n < 4u ? p.taps_low[n] : p.taps_high[n - 4u];
}

__device__ __forceinline__ unsigned int axis_stride(const Staggered3DParams& p) {
    unsigned int axis = p.dims_axis[3];
    if (axis == 0u) {
        return p.dims_axis[1] * p.dims_axis[2];
    }
    if (axis == 1u) {
        return p.dims_axis[2];
    }
    return 1u;
}

// One reflection step about the walls between cells: -1-m low, 2*extent-1-m
// high. Exact for |offset| <= extent, which Staggered3DParams guarantees.
__device__ __forceinline__ unsigned int reflect(int m, int extent) {
    if (m < 0) {
        return (unsigned int)(-1 - m);
    }
    if (m >= extent) {
        return (unsigned int)(2 * extent - 1 - m);
    }
    return (unsigned int)m;
}

struct Site {
    bool inside;
    unsigned int base;
    unsigned int stride;
    int coord;
    int extent;
    unsigned int taps;
    float scale;
};

__device__ __forceinline__ Site site(const Staggered3DParams& p) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    Site s;
    s.inside = !(i >= p.dims_axis[0] || j >= p.dims_axis[1] || k >= p.dims_axis[2]);
    s.stride = axis_stride(p);
    s.taps = p.order[0];

    unsigned int axis = p.dims_axis[3];
    unsigned int coord = k;
    if (axis == 0u) {
        coord = i;
    } else if (axis == 1u) {
        coord = j;
    }
    s.coord = (int)coord;
    s.extent = (int)p.dims_axis[axis];
    s.scale = p.inv_spacing[axis];

    unsigned int flat = (i * p.dims_axis[1] + j) * p.dims_axis[2] + k;
    // Strip the axis coordinate so a neighbour is base + m * stride.
    s.base = flat - coord * s.stride;
    return s;
}

extern "C" __global__ void staggered_gradient(
    const float* field,
    float* result,
    Staggered3DParams params
) {
    Site s = site(params);
    if (!s.inside) {
        return;
    }
    float sum = 0.0f;
    for (unsigned int n = 1u; n <= s.taps; ++n) {
        float c = tap(params, n - 1u);
        unsigned int hi = reflect(s.coord + (int)n, s.extent);
        unsigned int lo = reflect(s.coord - (int)n + 1, s.extent);
        sum += c * (field[s.base + hi * s.stride] - field[s.base + lo * s.stride]);
    }
    result[s.base + (unsigned int)s.coord * s.stride] = sum * s.scale;
}

extern "C" __global__ void staggered_divergence(
    const float* field,
    float* result,
    Staggered3DParams params
) {
    Site s = site(params);
    if (!s.inside) {
        return;
    }
    int j = s.coord;
    float sum = 0.0f;
    for (unsigned int n = 1u; n <= s.taps; ++n) {
        float c = tap(params, n - 1u);
        int step = (int)n;
        float gathered = 0.0f;

        // The unreflected preimage under i -> R(i - n + 1).
        int direct_low = j + step - 1;
        if (direct_low < s.extent) {
            gathered += field[s.base + (unsigned int)direct_low * s.stride];
        }
        // Its low-wall reflection.
        int mirrored_low = step - 2 - j;
        if (mirrored_low >= 0) {
            gathered += field[s.base + (unsigned int)mirrored_low * s.stride];
        }
        // The unreflected preimage under i -> R(i + n).
        int direct_high = j - step;
        if (direct_high >= 0) {
            gathered -= field[s.base + (unsigned int)direct_high * s.stride];
        }
        // Its high-wall reflection.
        if (j + step >= s.extent) {
            int mirrored_high = 2 * s.extent - 1 - j - step;
            gathered -= field[s.base + (unsigned int)mirrored_high * s.stride];
        }

        sum += c * gathered;
    }
    result[s.base + (unsigned int)j * s.stride] = sum * s.scale;
}
"#;

/// Compiled CUDA staggered gradient and divergence kernels.
#[derive(Debug)]
pub struct Staggered3DKernel {
    gradient: CudaMultiStorageKernel,
    divergence: CudaMultiStorageKernel,
}

impl Staggered3DKernel {
    /// Compile the staggered pair for a CUDA device.
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
            gradient: CudaMultiStorageKernel::new(
                "hephaestus-staggered-3d-gradient",
                STAGGERED_3D_KERNEL,
                "staggered_gradient",
                &[0, 1],
                block,
                0,
            )?,
            divergence: CudaMultiStorageKernel::new(
                "hephaestus-staggered-3d-divergence",
                STAGGERED_3D_KERNEL,
                "staggered_divergence",
                &[0, 1],
                block,
                0,
            )?,
        })
    }

    /// Dispatch the gradient over device-resident buffers.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the grid, or the launch
    /// failure.
    pub fn gradient(
        &self,
        device: &CudaDevice,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        self.dispatch(&self.gradient, device, input, output, params)
    }

    /// Dispatch the divergence over device-resident buffers.
    ///
    /// # Errors
    ///
    /// See [`Self::gradient`].
    pub fn divergence(
        &self,
        device: &CudaDevice,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        self.dispatch(&self.divergence, device, input, output, params)
    }

    fn dispatch(
        &self,
        kernel: &CudaMultiStorageKernel,
        device: &CudaDevice,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        params.validate_storage(input.len(), output.len())?;
        let mut dims = [0_usize; 3];
        for (slot, extent) in dims.iter_mut().zip(&params.dims_axis[..3]) {
            *slot = usize::try_from(*extent).map_err(|error| {
                HephaestusError::InvalidConfiguration {
                    message: format!("staggered grid extent does not fit usize: {error}"),
                }
            })?;
        }
        let grid = DispatchGrid::covering_domain(dims, WORKGROUP)?;
        MultiStorageKernel::<CudaDevice, Staggered3DParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            kernel,
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

/// Provider-owned implementation of [`hephaestus_core::Staggered3DOps`] for
/// CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaStaggered3DOps;

impl hephaestus_core::Staggered3DOps<CudaDevice> for CudaStaggered3DOps {
    type Staggered3D = Staggered3DKernel;

    fn prepare_staggered_3d(&self, device: &CudaDevice) -> Result<Self::Staggered3D> {
        Staggered3DKernel::new(device)
    }

    fn staggered_gradient_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::Staggered3D,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        kernel.gradient(device, input, output, params)
    }

    fn staggered_divergence_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::Staggered3D,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        kernel.divergence(device, input, output, params)
    }
}
