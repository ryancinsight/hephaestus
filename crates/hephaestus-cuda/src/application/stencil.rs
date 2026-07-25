//! CUDA 2D Laplacian stencil dispatch.

pub use hephaestus_core::{BoundaryCondition, Laplacian2DParams, LaplacianPolarity};
use hephaestus_core::{DispatchGrid, MultiStorageKernel, Result};

use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

const WORKGROUP: [usize; 3] = [8, 8, 1];

const LAPLACIAN_2D_KERNEL: &str = r#"
struct Laplacian2DParams {
    unsigned int dims_bc[4];
    float inv2[4];
};

extern "C" __global__ void laplacian_2d(
    const float* field,
    float* result,
    Laplacian2DParams params
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int nx = params.dims_bc[0];
    unsigned int ny = params.dims_bc[1];
    if (i >= nx || j >= ny) {
        return;
    }

    unsigned int idx = j * nx + i;
    unsigned int boundary = params.dims_bc[2];
    float laplacian = 0.0f;

    if (i > 0u && i < nx - 1u) {
        float left = field[j * nx + (i - 1u)];
        float center = field[idx];
        float right = field[j * nx + (i + 1u)];
        laplacian += (left - 2.0f * center + right) * params.inv2[0];
    } else if (i == 0u) {
        if (boundary == 0u) {
            float center = field[idx];
            laplacian += (-2.0f * center) * params.inv2[0];
        } else if (boundary == 1u) {
            float center = field[idx];
            if (nx >= 4u) {
                float u1 = field[j * nx + 1u];
                float u2 = field[j * nx + 2u];
                float u3 = field[j * nx + 3u];
                laplacian += (2.0f * center - 5.0f * u1 + 4.0f * u2 - u3) * params.inv2[0];
            } else {
                float right = field[j * nx + (i + 1u)];
                laplacian += (right - 2.0f * center + right) * params.inv2[0];
            }
        } else if (boundary == 2u) {
            float left = field[j * nx + (nx - 2u)];
            float center = field[idx];
            float right = field[j * nx + (i + 1u)];
            laplacian += (left - 2.0f * center + right) * params.inv2[0];
        }
    } else if (i == nx - 1u) {
        if (boundary == 0u) {
            float center = field[idx];
            laplacian += (-2.0f * center) * params.inv2[0];
        } else if (boundary == 1u) {
            float center = field[idx];
            if (nx >= 4u) {
                float u1 = field[j * nx + (nx - 2u)];
                float u2 = field[j * nx + (nx - 3u)];
                float u3 = field[j * nx + (nx - 4u)];
                laplacian += (2.0f * center - 5.0f * u1 + 4.0f * u2 - u3) * params.inv2[0];
            } else {
                float left = field[j * nx + (i - 1u)];
                laplacian += (left - 2.0f * center + left) * params.inv2[0];
            }
        } else if (boundary == 2u) {
            float left = field[j * nx + (i - 1u)];
            float center = field[idx];
            float right = field[j * nx + 1u];
            laplacian += (left - 2.0f * center + right) * params.inv2[0];
        }
    }

    if (j > 0u && j < ny - 1u) {
        float bottom = field[(j - 1u) * nx + i];
        float center = field[idx];
        float top = field[(j + 1u) * nx + i];
        laplacian += (bottom - 2.0f * center + top) * params.inv2[1];
    } else if (j == 0u) {
        if (boundary == 0u) {
            float center = field[idx];
            laplacian += (-2.0f * center) * params.inv2[1];
        } else if (boundary == 1u) {
            float center = field[idx];
            if (ny >= 4u) {
                float u1 = field[nx + i];
                float u2 = field[2u * nx + i];
                float u3 = field[3u * nx + i];
                laplacian += (2.0f * center - 5.0f * u1 + 4.0f * u2 - u3) * params.inv2[1];
            } else {
                float top = field[nx + i];
                laplacian += (top - 2.0f * center + top) * params.inv2[1];
            }
        } else if (boundary == 2u) {
            float bottom = field[(ny - 2u) * nx + i];
            float center = field[idx];
            float top = field[nx + i];
            laplacian += (bottom - 2.0f * center + top) * params.inv2[1];
        }
    } else if (j == ny - 1u) {
        if (boundary == 0u) {
            float center = field[idx];
            laplacian += (-2.0f * center) * params.inv2[1];
        } else if (boundary == 1u) {
            float center = field[idx];
            if (ny >= 4u) {
                float u1 = field[(ny - 2u) * nx + i];
                float u2 = field[(ny - 3u) * nx + i];
                float u3 = field[(ny - 4u) * nx + i];
                laplacian += (2.0f * center - 5.0f * u1 + 4.0f * u2 - u3) * params.inv2[1];
            } else {
                float bottom = field[(j - 1u) * nx + i];
                laplacian += (bottom - 2.0f * center + bottom) * params.inv2[1];
            }
        } else if (boundary == 2u) {
            float bottom = field[(j - 1u) * nx + i];
            float center = field[idx];
            float top = field[nx + i];
            laplacian += (bottom - 2.0f * center + top) * params.inv2[1];
        }
    }

    result[idx] = laplacian;
}
"#;

/// Compiled CUDA 2D Laplacian stencil kernel.
#[derive(Debug)]
pub struct Laplacian2DKernel {
    kernel: CudaMultiStorageKernel,
}

impl Laplacian2DKernel {
    /// Compile the stencil kernel for a CUDA device.
    pub fn new(_device: &CudaDevice) -> Result<Self> {
        let kernel = CudaMultiStorageKernel::new(
            "hephaestus-laplacian-2d",
            LAPLACIAN_2D_KERNEL,
            "laplacian_2d",
            &[0, 1],
            [8, 8, 1],
            0,
        )?;
        Ok(Self { kernel })
    }

    /// Dispatch the stencil over device-resident input and output buffers.
    pub fn dispatch(
        &self,
        device: &CudaDevice,
        input: &CudaBuffer<f32>,
        output: &CudaBuffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()> {
        params.validate_storage(input.len(), output.len())?;
        let grid = DispatchGrid::covering_domain([params.nx(), params.ny(), 1], WORKGROUP)?;
        MultiStorageKernel::<CudaDevice, Laplacian2DParams, [CudaStorageBinding<'_>; 2]>::dispatch(
            &self.kernel,
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
