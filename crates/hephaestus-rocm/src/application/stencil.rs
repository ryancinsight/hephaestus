//! ROCm/HIP 2D Laplacian stencil dispatch.

pub use hephaestus_core::{BoundaryCondition, Laplacian2DParams, LaplacianPolarity};
use hephaestus_core::{DeviceBuffer, DispatchGrid, MultiStorageKernel, Result};

use crate::RocmDevice;
use crate::application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
use crate::infrastructure::RocmBuffer;

const WORKGROUP: [usize; 3] = [8, 8, 1];

const LAPLACIAN_2D_KERNEL: &str = r#"
struct Laplacian2DParams {
    unsigned int dims_bc[4];
    float inv2[4];
};

__device__ float laplacian_x(
    const float* field,
    unsigned int i,
    unsigned int j,
    unsigned int nx,
    unsigned int boundary,
    float coefficient
) {
    unsigned int idx = j * nx + i;
    if (i > 0u && i < nx - 1u) {
        return (field[idx - 1u] - 2.0f * field[idx] + field[idx + 1u]) * coefficient;
    }
    if (i == 0u) {
        if (boundary == 0u) {
            return (-2.0f * field[idx]) * coefficient;
        }
        if (boundary == 1u) {
            if (nx >= 4u) {
                return (2.0f * field[idx] - 5.0f * field[j * nx + 1u]
                    + 4.0f * field[j * nx + 2u] - field[j * nx + 3u]) * coefficient;
            }
            return (field[idx + 1u] - 2.0f * field[idx] + field[idx + 1u]) * coefficient;
        }
        if (boundary == 2u) {
            return (field[j * nx + (nx - 2u)] - 2.0f * field[idx]
                + field[idx + 1u]) * coefficient;
        }
    } else if (i == nx - 1u) {
        if (boundary == 0u) {
            return (-2.0f * field[idx]) * coefficient;
        }
        if (boundary == 1u) {
            if (nx >= 4u) {
                return (2.0f * field[idx] - 5.0f * field[j * nx + (nx - 2u)]
                    + 4.0f * field[j * nx + (nx - 3u)] - field[j * nx + (nx - 4u)]) * coefficient;
            }
            return (field[idx - 1u] - 2.0f * field[idx] + field[idx - 1u]) * coefficient;
        }
        if (boundary == 2u) {
            return (field[idx - 1u] - 2.0f * field[idx]
                + field[j * nx + 1u]) * coefficient;
        }
    }
    return 0.0f;
}

__device__ float laplacian_y(
    const float* field,
    unsigned int i,
    unsigned int j,
    unsigned int nx,
    unsigned int ny,
    unsigned int boundary,
    float coefficient
) {
    unsigned int idx = j * nx + i;
    if (j > 0u && j < ny - 1u) {
        return (field[idx - nx] - 2.0f * field[idx] + field[idx + nx]) * coefficient;
    }
    if (j == 0u) {
        if (boundary == 0u) {
            return (-2.0f * field[idx]) * coefficient;
        }
        if (boundary == 1u) {
            if (ny >= 4u) {
                return (2.0f * field[idx] - 5.0f * field[nx + i]
                    + 4.0f * field[2u * nx + i] - field[3u * nx + i]) * coefficient;
            }
            return (field[idx + nx] - 2.0f * field[idx] + field[idx + nx]) * coefficient;
        }
        if (boundary == 2u) {
            return (field[(ny - 2u) * nx + i] - 2.0f * field[idx]
                + field[idx + nx]) * coefficient;
        }
    } else if (j == ny - 1u) {
        if (boundary == 0u) {
            return (-2.0f * field[idx]) * coefficient;
        }
        if (boundary == 1u) {
            if (ny >= 4u) {
                return (2.0f * field[idx] - 5.0f * field[(ny - 2u) * nx + i]
                    + 4.0f * field[(ny - 3u) * nx + i] - field[(ny - 4u) * nx + i]) * coefficient;
            }
            return (field[idx - nx] - 2.0f * field[idx] + field[idx - nx]) * coefficient;
        }
        if (boundary == 2u) {
            return (field[idx - nx] - 2.0f * field[idx]
                + field[nx + i]) * coefficient;
        }
    }
    return 0.0f;
}

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
    result[idx] = laplacian_x(field, i, j, nx, boundary, params.inv2[0])
        + laplacian_y(field, i, j, nx, ny, boundary, params.inv2[1]);
}
"#;

/// Compiled HIP 2D Laplacian stencil kernel.
#[derive(Debug)]
pub struct Laplacian2DKernel {
    kernel: RocmMultiStorageKernel,
}

impl Laplacian2DKernel {
    /// Compile the stencil kernel for a ROCm device.
    pub fn new(_device: &RocmDevice) -> Result<Self> {
        let kernel = RocmMultiStorageKernel::new(
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
        device: &RocmDevice,
        input: &RocmBuffer<f32>,
        output: &RocmBuffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()> {
        params.validate_storage(input.len(), output.len())?;
        let grid = DispatchGrid::covering_domain([params.nx(), params.ny(), 1], WORKGROUP)?;
        MultiStorageKernel::<RocmDevice, Laplacian2DParams, [RocmStorageBinding<'_>; 2]>::dispatch(
            &self.kernel,
            device,
            [
                RocmStorageBinding::new(0, input),
                RocmStorageBinding::new(1, output),
            ],
            params,
            grid,
        )
    }
}
