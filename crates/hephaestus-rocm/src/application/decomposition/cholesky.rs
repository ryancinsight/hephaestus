//! HIP Cholesky factorization for finite symmetric positive-definite matrices.
//!
//! The factorization uses the left-looking recurrence
//! `L[k,k] = sqrt(A[k,k] - sum(L[k,j]^2))` and
//! `L[i,k] = (A[i,k] - sum(L[i,j] * L[k,j])) / L[k,k]`.
//! Each recurrence step is ordered on HIP's default stream, so a later column
//! observes the diagonal and column writes from the preceding steps.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp,
    KernelDevice, Result,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::{validate_dense_operand, validate_square};
use crate::application::pipeline::{
    CholeskyStage, LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::{DevicePtr, RocmBuffer};

const WIDTH: BlockWidth = BlockWidth::DEFAULT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CholeskyMeta {
    n: u32,
    k: u32,
}

const _: () = assert!(core::mem::size_of::<CholeskyMeta>() == 8);

/// Lower-triangular Cholesky factor on the device.
///
/// The host-side factor is retained only for the existing solve, determinant,
/// and inverse methods. The factorization itself is performed by HIP kernels.
pub struct GpuCholesky {
    inner: leto_ops::CholeskyDecomposition<f32>,
    lower: RocmBuffer<f32>,
    n: usize,
}

impl GpuCholesky {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lower(&self) -> &RocmBuffer<f32> {
        &self.lower
    }

    /// Consume and return the lower-triangular factor buffer.
    #[must_use]
    #[inline]
    pub fn into_lower(self) -> RocmBuffer<f32> {
        self.lower
    }

    /// Compute the determinant from the retained Cholesky factor.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** using the retained factor.
    ///
    /// The factorization is device-resident and HIP-computed. This method
    /// follows the common CUDA/wgpu contract by downloading the RHS for the
    /// existing host-side substitution implementation and uploading the
    /// solution.
    pub fn solve(&self, device: &RocmDevice, rhs: &RocmBuffer<f32>) -> Result<RocmBuffer<f32>> {
        if rhs.len() != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len(),
            });
        }
        if self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0_f32; self.n];
        device.download(rhs, &mut rhs_host)?;
        let rhs_view = leto::ArrayView::<f32, 1>::new(
            Layout::c_contiguous([self.n]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("Cholesky solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution =
            self.inner
                .solve(&rhs_view)
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("Cholesky solve failed: {error}"),
                })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }

    /// Compute **A**⁻¹ from the retained Cholesky factor.
    pub fn inv(&self, device: &RocmDevice) -> Result<RocmBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inverse = self
            .inner
            .inv()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("Cholesky inverse failed: {error}"),
            })?;
        device.upload(leto::Storage::as_slice(inverse.storage()))
    }
}

fn kernel_source() -> String {
    r#"
struct CholeskyMeta {
    unsigned int n;
    unsigned int k;
};

extern "C" __global__ void cholesky_validate(
    const float* matrix,
    unsigned int* status,
    CholeskyMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.n * meta.n;
    if (index < elements && !isfinite(matrix[index])) {
        atomicExch(status, 1u);
    }
}

extern "C" __global__ void cholesky_diagonal(
    float* matrix,
    unsigned int* status,
    CholeskyMeta meta
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || status[0] != 0u) {
        return;
    }
    unsigned int k = meta.k;
    float sum = 0.0f;
    for (unsigned int j = 0u; j < k; j++) {
        float value = matrix[k * meta.n + j];
        sum += value * value;
    }
    float diagonal = matrix[k * meta.n + k] - sum;
    if (!isfinite(diagonal) || !(diagonal > 0.0f)) {
        atomicExch(status, 1u);
        return;
    }
    matrix[k * meta.n + k] = sqrtf(diagonal);
}

extern "C" __global__ void cholesky_column(
    float* matrix,
    unsigned int* status,
    CholeskyMeta meta
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= meta.k || i >= meta.n || status[0] != 0u) {
        return;
    }
    float sum = 0.0f;
    for (unsigned int j = 0u; j < meta.k; j++) {
        sum += matrix[i * meta.n + j] * matrix[meta.k * meta.n + j];
    }
    float diagonal = matrix[meta.k * meta.n + meta.k];
    float value = (matrix[i * meta.n + meta.k] - sum) / diagonal;
    if (!isfinite(value)) {
        atomicExch(status, 1u);
        return;
    }
    matrix[i * meta.n + meta.k] = value;
}

extern "C" __global__ void cholesky_clear_upper(
    float* matrix,
    unsigned int* status,
    CholeskyMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.n * meta.n;
    if (index < elements) {
        unsigned int row = index / meta.n;
        unsigned int col = index - row * meta.n;
        if (col > row) {
            matrix[index] = 0.0f;
        }
    }
}
"#
    .to_string()
}

fn checked_dimension(n: usize) -> Result<(u32, usize)> {
    let elements = n
        .checked_mul(n)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("Cholesky matrix element count overflows for dimension {n}"),
        })?;
    let n_u32 = u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("Cholesky dimension {n} exceeds the HIP argument range"),
    })?;
    u32::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("Cholesky matrix element count {elements} exceeds the HIP index range"),
    })?;
    Ok((n_u32, elements))
}

fn launch_stage(
    device: &RocmDevice,
    stage: CholeskyStage,
    matrix: &RocmBuffer<f32>,
    status: &RocmBuffer<u32>,
    meta: CholeskyMeta,
    work_items: usize,
) -> Result<()> {
    let entry = match stage {
        CholeskyStage::Validate => "cholesky_validate",
        CholeskyStage::Diagonal => "cholesky_diagonal",
        CholeskyStage::Column => "cholesky_column",
        CholeskyStage::ClearUpper => "cholesky_clear_upper",
    };
    let kernel = cached_kernel(device, PipelineKey::Cholesky(stage), entry, kernel_source)?;
    let grid = if matches!(stage, CholeskyStage::Diagonal) {
        1
    } else {
        grid_size(work_items, WIDTH)?
    };
    let mut matrix_ptr: DevicePtr = matrix.raw();
    let mut status_ptr: DevicePtr = status.raw();
    let mut meta = meta;
    let mut args: [*mut core::ffi::c_void; 3] = [
        (&mut matrix_ptr as *mut DevicePtr).cast(),
        (&mut status_ptr as *mut DevicePtr).cast(),
        (&mut meta as *mut CholeskyMeta).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, WIDTH),
        &mut args,
    )
}

fn factor_on_device(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
    require_dense: bool,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    if require_dense {
        validate_dense_operand("cholesky", &matrix)?;
    }
    let (n_u32, elements) = checked_dimension(n)?;
    if n == 0 {
        let lower = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::cholesky_decompose(&leto::ArrayView::<f32, 2>::new(
            Layout::c_contiguous([0, 0]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("empty Cholesky layout failed: {error}"),
            })?,
            &[],
        ))
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("empty Cholesky decomposition failed: {error}"),
        })?;
        return Ok(GpuCholesky { inner, lower, n: 0 });
    }

    let dense_layout =
        Layout::c_contiguous([n, n]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("Cholesky dense layout failed: {error}"),
        })?;
    let lower = device.alloc_zeroed::<f32>(elements)?;
    if !require_dense {
        unary_elementwise_strided_into::<IdentityOp, f32, 2>(
            device,
            matrix,
            StridedOperand {
                buffer: &lower,
                layout: &dense_layout,
            },
            WIDTH,
        )?;
    }
    if require_dense {
        let mut stream = device.stream()?;
        stream.copy(matrix.buffer, &lower)?;
        stream.submit()?;
    }

    let status = device.alloc_zeroed::<u32>(1)?;
    let meta = CholeskyMeta { n: n_u32, k: 0 };
    launch_stage(
        device,
        CholeskyStage::Validate,
        &lower,
        &status,
        meta,
        elements,
    )?;
    for k in 0..n {
        let k_u32 = u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("Cholesky step {k} exceeds the HIP argument range"),
        })?;
        let meta = CholeskyMeta { n: n_u32, k: k_u32 };
        launch_stage(device, CholeskyStage::Diagonal, &lower, &status, meta, 1)?;
        launch_stage(device, CholeskyStage::Column, &lower, &status, meta, n)?;
    }
    launch_stage(
        device,
        CholeskyStage::ClearUpper,
        &lower,
        &status,
        CholeskyMeta { n: n_u32, k: 0 },
        elements,
    )?;

    let mut status_host = [0_u32; 1];
    device.download(&status, &mut status_host)?;
    if status_host[0] != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: "Cholesky factorization failed: input is non-finite or not positive-definite"
                .to_string(),
        });
    }

    let mut host_factor = vec![0.0_f32; elements];
    device.download(&lower, &mut host_factor)?;
    let host_array = leto::Array2::from_shape_vec([n, n], host_factor).map_err(|error| {
        HephaestusError::DispatchFailed {
            message: format!("Cholesky factor shape failed: {error}"),
        }
    })?;
    let inner = leto_ops::CholeskyDecomposition::from_raw_parts(host_array);
    Ok(GpuCholesky { inner, lower, n })
}

/// Compute the Cholesky factorization **A** = **L** **L**ᵀ through HIP.
///
/// Strided rank-2 inputs are materialized into a dense device buffer by the
/// existing ROCm strided identity kernel. The factorization then runs as
/// ordered HIP diagonal and column launches.
pub fn cholesky_decompose(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    factor_on_device(device, matrix, false)
}

/// Compute the Cholesky factorization for a dense C-contiguous input.
///
/// This entry point preserves the CUDA/wgpu blocked-path contract and rejects
/// non-dense views before the device-to-device startup copy. ROCm uses the
/// same ordered HIP factor kernels for this first parity slice; its blocked
/// API is therefore contract-equivalent without claiming a separate tuning
/// strategy.
pub fn cholesky_decompose_blocked(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    factor_on_device(device, matrix, true)
}

#[cfg(test)]
mod tests {
    use super::kernel_source;

    #[test]
    fn source_contains_ordered_factorization_stages() {
        let source = kernel_source();
        assert!(source.contains("cholesky_validate"));
        assert!(source.contains("cholesky_diagonal"));
        assert!(source.contains("cholesky_column"));
        assert!(source.contains("cholesky_clear_upper"));
        assert!(source.contains("sqrtf(diagonal)"));
    }
}
