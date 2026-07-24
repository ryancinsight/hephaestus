//! HIP complete-pivoted LU factorization.
//!
//! Each ordered HIP step selects the largest-magnitude entry in the trailing
//! submatrix, swaps its row and column into the diagonal, and applies one
//! Gaussian-elimination update. The packed factors and both permutations stay
//! resident on the device. The retained host factor exists only for the
//! established scalar determinant, solve, and inverse contracts shared by the
//! CUDA and wgpu backends.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp,
    KernelDevice, Result,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::{validate_dense_operand, validate_square};
use crate::application::pipeline::{
    FullPivLuStage, LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::{DevicePtr, RocmBuffer};

const WIDTH: BlockWidth = BlockWidth::DEFAULT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FullPivLuMeta {
    n: u32,
    k: u32,
}

const _: () = assert!(core::mem::size_of::<FullPivLuMeta>() == 8);

/// Complete-pivoted LU result with packed factors and permutations on the device.
pub struct GpuFullPivLuDecomposition {
    /// Host-side reference retained for scalar solve, determinant, and inverse.
    inner: leto_ops::FullPivLuDecomposition<f32>,
    /// Packed unit-lower/upper factors.
    lu: RocmBuffer<f32>,
    /// Row permutation, where each entry is the original row at that position.
    row_perm: Vec<usize>,
    /// Column permutation, where each entry is the original column at that position.
    col_perm: Vec<usize>,
    rank: usize,
    n: usize,
}

impl GpuFullPivLuDecomposition {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Numerical rank reported by complete pivoting.
    #[must_use]
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Borrow the packed factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lu_buffer(&self) -> &RocmBuffer<f32> {
        &self.lu
    }

    /// Return the row permutation.
    #[must_use]
    #[inline]
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_perm
    }

    /// Return the column permutation.
    #[must_use]
    #[inline]
    pub fn col_permutation(&self) -> &[usize] {
        &self.col_perm
    }

    /// Compute the determinant from the retained factor.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** using the retained factor.
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
                message: format!("complete-pivoted LU solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution =
            self.inner
                .solve(&rhs_view)
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("complete-pivoted LU solve failed: {error}"),
                })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }

    /// Compute **A**⁻¹ from the retained factor.
    pub fn inv(&self, device: &RocmDevice) -> Result<RocmBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inverse = self
            .inner
            .inv()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("complete-pivoted LU inverse failed: {error}"),
            })?;
        device.upload(leto::Storage::as_slice(inverse.storage()))
    }
}

fn kernel_source() -> String {
    r#"
struct FullPivLuMeta {
    unsigned int n;
    unsigned int k;
};

extern "C" __global__ void full_piv_lu_validate(
    const float* matrix,
    unsigned int* status,
    FullPivLuMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.n * meta.n;
    if (index < elements && !isfinite(matrix[index])) {
        atomicExch(status, 1u);
    }
}

extern "C" __global__ void full_piv_lu_step(
    float* matrix,
    unsigned int* row_perm,
    unsigned int* col_perm,
    unsigned int* status,
    unsigned int* rank,
    float* threshold,
    FullPivLuMeta meta
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || status[0] != 0u || rank[0] != meta.n) {
        return;
    }

    unsigned int n = meta.n;
    unsigned int k = meta.k;
    if (k == 0u) {
        float global_max = 0.0f;
        for (unsigned int row = 0u; row < n; row++) {
            for (unsigned int col = 0u; col < n; col++) {
                float magnitude = fabsf(matrix[row * n + col]);
                if (magnitude > global_max) {
                    global_max = magnitude;
                }
            }
        }
        threshold[0] = global_max / 1000000000000.0f;
    }

    unsigned int pivot_row = k;
    unsigned int pivot_col = k;
    float pivot_mag = fabsf(matrix[k * n + k]);
    for (unsigned int row = k; row < n; row++) {
        for (unsigned int col = k; col < n; col++) {
            float candidate = fabsf(matrix[row * n + col]);
            if (candidate > pivot_mag) {
                pivot_mag = candidate;
                pivot_row = row;
                pivot_col = col;
            }
        }
    }
    if (!isfinite(pivot_mag)) {
        atomicExch(status, 1u);
        return;
    }
    if (pivot_mag <= threshold[0]) {
        rank[0] = k;
        return;
    }

    if (pivot_row != k) {
        for (unsigned int col = 0u; col < n; col++) {
            float value = matrix[k * n + col];
            matrix[k * n + col] = matrix[pivot_row * n + col];
            matrix[pivot_row * n + col] = value;
        }
        unsigned int value = row_perm[k];
        row_perm[k] = row_perm[pivot_row];
        row_perm[pivot_row] = value;
    }
    if (pivot_col != k) {
        for (unsigned int row = 0u; row < n; row++) {
            float value = matrix[row * n + k];
            matrix[row * n + k] = matrix[row * n + pivot_col];
            matrix[row * n + pivot_col] = value;
        }
        unsigned int value = col_perm[k];
        col_perm[k] = col_perm[pivot_col];
        col_perm[pivot_col] = value;
    }

    float pivot = matrix[k * n + k];
    if (!isfinite(pivot) || pivot == 0.0f) {
        atomicExch(status, 1u);
        return;
    }
    for (unsigned int row = k + 1u; row < n; row++) {
        float factor = matrix[row * n + k] / pivot;
        if (!isfinite(factor)) {
            atomicExch(status, 1u);
            return;
        }
        matrix[row * n + k] = factor;
        for (unsigned int col = k + 1u; col < n; col++) {
            float value = matrix[row * n + col] - factor * matrix[k * n + col];
            if (!isfinite(value)) {
                atomicExch(status, 1u);
                return;
            }
            matrix[row * n + col] = value;
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
            message: format!("complete-pivoted LU element count overflows for dimension {n}"),
        })?;
    let n_u32 = u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("complete-pivoted LU dimension {n} exceeds the HIP argument range"),
    })?;
    u32::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!(
            "complete-pivoted LU element count {elements} exceeds the HIP index range"
        ),
    })?;
    Ok((n_u32, elements))
}

#[allow(clippy::too_many_arguments)]
fn launch_stage(
    device: &RocmDevice,
    stage: FullPivLuStage,
    matrix: &RocmBuffer<f32>,
    row_perm: &RocmBuffer<u32>,
    col_perm: &RocmBuffer<u32>,
    status: &RocmBuffer<u32>,
    rank: &RocmBuffer<u32>,
    threshold: &RocmBuffer<f32>,
    meta: FullPivLuMeta,
    work_items: usize,
) -> Result<()> {
    let entry = match stage {
        FullPivLuStage::Validate => "full_piv_lu_validate",
        FullPivLuStage::Step => "full_piv_lu_step",
    };
    let kernel = cached_kernel(device, PipelineKey::FullPivLu(stage), entry, kernel_source)?;
    let grid = if matches!(stage, FullPivLuStage::Step) {
        1
    } else {
        grid_size(work_items, WIDTH)?
    };
    let mut matrix_ptr: DevicePtr = matrix.raw();
    let mut row_perm_ptr: DevicePtr = row_perm.raw();
    let mut col_perm_ptr: DevicePtr = col_perm.raw();
    let mut status_ptr: DevicePtr = status.raw();
    let mut rank_ptr: DevicePtr = rank.raw();
    let mut threshold_ptr: DevicePtr = threshold.raw();
    let mut meta = meta;
    let mut args: [*mut core::ffi::c_void; 7] = [
        (&mut matrix_ptr as *mut DevicePtr).cast(),
        (&mut row_perm_ptr as *mut DevicePtr).cast(),
        (&mut col_perm_ptr as *mut DevicePtr).cast(),
        (&mut status_ptr as *mut DevicePtr).cast(),
        (&mut rank_ptr as *mut DevicePtr).cast(),
        (&mut threshold_ptr as *mut DevicePtr).cast(),
        (&mut meta as *mut FullPivLuMeta).cast(),
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
) -> Result<GpuFullPivLuDecomposition> {
    let n = validate_square(&matrix)?;
    if require_dense {
        validate_dense_operand("complete-pivoted LU", &matrix)?;
    }
    let (n_u32, elements) = checked_dimension(n)?;
    if n == 0 {
        let factors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::full_piv_lu(&leto::ArrayView::<f32, 2>::new(
            Layout::c_contiguous([0, 0]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("empty complete-pivoted LU layout failed: {error}"),
            })?,
            &[],
        ))
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("empty complete-pivoted LU decomposition failed: {error}"),
        })?;
        return Ok(GpuFullPivLuDecomposition {
            inner,
            lu: factors,
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            rank: 0,
            n: 0,
        });
    }

    let dense_layout =
        Layout::c_contiguous([n, n]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("complete-pivoted LU dense layout failed: {error}"),
        })?;
    let factors = device.alloc_zeroed::<f32>(elements)?;
    if require_dense {
        let mut stream = device.stream()?;
        stream.copy(matrix.buffer, &factors)?;
        stream.submit()?;
    } else {
        unary_elementwise_strided_into::<IdentityOp, f32, 2>(
            device,
            matrix,
            StridedOperand {
                buffer: &factors,
                layout: &dense_layout,
            },
            WIDTH,
        )?;
    }

    let identity: Vec<u32> = (0..n)
        .map(|index| {
            u32::try_from(index).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("complete-pivoted LU index {index} exceeds HIP range"),
            })
        })
        .collect::<Result<_>>()?;
    let row_perm = device.upload(&identity)?;
    let col_perm = device.upload(&identity)?;
    let status = device.alloc_zeroed::<u32>(1)?;
    let rank = device.upload(&[n_u32])?;
    let threshold = device.alloc_zeroed::<f32>(1)?;
    launch_stage(
        device,
        FullPivLuStage::Validate,
        &factors,
        &row_perm,
        &col_perm,
        &status,
        &rank,
        &threshold,
        FullPivLuMeta { n: n_u32, k: 0 },
        elements,
    )?;
    for k in 0..n {
        let k_u32 = u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("complete-pivoted LU step {k} exceeds HIP range"),
        })?;
        launch_stage(
            device,
            FullPivLuStage::Step,
            &factors,
            &row_perm,
            &col_perm,
            &status,
            &rank,
            &threshold,
            FullPivLuMeta { n: n_u32, k: k_u32 },
            1,
        )?;
    }

    let mut status_host = [0_u32; 1];
    device.download(&status, &mut status_host)?;
    if status_host[0] != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: "complete-pivoted LU factorization failed: input is non-finite or unstable"
                .to_string(),
        });
    }
    let mut rank_host = [0_u32; 1];
    device.download(&rank, &mut rank_host)?;
    let rank = usize::try_from(rank_host[0]).map_err(|_| HephaestusError::DispatchFailed {
        message: "complete-pivoted LU rank exceeds host index range".to_string(),
    })?;
    let mut row_perm_host = vec![0_u32; n];
    let mut col_perm_host = vec![0_u32; n];
    device.download(&row_perm, &mut row_perm_host)?;
    device.download(&col_perm, &mut col_perm_host)?;
    let row_perm = row_perm_host
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
                message: "complete-pivoted LU row permutation exceeds host range".to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let col_perm = col_perm_host
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
                message: "complete-pivoted LU column permutation exceeds host range".to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let inner = leto_ops::full_piv_lu(&leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data))
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("complete-pivoted LU host contract failed: {error}"),
        })?;

    Ok(GpuFullPivLuDecomposition {
        inner,
        lu: factors,
        row_perm,
        col_perm,
        rank,
        n,
    })
}

/// Compute complete-pivoted LU through HIP for a strided rank-2 matrix.
pub fn full_piv_lu(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuFullPivLuDecomposition> {
    factor_on_device(device, matrix, false)
}

/// Compute complete-pivoted LU through HIP for a dense C-contiguous matrix.
pub fn full_piv_lu_blocked(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuFullPivLuDecomposition> {
    factor_on_device(device, matrix, true)
}

#[cfg(test)]
mod tests {
    use super::kernel_source;

    #[test]
    fn source_contains_complete_pivot_stages() {
        let source = kernel_source();
        assert!(source.contains("full_piv_lu_validate"));
        assert!(source.contains("full_piv_lu_step"));
        assert!(source.contains("pivot_row"));
        assert!(source.contains("pivot_col"));
        assert!(source.contains("threshold[0]"));
    }
}
