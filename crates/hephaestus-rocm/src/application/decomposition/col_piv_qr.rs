//! HIP column-pivoted Householder QR factorization.
//!
//! Each ordered HIP step selects the largest remaining column norm, swaps that
//! column into the active position, applies one Householder reflector to the
//! working matrix, and accumulates **Q**. The packed **R**, materialized **Q**,
//! column permutation, and rank are device-backed. The retained host factor
//! exists only for the established least-squares contract shared by CUDA and
//! wgpu.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp,
    KernelDevice, Result,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_dense_operand;
use crate::application::pipeline::{
    ColPivQrStage, LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::{DevicePtr, RocmBuffer};

const WIDTH: BlockWidth = BlockWidth::DEFAULT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColPivQrMeta {
    rows: u32,
    cols: u32,
    pivots: u32,
    k: u32,
}

const _: () = assert!(core::mem::size_of::<ColPivQrMeta>() == 16);

fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR layout error: {error}"),
    }
}

/// Column-pivoted QR result with device-backed **Q**, **R**, and permutation.
pub struct GpuColPivQrDecomposition {
    /// Host-side reference retained for scalar least-squares solves.
    inner: leto_ops::ColPivQrDecomposition<f32>,
    /// Materialized orthogonal factor **Q** on the device.
    q: RocmBuffer<f32>,
    /// Upper-triangular factor **R** on the device.
    r: RocmBuffer<f32>,
    /// Column permutation, where each entry is the original column at that position.
    permutation: Vec<usize>,
    rank: usize,
    rows: usize,
    cols: usize,
}

impl GpuColPivQrDecomposition {
    /// Numerical rank reported by the pivoted diagonal threshold.
    #[must_use]
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Borrow the orthogonal factor **Q** buffer on the device.
    #[must_use]
    #[inline]
    pub fn q(&self) -> &RocmBuffer<f32> {
        &self.q
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r(&self) -> &RocmBuffer<f32> {
        &self.r
    }

    /// Return the column permutation.
    #[must_use]
    #[inline]
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ using the retained factor.
    pub fn solve_least_squares(
        &self,
        device: &RocmDevice,
        rhs: &RocmBuffer<f32>,
    ) -> Result<RocmBuffer<f32>> {
        if rhs.len() != self.rows {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.rows,
                device_len: rhs.len(),
            });
        }
        if self.rows == 0 || self.cols == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0_f32; self.rows];
        device.download(rhs, &mut rhs_host)?;
        let rhs_view = leto::ArrayView::<f32, 1>::new(
            Layout::c_contiguous([self.rows]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution = self.inner.solve_least_squares(&rhs_view).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR least-squares solve failed: {error}"),
            }
        })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }
}

fn kernel_source() -> String {
    r#"
struct ColPivQrMeta {
    unsigned int rows;
    unsigned int cols;
    unsigned int pivots;
    unsigned int k;
};

extern "C" __global__ void col_piv_qr_validate(
    const float* matrix,
    unsigned int* status,
    ColPivQrMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.rows * meta.cols;
    if (index < elements && !isfinite(matrix[index])) {
        atomicExch(status, 1u);
    }
}

extern "C" __global__ void col_piv_qr_step(
    float* matrix,
    float* q,
    unsigned int* permutation,
    float* heads,
    float* betas,
    unsigned int* status,
    unsigned int* rank,
    float* threshold,
    ColPivQrMeta meta
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || status[0] != 0u || rank[0] != meta.pivots) {
        return;
    }

    unsigned int m = meta.rows;
    unsigned int n = meta.cols;
    unsigned int k = meta.k;
    if (k == 0u) {
        float reference_norm = 0.0f;
        for (unsigned int col = 0u; col < n; col++) {
            float norm_sq = 0.0f;
            for (unsigned int row = 0u; row < m; row++) {
                float value = matrix[row * n + col];
                norm_sq += value * value;
            }
            float norm = sqrtf(norm_sq);
            if (norm > reference_norm) {
                reference_norm = norm;
            }
        }
        threshold[0] = reference_norm / 1000000000000.0f;
    }

    unsigned int pivot_col = k;
    float pivot_norm_sq = 0.0f;
    for (unsigned int row = k; row < m; row++) {
        float value = matrix[row * n + k];
        pivot_norm_sq += value * value;
    }
    for (unsigned int col = k + 1u; col < n; col++) {
        float candidate_norm_sq = 0.0f;
        for (unsigned int row = k; row < m; row++) {
            float value = matrix[row * n + col];
            candidate_norm_sq += value * value;
        }
        if (candidate_norm_sq > pivot_norm_sq) {
            pivot_norm_sq = candidate_norm_sq;
            pivot_col = col;
        }
    }
    if (!isfinite(pivot_norm_sq) || sqrtf(pivot_norm_sq) <= threshold[0]) {
        rank[0] = k;
        return;
    }

    if (pivot_col != k) {
        for (unsigned int row = 0u; row < m; row++) {
            float value = matrix[row * n + k];
            matrix[row * n + k] = matrix[row * n + pivot_col];
            matrix[row * n + pivot_col] = value;
        }
        unsigned int value = permutation[k];
        permutation[k] = permutation[pivot_col];
        permutation[pivot_col] = value;
    }

    float norm = sqrtf(pivot_norm_sq);
    float pivot = matrix[k * n + k];
    float sign = pivot < 0.0f ? -1.0f : 1.0f;
    float alpha = -sign * norm;
    float head = pivot - alpha;
    float vector_norm_sq = head * head;
    for (unsigned int row = k + 1u; row < m; row++) {
        float value = matrix[row * n + k];
        vector_norm_sq += value * value;
    }
    float beta = 2.0f / vector_norm_sq;
    if (!isfinite(alpha) || !isfinite(head) || !isfinite(beta)) {
        atomicExch(status, 1u);
        return;
    }

    // Q ← Q H, using the reflector vector before the R column is cleared.
    for (unsigned int row = 0u; row < m; row++) {
        float dot = q[row * m + k] * head;
        for (unsigned int tail = k + 1u; tail < m; tail++) {
            dot += q[row * m + tail] * matrix[tail * n + k];
        }
        float scale = beta * dot;
        q[row * m + k] -= scale * head;
        for (unsigned int tail = k + 1u; tail < m; tail++) {
            q[row * m + tail] -= scale * matrix[tail * n + k];
        }
    }

    // R ← H R, including column k so its reflector tail is cleared.
    for (unsigned int col = k; col < n; col++) {
        float dot = head * matrix[k * n + col];
        for (unsigned int row = k + 1u; row < m; row++) {
            dot += matrix[row * n + k] * matrix[row * n + col];
        }
        float scale = beta * dot;
        float top = matrix[k * n + col] - scale * head;
        if (!isfinite(top)) {
            atomicExch(status, 1u);
            return;
        }
        matrix[k * n + col] = top;
        for (unsigned int row = k + 1u; row < m; row++) {
            float value = matrix[row * n + col] - scale * matrix[row * n + k];
            if (!isfinite(value)) {
                atomicExch(status, 1u);
                return;
            }
            matrix[row * n + col] = value;
        }
    }
    matrix[k * n + k] = alpha;
    heads[k] = head;
    betas[k] = beta;
}
"#
    .to_string()
}

fn checked_dimensions(rows: usize, cols: usize) -> Result<(u32, u32, u32, usize, usize)> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR element count overflows for [{rows}, {cols}]"),
        })?;
    let q_elements = rows
        .checked_mul(rows)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR Q size overflows for row count {rows}"),
        })?;
    let rows_u32 = u32::try_from(rows).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR row count {rows} exceeds HIP range"),
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR column count {cols} exceeds HIP range"),
    })?;
    let pivots_u32 =
        u32::try_from(rows.min(cols)).map_err(|_| HephaestusError::DispatchFailed {
            message: "column-pivoted QR pivot count exceeds HIP range".to_string(),
        })?;
    u32::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR element count {elements} exceeds HIP range"),
    })?;
    u32::try_from(q_elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR Q size {q_elements} exceeds HIP range"),
    })?;
    Ok((rows_u32, cols_u32, pivots_u32, elements, q_elements))
}

#[allow(clippy::too_many_arguments)]
fn launch_stage(
    device: &RocmDevice,
    stage: ColPivQrStage,
    matrix: &RocmBuffer<f32>,
    q: &RocmBuffer<f32>,
    permutation: &RocmBuffer<u32>,
    heads: &RocmBuffer<f32>,
    betas: &RocmBuffer<f32>,
    status: &RocmBuffer<u32>,
    rank: &RocmBuffer<u32>,
    threshold: &RocmBuffer<f32>,
    meta: ColPivQrMeta,
    work_items: usize,
) -> Result<()> {
    let entry = match stage {
        ColPivQrStage::Validate => "col_piv_qr_validate",
        ColPivQrStage::Step => "col_piv_qr_step",
    };
    let kernel = cached_kernel(device, PipelineKey::ColPivQr(stage), entry, kernel_source)?;
    let grid = if matches!(stage, ColPivQrStage::Step) {
        1
    } else {
        grid_size(work_items, WIDTH)?
    };
    let mut matrix_ptr: DevicePtr = matrix.raw();
    let mut q_ptr: DevicePtr = q.raw();
    let mut permutation_ptr: DevicePtr = permutation.raw();
    let mut heads_ptr: DevicePtr = heads.raw();
    let mut betas_ptr: DevicePtr = betas.raw();
    let mut status_ptr: DevicePtr = status.raw();
    let mut rank_ptr: DevicePtr = rank.raw();
    let mut threshold_ptr: DevicePtr = threshold.raw();
    let mut meta = meta;
    let mut args: [*mut core::ffi::c_void; 9] = [
        (&mut matrix_ptr as *mut DevicePtr).cast(),
        (&mut q_ptr as *mut DevicePtr).cast(),
        (&mut permutation_ptr as *mut DevicePtr).cast(),
        (&mut heads_ptr as *mut DevicePtr).cast(),
        (&mut betas_ptr as *mut DevicePtr).cast(),
        (&mut status_ptr as *mut DevicePtr).cast(),
        (&mut rank_ptr as *mut DevicePtr).cast(),
        (&mut threshold_ptr as *mut DevicePtr).cast(),
        (&mut meta as *mut ColPivQrMeta).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, WIDTH),
        &mut args,
    )
}

/// Compute column-pivoted QR through HIP for a strided rank-2 matrix.
pub fn col_piv_qr(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuColPivQrDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    let (rows_u32, cols_u32, pivots_u32, elements, q_elements) = checked_dimensions(rows, cols)?;

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    if rows == 0 || cols == 0 {
        let q = device.alloc_zeroed::<f32>(0)?;
        let r = device.alloc_zeroed::<f32>(0)?;
        device.download(matrix.buffer, &mut host_data)?;
        let inner =
            leto_ops::col_piv_qr(&leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data))
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("empty column-pivoted QR decomposition failed: {error}"),
                })?;
        return Ok(GpuColPivQrDecomposition {
            inner,
            q,
            r,
            permutation: Vec::new(),
            rank: 0,
            rows,
            cols,
        });
    }

    let dense_layout =
        Layout::c_contiguous([rows, cols]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR dense layout failed: {error}"),
        })?;
    let r = device.alloc_zeroed::<f32>(elements)?;
    unary_elementwise_strided_into::<IdentityOp, f32, 2>(
        device,
        matrix,
        StridedOperand {
            buffer: &r,
            layout: &dense_layout,
        },
        WIDTH,
    )?;
    let mut q_host = vec![0.0_f32; q_elements];
    for index in 0..rows {
        q_host[index * rows + index] = 1.0;
    }
    let q = device.upload(&q_host)?;
    let identity: Vec<u32> = (0..cols)
        .map(|index| {
            u32::try_from(index).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("column-pivoted QR index {index} exceeds HIP range"),
            })
        })
        .collect::<Result<_>>()?;
    let permutation = device.upload(&identity)?;
    let heads = device.alloc_zeroed::<f32>(rows.min(cols))?;
    let betas = device.alloc_zeroed::<f32>(rows.min(cols))?;
    let status = device.alloc_zeroed::<u32>(1)?;
    let rank = device.upload(&[pivots_u32])?;
    let threshold = device.alloc_zeroed::<f32>(1)?;
    launch_stage(
        device,
        ColPivQrStage::Validate,
        &r,
        &q,
        &permutation,
        &heads,
        &betas,
        &status,
        &rank,
        &threshold,
        ColPivQrMeta {
            rows: rows_u32,
            cols: cols_u32,
            pivots: pivots_u32,
            k: 0,
        },
        elements,
    )?;
    for k in 0..rows.min(cols) {
        let k_u32 = u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("column-pivoted QR step {k} exceeds HIP range"),
        })?;
        launch_stage(
            device,
            ColPivQrStage::Step,
            &r,
            &q,
            &permutation,
            &heads,
            &betas,
            &status,
            &rank,
            &threshold,
            ColPivQrMeta {
                rows: rows_u32,
                cols: cols_u32,
                pivots: pivots_u32,
                k: k_u32,
            },
            1,
        )?;
    }

    let mut status_host = [0_u32; 1];
    device.download(&status, &mut status_host)?;
    if status_host[0] != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: "column-pivoted QR factorization failed: input is non-finite or unstable"
                .to_string(),
        });
    }
    let mut rank_host = [0_u32; 1];
    device.download(&rank, &mut rank_host)?;
    let rank = usize::try_from(rank_host[0]).map_err(|_| HephaestusError::DispatchFailed {
        message: "column-pivoted QR rank exceeds host index range".to_string(),
    })?;
    let mut permutation_host = vec![0_u32; cols];
    device.download(&permutation, &mut permutation_host)?;
    let permutation = permutation_host
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
                message: "column-pivoted QR permutation exceeds host range".to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    device.download(matrix.buffer, &mut host_data)?;
    let inner = leto_ops::col_piv_qr(&leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data))
        .map_err(|error| HephaestusError::DispatchFailed {
        message: format!("column-pivoted QR host contract failed: {error}"),
    })?;
    Ok(GpuColPivQrDecomposition {
        inner,
        q,
        r,
        permutation,
        rank,
        rows,
        cols,
    })
}

/// Validate a dense C-contiguous matrix before computing column-pivoted QR.
pub fn col_piv_qr_blocked(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuColPivQrDecomposition> {
    validate_dense_operand("column-pivoted QR", &matrix)?;
    col_piv_qr(device, matrix)
}

#[cfg(test)]
mod tests {
    use super::kernel_source;

    #[test]
    fn source_contains_column_pivot_stages() {
        let source = kernel_source();
        assert!(source.contains("col_piv_qr_validate"));
        assert!(source.contains("col_piv_qr_step"));
        assert!(source.contains("pivot_norm_sq"));
        assert!(source.contains("permutation[k]"));
        assert!(source.contains("Q H"));
    }
}
