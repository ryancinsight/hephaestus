//! HIP Householder QR factorization.
//!
//! Each ordered HIP step computes one reflector in the packed convention
//! shared with `leto-ops`: the upper triangle stores **R**, the strict lower
//! triangle stores reflector tails, and heads/betas are separate device
//! metadata buffers until the host-side solve contract is constructed.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp,
    KernelDevice, Result,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::validate_dense_operand;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, QrStage, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::{DevicePtr, RocmBuffer};

const WIDTH: BlockWidth = BlockWidth::DEFAULT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QrMeta {
    rows: u32,
    cols: u32,
    k: u32,
}

const _: () = assert!(core::mem::size_of::<QrMeta>() == 12);

/// QR decomposition result with packed factors resident on the device.
pub struct GpuQrDecomposition {
    /// Host-side packed factor retained for least-squares solves.
    inner: leto_ops::QrDecomposition<f32>,
    /// Packed factor whose upper triangle is **R**.
    r: RocmBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuQrDecomposition {
    /// Return `(rows, cols)` for the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the packed **R** factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn r_buffer(&self) -> &RocmBuffer<f32> {
        &self.r
    }

    /// Borrow the retained host-side decomposition.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::QrDecomposition<f32> {
        &self.inner
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
                message: format!("QR solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution = self.inner.solve_least_squares(&rhs_view).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("QR least-squares solve failed: {error}"),
            }
        })?;
        device.upload(leto::Storage::as_slice(solution.storage()))
    }
}

fn kernel_source() -> String {
    r#"
struct QrMeta {
    unsigned int rows;
    unsigned int cols;
    unsigned int k;
};

extern "C" __global__ void qr_validate(
    const float* matrix,
    unsigned int* status,
    QrMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.rows * meta.cols;
    if (index < elements && !isfinite(matrix[index])) {
        atomicExch(status, 1u);
    }
}

extern "C" __global__ void qr_step(
    float* matrix,
    float* heads,
    float* betas,
    unsigned int* status,
    QrMeta meta
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || status[0] != 0u) {
        return;
    }

    unsigned int k = meta.k;
    unsigned int m = meta.rows;
    unsigned int n = meta.cols;
    float norm_sq = 0.0f;
    for (unsigned int row = k; row < m; row++) {
        float value = matrix[row * n + k];
        norm_sq += value * value;
    }
    float norm = sqrtf(norm_sq);
    if (!isfinite(norm) || norm == 0.0f) {
        atomicExch(status, 1u);
        return;
    }

    float pivot = matrix[k * n + k];
    float alpha = pivot > 0.0f ? -norm : norm;
    float head = pivot - alpha;
    float vector_norm_sq = head * head;
    for (unsigned int row = k + 1u; row < m; row++) {
        float value = matrix[row * n + k];
        vector_norm_sq += value * value;
    }
    float beta = 2.0f / vector_norm_sq;
    if (!isfinite(beta)) {
        atomicExch(status, 1u);
        return;
    }

    for (unsigned int col = k + 1u; col < n; col++) {
        float dot = head * matrix[k * n + col];
        for (unsigned int row = k + 1u; row < m; row++) {
            dot += matrix[row * n + k] * matrix[row * n + col];
        }
        float scaled = beta * dot;
        float top = matrix[k * n + col] - scaled * head;
        if (!isfinite(top)) {
            atomicExch(status, 1u);
            return;
        }
        matrix[k * n + col] = top;
        for (unsigned int row = k + 1u; row < m; row++) {
            float value = matrix[row * n + col] - scaled * matrix[row * n + k];
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

fn checked_dimensions(rows: usize, cols: usize) -> Result<(u32, u32, usize)> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("QR matrix element count overflows for shape [{rows}, {cols}]"),
        })?;
    let rows_u32 = u32::try_from(rows).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("QR row count {rows} exceeds the HIP argument range"),
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("QR column count {cols} exceeds the HIP argument range"),
    })?;
    u32::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("QR matrix element count {elements} exceeds the HIP index range"),
    })?;
    Ok((rows_u32, cols_u32, elements))
}

fn launch_stage(
    device: &RocmDevice,
    stage: QrStage,
    matrix: &RocmBuffer<f32>,
    heads: &RocmBuffer<f32>,
    betas: &RocmBuffer<f32>,
    status: &RocmBuffer<u32>,
    meta: QrMeta,
    work_items: usize,
) -> Result<()> {
    let entry = match stage {
        QrStage::Validate => "qr_validate",
        QrStage::Step => "qr_step",
    };
    let kernel = cached_kernel(device, PipelineKey::Qr(stage), entry, kernel_source)?;
    let grid = if matches!(stage, QrStage::Step) {
        1
    } else {
        grid_size(work_items, WIDTH)?
    };
    let mut matrix_ptr: DevicePtr = matrix.raw();
    let mut heads_ptr: DevicePtr = heads.raw();
    let mut betas_ptr: DevicePtr = betas.raw();
    let mut status_ptr: DevicePtr = status.raw();
    let mut meta = meta;
    let mut args: [*mut core::ffi::c_void; 5] = [
        (&mut matrix_ptr as *mut DevicePtr).cast(),
        (&mut heads_ptr as *mut DevicePtr).cast(),
        (&mut betas_ptr as *mut DevicePtr).cast(),
        (&mut status_ptr as *mut DevicePtr).cast(),
        (&mut meta as *mut QrMeta).cast(),
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
) -> Result<GpuQrDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    if require_dense {
        validate_dense_operand("QR", &matrix)?;
    }
    let (rows_u32, cols_u32, elements) = checked_dimensions(rows, cols)?;
    if rows == 0 || cols == 0 {
        let r = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::QrDecomposition::from_raw_parts(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            rows,
            cols,
        );
        return Ok(GpuQrDecomposition {
            inner,
            r,
            rows,
            cols,
        });
    }

    let dense_layout =
        Layout::c_contiguous([rows, cols]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("QR dense layout failed: {error}"),
        })?;
    let r = device.alloc_zeroed::<f32>(elements)?;
    if require_dense {
        let mut stream = device.stream()?;
        stream.copy(matrix.buffer, &r)?;
        stream.submit()?;
    } else {
        unary_elementwise_strided_into::<IdentityOp, f32, 2>(
            device,
            matrix,
            StridedOperand {
                buffer: &r,
                layout: &dense_layout,
            },
            WIDTH,
        )?;
    }

    let heads = device.alloc_zeroed::<f32>(cols)?;
    let betas = device.alloc_zeroed::<f32>(cols)?;
    let status = device.alloc_zeroed::<u32>(1)?;
    launch_stage(
        device,
        QrStage::Validate,
        &r,
        &heads,
        &betas,
        &status,
        QrMeta {
            rows: rows_u32,
            cols: cols_u32,
            k: 0,
        },
        elements,
    )?;
    for k in 0..cols {
        let k_u32 = u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("QR step {k} exceeds the HIP argument range"),
        })?;
        launch_stage(
            device,
            QrStage::Step,
            &r,
            &heads,
            &betas,
            &status,
            QrMeta {
                rows: rows_u32,
                cols: cols_u32,
                k: k_u32,
            },
            1,
        )?;
    }

    let mut status_host = [0_u32; 1];
    device.download(&status, &mut status_host)?;
    if status_host[0] != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: "QR factorization failed: input is non-finite or rank-deficient".to_string(),
        });
    }

    let mut host_r = vec![0.0_f32; elements];
    let mut host_heads = vec![0.0_f32; cols];
    let mut host_betas = vec![0.0_f32; cols];
    device.download(&r, &mut host_r)?;
    device.download(&heads, &mut host_heads)?;
    device.download(&betas, &mut host_betas)?;
    let inner =
        leto_ops::QrDecomposition::from_raw_parts(host_r, host_heads, host_betas, rows, cols);
    Ok(GpuQrDecomposition {
        inner,
        r,
        rows,
        cols,
    })
}

/// Compute Householder QR through HIP.
pub fn qr_decompose(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    factor_on_device(device, matrix, false)
}

/// Compute Householder QR through HIP for a dense C-contiguous input.
pub fn qr_decompose_blocked(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    factor_on_device(device, matrix, true)
}

#[cfg(test)]
mod tests {
    use super::kernel_source;

    #[test]
    fn source_contains_householder_factorization_stages() {
        let source = kernel_source();
        assert!(source.contains("qr_validate"));
        assert!(source.contains("qr_step"));
        assert!(source.contains("sqrtf(norm_sq)"));
        assert!(source.contains("heads[k]"));
        assert!(source.contains("betas[k]"));
    }
}
