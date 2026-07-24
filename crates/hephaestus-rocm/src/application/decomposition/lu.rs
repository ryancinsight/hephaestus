//! HIP LU factorization with partial pivoting.
//!
//! The factorization overwrites a dense device matrix with the packed
//! unit-lower/upper representation used by `leto-ops`. Each pivot step is
//! launched in order on HIP's default stream, so a later step observes the
//! previous row swap and elimination updates.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CommandStream, ComputeDevice, DeviceBuffer, HephaestusError, IdentityOp,
    KernelDevice, Result,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::decomposition::validate::{validate_dense_operand, validate_square};
use crate::application::pipeline::{
    LaunchConfig, LuStage, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::unary_elementwise_strided_into;
use crate::infrastructure::{DevicePtr, RocmBuffer};

const WIDTH: BlockWidth = BlockWidth::DEFAULT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LuMeta {
    n: u32,
    k: u32,
}

const _: () = assert!(core::mem::size_of::<LuMeta>() == 8);

/// LU decomposition result with packed factors resident on the device.
pub struct GpuLuDecomposition {
    /// Host-side factor retained for the established solve/determinant API.
    inner: leto_ops::LuDecomposition<f32>,
    /// Packed factors: strict lower triangle stores **L** and upper triangle
    /// including the diagonal stores **U**.
    factors: RocmBuffer<f32>,
    n: usize,
}

impl GpuLuDecomposition {
    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the packed factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn factors(&self) -> &RocmBuffer<f32> {
        &self.factors
    }

    /// Return the row pivots selected by partial pivoting.
    #[must_use]
    #[inline]
    pub fn pivots(&self) -> &[usize] {
        self.inner.pivots()
    }

    /// Compute the determinant from the retained packed factors.
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
                message: format!("LU solve RHS layout failed: {error}"),
            })?,
            &rhs_host,
        );
        let solution =
            self.inner
                .solve(&rhs_view)
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("LU solve failed: {error}"),
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
                message: format!("LU inverse failed: {error}"),
            })?;
        device.upload(leto::Storage::as_slice(inverse.storage()))
    }
}

fn kernel_source() -> String {
    r#"
struct LuMeta {
    unsigned int n;
    unsigned int k;
};

extern "C" __global__ void lu_validate(
    const float* matrix,
    unsigned int* status,
    LuMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.n * meta.n;
    if (index < elements && !isfinite(matrix[index])) {
        atomicExch(status, 1u);
    }
}

extern "C" __global__ void lu_step(
    float* matrix,
    unsigned int* pivots,
    unsigned int* status,
    LuMeta meta
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || status[0] != 0u) {
        return;
    }

    unsigned int k = meta.k;
    unsigned int n = meta.n;
    unsigned int pivot_row = k;
    float pivot_mag = fabsf(matrix[k * n + k]);
    for (unsigned int row = k + 1u; row < n; row++) {
        float candidate = fabsf(matrix[row * n + k]);
        if (candidate > pivot_mag) {
            pivot_mag = candidate;
            pivot_row = row;
        }
    }
    pivots[k] = pivot_row;
    if (!isfinite(pivot_mag) || pivot_mag == 0.0f) {
        atomicExch(status, 1u);
        return;
    }

    if (pivot_row != k) {
        for (unsigned int col = 0u; col < n; col++) {
            float value = matrix[k * n + col];
            matrix[k * n + col] = matrix[pivot_row * n + col];
            matrix[pivot_row * n + col] = value;
        }
    }

    float pivot = matrix[k * n + k];
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
            message: format!("LU matrix element count overflows for dimension {n}"),
        })?;
    let n_u32 = u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("LU dimension {n} exceeds the HIP argument range"),
    })?;
    u32::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("LU matrix element count {elements} exceeds the HIP index range"),
    })?;
    Ok((n_u32, elements))
}

fn launch_stage(
    device: &RocmDevice,
    stage: LuStage,
    matrix: &RocmBuffer<f32>,
    pivots: &RocmBuffer<u32>,
    status: &RocmBuffer<u32>,
    meta: LuMeta,
    work_items: usize,
) -> Result<()> {
    let entry = match stage {
        LuStage::Validate => "lu_validate",
        LuStage::Step => "lu_step",
    };
    let kernel = cached_kernel(device, PipelineKey::Lu(stage), entry, kernel_source)?;
    let grid = if matches!(stage, LuStage::Step) {
        1
    } else {
        grid_size(work_items, WIDTH)?
    };
    let mut matrix_ptr: DevicePtr = matrix.raw();
    let mut pivots_ptr: DevicePtr = pivots.raw();
    let mut status_ptr: DevicePtr = status.raw();
    let mut meta = meta;
    let mut args: [*mut core::ffi::c_void; 4] = [
        (&mut matrix_ptr as *mut DevicePtr).cast(),
        (&mut pivots_ptr as *mut DevicePtr).cast(),
        (&mut status_ptr as *mut DevicePtr).cast(),
        (&mut meta as *mut LuMeta).cast(),
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
) -> Result<GpuLuDecomposition> {
    let n = validate_square(&matrix)?;
    if require_dense {
        validate_dense_operand("LU", &matrix)?;
    }
    let (n_u32, elements) = checked_dimension(n)?;
    if n == 0 {
        let factors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::lu_decompose(&leto::ArrayView::<f32, 2>::new(
            Layout::c_contiguous([0, 0]).map_err(|error| HephaestusError::DispatchFailed {
                message: format!("empty LU layout failed: {error}"),
            })?,
            &[],
        ))
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("empty LU decomposition failed: {error}"),
        })?;
        return Ok(GpuLuDecomposition {
            inner,
            factors,
            n: 0,
        });
    }

    let dense_layout =
        Layout::c_contiguous([n, n]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("LU dense layout failed: {error}"),
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

    let pivots = device.alloc_zeroed::<u32>(n)?;
    let status = device.alloc_zeroed::<u32>(1)?;
    launch_stage(
        device,
        LuStage::Validate,
        &factors,
        &pivots,
        &status,
        LuMeta { n: n_u32, k: 0 },
        elements,
    )?;
    for k in 0..n {
        let k_u32 = u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("LU step {k} exceeds the HIP argument range"),
        })?;
        launch_stage(
            device,
            LuStage::Step,
            &factors,
            &pivots,
            &status,
            LuMeta { n: n_u32, k: k_u32 },
            1,
        )?;
    }

    let mut status_host = [0_u32; 1];
    device.download(&status, &mut status_host)?;
    if status_host[0] != 0 {
        return Err(HephaestusError::DispatchFailed {
            message: "LU factorization failed: input is non-finite or singular".to_string(),
        });
    }

    let mut pivots_host = vec![0_u32; n];
    device.download(&pivots, &mut pivots_host)?;
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut sign = 1_i8;
    for (k, pivot) in pivots_host.into_iter().enumerate() {
        let pivot = usize::try_from(pivot).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("LU pivot at column {k} exceeds host index range"),
        })?;
        if pivot != k {
            permutation.swap(k, pivot);
            sign = -sign;
        }
    }

    let mut host_factors = vec![0.0_f32; elements];
    device.download(&factors, &mut host_factors)?;
    let inner = leto_ops::LuDecomposition::from_raw_parts(
        leto::Array2::from_shape_vec([n, n], host_factors).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("LU factor shape failed: {error}"),
            }
        })?,
        permutation,
        sign,
    );
    Ok(GpuLuDecomposition { inner, factors, n })
}

/// Compute LU with partial pivoting through HIP.
pub fn lu_decompose(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    factor_on_device(device, matrix, false)
}

/// Compute LU through HIP for a dense C-contiguous zero-offset input.
pub fn lu_decompose_blocked(
    device: &RocmDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    factor_on_device(device, matrix, true)
}

#[cfg(test)]
mod tests {
    use super::kernel_source;

    #[test]
    fn source_contains_pivoted_factorization_stages() {
        let source = kernel_source();
        assert!(source.contains("lu_validate"));
        assert!(source.contains("lu_step"));
        assert!(source.contains("pivot_row"));
        assert!(source.contains("factor * matrix"));
    }
}
