//! Device accumulation of the orthogonal QR factor.

use core::ffi::c_void;
use std::sync::Arc;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use super::GpuQrDecomposition;
use crate::application::linalg::{device_identity, to_u32};
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::{CudaBuffer, CudaDevice};

const Q_ACCUMULATE_ENTRY: &str = "qr_accumulate_q_kernel";
const Q_ACCUMULATE_THREADS: u32 = 256;

fn q_accumulate_source() -> String {
    r#"extern "C" __global__ void qr_accumulate_q_kernel(
    const float* packed,
    float* q,
    const float* heads,
    const float* betas,
    unsigned int m,
    unsigned int n,
    unsigned int reflector_count
) {
    const unsigned int column = blockIdx.x;
    if (column >= m) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    __shared__ float partials[256];

    // Q = H_1(H_2(...(H_k I))), so the final stored reflector is applied
    // to the identity first.
    for (unsigned int index = 0; index < reflector_count; ++index) {
        const unsigned int k = reflector_count - 1U - index;
        const unsigned int span = m - k;

        float partial = 0.0f;
        for (unsigned int local_row = tid;
             local_row < span;
             local_row += blockDim.x) {
            const unsigned int row = k + local_row;
            const float v = local_row == 0U
                ? heads[k]
                : packed[(size_t)row * (size_t)n + (size_t)k];
            partial += v * q[(size_t)row * (size_t)m + (size_t)column];
        }
        partials[tid] = partial;
        __syncthreads();

        for (unsigned int width = 128U; width > 0U; width >>= 1U) {
            if (tid < width) {
                partials[tid] += partials[tid + width];
            }
            __syncthreads();
        }

        const float scaled = betas[k] * partials[0];
        __syncthreads();
        for (unsigned int local_row = tid;
             local_row < span;
             local_row += blockDim.x) {
            const unsigned int row = k + local_row;
            const float v = local_row == 0U
                ? heads[k]
                : packed[(size_t)row * (size_t)n + (size_t)k];
            q[(size_t)row * (size_t)m + (size_t)column] -= scaled * v;
        }
        __syncthreads();
    }
}
"#
    .to_string()
}

impl GpuQrDecomposition {
    /// Accumulate the orthogonal factor **Q** (*m* × *m*, row-major) on the
    /// CUDA device.
    ///
    /// The factorization retains only its compact Householder form because
    /// forming **Q** costs O(*m*² *n*) and least-squares callers do not need
    /// it. This method starts from a device identity and applies the retained
    /// reflectors in reverse order, one 256-thread block per Q column.
    ///
    /// The method uploads `4mn + 8·min(m, n)` bytes for the packed factor,
    /// heads, and scales. It replaces host Q accumulation and a `4m²`-byte Q
    /// upload at the Python boundary.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the decomposition
    /// belongs to another CUDA device or `m * m` overflows. Dimensions outside
    /// CUDA's `u32` kernel domain and provider allocation, compilation, and
    /// launch failures retain their typed errors.
    pub fn accumulate_q(&self, device: &CudaDevice) -> Result<CudaBuffer<f32>> {
        if !self
            .r
            .context
            .as_ref()
            .is_some_and(|context| Arc::ptr_eq(context, device.cuda_context()))
        {
            return Err(HephaestusError::InvalidConfiguration {
                message: "QR decomposition must belong to the dispatch device".to_string(),
            });
        }

        let q_len = self.rows.checked_mul(self.rows).ok_or_else(|| {
            HephaestusError::InvalidConfiguration {
                message: format!("QR Q dimension {} overflows an element count", self.rows),
            }
        })?;
        let mut m = to_u32(self.rows, "QR Q row count")?;
        let mut n = to_u32(self.cols, "QR Q column count")?;
        let reflector_count = self.rows.min(self.cols);
        let mut reflector_count_arg = to_u32(reflector_count, "QR Q reflector count")?;

        let q = device_identity::<f32>(device, self.rows, q_len)?;
        if reflector_count == 0 {
            return Ok(q);
        }

        // Upload existing contiguous slices directly. Keeping heads and betas
        // separate avoids constructing an interleaved transient host vector.
        let packed = device.upload(self.inner.packed())?;
        let heads = device.upload(self.inner.heads())?;
        let betas = device.upload(self.inner.betas())?;
        let kernel = cached_kernel(
            device,
            PipelineKey::QrAccumulateQ,
            Q_ACCUMULATE_ENTRY,
            q_accumulate_source,
        )?;

        let mut packed_ptr = packed.raw();
        let mut q_ptr = q.raw();
        let mut heads_ptr = heads.raw();
        let mut betas_ptr = betas.raw();
        let mut args: [*mut c_void; 7] = [
            (&raw mut packed_ptr).cast(),
            (&raw mut q_ptr).cast(),
            (&raw mut heads_ptr).cast(),
            (&raw mut betas_ptr).cast(),
            (&raw mut m).cast(),
            (&raw mut n).cast(),
            (&raw mut reflector_count_arg).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::planar(m, 1, Q_ACCUMULATE_THREADS, 1),
            &mut args,
        )?;

        Ok(q)
    }
}
