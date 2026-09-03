//! CUDA trailing-matrix Householder update for blocked QR.

use super::*;
use crate::application::linalg::to_u32;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(super) struct HhReflectorMeta {
    pub(super) vector_offset: u32,
    pub(super) beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct HhMeta {
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    matrix_cols: u32,
    k: u32,
    _pad: [u32; 3],
}

fn householder_source() -> String {
    r#"
struct ReflectorMeta {
    unsigned int vector_offset;
    float beta;
};

struct HhMeta {
    unsigned int panel_rows;
    unsigned int reflector_count;
    unsigned int trail_cols;
    unsigned int matrix_cols;
    unsigned int k;
    unsigned int _pad0;
    unsigned int _pad1;
    unsigned int _pad2;
};

extern "C" __global__ void householder_kernel(
    const float* v_buf,
    float* a_buf,
    const ReflectorMeta* reflector_buf,
    HhMeta meta
) {
    __shared__ float sdata[256];

    unsigned int wid_x = blockIdx.x;
    unsigned int col = meta.k + meta.reflector_count + wid_x;
    unsigned int tid = threadIdx.x;

    if (col >= meta.matrix_cols) {
        return;
    }

    unsigned int n_cols = meta.matrix_cols;
    unsigned int k_offset = meta.k;

    for (unsigned int reflector = 0u; reflector < meta.reflector_count; reflector++) {
        unsigned int n_rows = meta.panel_rows - reflector;
        unsigned int start_row = k_offset + reflector;
        unsigned int v_off = reflector_buf[reflector].vector_offset;
        float beta = reflector_buf[reflector].beta;

        // Phase 1: partial dot = v^T · A[start_row:m, col]
        float partial = 0.0f;
        unsigned int row = tid;
        while (row < n_rows) {
            unsigned int a_idx = (start_row + row) * n_cols + col;
            partial += v_buf[v_off + row] * a_buf[a_idx];
            row += 256u;
        }
        sdata[tid] = partial;
        __syncthreads();

        // Parallel tree reduction
        for (unsigned int s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        float dot = sdata[0];
        __syncthreads();

        // Phase 2: A[start_row:m, col] -= beta * v * dot
        row = tid;
        while (row < n_rows) {
            unsigned int a_idx = (start_row + row) * n_cols + col;
            a_buf[a_idx] -= beta * v_buf[v_off + row] * dot;
            row += 256u;
        }
        __syncthreads();
    }
}
    "#
    .to_string()
}

pub(super) struct HhTrailingUpdate<'a> {
    pub(super) vectors: &'a CudaBuffer<f32>,
    pub(super) matrix: &'a CudaBuffer<f32>,
    pub(super) reflectors: &'a CudaBuffer<HhReflectorMeta>,
    pub(super) panel_rows: usize,
    pub(super) trail_cols: usize,
    pub(super) matrix_cols: usize,
    pub(super) panel_start: usize,
    pub(super) vector_offsets: &'a [usize],
    pub(super) betas: &'a [f32],
}

pub(super) fn hh_trailing_update(device: &CudaDevice, update: HhTrailingUpdate<'_>) -> Result<()> {
    let HhTrailingUpdate {
        vectors,
        matrix,
        reflectors,
        panel_rows,
        trail_cols,
        matrix_cols,
        panel_start,
        vector_offsets,
        betas,
    } = update;
    let reflector_count = betas.len();
    debug_assert_eq!(vector_offsets.len(), reflector_count);
    if panel_rows == 0 || trail_cols == 0 || reflector_count == 0 {
        return Ok(());
    }

    let meta = HhMeta {
        panel_rows: to_u32(panel_rows, "HH panel_rows")?,
        reflector_count: to_u32(reflector_count, "HH reflector_count")?,
        trail_cols: to_u32(trail_cols, "HH trail_cols")?,
        matrix_cols: to_u32(matrix_cols, "HH matrix_cols")?,
        k: to_u32(panel_start, "HH panel start")?,
        _pad: [0; 3],
    };

    let reflector_host: Vec<HhReflectorMeta> = vector_offsets
        .iter()
        .copied()
        .zip(betas.iter().copied())
        .map(|(offset, beta)| {
            let vector_offset = to_u32(offset, "HH vector_offset")?;
            Ok(HhReflectorMeta {
                vector_offset,
                beta,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    device.write_sub_buffer(reflectors, 0, &reflector_host)?;

    let kernel = cached_kernel(
        device,
        PipelineKey::QrHouseholder,
        "householder_kernel",
        householder_source,
    )?;

    let mut v_ptr = vectors.raw();
    let mut a_ptr = matrix.raw();
    let mut reflector_ptr = reflectors.raw();
    let launch_columns = meta.trail_cols;
    let mut metadata = meta;
    let mut args: [*mut std::ffi::c_void; 4] = [
        (&raw mut v_ptr).cast(),
        (&raw mut a_ptr).cast(),
        (&raw mut reflector_ptr).cast(),
        (&raw mut metadata).cast(),
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::planar(launch_columns, 1, 256, 1),
        &mut args,
    )
}
