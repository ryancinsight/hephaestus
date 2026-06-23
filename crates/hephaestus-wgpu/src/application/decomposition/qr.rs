//! GPU-resident QR decomposition via Householder reflectors.
//!
//! Computes **A** = **Q R** where **Q** is orthogonal and **R** is
//! upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`qr_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`qr_decompose_blocked`] — blocked algorithm where panel
//!   factorization runs on the CPU but the trailing Householder application
//!   runs on the GPU via a dedicated compute kernel.
//!
//! # Mathematical Foundations
//!
//! ## Theorem — Householder QR Factorization
//!
//! Every **A** ∈ ℝᵐˣⁿ with *m* ≥ *n* factors as **A** = **Q R** where
//! **Q** ∈ ℝᵐˣᵐ is orthogonal and **R** ∈ ℝᵐˣⁿ is upper-triangular.
//!
//! **Proof.** At step *k*, the Householder reflector **Hₖ** = **I** − βₖ
//! **vₖ vₖ**ᵀ zeros the entries below the diagonal of column *k*.
//! **Hₖ** is orthogonal (**Hₖ**ᵀ = **Hₖ** and **Hₖ²** = **I**).
//! After *n* steps, **Hₙ ⋯ H₁ A** = **R** is upper-triangular, so
//! **Q** = **H₁ᵀ ⋯ Hₙᵀ** = **H₁ ⋯ Hₙ** (each reflector is symmetric). ∎
//!
//! ## Blocked QR with GPU Trailing Application
//!
//! For large *m*, the dominant cost is applying the *b* Householder
//! reflectors from each panel to the trailing *m × (n−k−b)* submatrix.
//! Each application costs O(m(n−k)) flops — b applications per panel gives
//! O(b·m·(n−k)) — and is embarrassingly parallel across columns.
//!
//! **Theorem (Blocked QR complexity).** For *m × n* with block size *b*,
//! the total flop count is 2n²(m − n/3), identical to unblocked QR.  The
//! blocked variant improves performance by:
//! (a) moving the O(b·(m−k)·(n−k)) trailing Householder application to
//!     the GPU, and
//! (b) improving CPU cache locality for the O(b²·(m−k)) panel operations.
//!
//! **Proof.** Each block iteration costs:
//! - Panel factor: 2b²(m−k) − 2b³/3
//! - Trailing apply: 2b(m−k)(n−k−b)  (b rank-1 updates of width n−k−b)
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n²(m − n/3) total flops. ∎

use std::any::TypeId;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use super::region::{
    download_matrix_region_compact_reusable, write_matrix_region_compact_reusable, MatrixRegion,
};
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use hephaestus_core::panel_qr_packed;

/// QR decomposition result: device-resident R factor with host-side
/// decomposition for solve_least_squares.
pub struct GpuQrDecomposition {
    /// Host-side leto-ops decomposition (owns packed/heads/betas).
    inner: leto_ops::QrDecomposition<f32>,
    /// Device-resident upper-triangular factor **R** (*m* × *n*, row-major).
    r: WgpuBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl GpuQrDecomposition {
    /// (rows, cols) of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow the upper-triangular factor **R** buffer on the device.
    #[must_use]
    #[inline]
    pub fn r_buffer(&self) -> &WgpuBuffer<f32> {
        &self.r
    }

    /// Borrow the host-side Leto decomposition.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::QrDecomposition<f32> {
        &self.inner
    }

    /// Solve min ‖**A** · **x** − **rhs**‖₂ (least squares).
    ///
    /// Downloads the RHS from the device, solves on the host using the
    /// stored Householder reflectors, and uploads the solution vector.
    pub fn solve_least_squares(
        &self,
        device: &WgpuDevice,
        rhs: &WgpuBuffer<f32>,
    ) -> Result<WgpuBuffer<f32>> {
        let (m, n) = (self.rows, self.cols);
        if rhs.len != m {
            return Err(HephaestusError::LengthMismatch {
                host_len: m,
                device_len: rhs.len,
            });
        }
        if m == 0 || n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0f32; m];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view =
            leto::ArrayView::<f32, 1>::new(leto::Layout::c_contiguous([m]).unwrap(), &rhs_host);
        let x = self.inner.solve_least_squares(&rhs_view).map_err(|e| {
            HephaestusError::DispatchFailed {
                message: format!("QR least-squares solve failed: {e}"),
            }
        })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }
}

// Custom gather/scatter compute kernels removed in favor of generic MatrixRegion transfers.

// ---------------------------------------------------------------------------
// Householder apply uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the panel Householder reflector application kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct HhMeta {
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    matrix_cols: u32,
    k: u32,
}

// SAFETY: HhMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for HhMeta {}

/// Per-reflector metadata consumed by the panel Householder kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct HhReflectorMeta {
    /// Offset of this reflector in the packed vector buffer.
    vector_offset: u32,
    /// Householder scale factor β.
    beta: f32,
}

// SAFETY: HhReflectorMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for HhReflectorMeta {}

// ---------------------------------------------------------------------------
// Householder apply kernel:  A[:, col] -= β · v · (vᵀ · A[:, col])
// ---------------------------------------------------------------------------

/// WGSL source for applying all panel Householder reflectors.
fn hh_shader_source() -> String {
    r#"struct HhMeta {
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    matrix_cols: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read>      v_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> a_buf: array<f32>;
struct ReflectorMeta {
    vector_offset: u32,
    beta: f32,
}
@group(0) @binding(2) var<storage, read>      reflector_buf: array<ReflectorMeta>;
@group(0) @binding(3) var<uniform>             params: HhMeta;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wid:  vec3<u32>,
) {
    let col = params.k + params.reflector_count + wid.x;
    let tid = lid.x;

    if (col >= params.matrix_cols) {
        return;
    }

    let n_cols = params.matrix_cols;
    let k_offset = params.k;

    for (var reflector = 0u; reflector < params.reflector_count; reflector = reflector + 1u) {
        let n_rows = params.panel_rows - reflector;
        let start_row = k_offset + reflector;
        let v_off = reflector_buf[reflector].vector_offset;
        let beta = reflector_buf[reflector].beta;

        // Phase 1: partial dot = vᵀ · A[start_row:m, col]
        var partial = f32(0.0);
        var row = tid;
        while (row < n_rows) {
            let a_idx = (start_row + row) * n_cols + col;
            partial = partial + v_buf[v_off + row] * a_buf[a_idx];
            row = row + 256u;
        }
        sdata[tid] = partial;
        workgroupBarrier();

        // Parallel tree reduction.
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (tid < s) {
                sdata[tid] = sdata[tid] + sdata[tid + s];
            }
            workgroupBarrier();
        }

        let dot = sdata[0];
        workgroupBarrier();

        // Phase 2: A[start_row:m, col] -= beta * v * dot
        row = tid;
        while (row < n_rows) {
            let a_idx = (start_row + row) * n_cols + col;
            a_buf[a_idx] = a_buf[a_idx] - beta * v_buf[v_off + row] * dot;
            row = row + 256u;
        }
        storageBarrier();
        workgroupBarrier();
    }
}
"#
    .to_string()
}

struct HhKernel;

// ---------------------------------------------------------------------------
// Inline panel Householder QR
// ---------------------------------------------------------------------------

// panel_qr_packed is re-exported from hephaestus_core::decomposition.

// ---------------------------------------------------------------------------
// Entry point 1 — host delegation
// ---------------------------------------------------------------------------

/// Compute the Householder QR factorization on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For tall matrices where the trailing Householder application
/// should run on the GPU, prefer [`qr_decompose_blocked`].
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-finite values in the input.
/// - Exactly-zero pivot column norm (rank-deficient input).
pub fn qr_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [rows, cols] = matrix.layout.shape;
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    if rows == 0 || cols == 0 {
        let r_buf = device.alloc_zeroed::<f32>(0)?;
        let placeholder = vec![0.0f32];
        let inner = leto_ops::qr_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([1, 1]).unwrap(),
            &placeholder,
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("QR decomposition failed: {e}"),
        })?;
        return Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
            rows,
            cols,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let qr = leto_ops::qr_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("QR decomposition failed: {e}"),
    })?;

    let r_host = qr.r();
    let r_buf = device.upload(leto::Storage::as_slice(r_host.storage()))?;

    Ok(GpuQrDecomposition {
        inner: qr,
        r: r_buf,
        rows,
        cols,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing Householder application
// ---------------------------------------------------------------------------

/// Panel block size for the blocked QR algorithm.
///
/// A value of 32 balances CPU panel factorisation cost against GPU kernel
/// launch overhead. Each panel produces *b* Householder reflectors that are
/// applied to the trailing columns in one GPU dispatch.
const QR_BLOCK_SIZE: usize = 32;

/// Blocked QR factorization **A = Q R** with GPU-accelerated trailing
/// Householder application.
///
/// The algorithm processes the matrix in panels of `QR_BLOCK_SIZE` columns.
/// For each panel *k*:
///
/// 1. The panel `A[k:m, k:k+b]` is gathered into a contiguous device buffer
///    and downloaded to the host.
/// 2. The panel is factored on the **CPU** via inline Householder QR.
/// 3. The factored panel is uploaded to the device, and a GPU kernel
///    scatters the factored upper triangle back into the main matrix and zeroes
///    out the sub-diagonal elements.
/// 4. The *b* Householder reflectors are applied to the trailing columns
///    `A[k:m, k+b:n]` directly on the **GPU** in-place.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-finite values in the input.
/// - Rank-deficient input (zero column norm).
pub fn qr_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [m, n] = matrix.layout.shape;
    if m < n {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{m}, {n}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    if m == 0 || n == 0 {
        let r_buf = device.alloc_zeroed::<f32>(0)?;
        let placeholder = vec![0.0f32];
        let inner = leto_ops::qr_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([1, 1]).unwrap(),
            &placeholder,
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("QR decomposition failed: {e}"),
        })?;
        return Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
            rows: m,
            cols: n,
        });
    }

    // Create a GPU working buffer that is a copy of the input matrix.
    let work_buf = device.alloc_zeroed::<f32>(m * n)?;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-qr-copy"),
        });
    encoder.copy_buffer_to_buffer(
        &matrix.buffer.buffer,
        0,
        &work_buf.buffer,
        0,
        WgpuDevice::byte_size::<f32>(m * n)?,
    );
    device.queue().submit(Some(encoder.finish()));

    let block_size = QR_BLOCK_SIZE.min(n);

    let mut packed = vec![0.0f32; m * n];
    let mut cumulative_heads = Vec::with_capacity(n.min(m));
    let mut cumulative_betas = Vec::with_capacity(n.min(m));

    // Pre-allocate vectors_dev of maximum needed size: m * block_size.
    let vectors_dev = device.alloc_zeroed::<f32>(m * block_size)?;

    // Pre-allocate reflector_dev of size block_size.
    let reflector_dev = device.alloc_zeroed::<HhReflectorMeta>(block_size)?;

    // Pre-allocate a single temp_compact_buf to avoid repeated allocations in the loop.
    let temp_compact_buf = device.alloc_zeroed::<f32>(m * block_size)?;

    let hh_pipeline = cached_pipeline(
        device,
        (TypeId::of::<HhKernel>(), TypeId::of::<f32>(), 256),
        "hephaestus-hh",
        hh_shader_source,
    );

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = m - k;
        let trail_cols = n - k - b;

        // ── Step 1 & 2: Gather panel from work_buf directly to host panel ──
        // We gather all m rows to preserve and download previous updates for upper rows.
        let panel_region = MatrixRegion {
            stride: n,
            row_start: 0,
            col_start: k,
            rows: m,
            cols: b,
        };
        let mut panel = download_matrix_region_compact_reusable(
            device,
            &work_buf,
            &temp_compact_buf,
            panel_region,
        )?;

        // ── Step 3: Factor the active panel region on the CPU ──
        // Only factor the sub-slice starting at row k.
        let (heads, betas) = panel_qr_packed(&mut panel[k * b..], panel_rows, b)?;

        cumulative_heads.extend_from_slice(&heads);
        cumulative_betas.extend_from_slice(&betas);

        // Copy the complete final values of these columns into the host-side packed matrix.
        for j in 0..b {
            let col = k + j;
            for r in 0..m {
                packed[r * n + col] = panel[r * b + j];
            }
        }

        // Extract packed vectors for Step 6 before zeroing sub-diagonal elements of panel
        let factored_panel = &mut panel[k * b..];
        let mut packed_vectors = Vec::with_capacity(panel_rows * b);
        let mut vector_offsets = Vec::with_capacity(b);
        for j in 0..b {
            let vec_len = panel_rows - j;
            vector_offsets.push(packed_vectors.len());
            packed_vectors.push(heads[j]);
            for i in 1..vec_len {
                packed_vectors.push(factored_panel[(j + i) * b + j]);
            }
        }

        // Zero out the strictly lower-triangular part of panel before writing back
        for r in 0..panel_rows {
            for c in 0..b {
                if c < r {
                    factored_panel[r * b + c] = 0.0;
                }
            }
        }

        // ── Step 4 & 5: Write the factored panel with sub-diagonal zeroes back to the device ──
        let panel_write_region = MatrixRegion {
            stride: n,
            row_start: k,
            col_start: k,
            rows: panel_rows,
            cols: b,
        };
        write_matrix_region_compact_reusable(
            device,
            &work_buf,
            &temp_compact_buf,
            factored_panel,
            panel_write_region,
        )?;

        if trail_cols > 0 {
            device.write_sub_buffer(&vectors_dev, 0, &packed_vectors)?;

            let reflector_host: Vec<HhReflectorMeta> = vector_offsets
                .iter()
                .copied()
                .zip(betas.iter().copied())
                .map(|(offset, beta)| {
                    let vector_offset =
                        u32::try_from(offset).map_err(|_| HephaestusError::DispatchFailed {
                            message: format!("HH vector offset {offset} exceeds u32"),
                        })?;
                    Ok(HhReflectorMeta {
                        vector_offset,
                        beta,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            device.write_sub_buffer(&reflector_dev, 0, &reflector_host)?;

            let hh_meta = HhMeta {
                panel_rows: u32::try_from(panel_rows).map_err(|_| {
                    HephaestusError::DispatchFailed {
                        message: format!("HH panel_rows {panel_rows} exceeds u32"),
                    }
                })?,
                reflector_count: u32::try_from(b).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("HH reflector_count {b} exceeds u32"),
                })?,
                trail_cols: u32::try_from(trail_cols).map_err(|_| {
                    HephaestusError::DispatchFailed {
                        message: format!("HH trail_cols {trail_cols} exceeds u32"),
                    }
                })?,
                matrix_cols: u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("HH matrix_cols {n} exceeds u32"),
                })?,
                k: u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("HH k {k} exceeds u32"),
                })?,
            };

            let raw_hh_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<HhMeta>(1)?)?;
            let hh_meta_buf = crate::infrastructure::pool::uniform_guard(device.clone(), raw_hh_meta_buf);
            device
                .queue()
                .write_buffer(&hh_meta_buf, 0, bytemuck::bytes_of(&hh_meta));

            let hh_bind_group = device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hephaestus-hh-panel"),
                    layout: &hh_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: vectors_dev.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: work_buf.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: reflector_dev.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: hh_meta_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut hh_encoder =
                device
                    .inner()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("hephaestus-qr-hh-update"),
                    });

            {
                let mut pass = hh_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hephaestus-hh-panel"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&hh_pipeline);
                pass.set_bind_group(0, &hh_bind_group, &[]);
                let wg_x =
                    u32::try_from(trail_cols).map_err(|_| HephaestusError::DispatchFailed {
                        message: format!("HH workgroup count {trail_cols} exceeds u32"),
                    })?;
                pass.dispatch_workgroups(wg_x, 1, 1);
            }

            device.queue().submit(Some(hh_encoder.finish()));
        }
    }

    let inner =
        leto_ops::QrDecomposition::from_raw_parts(packed, cumulative_heads, cumulative_betas, m, n);

    Ok(GpuQrDecomposition {
        inner,
        r: work_buf,
        rows: m,
        cols: n,
    })
}
