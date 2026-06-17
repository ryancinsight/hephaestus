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

// ---------------------------------------------------------------------------
// Householder apply uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the panel Householder reflector application kernel.
///
/// Applies every reflector from a factored panel to each trailing column. One
/// workgroup owns one trailing column and applies the panel reflectors
/// sequentially, preserving the Householder dependency order without requiring
/// cross-workgroup synchronization.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct HhMeta {
    /// Number of rows in the current compact panel tile.
    panel_rows: u32,
    /// Number of Householder reflectors in the panel.
    reflector_count: u32,
    /// Number of trailing columns.
    trail_cols: u32,
    /// Row stride of the compact trailing matrix.
    trail_stride: u32,
}

// SAFETY: HhMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for HhMeta {}

// ---------------------------------------------------------------------------
// Householder apply kernel:  A[:, col] -= β · v · (vᵀ · A[:, col])
// ---------------------------------------------------------------------------

/// WGSL source for applying all panel Householder reflectors.
///
/// One workgroup owns one trailing column. Within the workgroup, reflectors
/// are applied sequentially; each reflector uses a parallel dot-product
/// reduction across the active rows and then updates those same rows.
fn hh_shader_source() -> String {
    const TY: &str = "f32";
    const ZERO: &str = "0.0";

    format!(
        r#"struct HhMeta {{
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    trail_stride: u32,
}}

@group(0) @binding(0) var<storage, read>      v_buf: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> a_buf: array<{ty}>;
@group(0) @binding(2) var<storage, read>      offsets_buf: array<u32>;
@group(0) @binding(3) var<storage, read>      beta_buf: array<{ty}>;
@group(0) @binding(4) var<uniform>             params: HhMeta;

var<workgroup> sdata: array<{ty}, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wid:  vec3<u32>,
) {{
    let col = wid.x;
    let tid = lid.x;

    if (col >= params.trail_cols) {{
        return;
    }}

    let stride = params.trail_stride;

    for (var reflector = 0u; reflector < params.reflector_count; reflector = reflector + 1u) {{
        let n = params.panel_rows - reflector;
        let off = reflector * stride;
        let v_off = offsets_buf[reflector];
        let beta = beta_buf[reflector];

        // Phase 1: partial dot = vᵀ · A[:, col]
        var partial = {ty}({zero});
        var row = tid;
        while (row < n) {{
            partial = partial + v_buf[v_off + row] * a_buf[off + row * stride + col];
            row = row + 256u;
        }}
        sdata[tid] = partial;
        workgroupBarrier();

        // Parallel tree reduction.
        for (var s = 128u; s > 0u; s = s >> 1u) {{
            if (tid < s) {{
                sdata[tid] = sdata[tid] + sdata[tid + s];
            }}
            workgroupBarrier();
        }}

        let dot = sdata[0];
        workgroupBarrier();

        // Phase 2: A[:, col] -= beta * v * dot
        row = tid;
        while (row < n) {{
            let idx = off + row * stride + col;
            a_buf[idx] = a_buf[idx] - beta * v_buf[v_off + row] * dot;
            row = row + 256u;
        }}
        storageBarrier();
        workgroupBarrier();
    }}
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct HhKernel;

struct HouseholderPanelUpdate<'a> {
    a_buf: &'a WgpuBuffer<f32>,
    v_buf: &'a WgpuBuffer<f32>,
    panel_rows: usize,
    trail_cols: usize,
    trail_stride: usize,
    reflector_count: usize,
    vector_offsets: &'a [usize],
    betas: &'a [f32],
}

fn hh_trailing_update(device: &WgpuDevice, update: HouseholderPanelUpdate<'_>) -> Result<()> {
    if update.panel_rows == 0 || update.trail_cols == 0 || update.reflector_count == 0 {
        return Ok(());
    }

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<HhKernel>(), TypeId::of::<f32>(), 256),
        "hephaestus-hh",
        hh_shader_source,
    );

    let offset_host: Vec<u32> = update
        .vector_offsets
        .iter()
        .copied()
        .map(|offset| {
            u32::try_from(offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("HH vector offset {offset} exceeds u32"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let offsets_buf = device.upload(&offset_host)?;
    let betas_buf = device.upload(update.betas)?;

    let meta = HhMeta {
        panel_rows: u32::try_from(update.panel_rows).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("HH panel_rows {} exceeds u32", update.panel_rows),
            }
        })?,
        reflector_count: u32::try_from(update.reflector_count).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("HH reflector_count {} exceeds u32", update.reflector_count),
            }
        })?,
        trail_cols: u32::try_from(update.trail_cols).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("HH trail_cols {} exceeds u32", update.trail_cols),
            }
        })?,
        trail_stride: u32::try_from(update.trail_stride).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("HH trail_stride {} exceeds u32", update.trail_stride),
            }
        })?,
    };

    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<HhMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-hh-panel"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: update.v_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: update.a_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: offsets_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: betas_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-hh-panel"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-hh-panel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x =
            u32::try_from(update.trail_cols).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("HH workgroup count {} exceeds u32", update.trail_cols),
            })?;
        pass.dispatch_workgroups(wg_x, 1, 1);
    }

    device.queue().submit(Some(encoder.finish()));
    device.recycle_uniform_buffer(meta_buf);

    Ok(())
}

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

fn trailing_columns(host: &[f32], stride: usize, row_start: usize, col_start: usize) -> Vec<f32> {
    let rows = host.len() / stride - row_start;
    let cols = stride - col_start;
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let source = (row_start + row) * stride + col_start;
        let target = row * cols;
        out[target..target + cols].copy_from_slice(&host[source..source + cols]);
    }
    out
}

fn scatter_trailing_columns(
    host: &mut [f32],
    stride: usize,
    row_start: usize,
    col_start: usize,
    compact: &[f32],
) {
    let rows = host.len() / stride - row_start;
    let cols = stride - col_start;
    debug_assert_eq!(compact.len(), rows * cols);
    for row in 0..rows {
        let target = (row_start + row) * stride + col_start;
        let source = row * cols;
        host[target..target + cols].copy_from_slice(&compact[source..source + cols]);
    }
}

/// Blocked QR factorization **A = Q R** with GPU-accelerated trailing
/// Householder application.
///
/// The algorithm processes the matrix in panels of `QR_BLOCK_SIZE` columns.
/// For each panel *k*:
///
/// 1. The panel `A[k:m, k:k+b]` is factored on the **CPU** via inline
///    Householder QR (O(2b²(m−k) − 2b³/3)).
/// 2. Each of the *b* Householder reflectors is applied to the trailing
///    columns `A[k+j:m, k+b:n]` on the **GPU** via a dedicated kernel:
///    `A[:, col] -= β · v · (vᵀ · A[:, col])` (O((m−k−j)(n−k−b)) per
///    reflector).
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

    // Download the full matrix to host.
    let mut host = vec![0.0f32; m * n];
    device.download(matrix.buffer, &mut host)?;

    // Keep a copy for the host-side solve API.
    let original_host = host.clone();

    let block_size = QR_BLOCK_SIZE.min(n);

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = m - k;
        let trail_cols = n - k - b;

        // ── Step 1: Extract and factor the panel A[k:m, k:k+b] on CPU ──
        let mut panel = vec![0.0f32; panel_rows * b];
        for i in 0..panel_rows {
            for j in 0..b {
                panel[i * b + j] = host[(k + i) * n + (k + j)];
            }
        }
        let (heads, betas) = panel_qr_packed(&mut panel, panel_rows, b)?;

        // Write R (upper triangle, j >= i) back to host, zero tails (j < i).
        for i in 0..panel_rows {
            for j in 0..b {
                if j >= i {
                    host[(k + i) * n + (k + j)] = panel[i * b + j];
                } else {
                    host[(k + i) * n + (k + j)] = 0.0;
                }
            }
        }

        if trail_cols == 0 {
            continue;
        }

        // ── Step 2: Upload the consumed trailing columns as a compact tile ──
        let mut trailing = trailing_columns(&host, n, k, k + b);
        let work_buf = device.upload(&trailing)?;

        // ── Step 3: Apply b Householder reflectors to trailing columns ──
        let mut packed_vectors = Vec::with_capacity(panel_rows * b);
        let mut vector_offsets = Vec::with_capacity(b);
        for j in 0..b {
            let vec_len = panel_rows - j;
            vector_offsets.push(packed_vectors.len());
            packed_vectors.push(heads[j]);
            for i in 1..vec_len {
                packed_vectors.push(panel[(j + i) * b + j]);
            }
        }
        let vectors_dev = device.upload(&packed_vectors)?;

        hh_trailing_update(
            device,
            HouseholderPanelUpdate {
                a_buf: &work_buf,
                v_buf: &vectors_dev,
                panel_rows,
                trail_cols,
                trail_stride: trail_cols,
                reflector_count: b,
                vector_offsets: &vector_offsets,
                betas: &betas,
            },
        )?;

        // ── Step 4: Download the compact tile and patch host state.
        device.download(&work_buf, &mut trailing)?;
        scatter_trailing_columns(&mut host, n, k, k + b, &trailing);
    }

    // Build a leto-ops QR on the original (un-factored) matrix for the
    // host-side solve_least_squares API.
    let original_view =
        leto::ArrayView::<f32, 2>::new(leto::Layout::c_contiguous([m, n]).unwrap(), &original_host);
    let inner =
        leto_ops::qr_decompose(&original_view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("QR blocked finalisation failed: {e}"),
        })?;

    // Materialize R from the blocked factorization result.
    // The host buffer's upper triangle contains R from the blocked loop.
    let mut r_host = vec![0.0f32; m * n];
    for i in 0..m.min(n) {
        for j in i..n {
            r_host[i * n + j] = host[i * n + j];
        }
    }
    let r_buf = device.upload(&r_host)?;

    Ok(GpuQrDecomposition {
        inner,
        r: r_buf,
        rows: m,
        cols: n,
    })
}
