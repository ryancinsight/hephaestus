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

/// Packed metadata for the Householder reflector application kernel.
///
/// Applies **H = I − β v vᵀ** to each column of the trailing submatrix:
/// `A[:, col] -= β · v · (vᵀ · A[:, col])`
///
/// The trailing submatrix occupies `A[offset .. offset + vec_len * trail_stride]`
/// with elements at `offset + row * trail_stride + col`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct HhMeta {
    /// Length of the Householder vector v.
    vec_len: u32,
    /// Number of trailing columns.
    trail_cols: u32,
    /// Row stride of the trailing matrix in the full buffer.
    trail_stride: u32,
    /// Element offset to A[0, 0] of the trailing submatrix.
    c_offset: u32,
    /// Householder coefficient β = 2 / (vᵀv).
    beta: f32,
    _pad: [u32; 3],
}

// SAFETY: HhMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for HhMeta {}

// ---------------------------------------------------------------------------
// Householder apply kernel:  A[:, col] -= β · v · (vᵀ · A[:, col])
// ---------------------------------------------------------------------------

/// WGSL source for the Householder reflector application kernel.
///
/// For each column of the trailing matrix, computes
/// `dot = vᵀ · A[:, col]` via a parallel reduction, then applies
/// `A[:, col] -= β · v · dot`.
///
/// One workgroup (256 threads) per trailing column; threads cooperatively
/// iterate over the row dimension with stride 256.
fn hh_shader_source() -> String {
    const TY: &str = "f32";
    const ZERO: &str = "0.0";

    format!(
        r#"struct HhMeta {{
    vec_len: u32,
    trail_cols: u32,
    trail_stride: u32,
    c_offset: u32,
    beta: {ty},
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read>      v_buf: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> a_buf: array<{ty}>;
@group(0) @binding(2) var<uniform>             params: HhMeta;

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

    let n     = params.vec_len;
    let stride = params.trail_stride;
    let off   = params.c_offset;
    let beta  = params.beta;

    // Phase 1: partial dot = vᵀ · A[:, col]
    var partial = {ty}({zero});
    var row = tid;
    while (row < n) {{
        partial = partial + v_buf[row] * a_buf[off + row * stride + col];
        row = row + 256u;
    }}
    sdata[tid] = partial;
    workgroupBarrier();

    // Parallel tree reduction
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
        a_buf[idx] = a_buf[idx] - beta * v_buf[row] * dot;
        row = row + 256u;
    }}
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct HhKernel;

struct HouseholderTrailingUpdate<'a> {
    v_buf: &'a WgpuBuffer<f32>,
    a_buf: &'a WgpuBuffer<f32>,
    vec_len: usize,
    trail_cols: usize,
    trail_stride: usize,
    c_offset: usize,
    beta: f32,
}

/// GPU dispatch for one Householder reflector application:
///
/// ```text
/// A[offset + row * stride + col] -= β · v[row] · Σᵢ v[i] · A[offset + i * stride + col]
/// ```
///
/// One workgroup (256 threads) per trailing column.
fn hh_trailing_update(device: &WgpuDevice, update: HouseholderTrailingUpdate<'_>) -> Result<()> {
    let vec_len = update.vec_len;
    let trail_cols = update.trail_cols;
    if vec_len == 0 || trail_cols == 0 {
        return Ok(());
    }

    let meta = HhMeta {
        vec_len: u32::try_from(vec_len).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("HH vec_len {vec_len} exceeds u32"),
        })?,
        trail_cols: u32::try_from(trail_cols).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("HH trail_cols {trail_cols} exceeds u32"),
        })?,
        trail_stride: u32::try_from(update.trail_stride).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("HH trail_stride {} exceeds u32", update.trail_stride),
            }
        })?,
        c_offset: u32::try_from(update.c_offset).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("HH c_offset {} exceeds u32", update.c_offset),
        })?,
        beta: update.beta,
        _pad: [0; 3],
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<HhKernel>(), TypeId::of::<f32>(), 256),
        "hephaestus-hh",
        hh_shader_source,
    );

    let meta_buf = device.get_uniform_buffer(std::mem::size_of::<HhMeta>() as u64)?;
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-hh"),
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
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-hh"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-hh"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // One workgroup per trailing column.
        let wg_x = u32::try_from(trail_cols).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("HH workgroup count {trail_cols} exceeds u32"),
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
/// launch overhead.  Each panel produces *b* Householder reflectors that
/// are applied to the trailing columns via *b* GPU kernel launches.
const QR_BLOCK_SIZE: usize = 32;

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

    // Device-resident buffer for the working matrix.
    let work_buf = device.alloc_zeroed::<f32>(m * n)?;

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
            device.write_buffer(&work_buf, &host)?;
            continue;
        }

        // ── Step 2: Upload working matrix to device ──
        device.write_buffer(&work_buf, &host)?;

        // ── Step 3: Apply b Householder reflectors to trailing columns ──
        for j in 0..b {
            let vec_len = panel_rows - j;

            // Reconstruct v_j from the factored panel:
            //   v[0] = heads[j]           (head component, at row k+j)
            //   v[i] = panel[(j+i)*b + j] (tail, below diagonal)
            let mut v = vec![0.0f32; vec_len];
            v[0] = heads[j];
            for i in 1..vec_len {
                v[i] = panel[(j + i) * b + j];
            }

            let v_dev = device.upload(&v)?;

            // Trailing submatrix A[k+j : m, k+b : n] in the full buffer.
            let c_offset = (k + j) * n + (k + b);

            hh_trailing_update(
                device,
                HouseholderTrailingUpdate {
                    v_buf: &v_dev,
                    a_buf: &work_buf,
                    vec_len,
                    trail_cols,
                    trail_stride: n,
                    c_offset,
                    beta: betas[j],
                },
            )?;
        }

        // ── Step 4: Download updated working matrix back to host ──
        device.download(&work_buf, &mut host)?;
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
