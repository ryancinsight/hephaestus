//! GPU-resident QR decomposition via Householder reflectors.
//!
//! Computes **A** = **Q R** where **Q** is orthogonal and **R** is
//! upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`qr_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`qr_decompose_blocked`] — width-selected hybrid algorithm. Matrices up to
//!   four panels use one host factorization; wider matrices factor panels on the
//!   CPU, apply wide trailing Householder updates on the GPU, and finish the
//!   final at-most-one-block tail on the CPU after a paired readback.
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

use hephaestus_core::{
    BlockedDecompositionBackend, BlockedQrBackend, ComputeDevice, HephaestusError, Result,
    TrailingHh, blocked_qr,
};

use super::validate::validate_dense_operand;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

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
    ///
    /// Both entry points leave the same object here: [`qr_decompose`] uploads
    /// the host factor's **R**, and [`qr_decompose_blocked`] computes it in
    /// place — its panel write-back zeroes every `col < row` entry, so the
    /// buffer it returns carries no reflectors despite its working name. The
    /// contract is asserted across both paths by the
    /// `qr_r_buffer_is_upper_triangular_on_both_entry_points` contract case.
    #[must_use]
    #[inline]
    pub fn r_buffer(&self) -> &WgpuBuffer<f32> {
        &self.r
    }

    /// Take ownership of the device-resident **R** buffer.
    ///
    /// Callers that need to keep **R** past this decomposition — the Python
    /// binding returns it as a standalone device tensor — would otherwise
    /// re-upload `inner().r()`, sending `4mn` bytes the device already holds.
    #[must_use]
    #[inline]
    pub fn into_r_buffer(self) -> WgpuBuffer<f32> {
        self.r
    }

    /// Borrow the host-side Leto decomposition.
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &leto_ops::QrDecomposition<f32> {
        &self.inner
    }

    /// Accumulate the orthogonal factor **Q** (*m* × *m*, row-major) on the
    /// device.
    ///
    /// **Q** is built here rather than during factorisation because it is not
    /// free: accumulating it costs O(*m*² *n*), and a least-squares caller
    /// discards it entirely. Factorisation stores only the compact Householder
    /// form, and this method materialises **Q** for the callers that ask.
    ///
    /// Starting from a device identity, the stored reflectors apply in reverse
    /// — **Q** = **H₁**(**H₂**(⋯(**H_k I**))) — with one workgroup per column
    /// of **Q**, mirroring the panel kernel's reduce-then-update shape.
    ///
    /// The transfer trade is `4mn + 8·min(m, n)` bytes uploaded (the packed
    /// factor and the per-reflector head/β pairs) in place of the `4m²` bytes
    /// a host-accumulated **Q** costs to download and upload back, and the
    /// O(*m*² *n*) accumulation itself moves off the host.
    ///
    /// # Errors
    ///
    /// - Buffer allocation or dispatch failure.
    /// - Dimensions exceeding `u32`.
    pub fn accumulate_q(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        let (m, n) = (self.rows, self.cols);
        let layout = leto::Layout::c_contiguous([m, m]).map_err(map_layout_err)?;
        let q = crate::application::linalg::device_identity::<f32>(device, &layout)?;

        let reflector_count = m.min(n);
        // With no reflectors — an empty matrix, or no columns to factor — the
        // identity is already Q, matching `inner().q()`.
        if m == 0 || reflector_count == 0 {
            return Ok(q);
        }

        let packed_dev = device.upload(self.inner.packed())?;

        let heads = self.inner.heads();
        let betas = self.inner.betas();
        let reflector_host = (0..reflector_count)
            .map(|k| QReflector {
                head: heads[k],
                beta: betas[k],
            })
            .collect::<Vec<_>>();
        let reflector_dev = device.alloc_uninitialized::<QReflector>(reflector_count)?;
        device.write_sub_buffer(&reflector_dev, 0, &reflector_host)?;

        let to_u32 = |value: usize, what: &str| {
            u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("Q accumulation {what} {value} exceeds u32"),
            })
        };
        let meta = QMeta {
            m: to_u32(m, "m")?,
            n: to_u32(n, "n")?,
            reflector_count: to_u32(reflector_count, "reflector_count")?,
        };

        let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<QMeta>(1)?)?;

        device
            .queue()
            .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

        let pipeline = cached_pipeline(
            device,
            (TypeId::of::<QAccumulateKernel>(), TypeId::of::<f32>(), 256),
            "hephaestus-qr-q-accumulate",
            q_accumulate_shader_source,
        );
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-qr-q-accumulate"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: packed_dev.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: q.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: reflector_dev.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: meta_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-qr-q-accumulate"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-qr-q-accumulate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(meta.m, 1, 1);
        }
        device.queue().submit(Some(encoder.finish()));

        Ok(q)
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

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([m]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
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
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct HhMeta {
    panel_rows: u32,
    reflector_count: u32,
    trail_cols: u32,
    matrix_cols: u32,
    k: u32,
}

/// Per-reflector metadata consumed by the panel Householder kernel.
///
/// Public because the WGPU [`BlockedQrBackend`] impl names it as its
/// `Reflectors` buffer element, and a public trait impl cannot leak a
/// restricted type.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub struct HhReflectorMeta {
    /// Offset of this reflector in the packed vector buffer.
    pub(super) vector_offset: u32,
    /// Householder scale factor β.
    pub(super) beta: f32,
}

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

/// Apply the packed panel Householder reflectors to a matrix's trailing columns.
///
/// The shader pipeline is cached; the bind group and the uniform buffer are
/// built per call, because the trait method that reaches this takes no
/// loop-scoped descriptors.
fn householder_trailing_update(
    device: &WgpuDevice,
    vectors: &WgpuBuffer<f32>,
    matrix: &WgpuBuffer<f32>,
    reflectors: &WgpuBuffer<HhReflectorMeta>,
    spec: TrailingHh<'_>,
) -> Result<()> {
    fn dimension(value: usize, name: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("{name} {value} exceeds u32"),
        })
    }

    let reflector_host: Vec<HhReflectorMeta> = spec
        .vector_offsets
        .iter()
        .copied()
        .zip(spec.betas.iter().copied())
        .map(|(offset, beta)| -> Result<HhReflectorMeta> {
            Ok(HhReflectorMeta {
                vector_offset: dimension(offset, "HH vector offset")?,
                beta,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    device.write_sub_buffer(reflectors, 0, &reflector_host)?;

    let hh_meta = HhMeta {
        panel_rows: dimension(spec.panel_rows, "HH panel_rows")?,
        reflector_count: dimension(spec.betas.len(), "HH reflector_count")?,
        trail_cols: dimension(spec.trail_cols, "HH trail_cols")?,
        matrix_cols: dimension(spec.matrix_cols, "HH matrix_cols")?,
        k: dimension(spec.panel_start, "HH panel start")?,
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<HhKernel>(), TypeId::of::<f32>(), 256),
        "hephaestus-hh",
        hh_shader_source,
    );
    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<HhMeta>(1)?)?;
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-hh-panel"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vectors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reflectors.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });
    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&hh_meta));

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-qr-hh-update"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-hh-panel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dimension(spec.trail_cols, "HH workgroup count")?, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

// ---------------------------------------------------------------------------
// Q accumulation uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the **Q** accumulation kernel.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct QMeta {
    /// Row count *m* of the factored matrix, and the order of **Q**.
    m: u32,
    /// Column count *n*, the row stride of the packed factor.
    n: u32,
    /// Number of stored reflectors, `min(m, n)`.
    reflector_count: u32,
}

/// Per-reflector scalars the packed factor does not carry.
///
/// The head `v_k[k]` is displaced from the packed diagonal (which holds
/// `R[k][k]`), so it travels alongside β rather than being read from `packed`.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct QReflector {
    /// Householder vector head component `v_k[k]`.
    head: f32,
    /// Householder scale factor `β_k = 2 / (v_kᵀ v_k)`.
    beta: f32,
}

// ---------------------------------------------------------------------------
// Q accumulation kernel:  Q ← H₁ (H₂ (⋯ (H_k · I)))
// ---------------------------------------------------------------------------

/// WGSL source accumulating **Q** by applying the stored reflectors to an
/// identity, one workgroup per column of **Q**.
fn q_accumulate_shader_source() -> String {
    r#"struct QMeta {
    m: u32,
    n: u32,
    reflector_count: u32,
}
struct QReflector {
    head: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read>       packed: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read>       reflectors: array<QReflector>;
@group(0) @binding(3) var<uniform>             params: QMeta;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let col = wid.x;
    if (col >= params.m) {
        return;
    }
    let tid = lid.x;
    let m = params.m;
    let n = params.n;

    // Reverse order: Q = H_1 (H_2 (... (H_k I))), so the last reflector
    // reaches the identity first.
    for (var idx = 0u; idx < params.reflector_count; idx = idx + 1u) {
        let k = params.reflector_count - 1u - idx;
        let head = reflectors[k].head;
        let beta = reflectors[k].beta;
        let span = m - k;

        // Phase 1: dot = v_kᵀ · Q[k:m, col], reduced as a 256-way tree.
        var partial = f32(0.0);
        var row = tid;
        while (row < span) {
            let r = k + row;
            // The packed diagonal slot holds R[k][k]; the reflector head
            // arrives separately.
            var v = packed[r * n + k];
            if (row == 0u) { v = head; }
            partial = partial + v * q[r * m + col];
            row = row + 256u;
        }
        sdata[tid] = partial;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
            workgroupBarrier();
        }
        let scaled = beta * sdata[0];
        workgroupBarrier();

        // Phase 2: Q[k:m, col] -= β_k · v_k · dot.
        //
        // A zero β is deliberately *not* skipped with `continue`: β is a
        // storage read, so branching on it around the `workgroupBarrier()`
        // calls above would be non-uniform control flow, which WGSL forbids.
        // With β = 0 the update below is already the identity, so the
        // reflector costs a pass and changes nothing.
        row = tid;
        while (row < span) {
            let r = k + row;
            var v = packed[r * n + k];
            if (row == 0u) { v = head; }
            q[r * m + col] = q[r * m + col] - scaled * v;
            row = row + 256u;
        }
        storageBarrier();
        workgroupBarrier();
    }
}
"#
    .to_string()
}

struct QAccumulateKernel;

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
    let [rows, cols] = matrix.layout.shape();
    if rows < cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

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
// Entry point 2 — blocked with hybrid trailing Householder application
// ---------------------------------------------------------------------------

/// Panel block size for the blocked QR algorithm.
///
/// A value of 32 balances CPU panel factorisation cost against GPU kernel
/// launch overhead. Each panel produces *b* Householder reflectors that are
/// applied to the trailing columns in one GPU dispatch.
const QR_BLOCK_SIZE: usize = 32;

/// Bounded measured panel regime where one dense host factorization outperforms
/// blocked region transfers and per-panel synchronization.
const QR_DIRECT_PANEL_LIMIT: usize = 4;

/// Blocked QR factorization **A = Q R** with hybrid trailing Householder
/// application.
///
/// Dense matrices of at most four `QR_BLOCK_SIZE` panels use the canonical
/// [`qr_decompose`] host factorization directly. This regime has no wide
/// enough trailing work to repay the blocked path's region gather/scatter,
/// compact device scratch, and per-panel synchronization, so one dense download
/// and one `R` upload replace that schedule.
///
/// The algorithm processes the matrix in panels of `QR_BLOCK_SIZE` columns.
/// For wider matrices, each panel *k* performs:
///
/// 1. The panel `A[k:m, k:k+b]` is gathered into a contiguous device buffer
///    and downloaded to the host.
/// 2. The panel is factored on the **CPU** via inline Householder QR.
/// 3. The factored panel is uploaded to the device, and a GPU kernel
///    scatters the factored upper triangle back into the main matrix and zeroes
///    out the sub-diagonal elements.
/// 4. The *b* Householder reflectors are applied to the trailing columns
///    `A[k:m, k+b:n]` directly on the **GPU** in-place while the trailing width
///    exceeds one block.
/// 5. The final panel and at-most-one-block tail are gathered together and
///    finished on the CPU before one paired device write.
///
/// # Errors
///
/// - Underdetermined shape (*m* < *n*).
/// - Non-dense (non-C-contiguous / offset / broadcast) operand: the
///   blocked path bulk-copies the matrix storage on the device.
/// - Non-finite values in the input.
/// - Rank-deficient input (zero column norm).
pub fn qr_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuQrDecomposition> {
    let [m, n] = matrix.layout.shape();
    if m < n {
        return Err(HephaestusError::DispatchFailed {
            message: format!("QR requires m ≥ n, got shape [{m}, {n}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;
    validate_dense_operand("QR", &matrix)?;

    if m == 0 || n == 0 {
        let r_buf = device.alloc_zeroed::<f32>(0)?;
        let inner =
            leto_ops::QrDecomposition::from_raw_parts(Vec::new(), Vec::new(), Vec::new(), m, n);
        return Ok(GpuQrDecomposition {
            inner,
            r: r_buf,
            rows: m,
            cols: n,
        });
    }

    let block_size = QR_BLOCK_SIZE.min(n);
    // Component profiling shows the blocked schedule remains transfer-bound
    // through four panels. Route that bounded regime directly; wider shapes
    // keep the GPU path pending their own measured crossover.
    if n.div_ceil(block_size) <= QR_DIRECT_PANEL_LIMIT {
        return qr_decompose(device, matrix);
    }

    let work_buf = device.clone_device(matrix.buffer, m * n)?;
    let result = blocked_qr(device, work_buf, m, n, block_size)?;

    let inner =
        leto_ops::QrDecomposition::from_raw_parts(result.packed, result.heads, result.betas, m, n);

    Ok(GpuQrDecomposition {
        inner,
        r: result.r,
        rows: m,
        cols: n,
    })
}

/// The WGPU blocked-QR operations the shared [`blocked_qr`] loop drives
/// (ADR 0003).
///
/// Only the reflector-metadata buffer and the trailing Householder kernel are
/// WGPU-specific: the loop owns the panel walk, the CPU panel factorisation,
/// the reflector packing, the sub-diagonal zeroing, and the final gather. WGPU
/// also selects the host tail-finish policy, because its trailing kernel has a
/// fixed launch cost that the host path avoids for a final `≤ block_size` tail.
impl BlockedQrBackend for WgpuDevice {
    type Reflectors = WgpuBuffer<HhReflectorMeta>;

    fn alloc_reflectors(&self, len: usize) -> Result<Self::Reflectors> {
        self.alloc_uninitialized::<HhReflectorMeta>(len)
    }

    fn write_flat(&self, buf: &Self::Buffer, data: &[f32]) -> Result<()> {
        self.write_sub_buffer(buf, 0, data)
    }

    fn householder_trailing(
        &self,
        vectors: &Self::Buffer,
        matrix: &Self::Buffer,
        reflectors: &Self::Reflectors,
        spec: TrailingHh<'_>,
    ) -> Result<()> {
        householder_trailing_update(self, vectors, matrix, reflectors, spec)
    }

    fn finishes_tail_on_cpu(&self) -> bool {
        true
    }
}
