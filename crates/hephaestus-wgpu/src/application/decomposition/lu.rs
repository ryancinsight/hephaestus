//! GPU-resident LU decomposition with partial pivoting.
//!
//! Computes **P A** = **L U** where **P** is a permutation matrix, **L** is
//! unit lower-triangular, and **U** is upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`lu_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`lu_decompose_blocked`] — blocked algorithm where panel
//!   factorization uses the same partial-pivoting rule as Leto's LU and the
//!   O(2n³/3) trailing GEMM update (`A₂₂ -= L₂₁ · U₁₂`) runs on the GPU via a
//!   dedicated compute kernel.
//!
//! # Mathematical Foundations
//!
//! ## Blocked LU with GPU Trailing GEMM
//!
//! Partition **P A = L U** into *b × b* blocks:
//!
//! ```text
//! ┌           ┐   ┌       ┐ ┌       ┐
//! │ A₁₁  A₁₂ │   │ L₁₁ 0 │ │ U₁₁ U₁₂│
//! │ A₂₁  A₂₂ │ = │ L₂₁ I │ │  0  S₂₂│
//! └           ┘   └       ┘ └       ┘
//! ```
//!
//! The Schur complement is **S₂₂ = A₂₂ − L₂₁ U₁₂** and the dominant
//! cost is the rank-b GEMM update which runs on the GPU.
//!
//! **Theorem (Blocked LU complexity).** For *n × n* with block size *b*,
//! the total flop count is 2n³/3, identical to unblocked LU.  The blocked
//! variant improves performance by:
//! (a) moving the O(b(n−k)²) trailing GEMM to the GPU, and
//! (b) improving CPU cache locality for the O(b²(n−k)) panel operations.
//!
//! **Proof.** Each block iteration costs:
//! - Panel factor: 2b³/3
//! - Panel solve (L₂₁): b²(n−k−b)/2
//! - Panel solve (U₁₂): b²(n−k−b)/2
//! - Trailing GEMM: 2b(n−k−b)²
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n³/3 total flops (the same
//! as unblocked LU).  The key performance gain is that the trailing GEMM,
//! which dominates for large n, executes on the GPU's massively parallel
//! compute units rather than on the CPU's sequential cores.

use std::any::TypeId;

use hephaestus_core::blocked_lu;
use hephaestus_core::{BlockedDecompositionBackend, ComputeDevice, HephaestusError, Result};

use super::validate::{validate_dense_operand, validate_square};
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// LU decomposition result: device-resident packed factors with host-side
/// decomposition for solve/inv/det.
pub struct GpuLuDecomposition {
    /// Host-side leto-ops decomposition (owns pivots, sign, factors).
    inner: leto_ops::LuDecomposition<f32>,
    /// Device-resident packed L/U factors (*n* × *n*, row-major).
    factors: WgpuBuffer<f32>,
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
    pub fn factors(&self) -> &WgpuBuffer<f32> {
        &self.factors
    }

    /// Return the permutation pivots.
    #[must_use]
    #[inline]
    pub fn pivots(&self) -> &[usize] {
        self.inner.pivots()
    }

    /// Determinant: sign × Πᵢ Uᵢᵢ via the host-side decomposition.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.inner.det()
    }

    /// Solve **A** · **x** = **rhs** via host-side forward/back substitution.
    pub fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        if rhs.len != self.n {
            return Err(HephaestusError::LengthMismatch {
                host_len: self.n,
                device_len: rhs.len,
            });
        }
        if self.n == 0 {
            return device.upload(&[] as &[f32]);
        }

        let mut rhs_host = vec![0.0f32; self.n];
        device.download(rhs, &mut rhs_host)?;

        let rhs_view = leto::ArrayView::<f32, 1>::new(
            leto::Layout::c_contiguous([self.n]).expect("infallible: valid contiguous layout"),
            &rhs_host,
        );
        let x = self
            .inner
            .solve(&rhs_view)
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("LU solve failed: {e}"),
            })?;

        device.upload(leto::Storage::as_slice(x.storage()))
    }

    /// Compute the inverse **A**⁻¹ via the host-side decomposition.
    pub fn inv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inv = self
            .inner
            .inv()
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("LU inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

// ---------------------------------------------------------------------------
// GEMM uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the trailing GEMM compute kernel.
/// Computes **C -= A · B** where A is (m×k), B is (k×n), C is (m×n).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct GemmMeta {
    /// Shape: [m, n, k, padding].
    shape: [u32; 4],
    /// Row strides: [C row-stride, A row-stride, B row-stride, padding].
    strides: [u32; 4],
    /// Element offsets: [C offset, A offset, B offset, padding].
    offsets: [u32; 4],
}

// SAFETY: GemmMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for GemmMeta {}

// ---------------------------------------------------------------------------
// GEMM kernel  C -= A · B
// ---------------------------------------------------------------------------

/// WGSL source for the trailing GEMM kernel
///
/// ```text
/// C[i,j] -= Σₖ A[i,k] · B[k,j]
/// ```
///
/// where A is L₂₁ (m×k), B is U₁₂ (k×n), and C is the trailing
/// submatrix A₂₂ (m×n).  Uses 16×16 workgroup tiles with shared-memory
/// cooperative loading for both A and B tiles.
fn gemm_shader_source() -> String {
    const TY: &str = "f32";
    const ZERO: &str = "0.0";

    format!(
        r#"struct GemmMeta {{
    shape: vec4<u32>,
    strides: vec4<u32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read_write> a_buf: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> b_buf: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> c_buf: array<{ty}>;
@group(0) @binding(3) var<uniform>             params: GemmMeta;

var<workgroup> tile_a: array<{ty}, 256>;
var<workgroup> tile_b: array<{ty}, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wgid: vec3<u32>
) {{
    let row = gid.y;
    let col = gid.x;
    let m = params.shape.x;
    let n = params.shape.y;
    let k = params.shape.z;
    let c_stride = params.strides.x;
    let a_stride = params.strides.y;
    let b_stride = params.strides.z;
    let c_off = params.offsets.x;
    let a_off = params.offsets.y;
    let b_off = params.offsets.z;

    var sum = {ty}({zero});
    let num_tiles = (k + 15u) / 16u;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {{
        // Load tile of A: A[row, tile*16 + lid.x]
        let a_col = tile * 16u + lid.x;
        if (row < m && a_col < k) {{
            tile_a[lid.y * 16u + lid.x] = a_buf[a_off + row * a_stride + a_col];
        }} else {{
            tile_a[lid.y * 16u + lid.x] = {ty}({zero});
        }}

        // Load tile of B: B[tile*16 + lid.y, col]
        let b_row = tile * 16u + lid.y;
        if (b_row < k && col < n) {{
            tile_b[lid.y * 16u + lid.x] = b_buf[b_off + b_row * b_stride + col];
        }} else {{
            tile_b[lid.y * 16u + lid.x] = {ty}({zero});
        }}

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            sum = sum + tile_a[lid.y * 16u + i] * tile_b[i * 16u + lid.x];
        }}

        workgroupBarrier();
    }}

    // C -= A · B
    let c_idx = c_off + row * c_stride + col;
    if (row < m && col < n) {{
        c_buf[c_idx] = c_buf[c_idx] - sum;
    }}
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct GemmKernel;

pub(crate) struct GemmTrailingUpdate<'a> {
    pub(crate) a_buf: &'a WgpuBuffer<f32>,
    pub(crate) a_offset: usize,
    pub(crate) a_stride: usize,
    pub(crate) a_rows: usize,
    pub(crate) a_cols: usize,
    pub(crate) b_buf: &'a WgpuBuffer<f32>,
    pub(crate) b_offset: usize,
    pub(crate) b_stride: usize,
    pub(crate) b_cols: usize,
    pub(crate) c_buf: &'a WgpuBuffer<f32>,
    pub(crate) c_offset: usize,
    pub(crate) c_stride: usize,
}

/// GPU dispatch for the trailing GEMM:  **C -= A · B**
///
/// A is (m×k) starting at `a_offset` with row stride `a_stride`,
/// B is (k×n) starting at `b_offset` with row stride `b_stride`,
/// C is (m×n) starting at `c_offset` with row stride `c_stride`.
pub(crate) fn gemm_trailing_update(
    device: &WgpuDevice,
    update: GemmTrailingUpdate<'_>,
) -> Result<()> {
    let m = update.a_rows;
    let k = update.a_cols;
    let n = update.b_cols;
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    let meta = GemmMeta {
        shape: [
            u32::try_from(m).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM m {m} exceeds u32"),
            })?,
            u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM n {n} exceeds u32"),
            })?,
            u32::try_from(k).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM k {k} exceeds u32"),
            })?,
            0,
        ],
        strides: [
            u32::try_from(update.c_stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM C row stride {} exceeds u32", update.c_stride),
            })?,
            u32::try_from(update.a_stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM A row stride {} exceeds u32", update.a_stride),
            })?,
            u32::try_from(update.b_stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM B row stride {} exceeds u32", update.b_stride),
            })?,
            0,
        ],
        offsets: [
            u32::try_from(update.c_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM C offset {} exceeds u32", update.c_offset),
            })?,
            u32::try_from(update.a_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM A offset {} exceeds u32", update.a_offset),
            })?,
            u32::try_from(update.b_offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("GEMM B offset {} exceeds u32", update.b_offset),
            })?,
            0,
        ],
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<GemmKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-gemm",
        gemm_shader_source,
    );

    let raw_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<GemmMeta>(1)?)?;
    let meta_buf = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buf);
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-gemm"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: update.a_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: update.b_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: update.c_buf.buffer.as_entire_binding(),
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
            label: Some("hephaestus-gemm"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x = u32::try_from(n.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("GEMM workgroup x {} exceeds u32", n.div_ceil(16)),
        })?;
        let wg_y = u32::try_from(m.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("GEMM workgroup y {} exceeds u32", m.div_ceil(16)),
        })?;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point 1 — host delegation
// ---------------------------------------------------------------------------

/// Compute the LU decomposition with partial pivoting on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For large matrices where the O(2n³/3) trailing update should
/// run on the GPU, prefer [`lu_decompose_blocked`].
///
/// # Errors
///
/// - Non-square input.
/// - Non-finite values in the input.
/// - Singular matrix (exact zero pivot).
pub fn lu_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        let factors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::lu_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([0, 0]).expect("infallible: empty matrix layout"),
            &[],
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("LU decomposition failed: {e}"),
        })?;
        return Ok(GpuLuDecomposition {
            inner,
            factors,
            n: 0,
        });
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    let lu = leto_ops::lu_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("LU decomposition failed: {e}"),
    })?;

    let factors = device.upload(leto::Storage::as_slice(lu.factors().storage()))?;

    Ok(GpuLuDecomposition {
        inner: lu,
        factors,
        n,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing GEMM
// ---------------------------------------------------------------------------

/// Panel block size for the blocked LU algorithm.
///
/// A value of 64 balances CPU panel factorisation cost against GPU GEMM
/// launch overhead.
const LU_BLOCK_SIZE: usize = 64;

/// Blocked LU factorization **P A = L U** with GPU-accelerated trailing-matrix
/// GEMM updates.
///
/// The algorithm processes the matrix in `LU_BLOCK_SIZE × LU_BLOCK_SIZE`
/// panels.  For each panel *k*:
///
/// 1. The diagonal block is factored on the **CPU** via in-place partial
///    pivoting LU (O(2b³/3)).
/// 2. The row permutation is applied to the remaining rows of `A₂₁` and
///    `A₁₂`.
/// 3. The L₂₁ panel (below-diagonal) is solved on the **CPU** via
///    forward-substitution with U₁₁ (O(b²(n−k)/2)).
/// 4. The U₁₂ panel (right-of-diagonal) is solved on the **CPU** via
///    forward-substitution with L₁₁ (O(b²(n−k)/2)).
/// 5. The trailing submatrix is updated on the **GPU** via a dedicated GEMM
///    kernel: `A₂₂ -= L₂₁ · U₁₂` (O(2b(n−k)²)).
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-dense (non-C-contiguous / offset / broadcast) operand: the
///   blocked path bulk-copies the matrix storage on the device.
/// - Non-finite values in the input.
/// - Singular matrix (exact zero pivot).
pub fn lu_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuLuDecomposition> {
    let n = validate_square(&matrix)?;
    validate_dense_operand("LU", &matrix)?;
    if n == 0 {
        let factors = device.alloc_zeroed::<f32>(0)?;
        let inner = leto_ops::lu_decompose(&leto::ArrayView::<f32, 2>::new(
            leto::Layout::c_contiguous([0, 0]).expect("infallible: empty matrix layout"),
            &[],
        ))
        .map_err(|e| HephaestusError::DispatchFailed {
            message: format!("LU decomposition failed: {e}"),
        })?;
        return Ok(GpuLuDecomposition {
            inner,
            factors,
            n: 0,
        });
    }

    // Copy the (dense, validated) operand into a fresh device working buffer
    // on the GPU; the shared loop owns all host bookkeeping.
    let factors = device.clone_device(matrix.buffer, n * n)?;
    let result = blocked_lu(device, factors, n, LU_BLOCK_SIZE.min(n))?;

    let inner = leto_ops::LuDecomposition::from_raw_parts(
        leto::Array2::from_shape_vec([n, n], result.host).expect("valid square factor"),
        result.perm,
        result.sign,
    );

    Ok(GpuLuDecomposition {
        inner,
        factors: result.factors,
        n,
    })
}
