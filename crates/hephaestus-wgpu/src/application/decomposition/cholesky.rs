//! GPU-resident Cholesky decomposition.
//!
//! Computes **A** = **L** **L**ᵀ for symmetric positive-definite matrices.
//!
//! Two entry points are provided:
//!
//! - [`cholesky_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`cholesky_decompose_blocked`] — blocked algorithm where panel
//!   factorization runs on the CPU but the O(n³) trailing SYRK update
//!   (`A₂₂ -= L₂₁ L₂₁ᵀ`) runs on the GPU via a dedicated compute kernel.

use std::any::TypeId;
use std::sync::OnceLock;

use hephaestus_core::{ComputeDevice, HephaestusError, Result, factor_cholesky_panel};
use leto::Layout;

use super::region::{
    MatrixRegion, download_matrix_region_compact_into, write_matrix_region_compact_reusable,
};
use super::validate::{validate_dense_operand, validate_square};
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::uniform_guard;

// ---------------------------------------------------------------------------
// SYRK uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the SYRK compute kernel, matching the WGSL `SyrkMeta`
/// struct.  The matrix layout fields describe the **trailing matrix** C;
/// `panel_cols` is the rank-k dimension of the panel L₂₁.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct SyrkMeta {
    /// Shape of the trailing matrix: `[rows, cols]`.
    shape: [u32; 2],
    /// Row-major strides of the trailing matrix.
    strides: [u32; 2],
    /// Element offset into the trailing-matrix buffer.
    offset: u32,
    /// Rank-k dimension (number of columns in the panel).
    panel_cols: u32,
    /// Element offset in the panel buffer where the active panel begins.
    panel_offset: u32,
    /// Row stride of the panel buffer.
    panel_stride: u32,
}

// ---------------------------------------------------------------------------
// SYRK kernel
// ---------------------------------------------------------------------------

/// WGSL source for the rank-k symmetric update
///
/// ```text
/// C[i,j] -= Σₖ B[i,k] · B[j,k]
/// ```
///
/// where `C` is the trailing matrix and `B` is the panel (`rows × k`).
/// Each workgroup processes a 16×16 tile of `C` using shared-memory
/// cooperative loading of panel rows, identical in spirit to the tiled matmul
/// kernel but specialised for the symmetric case.
fn syrk_shader_source() -> String {
    // WGSL f32 literal is always "f32(...)".
    const TY: &str = "f32";
    const ZERO: &str = "0.0";

    format!(
        r#"struct SyrkMeta {{
    shape: vec2<u32>,
    strides: vec2<u32>,
    offset: u32,
    panel_cols: u32,
    panel_offset: u32,
    panel_stride: u32,
}}

@group(0) @binding(0) var<storage, read_write> trail:  array<{ty}>;
@group(0) @binding(1) var<uniform>             syrk_meta: SyrkMeta;

var<workgroup> panel_row: array<array<{ty}, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)  gid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:  vec3<u32>,
    @builtin(workgroup_id)          wid:  vec3<u32>,
) {{
    let col = gid.x;
    let row = gid.y;
    let local_col = lid.x;
    let local_row = lid.y;

    let rows = syrk_meta.shape.x;
    let cols = syrk_meta.shape.y;
    let k    = syrk_meta.panel_cols;
    let stride_row = syrk_meta.strides.x;
    let stride_col = syrk_meta.strides.y;
    let off        = syrk_meta.offset;

    var sum = {ty}({zero});
    let num_tiles = (k + 15u) / 16u;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {{
        // Load `panel[row, tile*16 + local_col]` into shared memory.
        let panel_col = tile * 16u + local_col;
        if (row < rows && panel_col < k) {{
            panel_row[local_row][local_col] = trail[syrk_meta.panel_offset + row * syrk_meta.panel_stride + panel_col];
        }} else {{
            panel_row[local_row][local_col] = {ty}({zero});
        }}
        workgroupBarrier();

        // Each thread computes its own output element.
        // Re-load `panel[row, t*16 + i]` for each i in the tile from shared
        // memory, and load `panel[col, t*16 + i]` directly from global memory.
        if (row < rows && col < cols && col <= row) {{
            for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
                let ki = tile * 16u + i;
                if (ki < k) {{
                    let a_val = panel_row[local_row][i];
                    let b_val = trail[syrk_meta.panel_offset + col * syrk_meta.panel_stride + ki];
                    sum = sum + a_val * b_val;
                }}
            }}
        }}

        workgroupBarrier();
    }}

    // Write back: C[row, col] -= sum
    if (row < rows && col < cols && col <= row) {{
        let c_off = off + row * stride_row + col * stride_col;
        trail[c_off] = trail[c_off] - sum;
    }}
}}
"#,
        ty = TY,
        zero = ZERO,
    )
}

struct SyrkKernel;

/// GPU dispatch for the rank-k symmetric trailing-matrix update
///
/// ```text
/// trail[row, col] -= Σₖ panel[row, k] · panel[col, k]
/// ```
///
/// Only the **lower triangle** (`col <= row`) of the trailing matrix is
/// updated, which is sufficient for the blocked Cholesky loop.
fn syrk_trailing_update(
    device: &WgpuDevice,
    encoder: &mut wgpu::CommandEncoder,
    trail: &WgpuBuffer<f32>,
    trail_layout: &Layout<2>,
    panel_cols: usize,
    panel_offset: usize,
    panel_stride: usize,
) -> Result<()> {
    let [rows, cols] = trail_layout.shape();
    if rows == 0 || cols == 0 || panel_cols == 0 {
        return Ok(());
    }

    let meta = SyrkMeta {
        shape: [
            u32::try_from(rows).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK row count {rows} exceeds u32"),
            })?,
            u32::try_from(cols).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK col count {cols} exceeds u32"),
            })?,
        ],
        strides: [
            u32::try_from(trail_layout.strides()[0]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK row stride {} exceeds u32", trail_layout.strides()[0]),
                }
            })?,
            u32::try_from(trail_layout.strides()[1]).map_err(|_| {
                HephaestusError::DispatchFailed {
                    message: format!("SYRK col stride {} exceeds u32", trail_layout.strides()[1]),
                }
            })?,
        ],
        offset: u32::try_from(trail_layout.offset()).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("SYRK offset {} exceeds u32", trail_layout.offset()),
            }
        })?,
        panel_cols: u32::try_from(panel_cols).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("SYRK panel cols {panel_cols} exceeds u32"),
        })?,
        panel_offset: u32::try_from(panel_offset).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("SYRK panel offset {panel_offset} exceeds u32"),
        })?,
        panel_stride: u32::try_from(panel_stride).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("SYRK panel stride {panel_stride} exceeds u32"),
        })?,
    };

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<SyrkKernel>(), TypeId::of::<f32>(), 16),
        "hephaestus-syrk",
        syrk_shader_source,
    );

    let raw_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<SyrkMeta>(1)?)?;
    let meta_buf = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buf);
    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-syrk"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: trail.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-syrk"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x =
            u32::try_from(cols.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK workgroup x {} exceeds u32", cols.div_ceil(16)),
            })?;
        let wg_y =
            u32::try_from(rows.div_ceil(16)).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("SYRK workgroup y {} exceeds u32", rows.div_ceil(16)),
            })?;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Lower-triangular Cholesky factor on the device, solving on the device and
/// inverting through a host-side decomposition without re-factorization.
///
/// # Host residency
///
/// [`solve`](Self::solve) substitutes on the device against the resident
/// factor and [`det`](Self::det) reads only the factor's diagonal; the
/// `n × n` host factor serves [`inv`](Self::inv) alone.
/// [`cholesky_decompose_blocked`] therefore keeps just that diagonal (`n`
/// elements, extracted from panels it already downloads) and leaves `inner`
/// empty, materializing it from the device factor on the first `inv` and
/// caching it thereafter. [`cholesky_decompose`] factors on the host to
/// begin with, so it populates `inner` eagerly and never downloads.
pub struct GpuCholesky {
    /// Host-side leto-ops decomposition, materialized on first host-side
    /// inversion. Pre-populated by the host-delegating entry point.
    inner: OnceLock<leto_ops::CholeskyDecomposition<f32>>,
    /// Diagonal of **L** in factor order, `n` elements.
    ///
    /// Bitwise equal to the device factor's diagonal: both are written from
    /// the same host panel values.
    diagonal: Vec<f32>,
    /// Device-resident lower-triangular factor **L** (*n* × *n*, row-major).
    lower: WgpuBuffer<f32>,
    n: usize,
}

impl GpuCholesky {
    /// The host factor for [`inv`](Self::inv), downloading and caching it on first use.
    ///
    /// The device factor is authoritative and complete once
    /// [`zero_strict_upper`] has run: the per-panel scatters write every cell
    /// with `row >= blockstart(col)` from the same host panel values the old
    /// eager array held, and that pass clears the rest. Downloading it
    /// therefore reconstructs exactly the decomposition the eager path built.
    ///
    /// # Errors
    ///
    /// Propagates the device readback failure.
    fn host_factor(&self, device: &WgpuDevice) -> Result<&leto_ops::CholeskyDecomposition<f32>> {
        if let Some(factor) = self.inner.get() {
            return Ok(factor);
        }

        let mut data = vec![0.0f32; self.n * self.n];
        device.download(&self.lower, &mut data)?;
        let factor = leto_ops::CholeskyDecomposition::from_raw_parts(
            leto::Array2::from_shape_vec([self.n, self.n], data)
                .expect("invariant: n*n elements form an n x n factor"),
        );
        // A racing initializer computed the same value from the same
        // immutable buffer, so either winner is correct.
        let _ = self.inner.set(factor);
        Ok(self
            .inner
            .get()
            .expect("invariant: set unconditionally above"))
    }

    /// Matrix dimension *n*.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Borrow the lower-triangular factor buffer on the device.
    #[must_use]
    #[inline]
    pub fn lower(&self) -> &WgpuBuffer<f32> {
        &self.lower
    }

    /// Consume and return the lower-triangular factor buffer.
    #[must_use]
    #[inline]
    pub fn into_lower(self) -> WgpuBuffer<f32> {
        self.lower
    }

    /// Determinant det(**A**) = Πᵢ Lᵢᵢ² from the retained factor diagonal.
    ///
    /// Multiplies in ascending `i` from `1.0`, the same order and the same
    /// operations as [`leto_ops::CholeskyDecomposition::det`], so the result
    /// is bitwise equal to the host reference's on an identical factor. No
    /// device dispatch and no readback: the diagonal is captured while the
    /// factorization already has each panel on the host.
    #[must_use]
    #[inline]
    pub fn det(&self) -> f32 {
        self.diagonal
            .iter()
            .fold(1.0f32, |product, &diagonal| product * (diagonal * diagonal))
    }

    /// Solve **A** · **x** = **rhs** on the device against the resident factor.
    ///
    /// Blocked forward substitution on **L** then backward substitution on
    /// **L**ᵀ, blocked by 256 rows; neither the factor nor the right-hand
    /// side crosses the host, and the host factor stays unmaterialized.
    ///
    /// # Errors
    ///
    /// - `LengthMismatch` when `rhs.len != n`.
    /// - `DispatchFailed` when a dimension or workgroup count exceeds `u32`.
    pub fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        device_solve(device, &self.lower, self.n, rhs)
    }

    /// Whether the host-side factor copy is resident.
    ///
    /// `false` after [`cholesky_decompose_blocked`] until the first
    /// [`inv`](Self::inv); `true` from construction for
    /// [`cholesky_decompose`]. [`solve`](Self::solve) never materializes it.
    #[must_use]
    #[inline]
    pub fn host_factor_materialized(&self) -> bool {
        self.inner.get().is_some()
    }

    /// Compute the inverse **A**⁻¹ via the host-side decomposition.
    ///
    /// On the blocked entry point's first call this materializes the host
    /// factor (see [`GpuCholesky`]); later calls reuse it. The `n` device
    /// solves that would retire this download are their own item.
    ///
    /// # Errors
    ///
    /// The device readback failure, or the host inversion failure.
    pub fn inv(&self, device: &WgpuDevice) -> Result<WgpuBuffer<f32>> {
        if self.n == 0 {
            return device.alloc_zeroed::<f32>(0);
        }
        let inv = self
            .host_factor(device)?
            .inv()
            .map_err(|e| HephaestusError::DispatchFailed {
                message: format!("Cholesky inverse failed: {e}"),
            })?;
        device.upload(leto::Storage::as_slice(inv.storage()))
    }
}

// ---------------------------------------------------------------------------
// Entry point 1 — host delegation (unchanged)
// ---------------------------------------------------------------------------

/// Compute the Cholesky factorization **A** = **L** **L**ᵀ on the GPU.
///
/// The entire factorization (panel + trailing) is delegated to the host via
/// [`leto_ops`].  The result is stored on the device for downstream GPU
/// consumers.  For large matrices where the O(n³) trailing update should run
/// on the GPU, prefer [`cholesky_decompose_blocked`].
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    if n == 0 {
        // An empty factor has an empty diagonal, so `det` yields the empty
        // product `1.0` and `solve`/`inv` short-circuit before reaching
        // `inner` — no host decomposition needs to exist.
        return Ok(GpuCholesky {
            inner: OnceLock::new(),
            diagonal: Vec::new(),
            lower: device.alloc_zeroed::<f32>(0)?,
            n: 0,
        });
    }

    // Download input to host.
    let mut host_data = vec![0.0f32; matrix.buffer.len];
    device.download(matrix.buffer, &mut host_data)?;

    // Create a leto ArrayView over the downloaded data.
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);

    // Factor on CPU using leto-ops.
    let chol =
        leto_ops::cholesky_decompose(&view).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Cholesky decomposition failed: {e}"),
        })?;

    // Upload the lower-triangular factor to the device.
    let factor = leto::Storage::as_slice(chol.lower().storage());
    let diagonal = (0..n).map(|k| factor[k * n + k]).collect();
    let lower = device.upload(factor)?;

    // This path factored on the host, so the decomposition already exists:
    // publish it rather than making `solve`/`inv` download it back.
    let inner = OnceLock::new();
    let _ = inner.set(chol);

    Ok(GpuCholesky {
        inner,
        diagonal,
        lower,
        n,
    })
}

// ---------------------------------------------------------------------------
// Entry point 2 — blocked with GPU trailing SYRK
// ---------------------------------------------------------------------------

/// Panel block size for the blocked Cholesky algorithm.
///
/// A value of 64 balances CPU panel factorisation cost against GPU SYRK
/// launch overhead.  For matrices smaller than `BLOCK_SIZE` the algorithm
/// degrades gracefully to a single CPU panel pass.
const BLOCK_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Strict-upper triangular zero
// ---------------------------------------------------------------------------

/// Packed metadata for the triangular-zero kernel, matching the WGSL
/// `TriangleMeta` struct.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct TriangleMeta {
    /// Order of the square matrix.
    n: u32,
    /// Element count, carried so the shader needs no multiply that could
    /// overflow silently at `u32` width.
    total: u32,
}

struct StrictUpperZeroKernel;

/// WGSL source for the in-place strict-upper triangular zero.
///
/// Each invocation owns one flat index of the row-major matrix and writes only
/// when that index lies strictly above the diagonal, so the factor itself is
/// never read back or rewritten.
fn strict_upper_zero_shader_source() -> String {
    r#"struct TriangleMeta {
    n: u32,
    total: u32,
}
@group(0) @binding(0) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(1) var<uniform>             params: TriangleMeta;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= params.total) {
        return;
    }
    let r = idx / params.n;
    let c = idx % params.n;
    if (c > r) {
        matrix[idx] = 0.0;
    }
}
"#
    .to_string()
}

/// Zero the strictly-upper triangle of a row-major `n` x `n` device buffer in
/// place.
///
/// The blocked factorisation leaves the strict upper triangle holding the
/// input's values outside the diagonal blocks (the per-panel scatters cover
/// only `row >= blockstart(col)`), while the diagonal blocks are already
/// zeroed there by [`hephaestus_core::factor_cholesky_panel`]. This pass
/// finishes the factor on the device, replacing an `n^2` host upload whose
/// only remaining effect was that zeroing.
///
/// # Errors
///
/// - `LengthMismatch` when `matrix.len != n * n`.
/// - `DispatchFailed` when the element count exceeds the shader's `u32` index
///   width or the workgroup count exceeds `u32::MAX`.
fn zero_strict_upper(device: &WgpuDevice, matrix: &WgpuBuffer<f32>, n: usize) -> Result<()> {
    let total = n
        .checked_mul(n)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("cholesky dimension {n} overflows an element count"),
        })?;
    if matrix.len != total {
        return Err(HephaestusError::LengthMismatch {
            host_len: total,
            device_len: matrix.len,
        });
    }
    if total == 0 {
        return Ok(());
    }

    let meta = TriangleMeta {
        n: u32::try_from(n).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("cholesky dimension {n} exceeds u32"),
        })?,
        total: u32::try_from(total).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("cholesky element count {total} exceeds u32"),
        })?,
    };

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<StrictUpperZeroKernel>(),
            TypeId::of::<f32>(),
            256,
        ),
        "hephaestus-cholesky-strict-upper-zero",
        strict_upper_zero_shader_source,
    );

    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<TriangleMeta>(1)?)?;
    let meta_buf = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-cholesky-strict-upper-zero-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let workgroups =
        u32::try_from(total.div_ceil(256)).map_err(|_| HephaestusError::DispatchFailed {
            message: format!(
                "cholesky triangular-zero workgroup count {} exceeds u32::MAX",
                total.div_ceil(256)
            ),
        })?;

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cholesky-strict-upper-zero"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-cholesky-strict-upper-zero-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

// ---------------------------------------------------------------------------
// Device-side triangular solves against the resident factor
// ---------------------------------------------------------------------------

/// Rows per triangular-solve block: the workgroup width of the block kernel,
/// so one workgroup owns one diagonal block of **L** with a lane per row.
const SOLVE_BLOCK_WIDTH: u32 = 256;
/// [`SOLVE_BLOCK_WIDTH`] as an index count for host-side blocking arithmetic.
const SOLVE_BLOCK: usize = SOLVE_BLOCK_WIDTH as usize;

/// Packed metadata for one triangular-solve dispatch, matching the WGSL
/// `SolveMeta` struct.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct SolveMeta {
    /// Matrix dimension *n* of the *n* × *n* factor.
    n: u32,
    /// First row of the diagonal block this dispatch works on.
    block_start: u32,
    /// Rows in the block: `SOLVE_BLOCK`, or the ragged tail for the last one.
    block_len: u32,
    /// Rows the trailing update touches: everything after the block going
    /// forward, everything before it going backward. Unused by the block
    /// solve, which pads the struct to its 16-byte uniform stride.
    update_rows: u32,
}

/// Which triangular system a solve pass works on.
///
/// Forward substitution solves **L** · **y** = **b** from the first block
/// down; backward substitution solves **L**ᵀ · **x** = **y** from the last
/// block up, reading **L** transposed so no second factor is materialized.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Substitution {
    Forward,
    Backward,
}

impl Substitution {
    /// WGSL expression for the factor element coupling `row` to column
    /// `col` of the current system: `L[row, col]` forward, `L[col, row]`
    /// backward.
    fn coefficient(self) -> &'static str {
        match self {
            Self::Forward => "lower[row * params.n + col]",
            Self::Backward => "lower[col * params.n + row]",
        }
    }
}

struct ForwardBlockSolveKernel;
struct ForwardTrailingUpdateKernel;
struct BackwardBlockSolveKernel;
struct BackwardTrailingUpdateKernel;

/// WGSL source for the diagonal-block solve.
///
/// One workgroup solves the block's `block_len × block_len` triangle against
/// its slice of `rhs` in place: lane `i` owns local row `i`, and each step
/// `c` finalizes row `c` (one division by the diagonal), publishes it through
/// workgroup memory, then every later row subtracts its coupling to it.
/// Forward substitution walks `c` upward and couples rows below the pivot;
/// backward substitution walks it downward and couples rows above. The
/// barrier sits in uniform control flow (`block_len` is a uniform), and each
/// workgroup slot is written exactly once per solve, so one barrier per step
/// orders the publish before its readers.
fn block_solve_shader_source(substitution: Substitution) -> String {
    let (pivot, coupled) = match substitution {
        Substitution::Forward => ("step", "i > c"),
        Substitution::Backward => ("params.block_len - 1u - step", "i < c"),
    };
    let coefficient = substitution.coefficient();
    format!(
        r#"struct SolveMeta {{
    n: u32,
    block_start: u32,
    block_len: u32,
    update_rows: u32,
}}
@group(0) @binding(0) var<storage, read>       lower: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs:   array<f32>;
@group(0) @binding(2) var<uniform>             params: SolveMeta;

var<workgroup> solved: array<f32, {SOLVE_BLOCK}u>;

@compute @workgroup_size({SOLVE_BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let i = lid.x;
    let row = params.block_start + i;
    let owns_row = i < params.block_len;
    var acc = 0.0;
    if (owns_row) {{
        acc = rhs[row];
    }}
    for (var step = 0u; step < params.block_len; step = step + 1u) {{
        let c = {pivot};
        let col = params.block_start + c;
        if (i == c) {{
            let value = acc / {coefficient};
            solved[c] = value;
            rhs[row] = value;
        }}
        workgroupBarrier();
        if (owns_row && {coupled}) {{
            acc = acc - {coefficient} * solved[c];
        }}
    }}
}}
"#
    )
}

/// WGSL source for the trailing update after a block solve.
///
/// One invocation per row outside the solved block subtracts that row's
/// coupling to every just-solved entry: `rhs[row] -= Σ_c coefficient(row, c)
/// · rhs[c]` over the block's columns. Forward substitution updates the rows
/// after the block; backward substitution updates the rows before it. The
/// block's own entries are only read, and every written row lies outside it,
/// so the dispatch is race-free without atomics.
fn trailing_update_shader_source(substitution: Substitution) -> String {
    let first_row = match substitution {
        Substitution::Forward => "params.block_start + params.block_len",
        Substitution::Backward => "0u",
    };
    let coefficient = substitution.coefficient();
    format!(
        r#"struct SolveMeta {{
    n: u32,
    block_start: u32,
    block_len: u32,
    update_rows: u32,
}}
@group(0) @binding(0) var<storage, read>       lower: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs:   array<f32>;
@group(0) @binding(2) var<uniform>             params: SolveMeta;

@compute @workgroup_size({SOLVE_BLOCK})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {{
    if (gid.x >= params.update_rows) {{
        return;
    }}
    let row = {first_row} + gid.x;
    var acc = rhs[row];
    for (var c = 0u; c < params.block_len; c = c + 1u) {{
        let col = params.block_start + c;
        acc = acc - {coefficient} * rhs[col];
    }}
    rhs[row] = acc;
}}
"#
    )
}

fn forward_block_solve_shader_source() -> String {
    block_solve_shader_source(Substitution::Forward)
}

fn forward_trailing_update_shader_source() -> String {
    trailing_update_shader_source(Substitution::Forward)
}

fn backward_block_solve_shader_source() -> String {
    block_solve_shader_source(Substitution::Backward)
}

fn backward_trailing_update_shader_source() -> String {
    trailing_update_shader_source(Substitution::Backward)
}

/// The four kernels of the blocked solve, in the order a solve issues them.
#[derive(Clone, Copy)]
enum SolveKernel {
    ForwardBlock,
    ForwardUpdate,
    BackwardBlock,
    BackwardUpdate,
}

/// One dispatch of the blocked solve: its kernel, metadata, and width.
struct SolveDispatch {
    kernel: SolveKernel,
    meta: SolveMeta,
    workgroups: u32,
}

/// Solve **A** · **x** = **rhs** on the device against the resident factor.
///
/// Forward substitution on **L** then backward substitution on **L**ᵀ, each
/// blocked by `SOLVE_BLOCK` rows: per block one workgroup solves the diagonal
/// triangle in place, then a row-parallel dispatch applies the rank-`block`
/// update to the rows still unsolved. That is `2 · ⌈n / SOLVE_BLOCK⌉` block
/// solves plus their trailing updates per solve, all recorded in one compute
/// pass so WebGPU's per-dispatch ordering serializes the chain. The right-hand
/// side is copied device-to-device into the solution buffer first, so the
/// caller's buffer is untouched and nothing crosses the host.
///
/// # Errors
///
/// - `LengthMismatch` when `rhs.len != n`.
/// - `DispatchFailed` when a dimension or workgroup count exceeds `u32`.
fn device_solve(
    device: &WgpuDevice,
    lower: &WgpuBuffer<f32>,
    n: usize,
    rhs: &WgpuBuffer<f32>,
) -> Result<WgpuBuffer<f32>> {
    if rhs.len != n {
        return Err(HephaestusError::LengthMismatch {
            host_len: n,
            device_len: rhs.len,
        });
    }
    let solution = device.alloc_zeroed::<f32>(n)?;
    if n == 0 {
        return Ok(solution);
    }
    let to_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("cholesky solve {what} {value} exceeds u32"),
        })
    };
    let n_u32 = to_u32(n, "dimension")?;

    let pipeline_key = |kernel: TypeId| (kernel, TypeId::of::<f32>(), SOLVE_BLOCK_WIDTH);
    let forward_block = cached_pipeline(
        device,
        pipeline_key(TypeId::of::<ForwardBlockSolveKernel>()),
        "hephaestus-cholesky-forward-block-solve",
        forward_block_solve_shader_source,
    );
    let forward_update = cached_pipeline(
        device,
        pipeline_key(TypeId::of::<ForwardTrailingUpdateKernel>()),
        "hephaestus-cholesky-forward-trailing-update",
        forward_trailing_update_shader_source,
    );
    let backward_block = cached_pipeline(
        device,
        pipeline_key(TypeId::of::<BackwardBlockSolveKernel>()),
        "hephaestus-cholesky-backward-block-solve",
        backward_block_solve_shader_source,
    );
    let backward_update = cached_pipeline(
        device,
        pipeline_key(TypeId::of::<BackwardTrailingUpdateKernel>()),
        "hephaestus-cholesky-backward-trailing-update",
        backward_trailing_update_shader_source,
    );

    let block_count = n.div_ceil(SOLVE_BLOCK);
    let mut dispatches = Vec::with_capacity(4 * block_count);
    let dispatch =
        |kernel: SolveKernel, block_start: usize, block_len: usize, update_rows: usize| {
            Ok::<_, HephaestusError>(SolveDispatch {
                kernel,
                meta: SolveMeta {
                    n: n_u32,
                    block_start: to_u32(block_start, "block start")?,
                    block_len: to_u32(block_len, "block length")?,
                    update_rows: to_u32(update_rows, "update rows")?,
                },
                // A block solve is exactly one workgroup; `update_rows` is zero
                // there and the block kernel never reads it.
                workgroups: to_u32(update_rows.div_ceil(SOLVE_BLOCK).max(1), "workgroup count")?,
            })
        };
    for k in 0..block_count {
        let block_start = k * SOLVE_BLOCK;
        let block_len = SOLVE_BLOCK.min(n - block_start);
        let remaining = n - block_start - block_len;
        dispatches.push(dispatch(
            SolveKernel::ForwardBlock,
            block_start,
            block_len,
            0,
        )?);
        if remaining > 0 {
            dispatches.push(dispatch(
                SolveKernel::ForwardUpdate,
                block_start,
                block_len,
                remaining,
            )?);
        }
    }
    for k in (0..block_count).rev() {
        let block_start = k * SOLVE_BLOCK;
        let block_len = SOLVE_BLOCK.min(n - block_start);
        dispatches.push(dispatch(
            SolveKernel::BackwardBlock,
            block_start,
            block_len,
            0,
        )?);
        if block_start > 0 {
            dispatches.push(dispatch(
                SolveKernel::BackwardUpdate,
                block_start,
                block_len,
                block_start,
            )?);
        }
    }

    let pipeline_for = |kernel: SolveKernel| match kernel {
        SolveKernel::ForwardBlock => &forward_block,
        SolveKernel::ForwardUpdate => &forward_update,
        SolveKernel::BackwardBlock => &backward_block,
        SolveKernel::BackwardUpdate => &backward_update,
    };
    let meta_size = WgpuDevice::byte_size::<SolveMeta>(1)?;
    let mut metas = Vec::with_capacity(dispatches.len());
    let mut bind_groups = Vec::with_capacity(dispatches.len());
    for dispatch in &dispatches {
        let raw_meta = device.get_uniform_buffer(meta_size)?;
        let meta_buf = uniform_guard(device.clone(), raw_meta);
        device
            .queue()
            .write_buffer(&meta_buf, 0, eunomia::layout::bytes_of(&dispatch.meta));
        bind_groups.push(
            device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hephaestus-cholesky-solve-bind-group"),
                    layout: &pipeline_for(dispatch.kernel).get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: lower.raw().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: solution.raw().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: meta_buf.as_entire_binding(),
                        },
                    ],
                }),
        );
        metas.push(meta_buf);
    }

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cholesky-solve"),
        });
    encoder.copy_buffer_to_buffer(
        rhs.raw(),
        0,
        solution.raw(),
        0,
        WgpuDevice::byte_size::<f32>(n)?,
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-cholesky-solve-pass"),
            timestamp_writes: None,
        });
        for (dispatch, bind_group) in dispatches.iter().zip(&bind_groups) {
            pass.set_pipeline(pipeline_for(dispatch.kernel));
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dispatch.workgroups, 1, 1);
        }
    }
    device.queue().submit(Some(encoder.finish()));
    // The uniform guards return their buffers to the pool on drop; the
    // submitted command buffer already holds its own references.
    drop(metas);

    Ok(solution)
}

/// Blocked Cholesky factorization **A** = **L** **L**ᵀ with GPU-accelerated
/// trailing-matrix SYRK updates.
///
/// The algorithm processes the matrix in `BLOCK_SIZE × BLOCK_SIZE` panels.
/// For each panel *k*:
///
/// 1. The diagonal block is factored on the **CPU** via [`leto_ops`]
///    (O(b³/3)).
/// 2. The off-diagonal panel is solved on the **CPU** via triangular solve
///    (O(b²·(n−k)/2)).
/// 3. The trailing submatrix is updated on the **GPU** via a dedicated SYRK
///    kernel: `A₂₂ -= L₂₁ · L₂₁ᵀ` (O(b·(n−k)²/2)).
///
/// The SYRK trailing update is the dominant cost for large *n* and is the
/// reason this entry point exists: unlike [`cholesky_decompose`] which
/// delegates *all* O(n³) work to the CPU, this function offloads the
/// rank-k update to the GPU compute pipeline.
///
/// # Block-size tuning
///
/// `BLOCK_SIZE` is currently a compile-time constant.  A future refinement
/// could auto-tune based on the device's preferred workgroup size and
/// available shared memory.
///
/// # Errors
///
/// - Non-square matrix.
/// - Non-dense (non-C-contiguous / offset / broadcast) operand: the
///   blocked path bulk-copies the matrix storage on the device.
/// - Non-finite values in the input.
/// - Matrix is not positive-definite.
pub fn cholesky_decompose_blocked(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<GpuCholesky> {
    let n = validate_square(&matrix)?;
    validate_dense_operand("cholesky", &matrix)?;
    if n == 0 {
        return Ok(GpuCholesky {
            inner: OnceLock::new(),
            diagonal: Vec::new(),
            lower: device.alloc_zeroed::<f32>(0)?,
            n: 0,
        });
    }

    // Allocate device-resident buffer and copy matrix.buffer into it on the GPU
    let lower_buf = device.alloc_uninitialized::<f32>(n * n)?;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cholesky-copy"),
        });
    // Raw whole-matrix copy: sound only for dense C-contiguous
    // zero-offset operands, enforced by `validate_dense_operand` at the
    // entry point (a strided/offset/broadcast view would copy the wrong
    // elements or exceed the operand's storage extent).
    encoder.copy_buffer_to_buffer(
        &matrix.buffer.buffer,
        0,
        &lower_buf.buffer,
        0,
        WgpuDevice::byte_size::<f32>(n * n)?,
    );
    device.queue().submit(Some(encoder.finish()));

    let block_size = BLOCK_SIZE.min(n);

    // Pre-allocate a reusable compact buffer for the maximum panel region:
    // panel_rows = n - k <= n, cols = b <= block_size  =>  max n * block_size elements.
    let panel_compact_buf = device.alloc_uninitialized::<f32>(n * block_size)?;

    // Only the factor's diagonal is retained. `det` reads nothing else, and
    // `solve`/`inv` reconstruct the full factor from `lower_buf` on demand —
    // so the superseded `n * n` host array, which this loop scattered every
    // panel column into, is never allocated.
    let mut diagonal = vec![0.0f32; n];

    // Per-panel host scratch, allocated once and resized by the panel download
    // each iteration instead of allocating a fresh `Vec` per panel.
    let mut panel: Vec<f32> = Vec::with_capacity(n * block_size);

    for k in (0..n).step_by(block_size) {
        let b = block_size.min(n - k);
        let panel_rows = n - k;

        // ── Step 1: Download active panel region to host ──
        let panel_region = MatrixRegion {
            stride: n,
            row_start: k,
            col_start: k,
            rows: panel_rows,
            cols: b,
        };
        download_matrix_region_compact_into(
            device,
            &lower_buf,
            &panel_compact_buf,
            panel_region,
            &mut panel,
        )?;

        let trail_rows = n - k - b;
        // Factor this panel on the host (diagonal-block Cholesky + off-diagonal
        // triangular solve) — the backend-neutral shared computation.
        factor_cholesky_panel(&mut panel, b, trail_rows)?;

        // Retain this panel's contribution to the factor diagonal. The panel
        // is `panel_rows × b` compact, so global cell `(k + j, k + j)` sits at
        // panel row `j`, column `j`.
        for j in 0..b {
            diagonal[k + j] = panel[j * b + j];
        }

        // Upload the entire updated active panel (diagonal + off-diagonal) back to device buffer
        write_matrix_region_compact_reusable(
            device,
            &lower_buf,
            &panel_compact_buf,
            &panel,
            panel_region,
        )?;

        if trail_rows == 0 {
            continue;
        }

        // ── Step 3: trailing SYRK update on GPU ──
        let trail_layout = leto::Layout::try_new(
            [trail_rows, trail_rows],
            [n as isize, 1],
            (k + b) * n + (k + b),
        )
        .expect("invariant: submatrix layout derives from a validated parent");

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cholesky-syrk-update"),
            });

        syrk_trailing_update(
            device,
            &mut encoder,
            &lower_buf,
            &trail_layout,
            b,
            (k + b) * n + k,
            n,
        )?;

        device.queue().submit(Some(encoder.finish()));
    }

    // Finish the factor on the device. The per-panel scatters already wrote
    // every cell with `row >= blockstart(col)`, and the panel factorisation
    // zeroes each diagonal block's strict upper, so the only cells still
    // holding input values are strictly above the diagonal outside those
    // blocks — exactly what this pass clears. The superseded
    // `write_buffer(&lower_buf, &host)` uploaded the whole `n^2` matrix to
    // achieve the same thing, re-sending the lower triangle the device had
    // already computed.
    zero_strict_upper(device, &lower_buf, n)?;

    // `inner` stays empty: with the factor complete on the device and its
    // diagonal retained, only a host-side substitution needs the `n * n`
    // array, and `host_factor` downloads it then.
    Ok(GpuCholesky {
        inner: OnceLock::new(),
        diagonal,
        lower: lower_buf,
        n,
    })
}
