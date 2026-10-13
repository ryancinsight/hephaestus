//! Unblocked Cholesky factor: host delegation with GPU-resident output.

use std::any::TypeId;
use std::sync::OnceLock;

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use super::super::validate::validate_square;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

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
/// [`super::blocked::cholesky_decompose_blocked`] therefore keeps just that diagonal (`n`
/// elements, extracted from panels it already downloads) and leaves `inner`
/// empty, materializing it from the device factor on the first `inv` and
/// caching it thereafter. [`cholesky_decompose`] factors on the host to
/// begin with, so it populates `inner` eagerly and never downloads.
pub struct GpuCholesky {
    /// Host-side leto-ops decomposition, materialized on first host-side
    /// inversion. Pre-populated by the host-delegating entry point.
    pub(crate) inner: OnceLock<leto_ops::CholeskyDecomposition<f32>>,
    /// Diagonal of **L** in factor order, `n` elements.
    ///
    /// Bitwise equal to the device factor's diagonal: both are written from
    /// the same host panel values.
    pub(crate) diagonal: Vec<f32>,
    /// Device-resident lower-triangular factor **L** (*n* × *n*, row-major).
    pub(crate) lower: WgpuBuffer<f32>,
    pub(crate) n: usize,
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
        super::solve::device_solve(device, &self.lower, self.n, rhs)
    }

    /// Whether the host-side factor copy is resident.
    ///
    /// `false` after [`super::blocked::cholesky_decompose_blocked`] until the first
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
/// on the GPU, prefer [`super::blocked::cholesky_decompose_blocked`].
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
pub(crate) const BLOCK_SIZE: usize = 64;

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
pub(crate) fn zero_strict_upper(
    device: &WgpuDevice,
    matrix: &WgpuBuffer<f32>,
    n: usize,
) -> Result<()> {
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

    let meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<TriangleMeta>(1)?)?;

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
