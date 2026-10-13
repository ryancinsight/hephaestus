//! Blocked panel QR factorization.

use super::accumulate::{QAccumulateKernel, QMeta, QReflector, q_accumulate_shader_source};
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::map_layout_err;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use std::any::TypeId;

/// QR decomposition result: device-resident R factor with host-side
/// decomposition for solve_least_squares.
pub struct GpuQrDecomposition {
    /// Host-side leto-ops decomposition (owns packed/heads/betas).
    pub(crate) inner: leto_ops::QrDecomposition<f32>,
    /// Device-resident upper-triangular factor **R** (*m* × *n*, row-major).
    pub(crate) r: WgpuBuffer<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
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
    /// Both entry points leave the same object here: [`qr_decompose`](super::api::qr_decompose)
    /// uploads the host factor's **R**, and
    /// [`qr_decompose_blocked`](super::api::qr_decompose_blocked) computes it in
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
