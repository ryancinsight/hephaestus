//! Fused map-reduction operations and fixed-buffer prepared dispatch.

use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DialectScalar, HephaestusError, IdentityToken,
    OpIdentity, Result, Wgsl,
};
use leto::Layout;

use crate::application::elementwise::SqrtOp;
use crate::application::elementwise::unary::unary_pipeline;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::reduction::{MaxOp, PreparedReduction, SumOp, prepare_reduction};
use crate::application::strided::{
    StridedMeta, StridedOperand, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::UniformBufferGuard;

struct MapReductionKernel<Op>(PhantomData<Op>);

mod shader;

trait MapReductionOp: Copy + Send + Sync + 'static {
    type ReduceOp: CombineExpr<Wgsl>;
    const WGSL_MAP_EXPR: &'static str;
}

#[derive(Clone, Copy, Debug, Default)]
struct TraceOp;

#[derive(Clone, Copy, Debug, Default)]
struct NormL1Op;

#[derive(Clone, Copy, Debug, Default)]
struct DotOp;

#[derive(Clone, Copy, Debug, Default)]
struct NormMaxOp;

impl MapReductionOp for TraceOp {
    type ReduceOp = SumOp;
    const WGSL_MAP_EXPR: &'static str = "lhs";
}

impl MapReductionOp for NormL1Op {
    type ReduceOp = SumOp;
    const WGSL_MAP_EXPR: &'static str = "abs(lhs)";
}

impl MapReductionOp for DotOp {
    type ReduceOp = SumOp;
    const WGSL_MAP_EXPR: &'static str = "lhs * rhs";
}

impl MapReductionOp for NormMaxOp {
    type ReduceOp = MaxOp;
    const WGSL_MAP_EXPR: &'static str = "abs(lhs)";
}

/// WGPU scalar whose shader type supports the real-valued square root needed
/// to finish an L2 / Frobenius norm.
pub trait L2NormScalar:
    DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>
{
}

impl L2NormScalar for f32 {}

struct PreparedMapReduction<T> {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    groups: u32,
    partial: WgpuBuffer<T>,
    reduction: Option<PreparedReduction<T>>,
    _meta_buffer: Option<UniformBufferGuard>,
}

impl<T> PreparedMapReduction<T> {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        if let (Some(pipeline), Some(bind_group)) = (&self.pipeline, &self.bind_group) {
            encode_compute_pass(
                encoder,
                pipeline,
                bind_group,
                self.groups,
                "hephaestus-prepared-map-reduction",
            );
        }
        if let Some(reduction) = &self.reduction {
            reduction.encode(encoder)?;
        }
        Ok(())
    }

    fn dispatch(&self, device: &WgpuDevice, label: &'static str) -> Result<()> {
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        self.encode(&mut encoder)?;
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    fn output(&self) -> &WgpuBuffer<T> {
        self.reduction
            .as_ref()
            .map_or(&self.partial, PreparedReduction::output)
    }

    fn into_output(self) -> WgpuBuffer<T> {
        match self.reduction {
            Some(reduction) => reduction.into_output(),
            None => self.partial,
        }
    }
}

fn prepare_map_reduction<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
) -> Result<PreparedMapReduction<T>>
where
    Op: MapReductionOp,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, Wgsl>,
{
    let len = a.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(PreparedMapReduction {
            pipeline: None,
            bind_group: None,
            groups: 0,
            partial: device.upload(&[T::IDENTITY])?,
            reduction: None,
            _meta_buffer: None,
        });
    }

    let b_layout = b.layout.broadcast(a.layout.shape).map_err(map_layout_err)?;
    a.layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.buffer.len)
        .map_err(map_layout_err)?;

    let width = BlockWidth::DEFAULT;
    let groups = workgroups(len, width)?;
    let partial_len = usize::try_from(groups)
        .expect("invariant: supported WGPU targets have at least 32-bit usize");
    let partial = device.alloc_zeroed::<T>(partial_len)?;
    let meta = StridedMeta {
        shape: pad_shape(a.layout.shape)?,
        a_strides: pad_strides(a.layout.strides)?,
        b_strides: pad_strides(b_layout.strides)?,
        out_strides: [1, 1, 1, 1],
        offsets: [
            to_u32(a.layout.offset, "input offset")?,
            to_u32(b_layout.offset, "input offset")?,
            0,
            to_u32(len, "dispatch size")?,
        ],
    };
    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<MapReductionKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-map-reduction",
        || shader::source::<Op, T>(width),
    );
    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<StridedMeta>(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-map-reduction"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial.buffer.as_entire_binding(),
                },
            ],
        });
    let reduction = if groups > 1 {
        Some(prepare_reduction::<Op::ReduceOp, T>(device, &partial)?)
    } else {
        None
    };

    Ok(PreparedMapReduction {
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups,
        partial,
        reduction,
        _meta_buffer: Some(meta_buffer),
    })
}

fn map_reduction<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    Op: MapReductionOp,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, Wgsl>,
{
    let prepared = prepare_map_reduction::<Op, T, N>(device, a, b)?;
    prepared.dispatch(device, "hephaestus-map-reduction")?;
    Ok(prepared.into_output())
}

/// Prepared vector dot product over fixed input and output buffers.
pub struct PreparedDot<T> {
    inner: PreparedMapReduction<T>,
}

impl<T> PreparedDot<T> {
    /// Dispatch the prepared dot product once.
    ///
    /// # Errors
    ///
    /// Returns an error if command encoding or submission fails.
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        self.inner.dispatch(device, "hephaestus-prepared-dot")
    }

    /// Return the stable scalar output buffer.
    #[must_use]
    pub fn output(&self) -> &WgpuBuffer<T> {
        self.inner.output()
    }
}

/// Prepare `Σᵢ a[i] * b[i]` over fixed device buffers.
///
/// # Errors
///
/// Returns an error when the operand shapes differ, either layout is invalid
/// for its buffer, or the required GPU resources cannot be allocated.
///
/// # Examples
///
/// ```
/// use hephaestus_wgpu::{
///     ComputeDevice, HephaestusError, StridedOperand, WgpuDevice,
///     prepare_dot, prepare_norm_l2,
/// };
/// use leto::Layout;
///
/// # fn run() -> hephaestus_wgpu::Result<()> {
/// let device = match WgpuDevice::try_default("prepared-reduction-doctest") {
///     Ok(device) => device,
///     Err(HephaestusError::AdapterUnavailable { .. }) => return Ok(()),
///     Err(error) => return Err(error),
/// };
/// let layout = Layout::c_contiguous([2])
///     .expect("invariant: fixed shape is valid");
/// let left = device.upload(&[3.0_f32, 4.0])?;
/// let right = device.upload(&[1.0_f32, 1.0])?;
/// let left = StridedOperand { buffer: &left, layout: &layout };
/// let right = StridedOperand { buffer: &right, layout: &layout };
/// let dot = prepare_dot(&device, left, right)?;
/// let norm = prepare_norm_l2(&device, left)?;
/// dot.dispatch(&device)?;
/// norm.dispatch(&device)?;
/// let mut dot_value = [0.0_f32];
/// let mut norm_value = [0.0_f32];
/// device.download(dot.output(), &mut dot_value)?;
/// device.download(norm.output(), &mut norm_value)?;
/// assert_eq!(dot_value, [7.0]);
/// assert_eq!(norm_value, [5.0]);
/// # Ok(())
/// # }
/// # run()?;
/// # Ok::<(), hephaestus_wgpu::HephaestusError>(())
/// ```
pub fn prepare_dot<T>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<PreparedDot<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    if a.layout.shape != b.layout.shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                a.layout.shape, b.layout.shape
            ),
        });
    }
    Ok(PreparedDot {
        inner: prepare_map_reduction::<DotOp, T, 1>(device, a, b)?,
    })
}

/// Compute the vector dot product `Σᵢ a[i] * b[i]` on the GPU.
pub fn dot<T>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    let prepared = prepare_dot(device, a, b)?;
    prepared.dispatch(device)?;
    Ok(prepared.inner.into_output())
}

/// Compute the trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix on the GPU.
pub fn trace<T>(device: &WgpuDevice, matrix: StridedOperand<'_, T, 2>) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "trace requires a square matrix, got shape {:?}",
                matrix.layout.shape
            ),
        });
    }
    if rows == 0 {
        return device.upload(&[T::IDENTITY]);
    }
    let diag_layout = Layout::new(
        [rows],
        [matrix.layout.strides[0] + matrix.layout.strides[1]],
        matrix.layout.offset,
    );
    let diag = StridedOperand {
        buffer: matrix.buffer,
        layout: &diag_layout,
    };
    map_reduction::<TraceOp, T, 1>(device, diag, diag)
}

/// Compute the L1 norm `Σ |x|` on the GPU.
pub fn norm_l1<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    map_reduction::<NormL1Op, T, N>(device, view, view)
}

/// Prepared L2/Frobenius norm over a fixed input and output allocation.
pub struct PreparedL2Norm<T> {
    squared_sum: PreparedMapReduction<T>,
    sqrt_pipeline: wgpu::ComputePipeline,
    sqrt_bind_group: wgpu::BindGroup,
    output: WgpuBuffer<T>,
}

impl<T> PreparedL2Norm<T> {
    /// Dispatch the fused map, reduction tree, and square root in one command buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if command encoding or submission fails.
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-l2-norm"),
            });
        self.squared_sum.encode(&mut encoder)?;
        encode_compute_pass(
            &mut encoder,
            &self.sqrt_pipeline,
            &self.sqrt_bind_group,
            1,
            "hephaestus-prepared-l2-norm-sqrt",
        );
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    /// Return the stable scalar output buffer.
    #[must_use]
    pub fn output(&self) -> &WgpuBuffer<T> {
        &self.output
    }
}

/// Prepare `sqrt(Σ x²)` over a fixed device buffer.
///
/// # Errors
///
/// Returns an error when the view layout is invalid for its buffer or the
/// required GPU resources cannot be allocated.
pub fn prepare_norm_l2<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<PreparedL2Norm<T>>
where
    T: L2NormScalar,
{
    let squared_sum = prepare_map_reduction::<DotOp, T, N>(device, view, view)?;
    let output = device.alloc_zeroed::<T>(1)?;
    let sqrt_pipeline = unary_pipeline::<SqrtOp, T>(device, BlockWidth::DEFAULT);
    let sqrt_bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-prepared-l2-norm-sqrt"),
            layout: &sqrt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: squared_sum.output().buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });
    Ok(PreparedL2Norm {
        squared_sum,
        sqrt_pipeline,
        sqrt_bind_group,
        output,
    })
}

/// Compute the L2 / Frobenius norm `sqrt(Σ x²)` on the GPU.
pub fn norm_l2<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    T: L2NormScalar,
{
    let prepared = prepare_norm_l2(device, view)?;
    prepared.dispatch(device)?;
    Ok(prepared.output)
}

/// Compute the Max norm `max |x|` on the GPU.
pub fn norm_max<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
{
    map_reduction::<NormMaxOp, T, N>(device, view, view)
}
