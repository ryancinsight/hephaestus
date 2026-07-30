//! Vector/matrix reductions on the CUDA device: dot product, trace, and norms.
//!
//! Each operation uses one fused strided map-reduction kernel for its first
//! reduction pass. Only the workgroup partials remain as a temporary, so the
//! operation does not materialize an elementwise result of the logical input
//! size before reducing it. Prepared plans use the same first-pass object and
//! retain its partials across dispatches.

use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    IdentityToken, MaxOp, OpIdentity, Result, SumOp,
};
use leto::Layout;

use super::map_layout_err;
use crate::CudaDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::prepared_reduction::PreparedReductionPlan;
use crate::application::strided::{MAX_STRIDED_RANK, StridedOperand, pad_shape, pad_strides};
use crate::infrastructure::buffer::CudaBuffer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MapReductionMeta {
    shape: [u32; 4],
    a_strides: [i32; 4],
    b_strides: [i32; 4],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<MapReductionMeta>() == 64);

pub(crate) trait MapReductionOp: Copy + Send + Sync + 'static {
    type ReduceOp: CombineExpr<CudaC>;

    const EXPR: &'static str;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IdentityMap;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DotMap;

#[derive(Clone, Copy, Debug, Default)]
struct AbsMap;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SquareMap;

#[derive(Clone, Copy, Debug, Default)]
struct MaxAbsMap;

impl MapReductionOp for IdentityMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "lhs";
}

impl MapReductionOp for DotMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "lhs * rhs";
}

impl MapReductionOp for AbsMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "abs(lhs)";
}

impl MapReductionOp for SquareMap {
    type ReduceOp = SumOp;

    const EXPR: &'static str = "lhs * rhs";
}

impl MapReductionOp for MaxAbsMap {
    type ReduceOp = MaxOp;

    const EXPR: &'static str = "abs(lhs)";
}

pub(crate) fn shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: MapReductionOp,
    T: DialectScalar<CudaC> + IdentityToken<Op::ReduceOp, CudaC>,
{
    format!(
        r#"
struct MapReductionMeta {{
    unsigned int shape[4];
    int a_strides[4];
    int b_strides[4];
    unsigned int offsets[4];
}};

extern "C" __global__ void map_reduction_kernel(
    MapReductionMeta meta,
    const {ty}* a,
    const {ty}* b,
    {ty}* output
) {{
    extern __shared__ {ty} shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    {ty} value = {identity};
    if (i < meta.offsets[3]) {{
        unsigned int rem = i;
        int a_off = (int)meta.offsets[0];
        int b_off = (int)meta.offsets[1];
        for (int dimension = 3; dimension >= 0; --dimension) {{
            unsigned int dim = meta.shape[dimension];
            unsigned int index = rem % dim;
            rem /= dim;
            a_off += (int)index * meta.a_strides[dimension];
            b_off += (int)index * meta.b_strides[dimension];
        }}
        {ty} lhs = a[a_off];
        {ty} rhs = b[b_off];
        value = {expr};
    }}
    shared_data[tid] = value;
    __syncthreads();

    for (unsigned int stride = {width}u / 2u; stride > 0u; stride /= 2u) {{
        if (tid < stride) {{
            {ty} lhs = shared_data[tid];
            {ty} rhs = shared_data[tid + stride];
            shared_data[tid] = {reduce};
        }}
        __syncthreads();
    }}

    if (tid == 0u) {{
        output[blockIdx.x] = shared_data[0];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = <T as IdentityToken<Op::ReduceOp, CudaC>>::TOKEN,
        expr = Op::EXPR,
        width = width.get(),
        reduce = <Op::ReduceOp as CombineExpr<CudaC>>::EXPR,
    )
}

pub(crate) fn checked_shared_bytes<T>(width: BlockWidth) -> Result<u32> {
    usize::try_from(width.get())
        .ok()
        .and_then(|width| width.checked_mul(core::mem::size_of::<T>()))
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!(
                "CUDA map-reduction shared-memory size overflows for width {} and element size {}",
                width.get(),
                core::mem::size_of::<T>()
            ),
        })
}

pub(crate) struct PreparedMapReduction<'a, Op, T, const N: usize> {
    device: CudaDevice,
    a: &'a CudaBuffer<T>,
    b: &'a CudaBuffer<T>,
    meta: MapReductionMeta,
    partial: CudaBuffer<T>,
    kernel: Option<Arc<SafeCachedKernel>>,
    groups: u32,
    width: BlockWidth,
    shared_bytes: u32,
    reduction: Option<PreparedReductionPlan<T>>,
    _operation: PhantomData<Op>,
}

impl<Op, T, const N: usize> PreparedMapReduction<'_, Op, T, N>
where
    Op: MapReductionOp,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, CudaC>,
{
    pub(crate) fn dispatch(&self) -> Result<()> {
        if let Some(kernel) = &self.kernel {
            let mut meta = self.meta;
            let mut a_ptr = self.a.raw();
            let mut b_ptr = self.b.raw();
            let mut output_ptr = self.partial.raw();
            let mut args: [*mut core::ffi::c_void; 4] = [
                (&mut meta as *mut MapReductionMeta).cast(),
                (&mut a_ptr as *mut u64).cast(),
                (&mut b_ptr as *mut u64).cast(),
                (&mut output_ptr as *mut u64).cast(),
            ];
            launch_kernel(
                &self.device,
                kernel,
                LaunchConfig::linear_shared(self.groups, self.width, self.shared_bytes),
                &mut args,
            )?;
        }
        if let Some(reduction) = &self.reduction {
            reduction.dispatch(&self.partial)?;
        }
        Ok(())
    }

    pub(crate) fn output(&self) -> &CudaBuffer<T> {
        self.reduction
            .as_ref()
            .map_or(&self.partial, PreparedReductionPlan::output)
    }

    pub(crate) fn into_output(self) -> CudaBuffer<T> {
        match self.reduction {
            Some(reduction) => reduction.into_output(),
            None => self.partial,
        }
    }
}

pub(crate) fn prepare_map_reduction<'a, Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'a, T, N>,
    b: StridedOperand<'a, T, N>,
) -> Result<PreparedMapReduction<'a, Op, T, N>>
where
    Op: MapReductionOp,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, CudaC>,
{
    prepare_map_reduction_with_layouts::<Op, T, N>(device, a.buffer, a.layout, b.buffer, b.layout)
}

pub(crate) fn prepare_map_reduction_with_layouts<'a, Op, T, const N: usize>(
    device: &CudaDevice,
    a: &'a CudaBuffer<T>,
    a_layout: &Layout<N>,
    b: &'a CudaBuffer<T>,
    b_layout: &Layout<N>,
) -> Result<PreparedMapReduction<'a, Op, T, N>>
where
    Op: MapReductionOp,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, CudaC>,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "CUDA strided reductions support rank <= 4"
        );
    }
    let len = a_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(PreparedMapReduction {
            device: device.clone(),
            a,
            b,
            meta: MapReductionMeta {
                shape: [1; 4],
                a_strides: [0; 4],
                b_strides: [0; 4],
                offsets: [0; 4],
            },
            partial: device.upload(&[T::IDENTITY])?,
            kernel: None,
            groups: 0,
            width: BlockWidth::DEFAULT,
            shared_bytes: 0,
            reduction: None,
            _operation: PhantomData,
        });
    }
    a_layout
        .validate_storage_len(a.len())
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.len())
        .map_err(map_layout_err)?;

    let width = BlockWidth::DEFAULT;
    let groups = grid_size(len, width)?;
    let partial_len = usize::try_from(groups).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("CUDA map-reduction group count {groups} exceeds usize range"),
    })?;
    let partial = device.alloc_uninitialized::<T>(partial_len)?;
    let meta = MapReductionMeta {
        shape: pad_shape(a_layout.shape)?,
        a_strides: pad_strides(a_layout.strides)?,
        b_strides: pad_strides(b_layout.strides)?,
        offsets: [
            u32::try_from(a_layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("input offset {} exceeds u32 range", a_layout.offset),
            })?,
            u32::try_from(b_layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("input offset {} exceeds u32 range", b_layout.offset),
            })?,
            0,
            u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("logical size {len} exceeds u32 range"),
            })?,
        ],
    };
    let key = PipelineKey::MapReduction {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "map_reduction_kernel", || {
        shader_source::<Op, T>(width)
    })?;
    let shared_bytes = checked_shared_bytes::<T>(width)?;

    let reduction = if groups > 1 {
        Some(PreparedReductionPlan::prepare::<Op::ReduceOp>(
            device,
            partial_len,
            width,
        )?)
    } else {
        None
    };

    Ok(PreparedMapReduction {
        device: device.clone(),
        a,
        b,
        meta,
        partial,
        kernel: Some(kernel),
        groups,
        width,
        shared_bytes,
        reduction,
        _operation: PhantomData,
    })
}

fn map_reduction<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    Op: MapReductionOp,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, CudaC>,
{
    let prepared = prepare_map_reduction::<Op, T, N>(device, a, b)?;
    prepared.dispatch()?;
    Ok(prepared.into_output())
}

/// Compute the vector dot product `Σᵢ a[i] * b[i]` on the CUDA device.
pub fn dot<T>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    if a.layout.shape != b.layout.shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                a.layout.shape, b.layout.shape
            ),
        });
    }

    map_reduction::<DotMap, T, 1>(device, a, b)
}

/// Compute the trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix on the CUDA device.
pub fn trace<T>(device: &CudaDevice, matrix: StridedOperand<'_, T, 2>) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
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

    let diagonal_stride = matrix.layout.strides[0]
        .checked_add(matrix.layout.strides[1])
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: "trace diagonal stride overflows isize".to_string(),
        })?;
    let diag_layout = Layout::new([rows], [diagonal_stride], matrix.layout.offset);
    let diag_operand = StridedOperand {
        buffer: matrix.buffer,
        layout: &diag_layout,
    };

    map_reduction::<IdentityMap, T, 1>(device, diag_operand, diag_operand)
}

/// Compute the L1 norm `Σ |x|` on the CUDA device.
pub fn norm_l1<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    map_reduction::<AbsMap, T, N>(device, view, view)
}

/// Compute the L2 / Frobenius norm `sqrt(Σ x²)` on the CUDA device.
pub fn norm_l2<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    let squared_sum = map_reduction::<SquareMap, T, N>(device, view, view)?;
    let out = device.alloc_uninitialized::<T>(1)?;
    unary_elementwise_into::<SqrtOp, T>(device, &squared_sum, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}

/// Compute the Max norm `max |x|` on the CUDA device.
pub fn norm_max<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
{
    map_reduction::<MaxAbsMap, T, N>(device, view, view)
}

#[cfg(test)]
mod tests {
    use super::{DotMap, shader_source};
    use hephaestus_core::BlockWidth;

    #[test]
    fn source_declares_strided_map_reduction_contract() {
        let source = shader_source::<DotMap, f32>(BlockWidth::DEFAULT);
        assert!(source.contains("shape[4]"));
        assert!(source.contains("a_strides[4]"));
        assert!(source.contains("b_strides[4]"));
        assert!(source.contains("shared_data"));
        assert!(source.contains("lhs * rhs"));
        assert!(source.contains("__syncthreads();"));
    }
}
