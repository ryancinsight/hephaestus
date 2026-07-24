//! Strided dot products, traces, and norms on ROCm.
//!
//! One HIP workgroup maps each logical input element and reduces its mapped
//! value in shared memory. The host packs leto::Layout metadata into a
//! rank-four representation, so transposed, sliced, and diagonal views use
//! the same device path as contiguous inputs.

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DialectScalar, HephaestusError, HipC, IdentityToken,
    MaxOp, OpIdentity, Result, SumOp,
};
use leto::Layout;

use super::map_layout_err;
use crate::RocmDevice;
use crate::application::elementwise::{SqrtOp, unary_elementwise_into};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::application::reduction::reduction;
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

const MAX_STRIDED_RANK: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MapReductionMeta {
    shape: [u32; 4],
    a_strides: [i32; 4],
    b_strides: [i32; 4],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<MapReductionMeta>() == 48);

trait MapReductionOp: Copy + Send + Sync + 'static {
    type ReduceOp: CombineExpr<HipC>;

    const EXPR: &'static str;
}

#[derive(Clone, Copy, Debug, Default)]
struct IdentityMap;

#[derive(Clone, Copy, Debug, Default)]
struct DotMap;

#[derive(Clone, Copy, Debug, Default)]
struct AbsMap;

#[derive(Clone, Copy, Debug, Default)]
struct SquareMap;

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

fn shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + IdentityToken<Op::ReduceOp, HipC>,
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
        for (int dimension = 3; dimension >= 0; dimension--) {{
            unsigned int dim = meta.shape[dimension];
            unsigned int index = rem % dim;
            rem = rem / dim;
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
        identity = <T as IdentityToken<Op::ReduceOp, HipC>>::TOKEN,
        expr = Op::EXPR,
        width = width.get(),
        reduce = <Op::ReduceOp as CombineExpr<HipC>>::EXPR,
    )
}

fn checked_shared_bytes<T>(width: BlockWidth) -> Result<u32> {
    let width = usize::try_from(width.get()).map_err(|_| HephaestusError::DispatchFailed {
        message: format!(
            "ROCm map-reduction width {} exceeds usize range",
            width.get()
        ),
    })?;
    width
        .checked_mul(core::mem::size_of::<T>())
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!(
                "ROCm map-reduction shared-memory size overflows for width {} and element size {}",
                width,
                core::mem::size_of::<T>()
            ),
        })
}

fn map_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 4]> {
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "ROCm strided reductions support rank <= 4"
        );
    }
    let mut mapped = [1_u32; 4];
    for (dimension, &extent) in shape.iter().enumerate() {
        mapped[4 - N + dimension] =
            u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dimension {extent} exceeds u32 range"),
            })?;
    }
    Ok(mapped)
}

fn map_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 4]> {
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "ROCm strided reductions support rank <= 4"
        );
    }
    let mut mapped = [0_i32; 4];
    for (dimension, &stride) in strides.iter().enumerate() {
        mapped[4 - N + dimension] =
            i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("stride {stride} exceeds i32 range"),
            })?;
    }
    Ok(mapped)
}

fn map_reduction<Op, T, const N: usize>(
    device: &RocmDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, HipC>,
{
    let len = a.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }
    a.layout
        .validate_storage_len(a.buffer.len())
        .map_err(map_layout_err)?;
    b.layout
        .validate_storage_len(b.buffer.len())
        .map_err(map_layout_err)?;

    let width = BlockWidth::DEFAULT;
    let groups = grid_size(len, width)?;
    let partial_len = usize::try_from(groups).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("ROCm map-reduction group count {groups} exceeds usize range"),
    })?;
    let partial = device.alloc_zeroed::<T>(partial_len)?;
    let meta = MapReductionMeta {
        shape: map_shape(a.layout.shape)?,
        a_strides: map_strides(a.layout.strides)?,
        b_strides: map_strides(b.layout.strides)?,
        offsets: [
            u32::try_from(a.layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("input offset {} exceeds u32 range", a.layout.offset),
            })?,
            u32::try_from(b.layout.offset).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("input offset {} exceeds u32 range", b.layout.offset),
            })?,
            0,
            u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("logical size {len} exceeds u32 range"),
            })?,
        ],
    };
    let shared_bytes = checked_shared_bytes::<T>(width)?;
    let key = PipelineKey::MapReduction {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "map_reduction_kernel", || {
        shader_source::<Op, T>(width)
    })?;

    let mut meta = meta;
    let mut a_ptr: DevicePtr = a.buffer.raw();
    let mut b_ptr: DevicePtr = b.buffer.raw();
    let mut output_ptr: DevicePtr = partial.raw();
    let mut args: [*mut core::ffi::c_void; 4] = [
        (&mut meta as *mut MapReductionMeta).cast(),
        (&mut a_ptr as *mut DevicePtr).cast(),
        (&mut b_ptr as *mut DevicePtr).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear_shared(groups, width, shared_bytes),
        &mut args,
    )?;

    if groups == 1 {
        Ok(partial)
    } else {
        reduction::<Op::ReduceOp, T>(device, &partial)
    }
}

/// Compute the vector dot product Σᵢ a[i] * b[i] on the ROCm device.
///
/// The two rank-1 operands must have equal logical shapes. Layout validation
/// occurs before the HIP launch, and the returned buffer contains one scalar.
pub fn dot<T>(
    device: &RocmDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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

/// Compute the trace tr(A) = Σᵢ aᵢᵢ of a square matrix on ROCm.
pub fn trace<T>(device: &RocmDevice, matrix: StridedOperand<'_, T, 2>) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
    let diagonal_layout = Layout::new([rows], [diagonal_stride], matrix.layout.offset);
    let diagonal = StridedOperand {
        buffer: matrix.buffer,
        layout: &diagonal_layout,
    };
    map_reduction::<IdentityMap, T, 1>(device, diagonal, diagonal)
}

/// Compute the entrywise L1 norm Σ |x| on ROCm.
pub fn norm_l1<T, const N: usize>(
    device: &RocmDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    map_reduction::<AbsMap, T, N>(device, view, view)
}

/// Scalar types supported by ROCm L2/Frobenius norms.
pub trait L2NormScalar:
    DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>
{
}

impl L2NormScalar for f32 {}

/// Compute the L2/Frobenius norm sqrt(Σ x²) on ROCm.
pub fn norm_l2<T, const N: usize>(
    device: &RocmDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    T: L2NormScalar,
{
    let squared_sum = map_reduction::<SquareMap, T, N>(device, view, view)?;
    let output = device.alloc_zeroed::<T>(1)?;
    unary_elementwise_into::<SqrtOp, T>(device, &squared_sum, &output, BlockWidth::DEFAULT)?;
    Ok(output)
}

/// Compute the max-magnitude norm max |x| on ROCm.
pub fn norm_max<T, const N: usize>(
    device: &RocmDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    map_reduction::<MaxAbsMap, T, N>(device, view, view)
}

#[cfg(test)]
mod tests {
    use super::{DotMap, shader_source};
    use hephaestus_core::BlockWidth;

    #[test]
    fn source_declares_strided_map_reduction_contract() {
        let source = shader_source::<DotMap, i32>(BlockWidth::DEFAULT);
        assert!(source.contains("shape[4]"));
        assert!(source.contains("a_strides[4]"));
        assert!(source.contains("b_strides[4]"));
        assert!(source.contains("shared_data"));
        assert!(source.contains("lhs * rhs"));
        assert!(source.contains("__syncthreads();"));
    }
}
