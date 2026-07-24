//! Rank-2 prefix/suffix scan kernels over strided matrix operands on ROCm.

use bytemuck::Pod;
use core::marker::PhantomData;
use core::mem::size_of;
use hephaestus_core::{
    AxisScanMeta, BlockWidth, CombineExpr, ComputeDevice, DeviceBuffer, DialectScalar,
    HephaestusError, HipC, IdentityToken, OpIdentity, Result, plan_axis_scan,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

pub use hephaestus_core::{CumProdOp, CumSumOp, ScanDirection};

struct AxisScanKernel<Op>(PhantomData<Op>);

#[inline]
fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {error}"),
    }
}

fn scan_shader_source<Op: CombineExpr<HipC>, T: IdentityToken<Op, HipC>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"
struct AxisScanMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

__device__ unsigned int source_offset(AxisScanMeta meta, unsigned int row, unsigned int col) {{
    int off = (int)meta.offsets[0]
        + (int)row * meta.input_strides[0]
        + (int)col * meta.input_strides[1];
    return (unsigned int)off;
}}

__device__ unsigned int dest_offset(AxisScanMeta meta, unsigned int row, unsigned int col) {{
    int off = (int)meta.offsets[1]
        + (int)row * meta.output_strides[0]
        + (int)col * meta.output_strides[1];
    return (unsigned int)off;
}}

// One block owns one scan line. Each lane folds a contiguous chunk in logical
// order, then lane zero folds chunk totals in order. The second pass applies
// the prefix of preceding chunks to each local prefix. This is the tiled scan
// theorem: every element receives the same mathematical fold as a sequential
// scan for associative `expr`; floating-point reassociation is explicit and
// is covered by the provider's derived error bound.
extern __shared__ {ty} partial[];

extern "C" __global__ void scan_kernel(
    AxisScanMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int line = blockIdx.x;
    unsigned int lane = threadIdx.x;

    unsigned int rows = meta.input_shape[0];
    unsigned int cols = meta.input_shape[1];
    unsigned int axis = meta.offsets[2] & 1u;
    bool reverse = (meta.offsets[2] & 2u) != 0u;
    unsigned int len = (axis == 0u) ? rows : cols;
    unsigned int chunk_len = (len + {wg}u - 1u) / {wg}u;
    unsigned int start = lane * chunk_len;
    unsigned int end = min(start + chunk_len, len);
    {ty} local_acc = {identity};

    // Empty lanes retain the identity. `axis` and `reverse` are uniform
    // across the launch, so only the loop bounds vary by lane.
    for (unsigned int s = start; s < end; s++) {{
        unsigned int idx = reverse ? (len - 1u - s) : s;
        unsigned int row = (axis == 0u) ? idx : line;
        unsigned int col = (axis == 0u) ? line : idx;
        {ty} lhs = local_acc;
        {ty} rhs = input[source_offset(meta, row, col)];
        local_acc = {expr};
        output[dest_offset(meta, row, col)] = local_acc;
    }}
    partial[lane] = local_acc;
    __syncthreads();

    if (lane == 0u) {{
        {ty} prefix = {identity};
        for (unsigned int chunk = 0u; chunk < {wg}u; chunk++) {{
            {ty} total = partial[chunk];
            partial[chunk] = prefix;
            {ty} lhs = prefix;
            {ty} rhs = total;
            prefix = {expr};
        }}
    }}
    __syncthreads();

    {ty} prefix = partial[lane];
    for (unsigned int s = start; s < end; s++) {{
        unsigned int idx = reverse ? (len - 1u - s) : s;
        unsigned int row = (axis == 0u) ? idx : line;
        unsigned int col = (axis == 0u) ? line : idx;
        {ty} lhs = prefix;
        {ty} rhs = output[dest_offset(meta, row, col)];
        output[dest_offset(meta, row, col)] = {expr};
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = T::TOKEN,
        expr = Op::EXPR,
    )
}

/// Scan a rank-2 strided operand along `axis`, preserving its shape.
pub fn scan_axis_into<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    let Some(dispatch) = plan_axis_scan(
        input.layout,
        input.buffer.len(),
        output.layout,
        output.buffer.len(),
        axis,
        direction,
        width,
        input.buffer.aliases(output.buffer),
    )?
    else {
        return Ok(());
    };

    let key = PipelineKey::AxisScan {
        marker: core::any::TypeId::of::<AxisScanKernel<Op>>(),
        scalar: core::any::TypeId::of::<T>(),
        direction,
        axis,
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "scan_kernel", || {
        scan_shader_source::<Op, T>(width)
    })?;

    let mut meta_val = dispatch.meta;
    let mut input_ptr: DevicePtr = input.buffer.raw();
    let mut output_ptr: DevicePtr = output.buffer.raw();
    let mut args: [*mut core::ffi::c_void; 3] = [
        (&mut meta_val as *mut AxisScanMeta).cast(),
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    let shared_bytes = width
        .get()
        .checked_mul(u32::try_from(size_of::<T>()).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: "scan scalar size exceeds ROCm shared-memory address range".to_string(),
            }
        })?)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: "scan shared-memory byte count overflows u32".to_string(),
        })?;

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear_shared(dispatch.groups, width, shared_bytes),
        &mut args,
    )
}

/// Scan a rank-2 strided operand along `axis`, allocating a C-contiguous output.
pub fn scan_axis<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    let len = input.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.alloc_zeroed::<T>(0);
    }
    let output = device.alloc_zeroed::<T>(len)?;
    let output_layout = Layout::c_contiguous(input.layout.shape).map_err(map_layout_err)?;
    scan_axis_into::<Op, T>(
        device,
        input,
        axis,
        direction,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

/// Forward cumulative sum over a rank-2 strided operand along `axis`.
#[inline]
pub fn cumsum_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<CumSumOp> + IdentityToken<CumSumOp, HipC>,
{
    scan_axis_into::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, output, width)
}

/// Forward cumulative sum over a rank-2 strided operand, allocating output.
#[inline]
pub fn cumsum<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<CumSumOp> + IdentityToken<CumSumOp, HipC>,
{
    scan_axis::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, width)
}

/// Reverse cumulative product over a rank-2 strided operand along `axis`.
#[inline]
pub fn cumprod_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<CumProdOp> + IdentityToken<CumProdOp, HipC>,
{
    scan_axis_into::<CumProdOp, T>(device, input, axis, ScanDirection::Reverse, output, width)
}

/// Reverse cumulative product over a rank-2 strided operand, allocating output.
#[inline]
pub fn cumprod<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<CumProdOp> + IdentityToken<CumProdOp, HipC>,
{
    scan_axis::<CumProdOp, T>(device, input, axis, ScanDirection::Reverse, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_declares_tiled_shared_memory_contract() {
        let width = BlockWidth::new(8).expect("non-zero test width");
        let source = scan_shader_source::<CumSumOp, f32>(width);
        assert!(source.contains("extern __shared__ float partial[];"));
        assert!(source.contains("unsigned int line = blockIdx.x;"));
        assert!(source.contains("__syncthreads();"));
        assert!(source.contains("unsigned int chunk_len = (len + 8u - 1u) / 8u;"));
    }
}
