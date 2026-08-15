//! Rank-2 prefix/suffix scans, written once over the [`DeviceApi`] seam.
//!
//! Kernel source is selected by *dialect* rather than by device, because two
//! backends compiling the same language want the same source: CUDA C++ and
//! HIP C++ share [`c_family_axis_scan_source`] verbatim. Host orchestration —
//! validation, cache keying, shared-memory sizing, launch, and the allocating
//! entry points — is selected by nothing at all; it is one implementation
//! monomorphized per backend.

use core::any::TypeId;
use core::marker::PhantomData;
use core::mem::size_of;

use bytemuck::Pod;
use leto::Layout;

use super::device_api::{DeviceApi, LaunchGeometry};
use crate::domain::dialect::{CudaC, DialectScalar, HipC, KernelDialect};
use crate::domain::error::{HephaestusError, Result};
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, CumProdOp, CumSumOp, IdentityToken, OpIdentity};
use crate::domain::scan::{AxisScanMeta, ScanDirection, ScanOps, plan_axis_scan};
use crate::domain::view::StridedView;

/// A kernel dialect that can express a rank-2 axis scan.
///
/// Source generation belongs to the dialect, not the device: every backend
/// compiling HIP C++ wants byte-identical HIP C++. The trait is open, so a
/// backend crate introducing its own dialect supplies its own scan source
/// without an upstream edit.
pub trait AxisScanDialect: KernelDialect {
    /// Emit the scan kernel for combining operator `Op` over scalar `T` at
    /// block width `width`. The entry point is named [`AXIS_SCAN_ENTRY`].
    fn axis_scan_source<Op, T>(width: BlockWidth) -> String
    where
        Op: CombineExpr<Self>,
        T: IdentityToken<Op, Self>;
}

/// Entry-point name every [`AxisScanDialect`] implementation must emit.
pub const AXIS_SCAN_ENTRY: &str = "scan_kernel";

/// C-family (CUDA C++ / HIP C++) axis-scan kernel source.
///
/// One block owns one scan line. Each lane folds a contiguous chunk in
/// logical order, then lane zero folds the chunk totals in order, and a second
/// pass applies the prefix of preceding chunks to each local prefix. This is
/// the tiled scan theorem: for associative `Op::EXPR` every element receives
/// the same mathematical fold as a sequential scan. Floating-point
/// reassociation is explicit and is covered by the provider's derived error
/// bound.
///
/// NVRTC and hipRTC accept the same text here, so both dialects delegate to
/// this function rather than carrying a copy.
#[must_use]
pub fn c_family_axis_scan_source<L, Op, T>(width: BlockWidth) -> String
where
    L: KernelDialect,
    Op: CombineExpr<L>,
    T: IdentityToken<Op, L>,
{
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

extern "C" __global__ void {entry}(
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
        ty = <T as DialectScalar<L>>::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, L>>::TOKEN,
        expr = <Op as CombineExpr<L>>::EXPR,
        entry = AXIS_SCAN_ENTRY,
    )
}

impl AxisScanDialect for CudaC {
    #[inline]
    fn axis_scan_source<Op, T>(width: BlockWidth) -> String
    where
        Op: CombineExpr<Self>,
        T: IdentityToken<Op, Self>,
    {
        c_family_axis_scan_source::<Self, Op, T>(width)
    }
}

impl AxisScanDialect for HipC {
    #[inline]
    fn axis_scan_source<Op, T>(width: BlockWidth) -> String
    where
        Op: CombineExpr<Self>,
        T: IdentityToken<Op, Self>,
    {
        c_family_axis_scan_source::<Self, Op, T>(width)
    }
}

/// Marker distinguishing one combining operator's scan kernels in the cache.
struct AxisScanKernel<Op>(PhantomData<Op>);

/// Pipeline-cache identity of one compiled scan kernel.
///
/// A backend admits generic scans by converting this into its own cache key
/// (`impl From<AxisScanKey> for BackendPipelineKey`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisScanKey {
    /// Identity of the combining operator.
    pub marker: TypeId,
    /// Identity of the element scalar.
    pub scalar: TypeId,
    /// Scan direction.
    pub direction: ScanDirection,
    /// Scanned axis.
    pub axis: usize,
    /// Block width the source was specialized to.
    pub width: u32,
}

/// Kernel, geometry, and metadata for one planned scan launch.
pub struct AxisScanLaunch<D: DeviceApi> {
    kernel: D::Kernel,
    meta: AxisScanMeta,
    geometry: LaunchGeometry,
}

#[inline]
fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {error}"),
    }
}

/// Resolve the kernel and launch geometry for a rank-2 scan, or `None` when
/// the scan is empty.
///
/// # Errors
///
/// Returns an out-of-range axis, a shape mismatch, an aliased output, a
/// layout validation failure, or the kernel compilation failure.
pub fn plan_axis_scan_launch<D, Op, T>(
    device: &D,
    input: StridedView<'_, D::Buffer<T>, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedView<'_, D::Buffer<T>, 2>,
    width: BlockWidth,
) -> Result<Option<AxisScanLaunch<D>>>
where
    D: DeviceApi,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    Op: CombineExpr<D::Dialect>,
    T: Pod + IdentityToken<Op, D::Dialect>,
{
    let Some(dispatch) = plan_axis_scan(
        input.layout,
        crate::DeviceBuffer::len(input.buffer),
        output.layout,
        crate::DeviceBuffer::len(output.buffer),
        axis,
        direction,
        width,
        D::buffers_alias(input.buffer, output.buffer),
    )?
    else {
        return Ok(None);
    };

    let key = AxisScanKey {
        marker: TypeId::of::<AxisScanKernel<Op>>(),
        scalar: TypeId::of::<T>(),
        direction,
        axis,
        width: width.get(),
    };

    let kernel = device.compile_cached(D::CacheKey::from(key), AXIS_SCAN_ENTRY, || {
        <D::Dialect as AxisScanDialect>::axis_scan_source::<Op, T>(width)
    })?;

    let scalar_bytes =
        u32::try_from(size_of::<T>()).map_err(|_| HephaestusError::DispatchFailed {
            message: format!(
                "scan scalar size exceeds {} shared-memory address range",
                device.backend_name()
            ),
        })?;
    let shared_bytes =
        width
            .get()
            .checked_mul(scalar_bytes)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: "scan shared-memory byte count overflows u32".to_string(),
            })?;

    Ok(Some(AxisScanLaunch {
        kernel,
        meta: dispatch.meta,
        geometry: LaunchGeometry::linear_shared(dispatch.groups, width, shared_bytes),
    }))
}

/// Launch a planned scan against device addresses read at call time, so a
/// caller holding buffer borrows observes writes made after planning.
///
/// # Errors
///
/// Returns the backend launch failure.
pub fn launch_planned_axis_scan<D>(
    device: &D,
    plan: &AxisScanLaunch<D>,
    input: D::DevicePtr,
    output: D::DevicePtr,
) -> Result<()>
where
    D: DeviceApi,
{
    // Argument list mirrors `scan_kernel(AxisScanMeta, const T*, T*)`.
    device.launch(&plan.kernel, plan.geometry, &plan.meta, &[input, output])
}

/// Scan a rank-2 strided operand along `axis`, preserving the input shape.
///
/// # Errors
///
/// Returns an out-of-range axis, a shape mismatch, an aliased output, a
/// layout validation failure, or the backend dispatch failure.
pub fn scan_axis_into<D, Op, T>(
    device: &D,
    input: StridedView<'_, D::Buffer<T>, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedView<'_, D::Buffer<T>, 2>,
    width: BlockWidth,
) -> Result<()>
where
    D: DeviceApi,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    Op: CombineExpr<D::Dialect>,
    T: Pod + IdentityToken<Op, D::Dialect>,
{
    let input_ptr = D::device_ptr(input.buffer);
    let output_ptr = D::device_ptr(output.buffer);
    let Some(plan) =
        plan_axis_scan_launch::<D, Op, T>(device, input, axis, direction, output, width)?
    else {
        return Ok(());
    };
    launch_planned_axis_scan(device, &plan, input_ptr, output_ptr)
}

/// Scan a rank-2 strided operand along `axis`, allocating a C-contiguous
/// output buffer.
///
/// # Errors
///
/// Returns an out-of-range axis, a layout validation failure, the allocation
/// failure, or the backend dispatch failure.
pub fn scan_axis<D, Op, T>(
    device: &D,
    input: StridedView<'_, D::Buffer<T>, 2>,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
) -> Result<D::Buffer<T>>
where
    D: DeviceApi,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    Op: CombineExpr<D::Dialect>,
    T: Pod + IdentityToken<Op, D::Dialect>,
{
    let len = input.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.alloc_zeroed::<T>(0);
    }
    let output_layout = Layout::c_contiguous(input.layout.shape()).map_err(map_layout_err)?;
    let output = device.alloc_uninitialized::<T>(len)?;
    scan_axis_into::<D, Op, T>(
        device,
        input,
        axis,
        direction,
        StridedView::new(&output, &output_layout),
        width,
    )?;
    Ok(output)
}

/// Generate the four direction/operator convenience pairs. Each expands to an
/// `_into` form writing a caller-supplied output and an allocating form; both
/// forward to the one generic entry point above with the operator and
/// direction fixed.
macro_rules! scan_convenience {
    ($($into_name:ident, $alloc_name:ident, $op:ty, $direction:expr, $summary:literal;)*) => {
        $(
            #[doc = concat!($summary, " over a rank-2 strided operand along `axis`.")]
            ///
            /// # Errors
            ///
            /// Returns an out-of-range axis, a shape mismatch, an aliased
            /// output, a layout validation failure, or the backend dispatch
            /// failure.
            #[inline]
            pub fn $into_name<D, T>(
                device: &D,
                input: StridedView<'_, D::Buffer<T>, 2>,
                axis: usize,
                output: StridedView<'_, D::Buffer<T>, 2>,
                width: BlockWidth,
            ) -> Result<()>
            where
                D: DeviceApi,
                D::CacheKey: From<AxisScanKey>,
                D::Dialect: AxisScanDialect,
                $op: CombineExpr<D::Dialect>,
                T: Pod + IdentityToken<$op, D::Dialect>,
            {
                scan_axis_into::<D, $op, T>(device, input, axis, $direction, output, width)
            }

            #[doc = concat!($summary, " over a rank-2 strided operand, allocating a C-contiguous output buffer.")]
            ///
            /// # Errors
            ///
            /// Returns an out-of-range axis, a layout validation failure, the
            /// allocation failure, or the backend dispatch failure.
            #[inline]
            pub fn $alloc_name<D, T>(
                device: &D,
                input: StridedView<'_, D::Buffer<T>, 2>,
                axis: usize,
                width: BlockWidth,
            ) -> Result<D::Buffer<T>>
            where
                D: DeviceApi,
                D::CacheKey: From<AxisScanKey>,
                D::Dialect: AxisScanDialect,
                $op: CombineExpr<D::Dialect>,
                T: Pod + IdentityToken<$op, D::Dialect>,
            {
                scan_axis::<D, $op, T>(device, input, axis, $direction, width)
            }
        )*
    };
}

scan_convenience! {
    cumsum_into, cumsum, CumSumOp, ScanDirection::Forward, "Forward cumulative sum";
    suffix_sum_into, suffix_sum, CumSumOp, ScanDirection::Reverse, "Reverse cumulative sum";
    cumprod_into, cumprod, CumProdOp, ScanDirection::Forward, "Forward cumulative product";
    suffix_prod_into, suffix_prod, CumProdOp, ScanDirection::Reverse, "Reverse cumulative product";
}

/// The [`ScanOps`] seam implemented once for every [`DeviceApi`] backend.
///
/// A zero-sized marker parameterized by the device it scans, so a bound of
/// `S: ScanOps<D, T>` still monomorphizes to that backend's own kernel
/// dispatch at no runtime cost.
pub struct AxisScanOps<D>(PhantomData<fn() -> D>);

// Hand-written rather than derived: a derive would bound each impl on `D`,
// but the marker holds no `D` value and every operation is available for any
// device.
impl<D> Clone for AxisScanOps<D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for AxisScanOps<D> {}

impl<D> Default for AxisScanOps<D> {
    #[inline]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D> core::fmt::Debug for AxisScanOps<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("AxisScanOps")
    }
}

impl<D> AxisScanOps<D> {
    /// The scan seam for device `D`.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// A scan bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound operands between dispatches are observed (the seam's
/// rebind contract). An empty scan prepares to a no-op.
pub struct PreparedAxisScan<'op, D: DeviceApi, T: Pod> {
    input: &'op D::Buffer<T>,
    output: &'op D::Buffer<T>,
    plan: Option<AxisScanLaunch<D>>,
}

// `D: 'static` is required because a prepared scan borrows the device's own
// buffers for `'op`; a GPU device handle owns refcounted context state and is
// always `'static`, and `T: Pod` already implies it for the scalar.
impl<D, T> ScanOps<D, T> for AxisScanOps<D>
where
    D: DeviceApi + 'static,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    T: Pod + DialectScalar<D::Dialect>,
{
    type Dialect = D::Dialect;
    type PreparedScan<'op, const N: usize>
        = PreparedAxisScan<'op, D, T>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, D::Buffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{} scan supports rank 2 only, got rank {N}",
                    device.backend_name()
                ),
            });
        }
        // The rank guard above proves N == 2, so the rank-2 components are
        // total; rebuilding a Layout<2> avoids reinterpreting &Layout<N> and
        // keeps the featureless build under forbid(unsafe_code).
        let input_layout = Layout::try_new(
            [input.layout.shape()[0], input.layout.shape()[1]],
            [input.layout.strides()[0], input.layout.strides()[1]],
            input.layout.offset(),
        )
        .expect("invariant: rank guard proves the rank-2 components are total");
        let output_layout = Layout::try_new(
            [output.layout.shape()[0], output.layout.shape()[1]],
            [output.layout.strides()[0], output.layout.strides()[1]],
            output.layout.offset(),
        )
        .expect("invariant: rank guard proves the rank-2 components are total");
        let plan = plan_axis_scan_launch::<D, Op, T>(
            device,
            StridedView::new(input.buffer, &input_layout),
            axis,
            direction,
            StridedView::new(output.buffer, &output_layout),
            BlockWidth::DEFAULT,
        )?;
        Ok(PreparedAxisScan {
            input: input.buffer,
            output: output.buffer,
            plan,
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        let Some(plan) = &prepared.plan else {
            return Ok(());
        };
        launch_planned_axis_scan(
            device,
            plan,
            D::device_ptr(prepared.input),
            D::device_ptr(prepared.output),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_family_source_declares_the_tiled_shared_memory_contract() {
        let width = BlockWidth::new(8).expect("non-zero test width");
        let source = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(width);
        assert!(source.contains("extern __shared__ float partial[];"));
        assert!(source.contains("unsigned int line = blockIdx.x;"));
        assert!(source.contains("__syncthreads();"));
        assert!(source.contains("unsigned int chunk_len = (len + 8u - 1u) / 8u;"));
        assert!(source.contains("void scan_kernel("));
    }

    #[test]
    fn cuda_and_hip_scan_sources_are_identical() {
        // Both dialects compile the same C++; the shared emitter is why
        // neither backend carries a copy of this template.
        let width = BlockWidth::new(64).expect("non-zero test width");
        assert_eq!(
            <CudaC as AxisScanDialect>::axis_scan_source::<CumSumOp, f32>(width),
            <HipC as AxisScanDialect>::axis_scan_source::<CumSumOp, f32>(width)
        );
    }

    #[test]
    fn source_is_sensitive_to_operator_and_scalar() {
        let width = BlockWidth::new(32).expect("non-zero test width");
        let sum = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(width);
        let prod = c_family_axis_scan_source::<CudaC, CumProdOp, f32>(width);
        let sum_u32 = c_family_axis_scan_source::<CudaC, CumSumOp, u32>(width);

        assert!(sum.contains(<CumSumOp as CombineExpr<CudaC>>::EXPR));
        assert!(prod.contains(<CumProdOp as CombineExpr<CudaC>>::EXPR));
        assert_ne!(sum, prod);
        assert!(sum.contains("float partial[]"));
        assert!(sum_u32.contains("unsigned int partial[]"));
        assert_ne!(sum, sum_u32);
    }

    #[test]
    fn width_specializes_the_emitted_chunking() {
        let narrow = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(
            BlockWidth::new(16).expect("non-zero test width"),
        );
        let wide = c_family_axis_scan_source::<CudaC, CumSumOp, f32>(
            BlockWidth::new(256).expect("non-zero test width"),
        );
        assert!(narrow.contains("(len + 16u - 1u) / 16u"));
        assert!(wide.contains("(len + 256u - 1u) / 256u"));
    }
}
