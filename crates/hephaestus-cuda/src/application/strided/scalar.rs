//! Scalar strided entry points.

use super::launch::launch_scalar_strided;
use super::meta::scalar_strided_meta;
use super::types::StridedOperand;
use super::{MAX_STRIDED_RANK, map_layout_err};
use crate::CudaDevice;
use crate::infrastructure::buffer::CudaBuffer;
use eunomia::Pod;
use hephaestus_core::{BinaryExpr, BlockWidth, ComputeDevice, CudaC, DialectScalar, Result};
use leto::Layout;

/// Run `out[idx] = op(a[idx], scalar)` over logical indices of `out_layout`.
pub fn scalar_elementwise_strided_into<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    out: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let Some((meta, len)) = scalar_strided_meta(&a, &out)? else {
        return Ok(());
    };
    launch_scalar_strided::<Op, T>(device, a.buffer, scalar, out.buffer, meta, width, len)
}

/// Run `out = op(a, scalar)` over `output_shape`, allocating a C-contiguous output buffer.
///
/// The scalar path delegates through [`scalar_elementwise_strided_into`], passing
/// the scalar as a kernel argument instead of allocating a one-element device
/// buffer.
pub fn scalar_elementwise_strided<Op, T, const N: usize>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, N>,
    scalar: T,
    output_shape: [usize; N],
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "strided dispatch exceeds metadata capacity"
        );
    }

    let out_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(len)?;
    scalar_elementwise_strided_into::<Op, T, N>(
        device,
        a,
        scalar,
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
        width,
    )?;
    Ok(out)
}
