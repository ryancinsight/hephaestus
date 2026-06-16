//! Vector/matrix reductions on the CUDA device: dot product, trace, and norms.
//!
//! Unlike [`matmul`](super::matmul), these author no bespoke kernel — each is a
//! composition of the elementwise (map a strided view) and reduction (tree
//! reduce) primitives, so they inherit every optimization of those primitives
//! and stay correct for any strided/broadcast layout.

use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};
use leto::Layout;

use super::map_layout_err;
use crate::application::cuda_type::CudaScalar;
use crate::application::elementwise::{unary_elementwise_into, AbsOp, IdentityOp, MulOp, SqrtOp};
use crate::application::reduction::{reduction, MaxOp, ReductionIdentity, SumOp};
use crate::application::strided::{
    binary_elementwise_strided_into, unary_elementwise_strided_into, StridedOperand,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

/// Compute the vector dot product `Σᵢ a[i] * b[i]` on the CUDA device.
pub fn dot<T>(
    device: &CudaDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
{
    if a.layout.shape != b.layout.shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                a.layout.shape, b.layout.shape
            ),
        });
    }

    let len = a.layout.shape[0];
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_prod = device.alloc_zeroed::<T>(len)?;
    let temp_prod_layout = Layout::c_contiguous([len]).map_err(map_layout_err)?;
    let temp_prod_operand = StridedOperand {
        buffer: &temp_prod,
        layout: &temp_prod_layout,
    };

    binary_elementwise_strided_into::<MulOp, T, 1>(
        device,
        a,
        b,
        temp_prod_operand,
        BlockWidth::DEFAULT,
    )?;

    reduction::<SumOp, T>(device, &temp_prod)
}

/// Compute the trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix on the CUDA device.
pub fn trace<T>(device: &CudaDevice, matrix: StridedOperand<'_, T, 2>) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
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

    let s0 = matrix.layout.strides[0];
    let s1 = matrix.layout.strides[1];
    let diag_layout = Layout::new([rows], [s0 + s1], matrix.layout.offset);
    let diag_operand = StridedOperand {
        buffer: matrix.buffer,
        layout: &diag_layout,
    };

    let temp_diag = device.alloc_zeroed::<T>(rows)?;
    let temp_diag_layout = Layout::c_contiguous([rows]).map_err(map_layout_err)?;
    let temp_diag_operand = StridedOperand {
        buffer: &temp_diag,
        layout: &temp_diag_layout,
    };

    unary_elementwise_strided_into::<IdentityOp, T, 1>(
        device,
        diag_operand,
        temp_diag_operand,
        BlockWidth::DEFAULT,
    )?;

    reduction::<SumOp, T>(device, &temp_diag)
}

/// Compute the L1 norm `Σ |x|` on the CUDA device.
pub fn norm_l1<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
{
    let len = view.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_abs = device.alloc_zeroed::<T>(len)?;
    let temp_abs_layout = Layout::c_contiguous(view.layout.shape).map_err(map_layout_err)?;
    let temp_abs_operand = StridedOperand {
        buffer: &temp_abs,
        layout: &temp_abs_layout,
    };

    unary_elementwise_strided_into::<AbsOp, T, N>(
        device,
        view,
        temp_abs_operand,
        BlockWidth::DEFAULT,
    )?;

    reduction::<SumOp, T>(device, &temp_abs)
}

/// Compute the L2 / Frobenius norm `sqrt(Σ x²)` on the CUDA device.
pub fn norm_l2<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
{
    let len = view.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_sq = device.alloc_zeroed::<T>(len)?;
    let temp_sq_layout = Layout::c_contiguous(view.layout.shape).map_err(map_layout_err)?;
    let temp_sq_operand = StridedOperand {
        buffer: &temp_sq,
        layout: &temp_sq_layout,
    };

    binary_elementwise_strided_into::<MulOp, T, N>(
        device,
        view,
        view,
        temp_sq_operand,
        BlockWidth::DEFAULT,
    )?;

    let squared_sum = reduction::<SumOp, T>(device, &temp_sq)?;
    let out = device.alloc_zeroed::<T>(1)?;
    unary_elementwise_into::<SqrtOp, T>(device, &squared_sum, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}

/// Compute the Max norm `max |x|` on the CUDA device.
pub fn norm_max<T, const N: usize>(
    device: &CudaDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<MaxOp>,
{
    let len = view.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_abs = device.alloc_zeroed::<T>(len)?;
    let temp_abs_layout = Layout::c_contiguous(view.layout.shape).map_err(map_layout_err)?;
    let temp_abs_operand = StridedOperand {
        buffer: &temp_abs,
        layout: &temp_abs_layout,
    };

    unary_elementwise_strided_into::<AbsOp, T, N>(
        device,
        view,
        temp_abs_operand,
        BlockWidth::DEFAULT,
    )?;

    reduction::<MaxOp, T>(device, &temp_abs)
}
