//! Batched dispatch for prepared sparse WGPU operations.

use super::{PreparedSpmm, PreparedSpmv};
use bytemuck::Pod;
use hephaestus_core::{DialectScalar, HephaestusError, Result, Wgsl};

/// A prepared sparse operation that can be recorded into a shared command
/// encoder.
///
/// The scalar type remains monomorphized; the closed operation set uses enum
/// dispatch instead of a vtable on the repeated-dispatch path.
pub enum PreparedSparseDispatch<'a, T> {
    /// Prepared CSR matrix-vector product.
    Spmv(&'a PreparedSpmv<T>),
    /// Prepared CSR matrix-matrix product.
    Spmm(&'a PreparedSpmm<T>),
}

impl<T> PreparedSparseDispatch<'_, T> {
    fn device(&self) -> &crate::infrastructure::device::WgpuDevice {
        match self {
            Self::Spmv(op) => op.device(),
            Self::Spmm(op) => op.device(),
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Spmv(op) => op.encode(encoder),
            Self::Spmm(op) => op.encode(encoder),
        }
    }
}

/// Submit multiple prepared sparse operations through one command buffer.
///
/// This amortizes WGPU queue submission overhead for repeated tiny sparse
/// products while preserving each operation's prepared bind group and metadata.
///
/// # Errors
///
/// Returns [`HephaestusError::DispatchFailed`] when the batch contains prepared
/// operations from different WGPU devices.
pub fn submit_prepared_sparse_batch<T: DialectScalar<Wgsl> + Pod>(
    operations: &[PreparedSparseDispatch<'_, T>],
) -> Result<()> {
    let Some((first, rest)) = operations.split_first() else {
        return Ok(());
    };

    let device = first.device();
    if rest
        .iter()
        .any(|op| !std::ptr::eq(op.device().inner(), device.inner()))
    {
        return Err(HephaestusError::DispatchFailed {
            message: "prepared sparse batch contains operations from different WGPU devices"
                .to_string(),
        });
    }

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-sparse-batch"),
        });
    for operation in operations {
        operation.encode(&mut encoder);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
